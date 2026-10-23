// Copyright (c) 2026, Lu Lu

// Execute the production Dispatch, transport completion, and memory layout.
// Only device primitives and the metadata pack/relay entry points are
// doubled. This validates byte ranges and protocol order, not NIC visibility.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>
#include "kernel_operator.h"
#include "../csrc/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc/kernels/mega_moe/common/mega_moe_roce.hpp"

#define MEGA_MOE_SEND_MASK_H
constexpr int AIC = 0;
constexpr int AIV = 1;
constexpr int g_coreType = AIV;

namespace MegaMoeImpl {
using namespace AscendC;
struct MoeStageCommonConfig {
    uint32_t rankId;
    uint32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
    uint32_t moeExpertPerRank;
};
struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};
struct Params {
    MegaMoeTilingData* tilingData;
    PeermemInfo peermemInfo;
};
struct SendMaskConfig {
    uint64_t railTopkIdsSlotBytes;
};
struct SendMaskScratch {};
inline GM_ADDR g_winRankAddr_[32]{};

inline int32_t* HierarchicalSignalAddress(const PeermemInfo& info, HierarchicalSignal signal)
{
    return reinterpret_cast<int32_t*>(info.hierarchicalControlPtr + HierarchicalSignalOffset(signal));
}
inline void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    using namespace DispatchTest;
    const auto offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(value));
    if (offset == relayReady || offset == metaReady) {
        Check(RoceTest::blockIndex == static_cast<int>(receiveLeaderJob),
              "readiness was not published by the receive leader");
        Check(waits >= 1U && RoceTest::mteCanRead && ReadWord(local, payloadCommit) == sequence,
              "metadata/payload readiness preceded the unified receive acquire");
    }
    if (offset == txDone) {
        Check(queued.empty() && ReadWord(remote, payloadCommit) == sequence,
              "TxDone preceded outgoing payload completion");
    }
    std::memcpy(address, &value, sizeof(value));
}
inline int32_t ReadGmByPassDCache(const int32_t* address)
{
    int32_t value;
    std::memcpy(&value, address, sizeof(value));
    return value;
}
inline void WaitUntilGmFlagEquals(int32_t* address, int32_t value)
{
    if (DispatchTest::completePacking) {
        auto pack = std::move(DispatchTest::completePacking);
        DispatchTest::completePacking = nullptr;
        pack();
    }
    DispatchTest::Check(ReadGmByPassDCache(address) == value, "Dispatch skipped local rail readiness");
    RoceTest::signalObserved = true;
}
template <HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    AscendC::SetFlag<Event>(Id);
    AscendC::WaitFlag<Event>(Id);
}

// Rail bytes are prepared by the fixture; validate the actual packer's identity,
// destination and publication order when production Dispatch invokes it.
inline void SendTopkIdsToRailServer(const MoeStageCommonConfig& common, GM_ADDR*, const SendMaskConfig&,
                                    const Params& params, SendMaskScratch&, uint32_t targetServerIdx, uint32_t jobIdx)
{
    using namespace DispatchTest;
    Check(RoceTest::blockIndex == 0 && jobIdx == 0U, "metadata packer/epoch owner is not physical AIV0");
    Check(targetServerIdx == HierarchicalHost(common.rankId), "packer used its AIV index as the host rail slot");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence))) == sequence,
          "packer started before the current sequence was published");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::LocalRailPacked))) != sequence,
          "rail readiness was published before packing finished");
    ++packCalls;
}
inline void CalcRailRelayExpertRange(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t& first, uint32_t& count)
{
    DispatchTest::Check(RoceTest::blockIndex != 1, "sender unexpectedly assigned relay work");
    first = 0U;
    count = 1U;
}
inline void RelayRailTopkIdsToExperts(const MoeStageCommonConfig&, GM_ADDR*, const SendMaskConfig&, const Params&,
                                      SendMaskScratch&, uint32_t, uint32_t, int32_t, int32_t)
{
    using namespace DispatchTest;
    Check(RoceTest::blockIndex != 1 && ReadWord(local, metaReady) == sequence && RoceTest::mteCanRead,
          "metadata relay preceded the unified receive acquire");
    ++relayCalls;
}
inline void GmSignalWaitBarrier(int32_t*, int32_t)
{
    throw std::runtime_error("Dispatch unexpectedly entered final host barrier");
}
template <bool>
void SyncAll()
{
    throw std::runtime_error("Dispatch unexpectedly synchronized all cores");
}
}  // namespace MegaMoeImpl

#include "../csrc/kernels/mega_moe/stage/mega_moe_hierarchical_dispatch.hpp"

using namespace MegaMoeImpl;

void CheckDispatch(uint32_t rank, uint32_t bs, uint32_t capacity, uint32_t hidden, uint32_t topk,
                   std::string_view pattern)
{
    using namespace DispatchTest;
    MegaMoeTilingData t{};
    t.bs = bs;
    t.numMaxTokensPerRank = capacity;
    t.h = hidden;
    t.topK = topk;
    t.moeExpertPerRank = 12U;
    t.epWorldSize = 32U;
    t.dispatchRanksPerNode = 16U;
    t.blockAivNum = 56U;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.roceScratchUbOffset = kScratchOffset;
    t.hierarchicalSequence = 257U;
    Check(UseDecodeRailRouting(&t), "fixture requires the production decode rail layout");
    const PeermemSizeParams sizes{capacity,           topk, hidden, 12U, 32U, 2U, 1U, false, false,
                                  TOPO_TYPE_MTE_ROCE, true, 16U};
    const auto heapBytes = static_cast<size_t>(CalcPeermemLeastSize(sizes));
    local.assign(heapBytes, 0x7BU);
    remote.assign(heapBytes, 0xA5U);
    posts.clear();
    queued.clear();
    quiets = waits = packCalls = relayCalls = 0U;
    peer = HierarchicalGateway(rank);
    sequence = t.hierarchicalSequence;
    receiveLeaderJob = 2U + (1U - HierarchicalHost(rank));
    completePacking = nullptr;
    RoceTest::operations.clear();
    RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    Params params{&t, PeermemInfo(local.data(), &t)};
    MoeStageCommonConfig common{rank, bs, topk, hidden, 12U};
    SendMaskConfig config{static_cast<uint64_t>(CalcRailTopkIdsSlotBytes(&t))};
    SendMaskScratch scratch;
    const auto flagOffset = [&](HierarchicalSignal signal) {
        return Offset(reinterpret_cast<uint8_t*>(HierarchicalSignalAddress(params.peermemInfo, signal)), 4U);
    };
    metaCommit = flagOffset(HierarchicalSignal::NetMetaCommit);
    payloadCommit = flagOffset(HierarchicalSignal::NetPayloadCommit);
    metaReady = flagOffset(HierarchicalSignal::MetaReady);
    relayReady = flagOffset(HierarchicalSignal::RelayPayloadReady);
    txDone = flagOffset(HierarchicalSignal::TxDone);
    // A stale unified commit must not be satisfied by an old split notification.
    WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit),
                        sequence - 1U);
    WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit), sequence);

    const uint32_t rowBytes = static_cast<uint32_t>(CalcQuantTokenScaleBytes(hidden, 1U, topk, false));
    const uint32_t activeBytes = bs * rowBytes;
    const uint32_t source = Offset(params.peermemInfo.quantTokenScalePtr, activeBytes);
    const uint32_t destination = Offset(params.peermemInfo.relayPayloadPtr, activeBytes);
    Check(params.peermemInfo.quantTokenScalePtr + activeBytes <= params.peermemInfo.combineSendPtr &&
              params.peermemInfo.relayPayloadPtr + activeBytes <= params.peermemInfo.gatewayCombinePtr,
          "actual batch exceeds its production allocation");
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t byte = 0U; byte < rowBytes; ++byte) {
            local[source + token * rowBytes + byte] = static_cast<uint8_t>((token * 37U + byte * 13U + rank) % 251U);
        }
    }
    const auto payload = std::vector<uint8_t>(local.begin() + source, local.begin() + source + activeBytes);
    const uint32_t railBytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(bs, topk));
    const uint32_t rail =
        Offset(params.peermemInfo.railTopkIdsRecvPtr + HierarchicalHost(rank) * config.railTopkIdsSlotBytes, railBytes);
    // Use real rail packet positions and both zero-remote and sparse routes.
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t slot = 0U; slot < topk; ++slot) {
            const bool other = pattern == "remote" || (pattern == "mixed" && (token + slot) % 3U == 0U) ||
                               (pattern == "last" && token + 1U == bs && slot + 1U == topk);
            const uint32_t targetHost = HierarchicalHost(rank) ^ static_cast<uint32_t>(other);
            const int32_t expert = static_cast<int32_t>((targetHost * 16U + (token + slot) % 16U) * 12U + slot % 12U);
            const uint32_t item = token * topk + slot;
            const uint32_t packetWord = (item / 127U) * 128U + item % 127U;
            std::memcpy(local.data() + rail + packetWord * 4U, &expert, sizeof(expert));
        }
    }
    auto expected = remote;
    for (uint32_t job = 0U; job < 56U; ++job) {
        if (job == 1U) continue;
        RoceTest::blockIndex = static_cast<int>(job);
        RoceTest::subBlockIndex = static_cast<int>(job % 2U);
        FinishHierarchicalDispatch({job, 56U}, common, params);
    }
    Check(posts.empty() && quiets == 0U && waits == 0U, "packer/nonowner touched the QP");
    RoceTest::blockIndex = RoceTest::subBlockIndex = 1;
    const AivJobContext owner{1U, 56U};
    // Schedule the packer only when the sender reaches the rail wait. Payload
    // must already be in flight, while metadata must still be unsubmitted.
    completePacking = [&] {
        Check(posts.size() == 1U && queued.size() == 1U && quiets == 0U && packCalls == 0U,
              "payload did not overlap unfinished metadata packing");
        Check(queued[0].source == source && queued[0].destination == destination && queued[0].bytes == activeBytes,
              "the first submission was not the actual-batch payload");
        RoceTest::blockIndex = RoceTest::subBlockIndex = 0;
        GatherAndPostHierarchicalDispatch({0U, 56U}, common, params, nullptr, config, scratch);
        Check(ReadWord(local, flagOffset(HierarchicalSignal::LocalRailPacked)) == sequence,
              "AIV0 did not finish packing and publish readiness");
        RoceTest::blockIndex = RoceTest::subBlockIndex = 1;
    };
    GatherAndPostHierarchicalDispatch(owner, common, params, nullptr, config, scratch);
    Check(packCalls == 1U, "AIV1 repeated AIV0's packing work");
    Check(remote == expected, "Dispatch data completed before the shared quiet");
    Check(posts.size() == 2U && queued.size() == 2U && quiets == 0U && waits == 0U,
          "Dispatch did not defer both data Puts to one completion");
    Check(queued[1].source == rail && queued[1].destination == rail && queued[1].bytes == railBytes,
          "Dispatch did not submit packed metadata after the rail acquire");
    Check((RoceTest::operations == std::vector<int>{1, 1}), "Dispatch inserted completion between the two data Puts");
    const AivJobContext receiver{2U + (1U - HierarchicalHost(rank)), 56U};
    RoceTest::blockIndex = static_cast<int>(receiver.jobIndex);
    RoceTest::subBlockIndex = static_cast<int>(receiver.jobIndex % 2U);
    RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    bool staleRejected = false;
    try {
        GatherAndPostHierarchicalDispatch(receiver, common, params, nullptr, config, scratch);
    } catch (const std::runtime_error&) {
        staleRejected = true;
    }
    Check(staleRejected && relayCalls == 0U && ReadWord(local, metaReady) != sequence &&
              ReadWord(local, relayReady) != sequence,
          "split metadata flag incorrectly released the unified receiver");
    // Incoming completion can arrive while our outgoing data is still in flight.
    // The independent receive leader must not touch the sender's QP/CQ or wait
    // for its local TxDone; this permits the sender to reach Finish independently.
    WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit), sequence);
    GatherAndPostHierarchicalDispatch(receiver, common, params, nullptr, config, scratch);
    Check(relayCalls == 1U && waits == 1U && quiets == 0U && posts.size() == 2U && queued.size() == 2U,
          "receive leader did not relay independently of outgoing completion");
    Check(ReadWord(local, relayReady) == sequence && ReadWord(local, txDone) != sequence,
          "incoming payload readiness was held behind outgoing completion");
    RoceTest::blockIndex = RoceTest::subBlockIndex = 1;
    // Finish must only complete outgoing work; the receive leader already
    // published readiness and must remain its sole writer.
    RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    // Both GM sources must remain stable until the unified completion.
    Check(payload == std::vector<uint8_t>(local.begin() + source, local.begin() + source + activeBytes),
          "posting changed the quantized source");
    FinishHierarchicalDispatch(owner, common, params);
    std::copy_n(local.data() + rail, railBytes, expected.data() + rail);
    std::copy(payload.begin(), payload.end(), expected.begin() + destination);
    std::memcpy(expected.data() + payloadCommit, &sequence, sizeof(sequence));
    Check(remote == expected, "whole-batch payload changed active data, capacity tail, or neighboring regions");
    Check(queued.empty() && posts.size() == 3U && quiets == 2U && waits == 1U,
          "Finish omitted completion/commit or zero-remote notification");
    Check((RoceTest::operations == std::vector<int>{1, 1, 2, 1, 2}), "unified payload/commit quiet ordering changed");
    Check(ReadWord(local, txDone) == sequence && ReadWord(local, relayReady) == sequence && !RoceTest::mteCanRead,
          "sender repeated the receive acquire or lost independent readiness");
    std::fill_n(local.data() + source, activeBytes, uint8_t{0xDDU});
    std::fill_n(local.data() + rail, railBytes, uint8_t{0xCCU});
    Check(remote == expected, "completed receive still referenced the reusable source");
}

int main()
{
    try {
        for (uint32_t rank : {5U, 21U}) {
            for (const auto shape : std::array<std::array<uint32_t, 2>, 4>{{{1, 32}, {7, 64}, {31, 32}, {32, 64}}}) {
                for (const auto pattern : {"local", "remote", "mixed", "last"}) {
                    CheckDispatch(rank, shape[0], shape[1], 3584U, 8U, pattern);
                }
            }
            CheckDispatch(rank, 31U, 64U, 1024U, 1U, "last");
            CheckDispatch(rank, 32U, 32U, 8192U, 32U, "remote");
            CheckDispatch(rank, 1U, 64U, 3584U, 32U, "local");
        }
        // Negative control: the asynchronous double must detect early source
        // reuse instead of accidentally behaving as a synchronous memcpy.
        auto* ub = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(DispatchTest::kScratchOffset));
        MegaMoeImpl::RocePutBytes(DispatchTest::local.data() + 64U, DispatchTest::local.data() + 128U, 16U,
                                  DispatchTest::peer, ub);
        DispatchTest::local[128U] ^= 1U;
        bool rejected = false;
        try {
            MegaMoeImpl::RoceQuiet(DispatchTest::peer, ub);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        DispatchTest::Check(rejected, "asynchronous test double failed to detect an invalid source lifetime");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
