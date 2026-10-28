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

inline int32_t* HierarchicalSignalAddress(const PeermemInfo& info, HierarchicalSignal signal, uint32_t lane = 0U)
{
    return reinterpret_cast<int32_t*>(info.hierarchicalControlPtr + HierarchicalSignalOffset(signal, lane));
}
inline void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    using namespace DispatchTest;
    const auto offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(value));
    if (offset == metaReady) {
        Check(RoceTest::blockIndex == static_cast<int>(receiveLeaderJob), "wrong metadata leader");
        Check(waits >= 1U && RoceTest::mteCanRead && ReadWord(local, metaCommit) == sequence,
              "metadata readiness preceded its own acquire");
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
                                    const Params& params, SendMaskScratch&, uint32_t targetServerIdx, uint32_t jobIdx,
                                    uint32_t* remoteTokens)
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
    *remoteTokens = 0U;
    const uint32_t rail = Offset(
        params.peermemInfo.railTopkIdsRecvPtr + targetServerIdx * CalcRailTopkIdsSlotBytes(params.tilingData), 4U);
    for (uint32_t item = 0U; item < common.tokenNum * common.topK; ++item) {
        const uint32_t word = item / 127U * 128U + item % 127U;
        const auto expert = static_cast<int32_t>(ReadWord(local, rail + word * 4U));
        if (expert >= 0 && static_cast<uint32_t>(expert) / (16U * common.moeExpertPerRank) == 1U - targetServerIdx) {
            *remoteTokens |= 1U << (item / common.topK);
        }
    }
    ++packCalls;
}
inline void CalcRailRelayExpertRange(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t& first, uint32_t& count)
{
    DispatchTest::Check(!IsHierarchicalQpOwner(static_cast<uint32_t>(RoceTest::blockIndex)),
                        "sender unexpectedly assigned relay work");
    first = 0U;
    count = 1U;
}
inline void RelayRailTopkIdsToExperts(const MoeStageCommonConfig&, GM_ADDR*, const SendMaskConfig&, const Params&,
                                      SendMaskScratch&, uint32_t, uint32_t, int32_t, int32_t, AscendTimerDevice*)
{
    using namespace DispatchTest;
    Check(!IsHierarchicalQpOwner(static_cast<uint32_t>(RoceTest::blockIndex)) &&
              ReadWord(local, metaReady) == sequence && RoceTest::mteCanRead,
          "metadata relay preceded its receive acquire");
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
    progressCalls = 0U;
    RoceTest::cycleStep = 1;
    peer = HierarchicalGateway(rank);
    sequence = t.hierarchicalSequence;
    receiveLeaderJob = 2U + 2U * (1U - HierarchicalHost(rank));
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
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit, lane),
                            sequence - 1U);
    }
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
    uint32_t validMask = 0U;
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t slot = 0U; slot < topk; ++slot) {
            const bool other = pattern == "remote" || (pattern == "mixed" && (token + slot) % 3U == 0U) ||
                               (pattern == "last" && token + 1U == bs && slot + 1U == topk) ||
                               (pattern == "balanced" && token % 4U >= 2U);
            if (other) validMask |= 1U << token;
            const uint32_t targetHost = HierarchicalHost(rank) ^ static_cast<uint32_t>(other);
            const int32_t expert = static_cast<int32_t>((targetHost * 16U + (token + slot) % 16U) * 12U + slot % 12U);
            const uint32_t item = token * topk + slot;
            const uint32_t packetWord = (item / 127U) * 128U + item % 127U;
            std::memcpy(local.data() + rail + packetWord * 4U, &expert, sizeof(expert));
        }
    }
    auto expected = remote;
    for (uint32_t token = 0U; token < bs; ++token) {
        if ((validMask & (1U << token)) != 0U) {
            std::copy_n(local.data() + source + token * rowBytes, rowBytes,
                        expected.data() + destination + token * rowBytes);
        }
    }
    std::copy_n(local.data() + rail, railBytes, expected.data() + rail);
    const auto setExpected = [&](uint32_t offset) {
        std::memcpy(expected.data() + offset, &sequence, sizeof(sequence));
    };
    setExpected(metaCommit);
    for (uint32_t lane = 0U; lane < 8U; ++lane) setExpected(payloadCommit + lane * 512U);

    // Pack once, post route QP9, and certify it independently of all payloads.
    RoceTest::blockIndex = RoceTest::subBlockIndex = 0;
    GatherAndPostHierarchicalDispatch({0U, 56U}, common, params, nullptr, config, scratch);
    FinishHierarchicalDispatch({0U, 56U}, common, params);
    DeliverPosted(9U);
    Check(ReadWord(remote, metaCommit) == sequence, "route QP did not independently complete");
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        Check(ReadWord(remote, payloadCommit + lane * 512U) != sequence, "metadata falsely completed a payload QP");
    }
    // Incoming metadata is ready while every incoming payload commit is stale.
    // The production Relay must still run to completion before the input barrier.
    RoceTest::blockIndex = static_cast<int>(receiveLeaderJob);
    GatherAndPostHierarchicalDispatch({receiveLeaderJob, 56U}, common, params, nullptr, config, scratch);
    Check(relayCalls == 1U && ReadWord(local, metaReady) == sequence, "early metadata relay was blocked by payload");

    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        RoceTest::blockIndex = static_cast<int>(HierarchicalSendJob(lane));
        RoceTest::subBlockIndex = 1;
        const AivJobContext owner{HierarchicalSendJob(lane), 56U};
        GatherAndPostHierarchicalDispatch(owner, common, params, nullptr, config, scratch);
        FinishHierarchicalDispatch(owner, common, params);
    }
    Check(packCalls == 1U, "senders rescanned/packed metadata");
    std::array<uint32_t, 8> payloadPosts{}, rows{};
    uint32_t commits = 0U;
    for (const auto& post : posts) {
        if (IsCommit(post)) {
            ++commits;
            Check(post.qp == (post.destination == metaCommit ? 9U : 1U + (post.destination - payloadCommit) / 512U),
                  "commit was submitted on another lane's QP");
            continue;
        }
        if (post.qp == 9U) {
            Check(post.source == rail && post.destination == rail && post.bytes == railBytes, "incorrect route Put");
            continue;
        }
        Check(post.source - source == post.destination - destination && post.bytes % rowBytes == 0U,
              "payload did not retain whole-token coordinates");
        ++payloadPosts[post.qp - 1U];
        rows[post.qp - 1U] += post.bytes / rowBytes;
    }
    Check(commits == 9U, "empty lanes must also publish their own commit");
    Check(*std::max_element(rows.begin(), rows.end()) - *std::min_element(rows.begin(), rows.end()) <= 1U,
          "payload byte load differs by more than one token");
    if (bs == 32U && pattern == "balanced") {
        for (uint32_t lane = 0U; lane < 8U; ++lane) {
            Check(payloadPosts[lane] == 1U && rows[lane] == 2U, "balanced dispatch lost its eight two-token Puts");
        }
    }
    // Arbitrary cross-QP delivery order; quiet only retires the current owner.
    for (uint32_t lane = 8U; lane-- > 0U;) DeliverPosted(lane + 1U);
    Check(remote == expected, "payload filter missed, duplicated or overwrote a token/guard region");
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        RoceTest::blockIndex = static_cast<int>(HierarchicalSendJob(lane));
        RoceFinishOutstanding(peer, reinterpret_cast<uint8_t*>(kScratchOffset), nullptr,
                              AscendTimer::RdmaPhase::Combine, lane + 1U);
    }
    RoceTest::blockIndex = 0;
    RoceFinishOutstanding(peer, reinterpret_cast<uint8_t*>(kScratchOffset), nullptr,
                          AscendTimer::RdmaPhase::DispatchMeta, 9U);
    Check(queued.empty(), "one or more QPs escaped tail completion");
    Check(std::equal(payload.begin(), payload.end(), local.begin() + source), "Dispatch changed its source");
}

int main()
{
    for (uint32_t rank : {0U, 15U, 16U, 31U})
        for (uint32_t bs : {1U, 7U, 17U, 32U})
            for (auto pattern : {"local", "remote", "mixed", "last", "balanced"}) {
                CheckDispatch(rank, bs, 32U, 3584U, 8U, pattern);
            }
    CheckDispatch(16U, 32U, 64U, 128U, 32U, "remote");
}
