// Copyright (c) 2026, Lu Lu

// Run the production Dispatch, QP completion, relay partition and heap layout.
// Only device primitives and the metadata pack/relay operations are doubled.
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
constexpr int AIC = 0, AIV = 1, g_coreType = AIV;

namespace MegaMoeImpl {
using namespace AscendC;
struct MoeStageCommonConfig {
    uint32_t rankId, tokenNum, topK, tokenHiddenDim, moeExpertPerRank;
};
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
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
inline std::vector<uint32_t> coverage;

inline int32_t* HierarchicalSignalAddress(const PeermemInfo& info, HierarchicalSignal signal)
{
    return reinterpret_cast<int32_t*>(info.hierarchicalControlPtr + HierarchicalSignalOffset(signal));
}
inline void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    using namespace DispatchTest;
    const auto offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(value));
    if (offset == metaReady) {
        Check(RoceTest::blockIndex == static_cast<int>(receiveLeaderJob) &&
                  metaAcquired[RoceTest::blockIndex] == sequence && RoceTest::mteCanRead,
              "metadata readiness preceded its own receive acquire");
    }
    if (offset == relayReady) {
        Check(RoceTest::blockIndex == 0 && payloadAcquired[0] == sequence && RoceTest::mteCanRead,
              "payload readiness preceded its independent receive acquire");
    }
    if (offset == txDone) {
        Check(RoceTest::blockIndex == 1 && Pending(1U) == 0U && ReadWord(remote, payloadCommit) == sequence,
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
    DispatchTest::Check(ReadGmByPassDCache(address) == value, "stale local sequence/rail readiness");
    RoceTest::signalObserved = true;
}
template <HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    AscendC::SetFlag<Event>(Id);
    AscendC::WaitFlag<Event>(Id);
}

inline void SendTopkIdsToRailServer(const MoeStageCommonConfig& common, GM_ADDR*, const SendMaskConfig&,
                                    const Params& params, SendMaskScratch&, uint32_t targetHost, uint32_t job)
{
    using namespace DispatchTest;
    Check(RoceTest::blockIndex == 0 && job == 0U, "metadata packer/epoch owner is not physical AIV0");
    Check(targetHost == HierarchicalHost(common.rankId), "packer used its AIV index as the rail slot");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence))) == sequence,
          "packer started before initializing the sequence");
    Check(static_cast<uint32_t>(ReadGmByPassDCache(
              HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::LocalRailPacked))) != sequence,
          "rail readiness was published before packing");
    ++packCalls;
}

/* PRODUCTION_RELAY_RANGE */

inline void RelayRailTopkIdsToExperts(const MoeStageCommonConfig& common, GM_ADDR*, const SendMaskConfig&,
                                      const Params& params, SendMaskScratch&, uint32_t sourceHost, uint32_t job,
                                      int32_t first, int32_t count)
{
    using namespace DispatchTest;
    Check(job >= 2U && job < metaSenderJob && job == static_cast<uint32_t>(RoceTest::blockIndex),
          "sender/packer was assigned relay work");
    const auto ready = sourceHost == HierarchicalHost(common.rankId) ? HierarchicalSignal::LocalRailPacked
                                                                     : HierarchicalSignal::MetaReady;
    Check(static_cast<uint32_t>(ReadGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, ready))) == sequence &&
              RoceTest::mteCanRead,
          "relay preceded metadata readiness");
    const uint32_t experts = 16U * common.moeExpertPerRank;
    const int32_t localFirst = first - static_cast<int32_t>(HierarchicalHost(common.rankId) * experts);
    Check(localFirst >= 0 && count > 0 && localFirst + count <= static_cast<int32_t>(experts),
          "invalid relay expert range");
    for (int32_t expert = localFirst; expert < localFirst + count; ++expert) {
        ++coverage[sourceHost * experts + static_cast<uint32_t>(expert)];
    }
    ++relayCalls;
}
inline void GmSignalWaitBarrier(int32_t*, int32_t)
{
    throw std::runtime_error("Dispatch entered an unexpected final host barrier");
}
template <bool>
void SyncAll()
{
    throw std::runtime_error("Dispatch unexpectedly synchronized all cores");
}
}  // namespace MegaMoeImpl

#include "../csrc/kernels/mega_moe/stage/mega_moe_hierarchical_dispatch.hpp"

using namespace MegaMoeImpl;

struct JobScope {
    int previous = RoceTest::blockIndex;
    explicit JobScope(uint32_t job)
    {
        RoceTest::blockIndex = static_cast<int>(job);
        RoceTest::subBlockIndex = static_cast<int>(job % 2U);
        RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    }
    ~JobScope()
    {
        RoceTest::blockIndex = previous;
        RoceTest::subBlockIndex = previous % 2;
        RoceTest::signalObserved = RoceTest::scalarEventSet = RoceTest::mteCanRead = false;
    }
};

void CheckDispatch(uint32_t rank, uint32_t bs, uint32_t capacity, uint32_t hidden, uint32_t topk,
                   std::string_view pattern, bool metadataFirst, uint32_t jobs = 56U, uint32_t epoch = 257U,
                   bool outgoingPayloadFirst = false)
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
    t.blockAivNum = jobs;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.roceScratchUbOffset = kScratchOffset;
    t.hierarchicalSequence = epoch;
    Check(UseDecodeRailRouting(&t), "fixture requires decode rail layout");
    const PeermemSizeParams sizes{capacity,           topk, hidden, 12U, 32U, 2U, 1U, false, false,
                                  TOPO_TYPE_MTE_ROCE, true, 16U};
    const auto heapBytes = static_cast<size_t>(CalcPeermemLeastSize(sizes));
    local.assign(heapBytes, 0x7BU);
    remote.assign(heapBytes, 0xA5U);
    posts.clear();
    queued.clear();
    quiets = waits = packCalls = relayCalls = 0U;
    peer = HierarchicalGateway(rank);
    sequence = epoch;
    receiveLeaderJob = 2U + (1U - HierarchicalHost(rank));
    metaSenderJob = HierarchicalMetaSendJob(jobs);
    metaAcquired.fill(0U);
    payloadAcquired.fill(0U);
    onPayloadWait = nullptr;
    coverage.assign(2U * 16U * 12U, 0U);
    RoceTest::operations.clear();
    Params params{&t, PeermemInfo(local.data(), &t)};
    MoeStageCommonConfig common{rank, bs, topk, hidden, 12U};
    SendMaskConfig config{static_cast<uint64_t>(CalcRailTopkIdsSlotBytes(&t))};
    SendMaskScratch scratch;
    const auto offset = [&](HierarchicalSignal signal) {
        return Offset(reinterpret_cast<uint8_t*>(HierarchicalSignalAddress(params.peermemInfo, signal)), 4U);
    };
    metaCommit = offset(HierarchicalSignal::NetMetaCommit);
    payloadCommit = offset(HierarchicalSignal::NetPayloadCommit);
    metaReady = offset(HierarchicalSignal::MetaReady);
    relayReady = offset(HierarchicalSignal::RelayPayloadReady);
    txDone = offset(HierarchicalSignal::TxDone);
    for (auto signal :
         {HierarchicalSignal::Sequence, HierarchicalSignal::LocalRailPacked, HierarchicalSignal::NetMetaCommit,
          HierarchicalSignal::NetPayloadCommit, HierarchicalSignal::MetaReady, HierarchicalSignal::RelayPayloadReady,
          HierarchicalSignal::TxDone}) {
        SetWord(local, offset(signal), sequence - 1U);
    }

    const uint32_t activeBytes = bs * static_cast<uint32_t>(CalcQuantTokenScaleBytes(hidden, 1U, topk, false));
    const uint32_t source = Offset(params.peermemInfo.quantTokenScalePtr, activeBytes);
    const uint32_t destination = Offset(params.peermemInfo.relayPayloadPtr, activeBytes);
    Check(params.peermemInfo.quantTokenScalePtr + activeBytes <= params.peermemInfo.combineSendPtr &&
              params.peermemInfo.relayPayloadPtr + activeBytes <= params.peermemInfo.gatewayCombinePtr,
          "actual batch exceeds its allocation");
    for (uint32_t byte = 0U; byte < activeBytes; ++byte) {
        local[source + byte] = static_cast<uint8_t>((byte * 13U + rank) % 251U);
    }
    const auto payload = std::vector<uint8_t>(local.begin() + source, local.begin() + source + activeBytes);
    const uint32_t railBytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(bs, topk));
    const uint32_t rail =
        Offset(params.peermemInfo.railTopkIdsRecvPtr + HierarchicalHost(rank) * config.railTopkIdsSlotBytes, railBytes);
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t slot = 0U; slot < topk; ++slot) {
            const bool other = pattern == "remote" || (pattern == "mixed" && (token + slot) % 3U == 0U) ||
                               (pattern == "last" && token + 1U == bs && slot + 1U == topk);
            const uint32_t host = HierarchicalHost(rank) ^ static_cast<uint32_t>(other);
            const int32_t expert = static_cast<int32_t>((host * 16U + (token + slot) % 16U) * 12U + slot % 12U);
            const uint32_t item = token * topk + slot;
            std::memcpy(local.data() + rail + ((item / 127U) * 128U + item % 127U) * 4U, &expert, sizeof(expert));
        }
    }
    auto expected = remote;
    const auto gather = [&](uint32_t job) {
        JobScope scope(job);
        GatherAndPostHierarchicalDispatch({job, jobs}, common, params, nullptr, config, scratch);
    };
    const auto finish = [&](uint32_t job) {
        JobScope scope(job);
        FinishHierarchicalDispatch({job, jobs}, common, params);
    };
    for (uint32_t job = 0U; job < jobs; ++job) {
        if (job != 1U) finish(job);
    }
    Check(posts.empty() && quiets == 0U && waits == 0U, "nonowner touched the payload completion path");

    gather(1U);
    Check(posts.size() == 1U && queued.size() == 1U && packCalls == 0U && quiets == 0U,
          "payload did not start before metadata packing");
    Check(queued[0].source == source && queued[0].destination == destination && queued[0].bytes == activeBytes,
          "payload did not use the actual batch range");
    bool stale = false;
    try {
        finish(1U);
    } catch (const std::runtime_error&) {
        stale = true;
    }
    Check(stale && quiets == 0U, "payload sender committed an uninitialized sequence");

    if (outgoingPayloadFirst) {
        // The packer initializes Sequence before packing. Hold LocalRailPacked
        // at the previous epoch: QP1 must still complete without QP2 or receive.
        SetWord(local, offset(HierarchicalSignal::Sequence), sequence);
        finish(1U);
        std::copy(payload.begin(), payload.end(), expected.begin() + destination);
        SetWord(expected, payloadCommit, sequence);
        Check(remote == expected && queued.empty() && quiets == 2U && posts.size() == 2U && waits == 0U &&
                  ReadWord(local, offset(HierarchicalSignal::LocalRailPacked)) != sequence,
              "payload completion depended on metadata packing or incoming traffic");
    }

    const auto receiveMeta = [&] {
        bool rejected = false;
        try {
            gather(receiveLeaderJob);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        Check(rejected && relayCalls == 0U && ReadWord(local, metaReady) != sequence,
              "payload or old metadata released routing");
        SetWord(local, metaCommit, sequence);
        gather(receiveLeaderJob);
        Check(relayCalls > 0U && ReadWord(local, metaReady) == sequence, "metadata did not release routing");
    };
    const auto sendMeta = [&] {
        gather(metaSenderJob);
        Check(Pending(1U) == (outgoingPayloadFirst ? 0U : 1U) && Pending(2U) == 0U &&
                  quiets == (outgoingPayloadFirst ? 4U : 2U) && posts.size() == (outgoingPayloadFirst ? 4U : 3U),
              "metadata completion drained or waited for payload");
        Check(ReadWord(remote, metaCommit) == sequence &&
                  (ReadWord(remote, payloadCommit) == sequence) == outgoingPayloadFirst,
              "metadata did not commit independently");
        std::copy_n(local.data() + rail, railBytes, expected.data() + rail);
        SetWord(expected, metaCommit, sequence);
        Check(remote == expected, "metadata changed payload, capacity tail, or neighboring memory");
    };
    onPayloadWait = [&] {
        Check(ReadWord(local, offset(HierarchicalSignal::LocalRailPacked)) == sequence && packCalls == 1U,
              "payload receive blocked the metadata packer before publication");
        if (metadataFirst) {
            receiveMeta();
            Check(ReadWord(local, relayReady) != sequence &&
                      (ReadWord(local, txDone) == sequence) == outgoingPayloadFirst,
                  "metadata falsely certified incoming or outgoing payload");
            sendMeta();
        }
        SetWord(local, payloadCommit, sequence);
    };
    gather(0U);
    Check(ReadWord(local, relayReady) == sequence && (ReadWord(local, txDone) == sequence) == outgoingPayloadFirst,
          "incoming payload waited for outgoing completion");
    if (!metadataFirst) {
        Check(ReadWord(local, metaReady) != sequence, "payload incorrectly released metadata");
        sendMeta();
        receiveMeta();
    }

    // Exercise every production relay assignment, including the old tail's
    // experts now redistributed over one fewer worker. No holes/duplicates.
    for (uint32_t job = 2U; job < metaSenderJob; ++job) {
        if (job != receiveLeaderJob) gather(job);
    }
    Check(std::all_of(coverage.begin(), coverage.end(), [](uint32_t count) { return count == 1U; }),
          "reserving the metadata sender lost or duplicated a source/expert range");

    if (!outgoingPayloadFirst) finish(1U);
    std::copy(payload.begin(), payload.end(), expected.begin() + destination);
    SetWord(expected, payloadCommit, sequence);
    Check(remote == expected, "payload changed active data, capacity tail, or neighboring memory");
    Check(queued.empty() && posts.size() == 4U && quiets == 4U && waits == 2U,
          "split completion has missing or duplicate operations");
    const auto expectedOperations = outgoingPayloadFirst ? std::vector<int>{11, 12, 11, 12, 21, 22, 21, 22}
                                                         : std::vector<int>{11, 21, 22, 21, 22, 12, 11, 12};
    Check(RoceTest::operations == expectedOperations, "per-QP data/commit ordering changed");
    Check(ReadWord(local, txDone) == sequence && ReadWord(local, relayReady) == sequence,
          "outgoing completion lost independently published incoming readiness");
    std::fill_n(local.data() + source, activeBytes, 0xDDU);
    std::fill_n(local.data() + rail, railBytes, 0xCCU);
    Check(remote == expected, "completed receive still referenced reusable source memory");
}

int main()
{
    try {
        for (uint32_t rank : {5U, 21U}) {
            for (bool metaFirst : {false, true}) {
                for (const auto shape :
                     std::array<std::array<uint32_t, 2>, 4>{{{1, 32}, {7, 64}, {31, 32}, {32, 64}}}) {
                    for (const auto pattern : {"local", "remote", "mixed", "last"}) {
                        CheckDispatch(rank, shape[0], shape[1], 3584U, 8U, pattern, metaFirst);
                    }
                }
                for (uint32_t epoch : {127U, 128U, 257U, 258U}) {
                    CheckDispatch(rank, 31U, 64U, 1024U, 1U, "last", metaFirst, 6U, epoch);
                }
                CheckDispatch(rank, 32U, 32U, 8192U, 32U, "remote", metaFirst, 64U);
                CheckDispatch(rank, 32U, 64U, 3584U, 8U, "mixed", metaFirst, 56U, 260U, true);
            }
        }
        // Ensure the asynchronous double detects source reuse on either QP.
        for (uint32_t qp : {1U, 2U}) {
            DispatchTest::queued.clear();
            JobScope scope(qp == 1U ? 1U : DispatchTest::metaSenderJob);
            auto* ub = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(DispatchTest::kScratchOffset));
            if (qp == 1U) {
                RocePutBytes(DispatchTest::local.data() + 64U, DispatchTest::local.data() + 128U, 16U,
                             DispatchTest::peer, ub);
            } else {
                RocePutBytes<MEGAMOE_ROCE_META_QP_INDEX>(
                    DispatchTest::local.data() + 64U, DispatchTest::local.data() + 128U, 16U, DispatchTest::peer, ub);
            }
            DispatchTest::local[128U] ^= 1U;
            bool rejected = false;
            try {
                if (qp == 1U)
                    RoceQuiet(DispatchTest::peer, ub);
                else
                    RoceQuiet<MEGAMOE_ROCE_META_QP_INDEX>(DispatchTest::peer, ub);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            DispatchTest::Check(rejected, "double failed to detect an invalid source lifetime");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
