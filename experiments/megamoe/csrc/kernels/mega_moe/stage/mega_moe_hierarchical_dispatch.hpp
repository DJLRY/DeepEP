// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_HIERARCHICAL_DISPATCH_H
#define MEGA_MOE_HIERARCHICAL_DISPATCH_H

#include "../common/mega_moe_roce.hpp"
#include "mega_moe_send_mask.hpp"

namespace MegaMoeImpl {

__aicore__ inline __ubuf__ uint8_t* HierarchicalRoceScratch(const Params& params)
{
    return reinterpret_cast<__ubuf__ uint8_t*>(static_cast<uint64_t>(params.tilingData->roceScratchUbOffset));
}

__aicore__ inline void PublishHierarchicalLocalSignal(const Params& params, HierarchicalSignal signal)
{
    WriteGmByPassDCache(HierarchicalSignalAddress(params.peermemInfo, signal),
                        static_cast<int32_t>(params.tilingData->hierarchicalSequence));
}

__aicore__ inline void WaitHierarchicalLocalSignal(const Params& params, HierarchicalSignal signal, uint32_t peer = 0U,
                                                   __ubuf__ uint8_t* scratch = nullptr,
                                                   uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceWaitFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal), params.tilingData->hierarchicalSequence,
                       peer, scratch, RoceWaitKind::LocalReadiness, qpIndex);
#else
    (void)peer;
    (void)scratch;
    (void)qpIndex;
    WaitUntilGmFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal),
                          static_cast<int32_t>(params.tilingData->hierarchicalSequence));
#endif
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
}

// Quantization completes on all AIVs before this stage. Each owner posts its
// contiguous share of the whole batch without waiting for route packing.
__aicore__ inline void PostHierarchicalDispatch(const MoeStageCommonConfig& common, const Params& params, uint32_t lane,
                                                AscendTimerDevice* timer = nullptr)
{
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    const uint32_t qp = ascend_deepep::roce_layout::kFirstPayloadQp + lane;
    auto* scratch = HierarchicalRoceScratch(params);
    const uint32_t firstToken = common.tokenNum * lane / HIERARCHICAL_PAYLOAD_LANES;
    const uint32_t endToken = common.tokenNum * (lane + 1U) / HIERARCHICAL_PAYLOAD_LANES;
    const uint32_t rowBytes =
        static_cast<uint32_t>(CalcQuantTokenScaleBytes(common.tokenHiddenDim, 1U, common.topK, false));
    if (firstToken != endToken) {
        const uint64_t offset = static_cast<uint64_t>(firstToken) * rowBytes;
        RocePutBytes(params.peermemInfo.relayPayloadPtr + offset, params.peermemInfo.quantTokenScalePtr + offset,
                     (endToken - firstToken) * rowBytes, gateway, scratch, timer,
                     AscendTimer::RdmaPhase::DispatchPayload, false, qp);
    }
}

__aicore__ inline void GatherAndPostHierarchicalDispatch(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                         const Params& params, GM_ADDR* winRankAddr,
                                                         const SendMaskConfig& config, SendMaskScratch& scratch,
                                                         AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint32_t myHost = HierarchicalHost(common.rankId);
    const uint32_t otherHost = 1U - myHost;
    if (job.jobIndex == HierarchicalPackJob()) {
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::Sequence);
        SendTopkIdsToRailServer(common, winRankAddr, config, params, scratch, myHost, job.jobIndex);
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::LocalRailPacked);
        GM_ADDR packedRail = params.peermemInfo.railTopkIdsRecvPtr + myHost * config.railTopkIdsSlotBytes;
        RocePutBytes(packedRail, packedRail,
                     static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK)),
                     HierarchicalGateway(common.rankId), HierarchicalRoceScratch(params), timer,
                     AscendTimer::RdmaPhase::DispatchMeta, false, ascend_deepep::roce_layout::kMetadataQp);
        return;
    }
    if (IsHierarchicalPayloadOwner(job.jobIndex)) {
        PostHierarchicalDispatch(common, params, job.jobIndex / 2U, timer);
        return;
    }
    if (job.jobIndex >= job.totalJobs || job.totalJobs <= HIERARCHICAL_PAYLOAD_LANES + 1U) {
        return;
    }

    const uint32_t relayCore = HierarchicalRelayJob(job.jobIndex);
    const uint32_t relayCores = job.totalJobs - HIERARCHICAL_PAYLOAD_LANES - 1U;
    // Routing has its own commit. Never wait for payload before the input
    // barrier: count/prefix work must be allowed to overlap those transfers.
    // This leader has no CQ ownership and only observes incoming metadata.
    if (relayCore == otherHost) {
        RoceWaitCommit(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit),
                       params.tilingData->hierarchicalSequence, timer, AscendTimer::RdmaPhase::DispatchMeta,
                       HierarchicalGateway(common.rankId), nullptr, ascend_deepep::roce_layout::kMetadataQp);
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::MetaReady);
    }
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    for (uint32_t sourceHost = relayCore % 2U; sourceHost < 2U; sourceHost += relayCores) {
        WaitHierarchicalLocalSignal(
            params, sourceHost == myHost ? HierarchicalSignal::LocalRailPacked : HierarchicalSignal::MetaReady,
            HierarchicalGateway(common.rankId));
        uint32_t firstExpert;
        uint32_t expertCount;
        CalcRailRelayExpertRange(expertsPerHost, relayCore, relayCores, 2U, sourceHost, firstExpert, expertCount);
        if (expertCount != 0U) {
            RelayRailTopkIdsToExperts(common, winRankAddr, config, params, scratch, sourceHost, job.jobIndex,
                                      static_cast<int32_t>(myHost * expertsPerHost + firstExpert),
                                      static_cast<int32_t>(expertCount), timer);
        }
    }
}

__aicore__ inline void FinishHierarchicalDispatch(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                  const Params& params, AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!IsHierarchicalQpOwner(job.jobIndex)) {
        return;
    }
    const bool metadata = job.jobIndex == HierarchicalPackJob();
    const uint32_t lane = metadata ? 0U : job.jobIndex / 2U;
    if (!metadata) {
        // Payload may be posted before the packer runs. Only its commit needs
        // the shared, immutable Sequence source published by that packer.
        WaitHierarchicalLocalSignal(params, HierarchicalSignal::Sequence, HierarchicalGateway(common.rankId),
                                    HierarchicalRoceScratch(params), HierarchicalOwnerQp(job.jobIndex));
    }
    // Empty payload lanes still commit. Every owner posts before any input
    // barrier, and each commit certifies only preceding work on its own QP.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(
            params.peermemInfo, metadata ? HierarchicalSignal::NetMetaCommit : HierarchicalSignal::NetPayloadCommit,
            lane)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)),
        HierarchicalGateway(common.rankId), HierarchicalRoceScratch(params), timer,
        metadata ? AscendTimer::RdmaPhase::DispatchMeta : AscendTimer::RdmaPhase::DispatchPayload,
        HierarchicalOwnerQp(job.jobIndex));
}

// All producers have drained before this call. Only local peer pointers are
// accessed; the existing per-AIV route epoch advances once for the output phase.
__aicore__ inline void CrossRankSyncInHierarchicalHost(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                       const Params& params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    auto* counter =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(job.jobIndex) * RANK_SYNC_COUNTER_SLOT_BYTES);
    const int32_t epoch = ReadGmByPassDCache(counter) + 1;
    const uint32_t firstRank = HierarchicalHost(common.rankId) * HIERARCHICAL_RANKS_PER_HOST;
    auto* localFlags = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr);
    for (uint32_t localRank = job.jobIndex; localRank < HIERARCHICAL_RANKS_PER_HOST; localRank += job.totalJobs) {
        const uint32_t peer = firstRank + localRank;
        auto* remoteFlag = reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[peer]) + common.rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteFlag, epoch);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        auto* scratch = IsHierarchicalQpOwner(job.jobIndex) ? HierarchicalRoceScratch(params) : nullptr;
        RoceWaitFlagEquals(localFlags + peer * INT_CACHELINE, static_cast<uint32_t>(epoch),
                           HierarchicalGateway(common.rankId), scratch, RoceWaitKind::HostProducerBarrier,
                           HierarchicalOwnerQp(job.jobIndex));
#else
        GmSignalWaitBarrier(localFlags + peer * INT_CACHELINE, epoch);
#endif
    }
    WriteGmByPassDCache(counter, epoch);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

}  // namespace MegaMoeImpl
#endif
