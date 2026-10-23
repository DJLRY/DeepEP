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

__aicore__ inline void WaitHierarchicalLocalSignal(const Params& params, HierarchicalSignal signal)
{
    WaitUntilGmFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal),
                          static_cast<int32_t>(params.tilingData->hierarchicalSequence));
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
}

// Local packing preserves the existing rail format. RDMA reads its stable GM
// copy, never the packer's reusable UB. Payload rows retain their original token
// indices at the gateway; route metadata determines which rows consumers read.
__aicore__ inline void PostHierarchicalDispatch(const MoeStageCommonConfig& common, const Params& params,
                                                AscendTimerDevice* timer = nullptr)
{
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    auto* roceScratch = HierarchicalRoceScratch(params);
    // All local token/scale rows are ready before this stage. Decode is bounded
    // to 32 tokens and both regions reserve the full batch, so one contiguous
    // Put avoids route rescans and per-run posting overhead. Send actual B, not
    // the persistent buffer capacity; unused remote rows have no consumers.
    const uint32_t rowBytes =
        static_cast<uint32_t>(CalcQuantTokenScaleBytes(common.tokenHiddenDim, 1U, common.topK, false));
    RocePutBytes(params.peermemInfo.relayPayloadPtr, params.peermemInfo.quantTokenScalePtr, common.tokenNum * rowBytes,
                 gateway, roceScratch, timer, AscendTimer::RdmaPhase::DispatchPayload);
    // QP1 starts before rail packing completes. Its owner overlaps this transfer
    // with reset/shared input work, then publishes only the payload commit.
}

__aicore__ inline void PostHierarchicalDispatchMetadata(const MoeStageCommonConfig& common, const Params& params,
                                                        const SendMaskConfig& config,
                                                        AscendTimerDevice* timer = nullptr)
{
    WaitHierarchicalLocalSignal(params, HierarchicalSignal::LocalRailPacked);
    const uint32_t myHost = HierarchicalHost(common.rankId);
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    auto* roceScratch = HierarchicalRoceScratch(params);
    GM_ADDR packedRail = params.peermemInfo.railTopkIdsRecvPtr + myHost * config.railTopkIdsSlotBytes;
    const uint32_t railBytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK));
    RocePutBytes<MEGAMOE_ROCE_META_QP_INDEX>(packedRail, packedRail, railBytes, gateway, roceScratch, timer,
                                             AscendTimer::RdmaPhase::DispatchMeta);
    // Release the small metadata immediately, on its own queue. Deferring this
    // quiet until after reset/shared work would delay all remote route workers.
    RocePublishCommit<MEGAMOE_ROCE_META_QP_INDEX>(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        roceScratch, timer, AscendTimer::RdmaPhase::DispatchMeta);
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
        // Reuse the packer as the independent payload receive leader. Packing
        // has released both local relays and the metadata sender before this
        // wait; no route worker or outgoing completion is held behind payload.
        RoceWaitCommit(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit),
                       params.tilingData->hierarchicalSequence, timer, AscendTimer::RdmaPhase::DispatchPayload,
                       HierarchicalGateway(common.rankId));
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::RelayPayloadReady);
        return;
    }
    if (job.jobIndex == HierarchicalSendJob()) {
        PostHierarchicalDispatch(common, params, timer);
        return;
    }
    if (job.jobIndex == HierarchicalMetaSendJob(job.totalJobs)) {
        PostHierarchicalDispatchMetadata(common, params, config, timer);
        return;
    }
    if (job.jobIndex >= job.totalJobs || job.totalJobs <= 3U) {
        return;
    }

    // The metadata receive leader releases routing without waiting for payload.
    // Neither receiver touches a sender's QP/CQ. Both senders publish before the
    // public input barrier, so receive waits cannot form a barrier cycle.
    if (job.jobIndex == 2U + otherHost) {
        RoceWaitCommit(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit),
                       params.tilingData->hierarchicalSequence, timer, AscendTimer::RdmaPhase::DispatchMeta,
                       HierarchicalGateway(common.rankId));
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::MetaReady);
    }
    const uint32_t relayCore = job.jobIndex - 2U;
    const uint32_t relayCores = job.totalJobs - 3U;
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    for (uint32_t sourceHost = relayCore % 2U; sourceHost < 2U; sourceHost += relayCores) {
        WaitHierarchicalLocalSignal(
            params, sourceHost == myHost ? HierarchicalSignal::LocalRailPacked : HierarchicalSignal::MetaReady);
        uint32_t firstExpert;
        uint32_t expertCount;
        CalcRailRelayExpertRange(expertsPerHost, relayCore, relayCores, 2U, sourceHost, firstExpert, expertCount);
        if (expertCount != 0U) {
            RelayRailTopkIdsToExperts(common, winRankAddr, config, params, scratch, sourceHost, job.jobIndex,
                                      static_cast<int32_t>(myHost * expertsPerHost + firstExpert),
                                      static_cast<int32_t>(expertCount));
        }
    }
}

__aicore__ inline void FinishHierarchicalDispatch(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                  const Params& params, AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.jobIndex != HierarchicalSendJob()) {
        return;
    }
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    // Sequence is published before packing, but this owner no longer waits for
    // LocalRailPacked. Acquire the current value before the commit reads it.
    WaitHierarchicalLocalSignal(params, HierarchicalSignal::Sequence);
    // QP1 completes only payload. The metadata owner independently drains QP2
    // before the public barrier; Sequence cannot be reused before both finish.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        HierarchicalRoceScratch(params), timer, AscendTimer::RdmaPhase::DispatchPayload);
    PublishHierarchicalLocalSignal(params, HierarchicalSignal::TxDone);
    // The packer independently acquires incoming payload and publishes ready.
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
        GmSignalWaitBarrier(localFlags + peer * INT_CACHELINE, epoch);
    }
    WriteGmByPassDCache(counter, epoch);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

}  // namespace MegaMoeImpl
#endif
