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
                                                   __ubuf__ uint8_t* scratch = nullptr)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceWaitFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal), params.tilingData->hierarchicalSequence,
                       peer, scratch, RoceWaitKind::LocalReadiness);
#else
    (void)peer;
    (void)scratch;
    WaitUntilGmFlagEquals(HierarchicalSignalAddress(params.peermemInfo, signal),
                          static_cast<int32_t>(params.tilingData->hierarchicalSequence));
#endif
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
}

// Local packing preserves the existing rail format. RDMA reads its stable GM
// copy, never the packer's reusable UB. Payload rows retain their original token
// indices at the gateway; route metadata determines which rows consumers read.
__aicore__ inline void PostHierarchicalDispatch(const MoeStageCommonConfig& common, const Params& params,
                                                const SendMaskConfig& config, AscendTimerDevice* timer = nullptr)
{
    const uint32_t myHost = HierarchicalHost(common.rankId);
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    auto* roceScratch = HierarchicalRoceScratch(params);
    // Put metadata first on the single owned QP. Its last packet carries a
    // full sequence: HNS can release Relay while the following payload is
    // still in flight, without a separate metadata commit WQE.
    WaitHierarchicalLocalSignal(params, HierarchicalSignal::LocalRailPacked, gateway, roceScratch);
    GM_ADDR packedRail = params.peermemInfo.railTopkIdsRecvPtr + myHost * config.railTopkIdsSlotBytes;
    const uint32_t railBytes = static_cast<uint32_t>(CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK));
    RocePutBytes(packedRail, packedRail, railBytes, gateway, roceScratch, timer, AscendTimer::RdmaPhase::DispatchMeta);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // XSCALE has no established packet/SO contract. Preserve its explicit
    // quiet/commit/quiet protocol before posting the independent payload.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        roceScratch, timer, AscendTimer::RdmaPhase::DispatchMeta);
#endif
    // One contiguous Put retains every actual token at its original offset.
    // The later NetPayloadCommit certifies this transfer, including empty or
    // all-local routing. Both GM sources stay immutable until tail quiet.
    const uint32_t rowBytes =
        static_cast<uint32_t>(CalcQuantTokenScaleBytes(common.tokenHiddenDim, 1U, common.topK, false));
    RocePutBytes(params.peermemInfo.relayPayloadPtr, params.peermemInfo.quantTokenScalePtr, common.tokenNum * rowBytes,
                 gateway, roceScratch, timer, AscendTimer::RdmaPhase::DispatchPayload);
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
        return;
    }
    if (job.jobIndex == HierarchicalSendJob()) {
        PostHierarchicalDispatch(common, params, config, timer);
        return;
    }
    if (job.jobIndex >= job.totalJobs || job.totalJobs <= 2U) {
        return;
    }

    // The leader acquires metadata only and never touches the sender's CQ.
    // HNS exposes each aligned 512B packet as a unit; every packet in the final
    // 2KiB fragment has SO. Its last marker certifies all preceding metadata.
    // Do not wait for payload here or put that dependency on the input barrier.
    if (job.jobIndex == 2U + otherHost) {
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        const uint64_t markerOffset = otherHost * config.railTopkIdsSlotBytes +
                                      CalcRailTopkIdsSlotDataBytes(common.tokenNum, common.topK) - sizeof(int32_t);
        auto* marker = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + markerOffset);
#else
        auto* marker = HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetMetaCommit);
#endif
        RoceWaitCommit(marker, params.tilingData->hierarchicalSequence, timer, AscendTimer::RdmaPhase::DispatchMeta,
                       HierarchicalGateway(common.rankId));
        PublishHierarchicalLocalSignal(params, HierarchicalSignal::MetaReady);
    }
    const uint32_t relayCore = job.jobIndex - 2U;
    const uint32_t relayCores = job.totalJobs - 2U;
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
    if (job.jobIndex != HierarchicalSendJob()) {
        return;
    }
    const uint32_t gateway = HierarchicalGateway(common.rankId);
    // Commit the payload before the input barrier can block this owner.
    // HNS posts only: the final kernel-tail quiet protects all GM sources and
    // Sequence. XSCALE retains the conservative phase completion boundary.
    RocePublishCommit(
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::NetPayloadCommit)),
        reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress(params.peermemInfo, HierarchicalSignal::Sequence)), gateway,
        HierarchicalRoceScratch(params), timer, AscendTimer::RdmaPhase::DispatchPayload);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    PublishHierarchicalLocalSignal(params, HierarchicalSignal::TxDone);
#endif
    // Expert ranks acquire this network flag directly at the gateway after
    // their counts are ready. No gateway AIV must republish payload readiness.
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
        auto* scratch = job.jobIndex == HierarchicalSendJob() ? HierarchicalRoceScratch(params) : nullptr;
        RoceWaitFlagEquals(localFlags + peer * INT_CACHELINE, static_cast<uint32_t>(epoch),
                           HierarchicalGateway(common.rankId), scratch, RoceWaitKind::HostProducerBarrier);
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
