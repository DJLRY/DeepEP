// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_HIERARCHICAL_COMBINE_SYNC_H
#define MEGA_MOE_HIERARCHICAL_COMBINE_SYNC_H

#include "mega_moe_hierarchical_reduce.hpp"

namespace MegaMoeImpl {

// All callers have finished activation, Dispatch and their local Combine rows.
// Only a QP owner can poll/post that QP. One ready token per call is the bound;
// no token-ready condition is allowed to block producer rendezvous.
__aicore__ inline bool ProgressCombineDuringSync(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                 const Params& params, HierarchicalCombineProgress& progress,
                                                 AscendTimerDevice* timer)
{
    if (!IsHierarchicalCombineOwner(params.tilingData, job.jobIndex)) {
        return false;
    }
    return TryProgressHierarchicalReducedCombine(common, params, progress,
                                                 HierarchicalCombineLane(params.tilingData, job.jobIndex),
                                                 HierarchicalRoceScratch(params), timer, 1U);
}

// Every AIV publishes its own arrival before any owner waits. Nonowners can
// enter SyncAll immediately; owners continue bounded progress until every AIV
// has arrived, then join the SAME hardware barrier. Distinct phase/worker lines
// never reset and use the invocation sequence, so no launch clears a late store.
__aicore__ inline void JoinHierarchicalCombineProducers(const AivJobContext& job, const MoeStageCommonConfig& common,
                                                        const Params& params, HierarchicalCombineProgress& progress,
                                                        AscendTimerDevice* timer, uint32_t phase)
{
    auto* base = params.peermemInfo.combineProducerDonePtr +
                 static_cast<uint64_t>(phase) * HIERARCHICAL_MAX_AIV_WORKERS * HIERARCHICAL_FLAG_BYTES;
    PipeBarrier<PIPE_ALL>();
    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
    DataSyncBarrier<MemDsbT::DDR>();
    WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(base + job.jobIndex * HIERARCHICAL_FLAG_BYTES),
                        static_cast<int32_t>(params.tilingData->hierarchicalSequence));
    DataSyncBarrier<MemDsbT::DDR>();
    if (IsHierarchicalCombineOwner(params.tilingData, job.jobIndex)) {
        RoceWaitState wait{};
        const uint32_t lane = HierarchicalCombineLane(params.tilingData, job.jobIndex);
        for (uint32_t worker = 0U; worker < job.totalJobs; ++worker) {
            auto* flag = reinterpret_cast<__gm__ int32_t*>(base + worker * HIERARCHICAL_FLAG_BYTES);
            while (static_cast<uint32_t>(ReadGmByPassDCache(flag)) != params.tilingData->hierarchicalSequence) {
                ProgressCombineDuringSync(job, common, params, progress, timer);
                RoceCheckWait(wait, HierarchicalGateway(common.rankId), HierarchicalRoceScratch(params),
                              RoceWaitKind::HostProducerBarrier, params.tilingData->hierarchicalSequence, worker,
                              nullptr, lane + 1U);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// Same host handshake, epochs and final SyncAll as the baseline. Additionally
// progress while waiting on this worker's peer and on other local handshake
// workers. Commits and the blocking row drain remain AFTER this function.
__aicore__ inline void CrossRankSyncInHierarchicalHostWithProgress(const AivJobContext& job,
                                                                   const MoeStageCommonConfig& common,
                                                                   const Params& params,
                                                                   HierarchicalCombineProgress& progress,
                                                                   AscendTimerDevice* timer)
{
    auto* counter =
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                          static_cast<uint64_t>(job.jobIndex) * RANK_SYNC_COUNTER_SLOT_BYTES);
    const int32_t epoch = ReadGmByPassDCache(counter) + 1;
    const uint32_t firstRank = HierarchicalHost(common.rankId) * HIERARCHICAL_RANKS_PER_HOST;
    auto* localFlags = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.rankSyncInWorldPtr);
    const bool owner = IsHierarchicalCombineOwner(params.tilingData, job.jobIndex);
    const uint32_t lane = HierarchicalCombineLane(params.tilingData, job.jobIndex);
    for (uint32_t localRank = job.jobIndex; localRank < HIERARCHICAL_RANKS_PER_HOST; localRank += job.totalJobs) {
        const uint32_t peer = firstRank + localRank;
        auto* remote = reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[peer]) + common.rankId * INT_CACHELINE;
        WriteGmByPassDCache(remote, epoch);
        DataSyncBarrier<MemDsbT::DDR>();
        RoceWaitState wait{};
        auto* flag = localFlags + peer * INT_CACHELINE;
        while (ReadGmByPassDCache(flag) != epoch) {
            ProgressCombineDuringSync(job, common, params, progress, timer);
            RoceCheckWait(wait, HierarchicalGateway(common.rankId), owner ? HierarchicalRoceScratch(params) : nullptr,
                          RoceWaitKind::HostProducerBarrier, static_cast<uint32_t>(epoch), 0U, nullptr,
                          owner ? lane + 1U : 1U);
        }
    }
    WriteGmByPassDCache(counter, epoch);
    JoinHierarchicalCombineProducers(job, common, params, progress, timer, 1U);
}

}  // namespace MegaMoeImpl
#endif
