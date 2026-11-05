// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_HIERARCHICAL_REDUCE_H
#define MEGA_MOE_HIERARCHICAL_REDUCE_H

#include "mega_moe_hierarchical_combine.hpp"

namespace MegaMoeImpl {

__aicore__ inline uint32_t CombineMaskCount(uint32_t mask)
{
    uint32_t count = 0U;
    while (mask != 0U) {
        mask &= mask - 1U;
        ++count;
    }
    return count;
}

__aicore__ inline void InitializeHierarchicalReducedCombine(const MoeStageCommonConfig& common, const Params& params,
                                                            HierarchicalCombineProgress& progress, uint32_t lane,
                                                            AscendTimerDevice* timer)
{
    if (progress.initialized) {
        return;
    }
    const uint32_t owners = params.tilingData->combineReduceQps;
    const uint32_t first = common.tokenNum * lane / owners;
    const uint32_t end = common.tokenNum * (lane + 1U) / owners;
    if ((params.tilingData->pipelineFlags & 2U) != 0U) {
        InitializeHierarchicalCombineProgressRange(common, params, progress, first, end, timer);
        progress.nextRow = first * common.topK;
        return;
    }
    InitializeHierarchicalCombineProgress(common, params, progress, timer);
    progress.expectedRows = 0U;
    for (uint32_t token = 0U; token < common.tokenNum; ++token) {
        if (token < first || token >= end) {
            progress.pendingTopkMasks[token] = 0U;
        }
        progress.expectedRows += CombineMaskCount(progress.pendingTopkMasks[token]);
    }
    progress.nextRow = first * common.topK;
}

// One complete token is a nonpreemptible task. AIV1 owners must finish ALL
// routed/shared activation and Dispatch; AIV0 owners must finish ALL routed
// and shared prologues. Both enter at a drained UB/event boundary.
// Preserve the original BF16 expert outputs and FP32 multiply/add ordering within
// this host's top-k subset. Only the final source-rank output is cast to BF16.
template <bool DoubleBuffered = false>
__aicore__ inline void ReduceAndPostHierarchicalToken(const MoeStageCommonConfig& common, const Params& params,
                                                      uint32_t token, uint32_t mask, uint32_t lane,
                                                      __ubuf__ uint8_t* roceScratch, AscendTimerDevice* timer)
{
    const uint32_t hidden = common.tokenHiddenDim;
    constexpr uint32_t inputBuffers = DoubleBuffered ? 2U : 1U;
    LocalTensor<bfloat16_t> input(TPosition::VECCALC, 0U, hidden * inputBuffers);
    LocalTensor<float> value(TPosition::VECCALC, hidden * inputBuffers * sizeof(bfloat16_t), hidden);
    LocalTensor<float> sum(TPosition::VECCALC, hidden * (inputBuffers * sizeof(bfloat16_t) + sizeof(float)), hidden);
    LocalTensor<float> weights(TPosition::VECCALC, hidden * (inputBuffers * sizeof(bfloat16_t) + 2U * sizeof(float)),
                               Ops::Base::CeilAlign(common.topK, 8U));
    GlobalTensor<float> weightGm, partialGm;
    const uint32_t sourceHost = HierarchicalHost(HierarchicalGateway(common.rankId));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.peermemInfo.railTopkIdsRecvPtr +
                                                             sourceHost * CalcRailTopkIdsSlotBytes(params.tilingData) -
                                                             HierarchicalMetadataPrefixBytes(params.tilingData)));
    partialGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.peermemInfo.combinePartialSendPtr));
    GlobalTensor<bfloat16_t> rows;
    rows.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.peermemInfo.gatewayCombinePtr));
    const int64_t start = timer != nullptr && timer->Enabled() ? GetSystemCycle() : 0;
    {
        DataCopyPad(weights, weightGm[static_cast<uint64_t>(token) * common.topK], {1U, common.topK * 4U, 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();
        bool first = true;
        uint32_t consumed = 0U;
        if constexpr (DoubleBuffered) {
            // The caller supplies a nonempty ready mask. Seed one load, then
            // prefetch only the next selected expert in the original top-k order.
            uint32_t topk = 0U;
            while ((mask & (uint32_t{1} << topk)) == 0U) ++topk;
            DataCopy(input, rows[(static_cast<uint64_t>(token) * common.topK + topk) * hidden], hidden);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        for (uint32_t topk = 0U; topk < common.topK; ++topk) {
            if ((mask & (uint32_t{1} << topk)) == 0U) {
                continue;
            }
            if constexpr (DoubleBuffered) {
                const uint32_t slot = consumed & 1U;
                uint32_t next = topk + 1U;
                while (next < common.topK && (mask & (uint32_t{1} << next)) == 0U) ++next;
                if (next < common.topK) {
                    const uint32_t nextSlot = slot ^ 1U;
                    if (consumed != 0U) {
                        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(nextSlot));
                    }
                    DataCopy(input[nextSlot * hidden],
                             rows[(static_cast<uint64_t>(token) * common.topK + next) * hidden], hidden);
                    SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(nextSlot));
                }
                WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(slot));
                Cast(value, input[slot * hidden], RoundMode::CAST_NONE, hidden);
                // Only Cast reads the BF16 input. Its completion permits reuse
                // while the subsequent FP32 multiply/add works in other buffers.
                SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
                ++consumed;
            } else {
                const uint64_t row = static_cast<uint64_t>(token) * common.topK + topk;
                DataCopy(input, rows[row * hidden], hidden);
                SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
                Cast(value, input, RoundMode::CAST_NONE, hidden);
            }
            PipeBarrier<PIPE_V>();
            const float weight = weights.GetValue(topk);
            SyncFuncStatic<HardEvent::S_V, EVENT_ID0>();
            if (first) {
                Muls(sum, value, weight, hidden);
                first = false;
            } else {
                Muls(value, value, weight, hidden);
                PipeBarrier<PIPE_V>();
                Add(sum, sum, value, hidden);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (!DoubleBuffered) {
                SyncFuncStatic<HardEvent::V_MTE2, EVENT_ID0>();
            }
        }
        if constexpr (DoubleBuffered) {
            // Consume the last release event of each used slot before returning
            // to a caller that may reuse UB for ready scans or local Combine.
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            if (consumed > 1U) WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(1U));
        }
        SyncFuncStatic<HardEvent::V_MTE3, EVENT_ID0>();
        DataCopy(partialGm[static_cast<uint64_t>(token) * hidden], sum, hidden);
        SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
    }
    if (timer != nullptr && timer->Enabled()) {
        timer->RecordDetail(AscendTimer::AIV_COMBINE_REDUCE, start, GetSystemCycle(),
                            (static_cast<uint64_t>(lane + 1U) << 32) | token,
                            (static_cast<uint64_t>(mask) << 32) | hidden);
    }
    // The same AIV produced this GM row and owns this QP. No cross-core ready
    // flag or separate sender is needed. Each row stays immutable until quiet.
    const uint64_t offset = static_cast<uint64_t>(token) * hidden * sizeof(float);
    RocePutBytes(params.peermemInfo.combinePartialRecvPtr + offset, params.peermemInfo.combinePartialSendPtr + offset,
                 hidden * sizeof(float), HierarchicalGateway(common.rankId), roceScratch, timer,
                 AscendTimer::RdmaPhase::Combine, false, lane + 1U);
}

__aicore__ inline bool TryProgressHierarchicalReducedCombine(const MoeStageCommonConfig& common, const Params& params,
                                                             HierarchicalCombineProgress& progress, uint32_t lane,
                                                             __ubuf__ uint8_t* roceScratch,
                                                             AscendTimerDevice* timer = nullptr, uint32_t maxPosts = 1U)
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (!UseHierarchicalCombineReduction(params.tilingData) || lane >= params.tilingData->combineReduceQps ||
        maxPosts == 0U) {
        return false;
    }
    InitializeHierarchicalReducedCombine(common, params, progress, lane, timer);
    if (progress.sentRows == progress.expectedRows) {
        return false;
    }
    const bool diagnostic = timer != nullptr && timer->Enabled() && (params.tilingData->pipelineFlags & 64U) != 0U;
    AscendTimerCombineProgressScope timing(diagnostic ? timer : nullptr, AscendTimer::AIV_COMBINE_PROGRESS);
    const uint32_t pendingBefore = progress.expectedRows - progress.sentRows;
    uint32_t scannedRows = 0U;
    timing.StartScan();
    const uint32_t owners = params.tilingData->combineReduceQps;
    const uint32_t first = common.tokenNum * lane / owners;
    const uint32_t end = common.tokenNum * (lane + 1U) / owners;
    uint32_t token = progress.nextRow / common.topK;
    uint32_t posts = 0U;
    HierarchicalCombineReadySnapshot snapshot;
    for (uint32_t checked = 0U; checked < end - first && posts < maxPosts; ++checked) {
        const uint32_t mask = progress.pendingTopkMasks[token];
        bool ready = mask != 0U;
        uint32_t observed = 0U;
        const int64_t scanStart = diagnostic ? GetSystemCycle() : 0;
        for (uint32_t topk = 0U; (ready || diagnostic) && topk < common.topK; ++topk) {
            if ((mask & (uint32_t{1} << topk)) != 0U) {
                const uint32_t row = token * common.topK + topk;
                const bool rowReady = snapshot.Ready(params, row, common.tokenNum * common.topK - row, common.topK);
                ready = ready && rowReady;
                observed |= rowReady ? uint32_t{1} << topk : 0U;
                ++scannedRows;
            }
        }
        if (diagnostic && mask != 0U && observed != progress.deferredTopkMasks[token]) {
            timer->RecordDetail(AscendTimer::AIV_COMBINE_READY, scanStart, GetSystemCycle(),
                                (static_cast<uint64_t>(lane + 1U) << 32) | token,
                                (static_cast<uint64_t>(mask) << 32) | observed);
            progress.deferredTopkMasks[token] = observed;
        }
        if (ready) {
            timing.StopScan();
            if ((params.tilingData->pipelineFlags & 256U) != 0U) {
                ReduceAndPostHierarchicalToken<true>(common, params, token, mask, lane, roceScratch, timer);
            } else {
                ReduceAndPostHierarchicalToken<false>(common, params, token, mask, lane, roceScratch, timer);
            }
            timing.StartScan();
            progress.pendingTopkMasks[token] = 0U;
            progress.sentRows += CombineMaskCount(mask);
            ++posts;
            // The vector task overwrote the UB used for flag snapshots. Scalar
            // positive bits remain valid, but use a fresh snapshot for each token.
            snapshot = {};
        }
        if (++token == end) {
            token = first;
        }
    }
    progress.nextRow = token * common.topK;
    timing.Finish(pendingBefore, progress.expectedRows - progress.sentRows, posts, scannedRows);
    return posts != 0U;
}

// Explicit snapshots at synchronization edges include zero-ready tokens. A
// snapshot is an observation at poll time, not the exact producer-ready time.
// Phase: 1=before export sync, 2=after export sync, 3=before host barrier,
// 4=after host barrier. Progress envelopes show the last poll before each edge.
__aicore__ inline void TraceHierarchicalCombineBoundary(const MoeStageCommonConfig& common, const Params& params,
                                                        HierarchicalCombineProgress& progress, uint32_t lane,
                                                        AscendTimerDevice* timer, uint32_t phase)
{
    if (timer == nullptr || !timer->Enabled() || (params.tilingData->pipelineFlags & 64U) == 0U) {
        return;
    }
    InitializeHierarchicalReducedCombine(common, params, progress, lane, timer);
    const uint32_t owners = params.tilingData->combineReduceQps;
    const uint32_t first = common.tokenNum * lane / owners;
    const uint32_t end = common.tokenNum * (lane + 1U) / owners;
    HierarchicalCombineReadySnapshot snapshot;
    for (uint32_t token = first; token < end; ++token) {
        const int64_t start = GetSystemCycle();
        const uint32_t mask = progress.pendingTopkMasks[token];
        uint32_t observed = 0U;
        for (uint32_t topk = 0U; topk < common.topK; ++topk) {
            const uint32_t row = token * common.topK + topk;
            if ((mask & (uint32_t{1} << topk)) != 0U &&
                snapshot.Ready(params, row, common.tokenNum * common.topK - row, common.topK)) {
                observed |= uint32_t{1} << topk;
            }
        }
        timer->RecordDetail(AscendTimer::AIV_COMBINE_READY, start, GetSystemCycle(),
                            (static_cast<uint64_t>(lane + 1U) << 32) | token,
                            (static_cast<uint64_t>(mask) << 32) | observed);
        progress.deferredTopkMasks[token] = observed;
    }
    const int64_t start = GetSystemCycle();
    timer->RecordDetail(AscendTimer::AIV_COMBINE_SYNC, start, GetSystemCycle(),
                        (static_cast<uint64_t>(lane + 1U) << 32) | phase, progress.expectedRows - progress.sentRows);
}

// AIV1 owners may block only after the producer host barrier. A dedicated
// AIV0 owner may also drain after ALL its prologues: AIV1 producers make
// progress independently and publish each row without waiting for this AIV0.
// Commits still require the host barrier, including lanes with no remote rows.
__aicore__ inline void DrainHierarchicalReducedCombine(const MoeStageCommonConfig& common, const Params& params,
                                                       HierarchicalCombineProgress& progress, uint32_t lane,
                                                       __ubuf__ uint8_t* roceScratch,
                                                       AscendTimerDevice* timer = nullptr)
{
    InitializeHierarchicalReducedCombine(common, params, progress, lane, timer);
    RoceWaitState wait{};
    while (progress.sentRows < progress.expectedRows) {
        if (!TryProgressHierarchicalReducedCombine(common, params, progress, lane, roceScratch, timer, 8U)) {
            RoceCheckWait(wait, HierarchicalGateway(common.rankId), roceScratch, RoceWaitKind::CombineRows,
                          progress.expectedRows, progress.sentRows, nullptr, lane + 1U);
        } else {
            wait = {};
        }
    }
}

}  // namespace MegaMoeImpl
#endif
