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
    InitializeHierarchicalCombineProgress(common, params, progress, timer);
    const uint32_t owners = params.tilingData->combineReduceQps;
    const uint32_t first = common.tokenNum * lane / owners;
    const uint32_t end = common.tokenNum * (lane + 1U) / owners;
    progress.expectedRows = 0U;
    for (uint32_t token = 0U; token < common.tokenNum; ++token) {
        if (token < first || token >= end) {
            progress.pendingTopkMasks[token] = 0U;
        }
        progress.expectedRows += CombineMaskCount(progress.pendingTopkMasks[token]);
    }
    progress.nextRow = first * common.topK;
}

// One complete token is a nonpreemptible task. Call only after this AIV has
// finished ALL routed/shared activation and Dispatch, at a drained UB boundary.
// Preserve the original BF16 expert outputs and FP32 multiply/add ordering within
// this host's top-k subset. Only the final source-rank output is cast to BF16.
__aicore__ inline void ReduceAndPostHierarchicalToken(const MoeStageCommonConfig& common, const Params& params,
                                                      uint32_t token, uint32_t mask, uint32_t lane,
                                                      __ubuf__ uint8_t* roceScratch, AscendTimerDevice* timer)
{
    const uint32_t hidden = common.tokenHiddenDim;
    LocalTensor<bfloat16_t> input(TPosition::VECCALC, 0U, hidden);
    LocalTensor<float> value(TPosition::VECCALC, hidden * sizeof(bfloat16_t), hidden);
    LocalTensor<float> sum(TPosition::VECCALC, hidden * (sizeof(bfloat16_t) + sizeof(float)), hidden);
    LocalTensor<float> weights(TPosition::VECCALC, hidden * (sizeof(bfloat16_t) + 2U * sizeof(float)),
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
        for (uint32_t topk = 0U; topk < common.topK; ++topk) {
            if ((mask & (uint32_t{1} << topk)) == 0U) {
                continue;
            }
            const uint64_t row = static_cast<uint64_t>(token) * common.topK + topk;
            DataCopy(input, rows[row * hidden], hidden);
            SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
            Cast(value, input, RoundMode::CAST_NONE, hidden);
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
            SyncFuncStatic<HardEvent::V_MTE2, EVENT_ID0>();
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
    const uint32_t owners = params.tilingData->combineReduceQps;
    const uint32_t first = common.tokenNum * lane / owners;
    const uint32_t end = common.tokenNum * (lane + 1U) / owners;
    uint32_t token = progress.nextRow / common.topK;
    uint32_t posts = 0U;
    HierarchicalCombineReadySnapshot snapshot;
    for (uint32_t checked = 0U; checked < end - first && posts < maxPosts; ++checked) {
        const uint32_t mask = progress.pendingTopkMasks[token];
        bool ready = mask != 0U;
        for (uint32_t topk = 0U; ready && topk < common.topK; ++topk) {
            if ((mask & (uint32_t{1} << topk)) != 0U) {
                const uint32_t row = token * common.topK + topk;
                ready = snapshot.Ready(params, row, common.tokenNum * common.topK - row, common.topK);
            }
        }
        if (ready) {
            ReduceAndPostHierarchicalToken(common, params, token, mask, lane, roceScratch, timer);
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
    return posts != 0U;
}

// Only the post-producer host barrier permits blocking drain. Each owner posts
// its own independent commit even when this lane has no remote tokens.
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
