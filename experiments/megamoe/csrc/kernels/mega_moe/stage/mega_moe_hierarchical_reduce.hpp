// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_HIERARCHICAL_REDUCE_H
#define MEGA_MOE_HIERARCHICAL_REDUCE_H

#include "mega_moe_hierarchical_combine.hpp"
#include "mega_moe_combine_packet.hpp"
#include "mega_moe_combine_vf.hpp"

namespace MegaMoeImpl {

// Acquire EVERY selected packet: MTE has no cross-packet SO guarantee. One
// strided DMA per contiguous selected-row run gathers 32B packet tails. Queue
// every run before a single MTE2_V / VF / V_S round trip for this token.
// Main UB is free after the owner's foreground work; reduction may overwrite
// it only after this function returns the ready mask in a scalar register.
__aicore__ inline uint32_t GatewayCombinePacketReadyMask(const Params& params, uint32_t token, uint32_t mask)
{
    if (mask == 0U) return 0U;
    constexpr uint32_t blockBytes = 32U;
    const uint32_t rowBytes = CalcGatewayCombinePacketRowBytes(params.tilingData->h);
    const uint32_t packets = rowBytes / RAIL_PACKET_BYTES;
    const uint32_t topK = params.tilingData->topK;
    const uint32_t snapshotBytes = topK * packets * blockBytes;
    GlobalTensor<uint32_t> tails;
    tails.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(params.peermemInfo.gatewayCombinePtr +
                                                             static_cast<uint64_t>(token) * topK * rowBytes +
                                                             RAIL_PACKET_BYTES - blockBytes));
    LocalTensor<uint32_t> snapshot(TPosition::VECCALC, 0U, snapshotBytes / sizeof(uint32_t));
    LocalTensor<uint32_t> result(TPosition::VECCALC, snapshotBytes, blockBytes / sizeof(uint32_t));
    SyncFuncStatic<HardEvent::V_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    for (uint32_t topk = 0U; topk < topK;) {
        if ((mask & (uint32_t{1} << topk)) == 0U) {
            ++topk;
            continue;
        }
        const uint32_t first = topk++;
        while (topk < topK && (mask & (uint32_t{1} << topk)) != 0U) ++topk;
        const uint32_t count = (topk - first) * packets;
        // h <= 8192 and topK <= 32 bound count to 1056, within uint16_t.
        DataCopy(snapshot[first * packets * (blockBytes / sizeof(uint32_t))],
                 tails[first * packets * RAIL_PACKET_WORDS],
                 DataCopyParams{static_cast<uint16_t>(count), 1U,
                                static_cast<uint16_t>(RAIL_PACKET_BYTES / blockBytes - 1U), 0U});
    }
    SyncFuncStatic<HardEvent::MTE2_V, MEGAMOE_ROCE_EVENT_ID>();
    asc_vf_call<CheckGatewayCombinePacketsVF>(reinterpret_cast<__ubuf__ uint32_t*>(snapshot.GetPhyAddr()),
                                              reinterpret_cast<__ubuf__ uint32_t*>(result.GetPhyAddr()), packets, topK,
                                              mask, params.tilingData->hierarchicalSequence);
    SyncFuncStatic<HardEvent::V_S, MEGAMOE_ROCE_EVENT_ID>();
    const uint32_t ready = result.GetValue(0U);
    // Reduction's MTE2 may reuse the status bytes as input immediately.
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    return ready;
}

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
    const bool packetized = UsePacketizedGatewayCombine(params.tilingData);
    const uint32_t inputStride = CalcGatewayCombineInputUbRowBytes(params.tilingData) / sizeof(bfloat16_t);
    const uint32_t inputBytes = inputStride * inputBuffers * sizeof(bfloat16_t);
    const uint32_t sumBytes = CalcGatewayCombineSumUbBytes(params.tilingData);
    LocalTensor<bfloat16_t> input(TPosition::VECCALC, 0U, inputStride * inputBuffers);
    LocalTensor<float> sum(TPosition::VECCALC, inputBytes, sumBytes / sizeof(float));
    LocalTensor<float> weights(TPosition::VECCALC, inputBytes + sumBytes, Ops::Base::CeilAlign(common.topK, 8U));
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
        // The first row's MTE2_V also acquires these weights. No Scalar weight
        // read or per-row S_V handoff is needed by the fused row VF.
        bool first = true;
        uint32_t consumed = 0U;
        if constexpr (DoubleBuffered) {
            // The caller supplies a nonempty ready mask. Seed one load, then
            // prefetch only the next selected expert in the original top-k order.
            uint32_t topk = 0U;
            while ((mask & (uint32_t{1} << topk)) == 0U) ++topk;
            LoadGatewayCombineRow(input, rows, static_cast<uint64_t>(token) * common.topK + topk, hidden, packetized);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        for (uint32_t topk = 0U; topk < common.topK; ++topk) {
            if ((mask & (uint32_t{1} << topk)) == 0U) {
                continue;
            }
            const uint32_t slot = DoubleBuffered ? consumed & 1U : 0U;
            if constexpr (DoubleBuffered) {
                uint32_t next = topk + 1U;
                while (next < common.topK && (mask & (uint32_t{1} << next)) == 0U) ++next;
                if (next < common.topK) {
                    const uint32_t nextSlot = slot ^ 1U;
                    if (consumed != 0U) {
                        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(nextSlot));
                    }
                    LoadGatewayCombineRow(input[nextSlot * inputStride], rows,
                                          static_cast<uint64_t>(token) * common.topK + next, hidden, packetized);
                    SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(nextSlot));
                }
                WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(slot));
            } else {
                const uint64_t row = static_cast<uint64_t>(token) * common.topK + topk;
                LoadGatewayCombineRow(input, rows, row, hidden, packetized);
                SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
            }
            auto* rowUb = reinterpret_cast<__ubuf__ bfloat16_t*>(input[slot * inputStride].GetPhyAddr());
            auto* sumUb = reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr());
            auto* weightUb = reinterpret_cast<__ubuf__ float*>(weights.GetPhyAddr()) + topk;
            if (first) {
                asc_vf_call<AccumulateGatewayCombineRowVF<true>>(rowUb, sumUb, weightUb, hidden);
                first = false;
            } else {
                asc_vf_call<AccumulateGatewayCombineRowVF<false>>(rowUb, sumUb, weightUb, hidden);
            }
            // Order sum stores before the next VF's sum loads. Within the VF,
            // conversion/product stay in registers instead of a temporary UB row.
            PipeBarrier<PIPE_V>();
            if constexpr (DoubleBuffered) {
                SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
                ++consumed;
            } else {
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
        if (UsePacketizedGatewayCombine(params.tilingData)) {
            observed = GatewayCombinePacketReadyMask(params, token, mask);
            ready = ready && observed == mask;
            scannedRows += CombineMaskCount(mask);
        } else {
            for (uint32_t topk = 0U; (ready || diagnostic) && topk < common.topK; ++topk) {
                if ((mask & (uint32_t{1} << topk)) != 0U) {
                    const uint32_t row = token * common.topK + topk;
                    const bool rowReady = snapshot.Ready(params, row, common.tokenNum * common.topK - row, common.topK);
                    ready = ready && rowReady;
                    observed |= rowReady ? uint32_t{1} << topk : 0U;
                    ++scannedRows;
                }
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
        if (UsePacketizedGatewayCombine(params.tilingData)) {
            observed = GatewayCombinePacketReadyMask(params, token, mask);
        } else {
            for (uint32_t topk = 0U; topk < common.topK; ++topk) {
                const uint32_t row = token * common.topK + topk;
                if ((mask & (uint32_t{1} << topk)) != 0U &&
                    snapshot.Ready(params, row, common.tokenNum * common.topK - row, common.topK)) {
                    observed |= uint32_t{1} << topk;
                }
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
