/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */

#ifndef MEGA_MOE_HIERARCHICAL_COMBINE_H
#define MEGA_MOE_HIERARCHICAL_COMBINE_H

#include "../common/mega_moe_roce.hpp"
#include "../common/mega_moe_utils.hpp"

namespace MegaMoeImpl {

// Private scalar state of the single per-rank QP owner. One uint32 mask covers
// every supported top-k slot of a token. Shared epoch flags are never cleared:
// only this private pending mask changes as rows are submitted to the NIC.
struct HierarchicalCombineProgress {
    uint32_t pendingTopkMasks[HIERARCHICAL_MAX_TOKENS + 1U]{};
    uint32_t expectedRows = 0U;
    uint32_t sentRows = 0U;
    uint32_t nextRow = 0U;
    bool initialized = false;
};

// The default retains a full sweep for callers with infrequent progress points.
// A8W4 supplies separate budgets for its priority scheduler and prologue owner.
struct HierarchicalCombineProgressBudget {
    uint32_t maxPosts = 0xffffffffU;
    uint32_t maxScannedRows = 0xffffffffU;
};

// The caller has already acquired the paired source's metadata during Dispatch.
// Count only experts on this gateway's host. Original (token, top-k) coordinates
// are retained, including holes and duplicate destination ranks.
__aicore__ inline void InitializeHierarchicalCombineProgress(const MoeStageCommonConfig& common, const Params& params,
                                                             HierarchicalCombineProgress& progress,
                                                             AscendTimerDevice* timer = nullptr)
{
    if (progress.initialized) {
        return;
    }
    AscendTimerFixedScope initialization(timer, AscendTimer::AIV_COMBINE_INIT);
    const uint32_t localHost = HierarchicalHost(common.rankId);
    const uint32_t sourceHost = HierarchicalHost(HierarchicalGateway(common.rankId));
    const uint64_t sourceSlotOffset = static_cast<uint64_t>(sourceHost) * CalcRailTopkIdsSlotBytes(params.tilingData);
    GlobalTensor<int32_t> sourceTopk;
    sourceTopk.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + sourceSlotOffset));
    constexpr uint32_t scratchWords = MEGAMOE_ROCE_SCRATCH_BYTES / sizeof(int32_t);
    LocalTensor<int32_t> topkScratch(TPosition::VECCALC, params.tilingData->roceScratchUbOffset, scratchWords);
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    progress.expectedRows = 0U;
    for (uint32_t token = 0U; token < common.tokenNum; ++token) {
        progress.pendingTopkMasks[token] = 0U;
    }
    // Dispatch transferred whole 512B packets. Read each in two aligned chunks
    // using the owner's reserved 256B UB; the last word is a marker, not an ID.
    // This runs only before the first Combine Post, after Dispatch is drained.
    const uint32_t rowCount = common.tokenNum * common.topK;
    for (uint32_t packet = 0U, firstRow = 0U; firstRow < rowCount;
         ++packet, firstRow += RAIL_PACKET_DATA_WORDS) {
        const uint32_t remaining = rowCount - firstRow;
        const uint32_t validPacket = remaining < RAIL_PACKET_DATA_WORDS ? remaining : RAIL_PACKET_DATA_WORDS;
        for (uint32_t firstWord = 0U; firstWord < validPacket; firstWord += scratchWords) {
            SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
            DataCopy(topkScratch, sourceTopk[packet * RAIL_PACKET_WORDS + firstWord], scratchWords);
            SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
            const uint32_t left = validPacket - firstWord;
            const uint32_t valid = left < scratchWords ? left : scratchWords;
            for (uint32_t i = 0U; i < valid; ++i) {
                const int32_t expert = topkScratch.GetValue(i);
                if (expert >= 0 && static_cast<uint32_t>(expert) / expertsPerHost == localHost) {
                    const uint32_t row = firstRow + firstWord + i;
                    progress.pendingTopkMasks[row / common.topK] |= uint32_t(1) << (row % common.topK);
                    ++progress.expectedRows;
                }
            }
        }
    }
    // Finish scalar reads before the RDMA backend reuses this UB for its WQE.
    SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
    progress.initialized = true;
}

// One small DMA gathers the first 32B of up to eight 512B flag slots. Keep only
// the epoch matches in scalar state: Put can immediately overwrite the shared
// 256B scratch. This snapshot lives for one progress call, so a negative result
// never hides a later publication on the next sweep.
struct HierarchicalCombineReadySnapshot {
    uint32_t firstRow = 0U;
    uint32_t rowCount = 0U;
    uint32_t readyMask = 0U;

    __aicore__ inline bool Ready(const Params& params, uint32_t row, uint32_t rowsLeft, uint32_t scansLeft)
    {
        if (row < firstRow || row - firstRow >= rowCount) {
            constexpr uint32_t blockBytes = 32U;
            constexpr uint32_t batchRows = MEGAMOE_ROCE_SCRATCH_BYTES / blockBytes;
            static_assert(batchRows <= 32U && HIERARCHICAL_FLAG_BYTES % blockBytes == 0U);
            rowCount = rowsLeft < batchRows ? rowsLeft : batchRows;
            rowCount = scansLeft < rowCount ? scansLeft : rowCount;
            firstRow = row;
            GlobalTensor<int32_t> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                params.peermemInfo.gatewayCombineReadyPtr + static_cast<uint64_t>(row) * HIERARCHICAL_FLAG_BYTES));
            flags.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            LocalTensor<int32_t> snapshot(TPosition::VECCALC, params.tilingData->roceScratchUbOffset,
                                           MEGAMOE_ROCE_SCRATCH_BYTES / sizeof(int32_t));
            SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
            // blockLen/srcStride/dstStride are measured in 32B blocks.
            DataCopy(snapshot, flags,
                     DataCopyParams{static_cast<uint16_t>(rowCount), 1U,
                                    static_cast<uint16_t>(HIERARCHICAL_FLAG_BYTES / blockBytes - 1U), 0U});
            SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
            readyMask = 0U;
            for (uint32_t i = 0U; i < rowCount; ++i) {
                if (static_cast<uint32_t>(snapshot.GetValue(i * (blockBytes / sizeof(int32_t)))) ==
                    params.tilingData->hierarchicalSequence) {
                    readyMask |= uint32_t(1) << i;
                }
            }
            SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
        }
        return (readyMask & (uint32_t(1) << (row - firstRow))) != 0U;
    }
};

// Keep the current token's remaining bits in scalar state. Advancing still
// visits every row for exact scan-budget accounting, including holes and sent
// rows, but needs no per-row division or pending-mask reload.
struct HierarchicalCombineScanCursor {
    uint32_t row;
    uint32_t token;
    uint32_t topk;
    uint32_t pending;

    __aicore__ inline HierarchicalCombineScanCursor(const HierarchicalCombineProgress& progress, uint32_t topK)
        : row(progress.nextRow),
          token(row / topK),
          topk(row - token * topK),
          pending(progress.pendingTopkMasks[token] >> topk)
    {
    }

    __aicore__ inline bool Ready(const Params& params, HierarchicalCombineReadySnapshot& snapshot,
                                 uint32_t rowsLeft, uint32_t scansLeft) const
    {
        if ((pending & 1U) == 0U) {
            return false;
        }
        return snapshot.Ready(params, row, rowsLeft, scansLeft);
    }

    __aicore__ inline void Advance(const MoeStageCommonConfig& common, const HierarchicalCombineProgress& progress)
    {
        ++row;
        pending >>= 1U;
        if (++topk == common.topK) {
            topk = 0U;
            ++token;
            pending = token < common.tokenNum ? progress.pendingTopkMasks[token] : 0U;
        }
    }

    __aicore__ inline void Wrap(const HierarchicalCombineProgress& progress)
    {
        row = token = topk = 0U;
        pending = progress.pendingTopkMasks[0];
    }
};

// Called only after Put returns. A run may cross tokens, but clearing each
// token once avoids repeated division and read/modify/write for every sent row.
__aicore__ inline void ClearHierarchicalCombineRun(HierarchicalCombineProgress& progress, uint32_t topK, uint32_t token,
                                                   uint32_t topk, uint32_t rows)
{
    while (rows != 0U) {
        const uint32_t available = topK - topk;
        const uint32_t count = rows < available ? rows : available;
        // count is in [1, 32]; neither shift can be 32, including a full token.
        const uint32_t mask = (0xffffffffU >> (32U - count)) << topk;
        progress.pendingTopkMasks[token] &= ~mask;
        rows -= count;
        ++token;
        topk = 0U;
    }
}

// Probe at most one sweep without waiting for unfinished producers. The cursor
// survives a bounded call, so late slots cannot starve behind early ready rows.
// Adjacent ready slots form one Put; holes, the row budget, and the tensor end
// terminate a run. No weighting, reduction, padding copy, or quantization occurs.
//
// Only the rank's designated communication owner may call this routine. Its UB
// scratch and QP are the same resources used by the completed Dispatch stage.
__aicore__ inline bool TryProgressHierarchicalCombine(const MoeStageCommonConfig& common, const Params& params,
                                                      HierarchicalCombineProgress& progress,
                                                      __ubuf__ uint8_t* roceScratch, AscendTimerDevice* timer = nullptr,
                                                      HierarchicalCombineProgressBudget budget = {})
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (!UseHierarchicalRoce(params.tilingData)) {
        return false;
    }
    if (budget.maxPosts == 0U || budget.maxScannedRows == 0U) {
        return false;
    }
    InitializeHierarchicalCombineProgress(common, params, progress, timer);
    if (progress.sentRows == progress.expectedRows) {
        return false;
    }
    const uint32_t pendingBefore = progress.expectedRows - progress.sentRows;
    AscendTimerCombineProgressScope timing(timer, AscendTimer::AIV_COMBINE_PROGRESS);
    timing.StartScan();
    const uint32_t rowCount = common.tokenNum * common.topK;
    const uint32_t scanLimit = budget.maxScannedRows < rowCount ? budget.maxScannedRows : rowCount;
    const uint32_t rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    const uint32_t destination = HierarchicalGateway(common.rankId);
    HierarchicalCombineScanCursor cursor(progress, common.topK);
    HierarchicalCombineReadySnapshot snapshot;
    uint32_t scanned = 0U;
    uint32_t posts = 0U;
    while (scanned < scanLimit && posts < budget.maxPosts) {
        ++scanned;
        if (!cursor.Ready(params, snapshot, rowCount - cursor.row, scanLimit - scanned + 1U)) {
            cursor.Advance(common, progress);
            if (cursor.row == rowCount) {
                cursor.Wrap(progress);
            }
            continue;
        }
        const uint32_t runStart = cursor.row;
        const uint32_t runToken = cursor.token;
        const uint32_t runTopk = cursor.topk;
        cursor.Advance(common, progress);
        // Probe a terminating hole only once. It remains pending and is retried
        // on a later sweep if its producer publishes readiness in the meantime.
        bool skippedHole = false;
        while (cursor.row < rowCount && scanned < scanLimit) {
            ++scanned;
            if (!cursor.Ready(params, snapshot, rowCount - cursor.row, scanLimit - scanned + 1U)) {
                skippedHole = true;
                break;
            }
            cursor.Advance(common, progress);
        }
        const uint32_t runRows = cursor.row - runStart;
        const uint64_t byteOffset = static_cast<uint64_t>(runStart) * rowBytes;
        timing.StopScan();
        RocePutBytes(params.peermemInfo.combineSendPtr + byteOffset, params.peermemInfo.gatewayCombinePtr + byteOffset,
                     runRows * rowBytes, destination, roceScratch, timer, AscendTimer::RdmaPhase::Combine);
        timing.StartScan();
        ClearHierarchicalCombineRun(progress, common.topK, runToken, runTopk, runRows);
        // The cached bits begin at cursor.row, beyond the cleared run. A wrap
        // reloads token zero so previously submitted rows cannot reappear.
        progress.sentRows += runRows;
        ++posts;
        if (skippedHole) {
            cursor.Advance(common, progress);
        }
        if (cursor.row == rowCount) {
            cursor.Wrap(progress);
        }
    }
    progress.nextRow = cursor.row;
    timing.Finish(pendingBefore, progress.expectedRows - progress.sentRows, posts, scanned);
    return posts != 0U;
}

// Final drain is legal only after all local AIV producer work is complete and
// the caller has passed the 16-rank host barrier. Calling it in the producer loop
// can prevent that same owner from producing rows needed by a gateway.
//
// This only posts the outstanding rows; it does not complete the transfers.
// The caller must use RocePublishCommit after this returns to quiet the payload,
// publish CombineCommit, and quiet the commit before waiting for its paired
// gateway's commit. Outbound-before-inbound ordering on both hosts avoids
// a cycle. The commit is required even when expectedRows is zero and also closes
// this invocation's lifetime before the next one can reuse Dispatch buffers.
__aicore__ inline void DrainHierarchicalCombine(const MoeStageCommonConfig& common, const Params& params,
                                                HierarchicalCombineProgress& progress, __ubuf__ uint8_t* roceScratch,
                                                AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!UseHierarchicalRoce(params.tilingData)) {
        return;
    }
    InitializeHierarchicalCombineProgress(common, params, progress, timer);
    const uint32_t pendingBefore = progress.expectedRows - progress.sentRows;
    AscendTimerCombineProgressScope drain(timer, AscendTimer::AIV_COMBINE_DRAIN);
    while (progress.sentRows < progress.expectedRows) {
        if (!TryProgressHierarchicalCombine(common, params, progress, roceScratch, timer, {})) {
            const int64_t waitStart = GetSystemCycle();
            while (GetSystemCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        }
    }
    // The parent includes child progress calls and backoff. Their scan/POST
    // counters belong to the child records, not to this completion envelope.
    drain.Finish(pendingBefore, progress.expectedRows - progress.sentRows, 0U, 0U);
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_HIERARCHICAL_COMBINE_H
