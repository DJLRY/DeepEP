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
                                                             HierarchicalCombineProgress& progress)
{
    if (progress.initialized) {
        return;
    }
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

__aicore__ inline bool IsHierarchicalCombineRowReady(const MoeStageCommonConfig& common, const Params& params,
                                                     const HierarchicalCombineProgress& progress, uint32_t row)
{
    const uint32_t token = row / common.topK;
    const uint32_t topk = row % common.topK;
    if ((progress.pendingTopkMasks[token] & (uint32_t(1) << topk)) == 0U) {
        return false;
    }
    auto* ready = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.gatewayCombineReadyPtr +
                                                    static_cast<uint64_t>(row) * HIERARCHICAL_FLAG_BYTES);
    return static_cast<uint32_t>(ReadGmByPassDCache(ready)) == params.tilingData->hierarchicalSequence;
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
    InitializeHierarchicalCombineProgress(common, params, progress);
    if (progress.sentRows == progress.expectedRows) {
        return false;
    }
    const uint32_t rowCount = common.tokenNum * common.topK;
    const uint32_t scanLimit = budget.maxScannedRows < rowCount ? budget.maxScannedRows : rowCount;
    const uint32_t rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    const uint32_t destination = HierarchicalGateway(common.rankId);
    uint32_t row = progress.nextRow;
    uint32_t scanned = 0U;
    uint32_t posts = 0U;
    while (scanned < scanLimit && posts < budget.maxPosts) {
        ++scanned;
        if (!IsHierarchicalCombineRowReady(common, params, progress, row)) {
            ++row;
            if (row == rowCount) {
                row = 0U;
            }
            continue;
        }
        const uint32_t runStart = row++;
        // Probe a terminating hole only once. It remains pending and is retried
        // on a later sweep if its producer publishes readiness in the meantime.
        bool skippedHole = false;
        while (row < rowCount && scanned < scanLimit) {
            ++scanned;
            if (!IsHierarchicalCombineRowReady(common, params, progress, row)) {
                skippedHole = true;
                break;
            }
            ++row;
        }
        const uint64_t byteOffset = static_cast<uint64_t>(runStart) * rowBytes;
        RocePutBytes(params.peermemInfo.combineSendPtr + byteOffset, params.peermemInfo.gatewayCombinePtr + byteOffset,
                     (row - runStart) * rowBytes, destination, roceScratch, timer, AscendTimer::RdmaPhase::Combine);
        for (uint32_t sent = runStart; sent < row; ++sent) {
            progress.pendingTopkMasks[sent / common.topK] &= ~(uint32_t(1) << (sent % common.topK));
        }
        progress.sentRows += row - runStart;
        ++posts;
        if (skippedHole) {
            ++row;
        }
        if (row == rowCount) {
            row = 0U;
        }
    }
    progress.nextRow = row;
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
    InitializeHierarchicalCombineProgress(common, params, progress);
    while (progress.sentRows < progress.expectedRows) {
        if (!TryProgressHierarchicalCombine(common, params, progress, roceScratch, timer, {})) {
            const int64_t waitStart = GetSystemCycle();
            while (GetSystemCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        }
    }
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_HIERARCHICAL_COMBINE_H
