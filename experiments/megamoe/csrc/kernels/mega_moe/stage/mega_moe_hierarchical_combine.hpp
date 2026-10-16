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
    bool initialized = false;
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
    auto* sourceTopk = reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + sourceSlotOffset);
    const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
    for (uint32_t token = 0U; token < common.tokenNum; ++token) {
        uint32_t pending = 0U;
        for (uint32_t topk = 0U; topk < common.topK; ++topk) {
            const uint32_t row = token * common.topK + topk;
            const uint32_t packedOffset =
                row / RAIL_PACKET_DATA_WORDS * RAIL_PACKET_WORDS + row % RAIL_PACKET_DATA_WORDS;
            const int32_t expert = ReadGmByPassDCache(sourceTopk + packedOffset);
            if (expert >= 0 && static_cast<uint32_t>(expert) / expertsPerHost == localHost) {
                pending |= uint32_t(1) << topk;
                ++progress.expectedRows;
            }
        }
        progress.pendingTopkMasks[token] = pending;
    }
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

// Probe once without waiting for any unfinished producer. Adjacent ready slots
// form one Put; an unready/unused slot ends the run. No top-k weighting, local
// reduction, padding copy, or quantization is introduced by the gateway.
//
// Only the rank's designated communication owner may call this routine. Its UB
// scratch and QP are the same resources used by the completed Dispatch stage.
__aicore__ inline bool TryProgressHierarchicalCombine(const MoeStageCommonConfig& common, const Params& params,
                                                      HierarchicalCombineProgress& progress,
                                                      __ubuf__ uint8_t* roceScratch)
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (!UseHierarchicalRoce(params.tilingData)) {
        return false;
    }
    InitializeHierarchicalCombineProgress(common, params, progress);
    const uint32_t rowCount = common.tokenNum * common.topK;
    const uint32_t rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    const uint32_t destination = HierarchicalGateway(common.rankId);
    bool madeProgress = false;
    for (uint32_t row = 0U; row < rowCount;) {
        if (!IsHierarchicalCombineRowReady(common, params, progress, row)) {
            ++row;
            continue;
        }
        const uint32_t runStart = row++;
        while (row < rowCount && IsHierarchicalCombineRowReady(common, params, progress, row)) {
            ++row;
        }
        const uint64_t byteOffset = static_cast<uint64_t>(runStart) * rowBytes;
        RocePutBytes(params.peermemInfo.combineSendPtr + byteOffset, params.peermemInfo.gatewayCombinePtr + byteOffset,
                     (row - runStart) * rowBytes, destination, roceScratch);
        for (uint32_t sent = runStart; sent < row; ++sent) {
            progress.pendingTopkMasks[sent / common.topK] &= ~(uint32_t(1) << (sent % common.topK));
        }
        progress.sentRows += row - runStart;
        madeProgress = true;
    }
    return madeProgress;
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
                                                HierarchicalCombineProgress& progress, __ubuf__ uint8_t* roceScratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!UseHierarchicalRoce(params.tilingData)) {
        return;
    }
    InitializeHierarchicalCombineProgress(common, params, progress);
    while (progress.sentRows < progress.expectedRows) {
        if (!TryProgressHierarchicalCombine(common, params, progress, roceScratch)) {
            const int64_t waitStart = GetSystemCycle();
            while (GetSystemCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        }
    }
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_HIERARCHICAL_COMBINE_H
