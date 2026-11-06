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
    // Bounded progress: deferred run starts. Completion progress: positive
    // ready observations for the current epoch, cleared when a whole run posts.
    uint32_t deferredTopkMasks[HIERARCHICAL_MAX_TOKENS + 1U]{};
    uint32_t expectedRows = 0U;
    uint32_t sentRows = 0U;
    uint32_t nextRow = 0U;
    bool initialized = false;
    bool retryDeferred = false;
};

// The default retains a full sweep for callers with infrequent progress points.
// A8W4 supplies separate budgets for its priority scheduler and prologue owner.
struct HierarchicalCombineProgressBudget {
    uint32_t maxPosts = 0xffffffffU;
    uint32_t maxScannedRows = 0xffffffffU;
    // Hold any incomplete contiguous region at most once per start row, even
    // when its ready fragment is large. Route/sent gaps end a region; a complete
    // region posts immediately. Never wait here: the owner may be its producer.
    bool deferPartialRuns = false;
    // After an empty full sweep, let one scheduler visit pass without flag DMA.
    // Partial sweeps keep advancing; the final drain disables both policies.
    bool backoffAfterEmptySweep = false;
};

// The caller has already acquired the paired source's metadata during Dispatch.
// Count only experts on this gateway's host. Original (token, top-k) coordinates
// are retained, including holes and duplicate destination ranks.
__aicore__ inline void InitializeHierarchicalCombineProgressRange(const MoeStageCommonConfig& common,
                                                                  const Params& params,
                                                                  HierarchicalCombineProgress& progress,
                                                                  uint32_t firstToken, uint32_t endToken,
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
    for (uint32_t token = firstToken; token < endToken; ++token) {
        progress.pendingTopkMasks[token] = 0U;
        progress.deferredTopkMasks[token] = 0U;
    }
    // Dispatch transferred whole 512B packets. Read each in two aligned chunks
    // using the owner's reserved 256B UB; the last word is a marker, not an ID.
    // This runs only before the first Combine Post, after Dispatch is drained.
    const uint32_t rowBegin = firstToken * common.topK;
    const uint32_t rowEnd = endToken * common.topK;
    for (uint32_t packet = rowBegin / RAIL_PACKET_DATA_WORDS;
         rowBegin < rowEnd && packet * RAIL_PACKET_DATA_WORDS < rowEnd; ++packet) {
        const uint32_t firstRow = packet * RAIL_PACKET_DATA_WORDS;
        const uint32_t remaining = rowEnd - firstRow;
        const uint32_t validPacket = remaining < RAIL_PACKET_DATA_WORDS ? remaining : RAIL_PACKET_DATA_WORDS;
        const uint32_t beginWord = rowBegin > firstRow ? rowBegin - firstRow : 0U;
        // Read only aligned 256B chunks intersecting this lane. Never interpret
        // the packet's sequence marker or neighbouring lanes' IDs as routes.
        for (uint32_t firstWord = beginWord / scratchWords * scratchWords; firstWord < validPacket;
             firstWord += scratchWords) {
            SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
            DataCopy(topkScratch, sourceTopk[packet * RAIL_PACKET_WORDS + firstWord], scratchWords);
            SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
            const uint32_t left = validPacket - firstWord;
            const uint32_t valid = left < scratchWords ? left : scratchWords;
            const uint32_t firstValid = beginWord > firstWord ? beginWord - firstWord : 0U;
            for (uint32_t i = firstValid; i < valid; ++i) {
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

__aicore__ inline void InitializeHierarchicalCombineProgress(const MoeStageCommonConfig& common, const Params& params,
                                                             HierarchicalCombineProgress& progress,
                                                             AscendTimerDevice* timer = nullptr)
{
    InitializeHierarchicalCombineProgressRange(common, params, progress, 0U, common.tokenNum, timer);
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

// Keep the current token's remaining bits in scalar state. Scan budgets count
// logical row positions, including holes and sent rows. The sparse path skips
// an empty token suffix in one step while charging the same logical budget.
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

    __aicore__ inline bool Ready(const Params& params, HierarchicalCombineReadySnapshot& snapshot, uint32_t rowsLeft,
                                 uint32_t scansLeft) const
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

    // Only call when pending == 0: earlier slots in this token may still be
    // pending, but no slot at or after the cursor can become pending again.
    // Respect partial budgets and avoid shifting by 32 for a full top-k token.
    __aicore__ inline uint32_t SkipEmptySuffix(const MoeStageCommonConfig& common,
                                               const HierarchicalCombineProgress& progress, uint32_t scansLeft)
    {
        const uint32_t remaining = common.topK - topk;
        const uint32_t skipped = remaining < scansLeft ? remaining : scansLeft;
        row += skipped;
        topk += skipped;
        if (topk == common.topK) {
            topk = 0U;
            ++token;
            pending = token < common.tokenNum ? progress.pendingTopkMasks[token] : 0U;
        }
        return skipped;
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

// Zero-safe, portable bit scan; avoids depending on a device-specific scalar
// intrinsic when walking whole route segments rather than individual slots.
__aicore__ inline uint32_t HierarchicalCombineTrailingZeros(uint32_t bits)
{
    if (bits == 0U) return 32U;
    uint32_t count = 0U;
    if ((bits & 0xffffU) == 0U) {
        count += 16U;
        bits >>= 16U;
    }
    if ((bits & 0xffU) == 0U) {
        count += 8U;
        bits >>= 8U;
    }
    if ((bits & 0xfU) == 0U) {
        count += 4U;
        bits >>= 4U;
    }
    if ((bits & 0x3U) == 0U) {
        count += 2U;
        bits >>= 2U;
    }
    return count + ((bits & 1U) == 0U ? 1U : 0U);
}

__aicore__ inline uint32_t HierarchicalCombineLowBits(uint32_t count)
{
    return count == 32U ? 0xffffffffU : (uint32_t(1) << count) - 1U;
}

// Completion owns no foreground work. Route holes define immutable run
// boundaries; a late producer delays its whole run, never splits it into POSTs.
// Reuse the two existing per-token bitmaps instead of allocating a worst-case
// B*topK/2 descriptor array. Only whole runs are removed from pendingTopkMasks,
// so their boundaries survive retries without rereading routing metadata.
struct HierarchicalCombineRunSender {
    const MoeStageCommonConfig& common;
    const Params& params;
    HierarchicalCombineProgress& progress;
    __ubuf__ uint8_t* scratch;
    AscendTimerDevice* timer;
    AscendTimerCombineProgressScope& timing;
    uint32_t posts = 0U;
    uint32_t scanned = 0U;

    __aicore__ inline bool TrySend(uint32_t first, uint32_t end)
    {
        const int64_t start = timer != nullptr && timer->Enabled() ? GetSystemCycle() : 0;
        HierarchicalCombineReadySnapshot snapshot;
        bool segmentReady = true;
        for (uint32_t row = first; row < end;) {
            const uint32_t token = row / common.topK;
            const uint32_t topk = row % common.topK;
            const uint32_t available = common.topK - topk;
            const uint32_t count = end - row < available ? end - row : available;
            uint32_t unknown = HierarchicalCombineLowBits(count) & ~(progress.deferredTopkMasks[token] >> topk);
            while (unknown != 0U) {
                const uint32_t bit = HierarchicalCombineTrailingZeros(unknown);
                const uint32_t current = row + bit;
                if (scanned < 0xffffU) ++scanned;
                if (snapshot.Ready(params, current, end - current, end - current)) {
                    progress.deferredTopkMasks[token] |= uint32_t(1) << (topk + bit);
                } else {
                    segmentReady = false;
                }
                unknown &= unknown - 1U;
            }
            row += count;
        }
        if (!segmentReady) return false;
        const uint32_t rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
        const uint64_t offset = static_cast<uint64_t>(first) * rowBytes;
        timing.StopScan();
        RocePutBytes(params.peermemInfo.combineSendPtr + offset, params.peermemInfo.gatewayCombinePtr + offset,
                     (end - first) * rowBytes, HierarchicalGateway(common.rankId), scratch, timer,
                     AscendTimer::RdmaPhase::Combine);
        timing.StartScan();
        ClearHierarchicalCombineRun(progress, common.topK, first / common.topK, first % common.topK, end - first);
        for (uint32_t row = first; row < end;) {
            const uint32_t token = row / common.topK;
            const uint32_t topk = row % common.topK;
            const uint32_t available = common.topK - topk;
            const uint32_t count = end - row < available ? end - row : available;
            progress.deferredTopkMasks[token] &= ~(HierarchicalCombineLowBits(count) << topk);
            row += count;
        }
        progress.sentRows += end - first;
        ++posts;
        if (timer != nullptr && timer->Enabled()) {
            timer->RecordDetail(AscendTimer::AIV_COMBINE_RUN, start, GetSystemCycle(),
                                (static_cast<uint64_t>(first) << 32) | (end - first),
                                end == common.tokenNum * common.topK ? 0U : 1U);
        }
        return true;
    }

    __aicore__ inline void Discover(uint32_t first, uint32_t end)
    {
        for (uint32_t row = first; row < end;) {
            const uint32_t token = row / common.topK;
            const uint32_t topk = row % common.topK;
            const uint32_t available = common.topK - topk;
            const uint32_t count = end - row < available ? end - row : available;
            progress.pendingTopkMasks[token] |= HierarchicalCombineLowBits(count) << topk;
            row += count;
        }
        progress.expectedRows += end - first;
        TrySend(first, end);
    }

    // The whole metadata DMA has already been reduced to scalar bits before
    // this method runs. Ready gathers and POSTs may overwrite its UB immediately.
    __aicore__ inline void DiscoverBits(uint32_t bits, uint32_t first, uint32_t count, uint32_t& openRun)
    {
        const uint32_t noRun = common.tokenNum * common.topK;
        for (uint32_t offset = 0U; offset < count;) {
            const uint32_t suffix = bits >> offset;
            const bool routed = (suffix & 1U) != 0U;
            uint32_t length = HierarchicalCombineTrailingZeros(routed ? ~suffix : suffix);
            if (length > count - offset) length = count - offset;
            if (routed) {
                if (openRun == noRun) openRun = first + offset;
            } else if (openRun != noRun) {
                Discover(openRun, first + offset);
                openRun = noRun;
            }
            offset += length;
        }
    }

    __aicore__ inline void RetryPending()
    {
        const uint32_t rowCount = common.tokenNum * common.topK;
        uint32_t row = 0U;
        while (row < rowCount) {
            uint32_t token = row / common.topK;
            uint32_t topk = row % common.topK;
            uint32_t bits = progress.pendingTopkMasks[token] >> topk;
            if (bits == 0U) {
                row += common.topK - topk;
                continue;
            }
            row += HierarchicalCombineTrailingZeros(bits);
            const uint32_t first = row;
            do {
                token = row / common.topK;
                topk = row % common.topK;
                bits = progress.pendingTopkMasks[token] >> topk;
                uint32_t count = HierarchicalCombineTrailingZeros(~bits);
                if (count > common.topK - topk) count = common.topK - topk;
                row += count;
                if (count < common.topK - topk || row == rowCount) break;
            } while ((progress.pendingTopkMasks[row / common.topK] & 1U) != 0U);
            TrySend(first, row);  // Continue past an unready run; never wait at its head.
        }
    }
};

__aicore__ inline bool CompleteHierarchicalCombineRuns(const MoeStageCommonConfig& common, const Params& params,
                                                       HierarchicalCombineProgress& progress, __ubuf__ uint8_t* scratch,
                                                       AscendTimerDevice* timer)
{
    if (progress.initialized && progress.sentRows == progress.expectedRows) return false;
    const uint32_t initiallySent = progress.sentRows;
    AscendTimerCombineProgressScope timing(timer, AscendTimer::AIV_COMBINE_PROGRESS);
    HierarchicalCombineRunSender sender{common, params, progress, scratch, timer, timing};
    timing.StartScan();
    // Positive ready observations are private to this completion call. Legacy
    // deferral bits must not be interpreted as evidence that a producer is ready.
    for (uint32_t token = 0U; token < common.tokenNum; ++token) progress.deferredTopkMasks[token] = 0U;
    if (!progress.initialized) {
        for (uint32_t token = 0U; token < common.tokenNum; ++token) progress.pendingTopkMasks[token] = 0U;
        progress.expectedRows = 0U;
        const uint32_t rowCount = common.tokenNum * common.topK;
        uint32_t openRun = rowCount;
        const uint32_t expertsPerHost = HIERARCHICAL_RANKS_PER_HOST * common.moeExpertPerRank;
        const uint32_t firstExpert = HierarchicalHost(common.rankId) * expertsPerHost;
        const uint32_t endExpert = firstExpert + expertsPerHost;
        const uint64_t sourceOffset = static_cast<uint64_t>(HierarchicalHost(HierarchicalGateway(common.rankId))) *
                                      CalcRailTopkIdsSlotBytes(params.tilingData);
        GlobalTensor<int32_t> routes;
        routes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.railTopkIdsRecvPtr + sourceOffset));
        constexpr uint32_t words = MEGAMOE_ROCE_SCRATCH_BYTES / sizeof(int32_t);
        static_assert(words == 64U);
        LocalTensor<int32_t> metadata(TPosition::VECCALC, params.tilingData->roceScratchUbOffset, words);
        for (uint32_t packet = 0U; packet * RAIL_PACKET_DATA_WORDS < rowCount; ++packet) {
            const uint32_t firstRow = packet * RAIL_PACKET_DATA_WORDS;
            const uint32_t valid =
                rowCount - firstRow < RAIL_PACKET_DATA_WORDS ? rowCount - firstRow : RAIL_PACKET_DATA_WORDS;
            for (uint32_t word = 0U; word < valid; word += words) {
                SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
                DataCopy(metadata, routes[packet * RAIL_PACKET_WORDS + word], words);
                SyncFuncStatic<HardEvent::MTE2_S, MEGAMOE_ROCE_EVENT_ID>();
                const uint32_t count = valid - word < words ? valid - word : words;
                uint32_t low = 0U, high = 0U;
                for (uint32_t i = 0U; i < count; ++i) {
                    const uint32_t expert = static_cast<uint32_t>(metadata.GetValue(i));
                    if (expert >= firstExpert && expert < endExpert) {
                        if (i < 32U)
                            low |= uint32_t(1) << i;
                        else
                            high |= uint32_t(1) << (i - 32U);
                    }
                }
                SyncFuncStatic<HardEvent::S_MTE2, MEGAMOE_ROCE_EVENT_ID>();
                sender.DiscoverBits(low, firstRow + word, count < 32U ? count : 32U, openRun);
                if (count > 32U) sender.DiscoverBits(high, firstRow + word + 32U, count - 32U, openRun);
            }
        }
        if (openRun != rowCount) sender.Discover(openRun, rowCount);
        progress.initialized = true;
    }
    RoceWaitState waitState{};
    while (progress.sentRows < progress.expectedRows) {
        const uint32_t before = sender.posts;
        sender.RetryPending();
        if (progress.sentRows == progress.expectedRows) break;
        timing.StopScan();
        if (sender.posts == before) {
            RoceCheckWait(waitState, HierarchicalGateway(common.rankId), scratch, RoceWaitKind::CombineRows,
                          progress.expectedRows, progress.sentRows);
            const int64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        } else {
            waitState = {};
        }
        timing.StartScan();
    }
    progress.nextRow = 0U;
    progress.retryDeferred = false;
    timing.Finish(progress.expectedRows - initiallySent, 0U, sender.posts, sender.scanned);
    return sender.posts != 0U;
}

// By default probe at most one sweep without waiting for unfinished producers. The cursor
// survives a bounded call, so late slots cannot starve behind early ready rows.
// Adjacent ready slots form one Put; holes, the row budget, and the tensor end
// terminate a run. No weighting, reduction, padding copy, or quantization occurs.
//
// Only the rank's designated communication owner may call this routine. Its UB
// scratch and QP are the same resources used by the completed Dispatch stage.
// waitForAll selects route-run completion above, only after this owner has no
// unfinished foreground producer work. Bounded/A8W8 callers retain their policy.
__aicore__ inline bool TryProgressHierarchicalCombine(const MoeStageCommonConfig& common, const Params& params,
                                                      HierarchicalCombineProgress& progress,
                                                      __ubuf__ uint8_t* roceScratch, AscendTimerDevice* timer = nullptr,
                                                      HierarchicalCombineProgressBudget budget = {},
                                                      bool waitForAll = false)
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (!UseHierarchicalRoce(params.tilingData)) {
        return false;
    }
    if (waitForAll) {
        return CompleteHierarchicalCombineRuns(common, params, progress, roceScratch, timer);
    }
    if (budget.maxPosts == 0U || budget.maxScannedRows == 0U) {
        return false;
    }
    InitializeHierarchicalCombineProgress(common, params, progress, timer);
    if (progress.sentRows == progress.expectedRows) {
        return false;
    }
    if (budget.backoffAfterEmptySweep && progress.retryDeferred) {
        progress.retryDeferred = false;
        return false;
    }
    const uint32_t pendingBefore = progress.expectedRows - progress.sentRows;
    AscendTimerCombineProgressScope timing(timer, AscendTimer::AIV_COMBINE_PROGRESS);
    timing.StartScan();
    const uint32_t rowCount = common.tokenNum * common.topK;
    const uint32_t scanLimit = budget.maxScannedRows < rowCount ? budget.maxScannedRows : rowCount;
    const uint32_t rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    const uint32_t destination = HierarchicalGateway(common.rankId);
    const bool sparseScan = (params.tilingData->pipelineFlags & 4U) != 0U;
    HierarchicalCombineScanCursor cursor(progress, common.topK);
    HierarchicalCombineReadySnapshot snapshot;
    uint32_t scanned = 0U;
    uint32_t posts = 0U;
    while (scanned < scanLimit && posts < budget.maxPosts) {
        if (sparseScan && cursor.pending == 0U) {
            scanned += cursor.SkipEmptySuffix(common, progress, scanLimit - scanned);
            if (cursor.row == rowCount) {
                cursor.Wrap(progress);
            }
            continue;
        }
        ++scanned;
        if (!cursor.Ready(params, snapshot, rowCount - cursor.row, scanLimit - scanned + 1U)) {
            cursor.Advance(common, progress);
            if (cursor.row == rowCount) {
                cursor.Wrap(progress);
            }
            continue;
        }
        const uint32_t runStart = cursor.row;
        const int64_t runCycle = timer != nullptr && timer->Enabled() ? GetSystemCycle() : 0;
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
        const bool nextPending = cursor.row < rowCount && (cursor.pending & 1U) != 0U;
        const bool alreadyDeferred = (progress.deferredTopkMasks[runToken] & (uint32_t(1) << runTopk)) != 0U;
        // 0=tensor end, 1=route/sent gap, 2=producer not ready, 3=scan budget.
        const uint32_t stopReason = cursor.row == rowCount ? 0U : (skippedHole ? (nextPending ? 2U : 1U) : 3U);
        bool defer = false;
        if (budget.deferPartialRuns && !alreadyDeferred) {
            const uint32_t previous = runStart == 0U ? 0U : runStart - 1U;
            const bool previousPending = runStart != 0U && (progress.pendingTopkMasks[previous / common.topK] &
                                                            (uint32_t(1) << (previous % common.topK))) != 0U;
            defer = nextPending || previousPending;
        }
        if (defer) {
            progress.deferredTopkMasks[runToken] |= uint32_t(1) << runTopk;
        } else {
            const uint64_t byteOffset = static_cast<uint64_t>(runStart) * rowBytes;
            timing.StopScan();
            RocePutBytes(params.peermemInfo.combineSendPtr + byteOffset,
                         params.peermemInfo.gatewayCombinePtr + byteOffset, runRows * rowBytes, destination,
                         roceScratch, timer, AscendTimer::RdmaPhase::Combine);
            timing.StartScan();
            ClearHierarchicalCombineRun(progress, common.topK, runToken, runTopk, runRows);
            // The cached bits begin at cursor.row, beyond the cleared run. A
            // wrap reloads token zero so sent rows cannot reappear.
            progress.sentRows += runRows;
            ++posts;
        }
        if (timer != nullptr && timer->Enabled()) {
            timer->RecordDetail(AscendTimer::AIV_COMBINE_RUN, runCycle, GetSystemCycle(),
                                (static_cast<uint64_t>(runStart) << 32) | runRows,
                                stopReason | (static_cast<uint64_t>(defer) << 8) |
                                    (static_cast<uint64_t>(nextPending) << 9) |
                                    (static_cast<uint64_t>(alreadyDeferred) << 10));
        }
        if (skippedHole) {
            cursor.Advance(common, progress);
        }
        if (cursor.row == rowCount) {
            cursor.Wrap(progress);
        }
        if (sparseScan && progress.sentRows == progress.expectedRows) {
            break;
        }
    }
    progress.nextRow = cursor.row;
    progress.retryDeferred = budget.backoffAfterEmptySweep && posts == 0U && scanned == rowCount;
    timing.Finish(pendingBefore, progress.expectedRows - progress.sentRows, posts, scanned);
    return posts != 0U;
}

// Final drain is legal only after all local AIV producer work is complete and
// the caller has passed the 16-rank host barrier. Calling it in the producer loop
// can prevent that same owner from producing rows needed by a gateway.
//
// This only posts the outstanding rows; it does not complete the transfers.
// The caller must publish CombineCommit before waiting for its paired gateway.
// HNS uses the NIC's ordered commit and retires outgoing work at kernel exit;
// other backends retain the conservative completion/commit protocol. The commit
// is required even when expectedRows is zero. Together with the host barrier,
// the reciprocal commit closes old Dispatch readers before the next invocation.
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
    RoceWaitState waitState{};
    while (progress.sentRows < progress.expectedRows) {
        if (!TryProgressHierarchicalCombine(common, params, progress, roceScratch, timer, {})) {
            RoceCheckWait(waitState, HierarchicalGateway(common.rankId), roceScratch, RoceWaitKind::CombineRows,
                          progress.expectedRows, progress.sentRows);
            const int64_t waitStart = GetSystemCycle();
            while (GetSystemCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        } else {
            waitState = {};
        }
    }
    // The parent includes child progress calls and backoff. Their scan/POST
    // counters belong to the child records, not to this completion envelope.
    drain.Finish(pendingBefore, progress.expectedRows - progress.sentRows, 0U, 0U);
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_HIERARCHICAL_COMBINE_H
