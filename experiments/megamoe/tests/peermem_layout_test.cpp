// Copyright (c) 2026, Lu Lu
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include "../csrc/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc/kernels/mega_moe/common/mega_moe_workspace.hpp"

using namespace MegaMoeImpl;

static_assert(sizeof(MegaMoeTilingData) == 232);
static_assert(offsetof(MegaMoeTilingData, dispatchRanksPerNode) == 216);
static_assert(offsetof(MegaMoeTilingData, roceScratchUbOffset) == 220);
static_assert(offsetof(MegaMoeTilingData, hierarchicalSequence) == 224);

PeermemSizeParams parameters(const MegaMoeTilingData& t)
{
    return {t.numMaxTokensPerRank,      t.topK, t.h,        t.moeExpertPerRank,       t.epWorldSize,         2, 1,
            t.topkWeightsPrefetch != 0, false,  t.topoType, UseDecodeRailRouting(&t), t.dispatchRanksPerNode};
}

MegaMoeTilingData make_tiling(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch,
                              uint8_t mode = GROUPED_MATMUL_MODE_A8W4)
{
    MegaMoeTilingData t{};
    t.numMaxTokensPerRank = capacity;
    t.bs = std::min(capacity, HIERARCHICAL_MAX_TOKENS);
    t.h = h;
    t.topK = topk;
    t.topkWeightsPrefetch = prefetch;
    t.moeExpertPerRank = 12;
    t.epWorldSize = 32;
    t.dispatchRanksPerNode = 16;
    t.blockAivNum = 48;
    t.groupedMatmulMode = mode;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    return t;
}

void check_hybrid(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch, uint8_t mode)
{
    auto t = make_tiling(capacity, h, topk, prefetch, mode);
    assert(UseDecodeRailRouting(&t));
    const auto size = CalcPeermemLeastSize(parameters(t));
    std::vector<uint8_t> allocation(size + 511);
    auto* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(allocation.data()) + 511) & ~uintptr_t{511});
    PeermemInfo hybrid(base, &t);
    const std::array<uint8_t*, 10> boundaries = {hybrid.maskRecvPtr,
                                                 hybrid.expertCountRecvPtr,
                                                 hybrid.railTopkIdsRecvPtr,
                                                 hybrid.quantTokenScalePtr,
                                                 hybrid.combineSendPtr,
                                                 hybrid.relayPayloadPtr,
                                                 hybrid.gatewayCombinePtr,
                                                 hybrid.gatewayCombineReadyPtr,
                                                 hybrid.hierarchicalControlPtr,
                                                 base + size};
    assert(hybrid.rankSyncInWorldPtr == base);
    assert(boundaries.front() == base + PEERMEM_DATA_OFFSET);
    for (size_t i = 0; i < boundaries.size(); ++i) {
        assert(reinterpret_cast<uintptr_t>(boundaries[i]) % 512 == 0);
        assert(boundaries[i] >= base && boundaries[i] <= base + size);
        if (i) assert(boundaries[i - 1] < boundaries[i]);
    }
    assert(hybrid.relayPayloadPtr - hybrid.combineSendPtr == hybrid.gatewayCombineReadyPtr - hybrid.gatewayCombinePtr);
    assert(hybrid.gatewayCombinePtr - hybrid.relayPayloadPtr == hybrid.combineSendPtr - hybrid.quantTokenScalePtr);
    assert(hybrid.hierarchicalControlPtr - hybrid.gatewayCombineReadyPtr ==
           static_cast<int64_t>(capacity) * topk * HIERARCHICAL_FLAG_BYTES);
    assert(base + size - hybrid.hierarchicalControlPtr ==
           static_cast<int64_t>(HierarchicalSignal::Count) * HIERARCHICAL_FLAG_BYTES);
    assert(hybrid.quantTokenScalePtr - hybrid.railTopkIdsRecvPtr == 2 * CalcRailTopkIdsSlotBytes(&t));
    // Raw counts retain every source rank, even though rail metadata has two slots.
    assert(hybrid.railTopkIdsRecvPtr - hybrid.expertCountRecvPtr == CalcMteExpertCountRecvSize(t.moeExpertPerRank, 32));
    assert(CalcSendMaskExpertCapacity(&t, 46) >= 16 * t.moeExpertPerRank);

    // Switching actual BS must leave persistent regions at identical addresses.
    for (uint32_t bs : {1U, 17U, 31U, 32U, 1U, 32U}) {
        if (bs > capacity) continue;
        t.bs = bs;
        assert(UseDecodeRailRouting(&t));
        PeermemInfo next(base, &t);
        assert(CalcPeermemLeastSize(parameters(t)) == size);
        assert(next.railTopkIdsRecvPtr == hybrid.railTopkIdsRecvPtr);
        assert(next.quantTokenScalePtr == hybrid.quantTokenScalePtr);
        assert(next.combineSendPtr == hybrid.combineSendPtr);
        assert(next.relayPayloadPtr == hybrid.relayPayloadPtr);
        assert(next.gatewayCombineReadyPtr == hybrid.gatewayCombineReadyPtr);
        assert(next.hierarchicalControlPtr == hybrid.hierarchicalControlPtr);
    }

    if (capacity == 32 && topk == 32) {
        // 1024 IDs require nine packets, including the final eight IDs. The
        // final BF16 result and its 512B flag line must fit without aliasing.
        assert(CalcRailTopkIdsSlotBytes(&t) == 9 * 512);
        assert(hybrid.quantTokenScalePtr - hybrid.railTopkIdsRecvPtr == 18 * 512);
        assert(hybrid.gatewayCombineReadyPtr - hybrid.gatewayCombinePtr == 1024LL * h * 2);
        assert(hybrid.hierarchicalControlPtr - hybrid.gatewayCombineReadyPtr == 1024 * 512);
        const auto rowBytes = CalcQuantTokenScaleBytes(h, 1, topk, prefetch);
        assert(hybrid.relayPayloadPtr + 32 * rowBytes <= hybrid.gatewayCombinePtr);
    }

    // A hybrid mode change must not move persistent peer-memory regions. Both
    // modes dispatch the same FP8 records and combine full BF16 output rows.
    t.groupedMatmulMode = mode == GROUPED_MATMUL_MODE_A8W4 ? GROUPED_MATMUL_MODE_GENERAL : GROUPED_MATMUL_MODE_A8W4;
    PeermemInfo otherMode(base, &t);
    assert(UseDecodeRailRouting(&t));
    assert(CalcPeermemLeastSize(parameters(t)) == size);
    assert(otherMode.expertCountRecvPtr == hybrid.expertCountRecvPtr);
    assert(otherMode.railTopkIdsRecvPtr == hybrid.railTopkIdsRecvPtr);
    assert(otherMode.quantTokenScalePtr == hybrid.quantTokenScalePtr);
    assert(otherMode.combineSendPtr == hybrid.combineSendPtr);
    assert(otherMode.relayPayloadPtr == hybrid.relayPayloadPtr);
    assert(otherMode.gatewayCombinePtr == hybrid.gatewayCombinePtr);
    assert(otherMode.gatewayCombineReadyPtr == hybrid.gatewayCombineReadyPtr);
    assert(otherMode.hierarchicalControlPtr == hybrid.hierarchicalControlPtr);

    // Legacy MTE retains its original selection: A8W4 uses rail metadata at
    // these decode sizes; A8W8 must keep its direct-routing layout.
    t.groupedMatmulMode = mode;
    t.topoType = TOPO_TYPE_MTE;
    PeermemInfo local(base, &t);
    assert(UseDecodeRailRouting(&t) == (mode == GROUPED_MATMUL_MODE_A8W4));
    assert(local.maskRecvPtr == hybrid.maskRecvPtr);
    assert(local.expertCountRecvPtr == hybrid.expertCountRecvPtr);
    if (UseDecodeRailRouting(&t)) {
        assert(local.railTopkIdsRecvPtr == hybrid.railTopkIdsRecvPtr);
        assert(local.quantTokenScalePtr == hybrid.quantTokenScalePtr);
        assert(local.combineSendPtr == hybrid.combineSendPtr);
        assert(base + CalcPeermemLeastSize(parameters(t)) == hybrid.relayPayloadPtr);
    } else {
        assert(local.railTopkIdsRecvPtr == nullptr);
        assert(local.quantTokenScalePtr - local.expertCountRecvPtr == CalcExpertCountRecvSize(t.moeExpertPerRank, 32));
        assert(local.quantTokenScalePtr < hybrid.quantTokenScalePtr);
    }
    assert(local.relayPayloadPtr == nullptr && local.gatewayCombinePtr == nullptr);
    assert(local.gatewayCombineReadyPtr == nullptr && local.hierarchicalControlPtr == nullptr);
}

void check_workspace_ready_region(uint8_t mode)
{
    for (const auto [maxOutput, groups] :
         std::array<std::array<uint32_t, 2>, 5>{{{1, 1}, {256, 1}, {257, 2}, {512, 2}, {513, 3}}}) {
        auto t = make_tiling(32, 1024, 32, false, mode);
        t.maxOutputSize = maxOutput;
        t.hiddenDim = 256;
        t.moeExpertPerRank = 2;
        t.aicNum = 3;
        t.combineQuantMode = COMBINE_NO_QUANT;
        std::vector<uint8_t> allocation(4 * 1024 * 1024 + 511, 0xA5);
        auto* base =
            reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(allocation.data()) + 511) & ~uintptr_t{511});
        WorkspaceInfo hybrid(base, &t);
        assert(hybrid.workspaceSize <= 4 * 1024 * 1024);
        assert(hybrid.gmm2ReadyPtr != nullptr);
        assert(hybrid.gmm2MmadResPtr != nullptr);
        assert(reinterpret_cast<uintptr_t>(hybrid.gmm2ReadyPtr) % 64 == 0);
        assert(hybrid.gmm2ReadyPtr >= hybrid.flagActivationToGmm2Ptr);
        auto* flagsEnd = hybrid.flagActivationToGmm2Ptr + hybrid.flagResetElementCount * sizeof(int32_t);
        const int64_t readyBytes = static_cast<int64_t>(t.moeExpertPerRank) * groups * 64;
        auto* priorFlagsEnd = hybrid.gmm2ReadyPtr + readyBytes;
        const int64_t gatewayBytes = HIERARCHICAL_RANKS_PER_HOST * HIERARCHICAL_FLAG_BYTES;
        assert(hybrid.gatewayPayloadReadyPtr >= priorFlagsEnd);
        assert(hybrid.gatewayPayloadReadyPtr - priorFlagsEnd < HIERARCHICAL_FLAG_BYTES);
        assert(reinterpret_cast<uintptr_t>(hybrid.gatewayPayloadReadyPtr) % HIERARCHICAL_FLAG_BYTES == 0);
        assert(hybrid.gatewayPayloadReadyPtr + gatewayBytes == flagsEnd);
        assert(flagsEnd <= hybrid.gmm2MmadResPtr);
        // The reset span must include the final expert's final M-group, while
        // leaving the subsequent data/workspace byte untouched.
        std::fill(hybrid.flagActivationToGmm2Ptr, flagsEnd, uint8_t{0});
        assert(hybrid.gmm2ReadyPtr[readyBytes - 1] == 0);
        for (uint32_t slot = 0; slot < HIERARCHICAL_RANKS_PER_HOST; ++slot) {
            assert(hybrid.gatewayPayloadReadyPtr[slot * HIERARCHICAL_FLAG_BYTES] == 0);
        }
        assert(hybrid.gatewayPayloadReadyPtr[gatewayBytes - 1] == 0);
        assert(flagsEnd[0] == 0xA5);

        t.topoType = TOPO_TYPE_MTE;
        WorkspaceInfo legacy(base, &t);
        assert(legacy.gatewayPayloadReadyPtr == nullptr);
        const int64_t priorElements = (priorFlagsEnd - hybrid.flagActivationToGmm2Ptr) / sizeof(int32_t);
        if (mode == GROUPED_MATMUL_MODE_GENERAL) {
            assert(legacy.gmm2ReadyPtr == nullptr);
            assert(priorElements - legacy.flagResetElementCount == readyBytes / 4);
        } else {
            assert(legacy.gmm2ReadyPtr == hybrid.gmm2ReadyPtr);
            assert(legacy.flagResetElementCount == priorElements);
        }
        assert(hybrid.flagResetElementCount == CalcGatewayPayloadReadyFlagEnd(priorElements));
    }
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string_view selected(argv[1]);
    assert(selected == "a8w4" || selected == "a8w8");
    const uint8_t mode = selected == "a8w4" ? GROUPED_MATMUL_MODE_A8W4 : GROUPED_MATMUL_MODE_GENERAL;
    assert(HIERARCHICAL_MAX_TOKENS >= 32);
    for (uint32_t capacity : {1U, 17U, 31U, 32U, 64U}) {
        for (uint32_t h : {1024U, 7168U, 8192U}) {
            for (uint32_t topk : {1U, 6U, 32U}) {
                for (bool prefetch : {false, true}) check_hybrid(capacity, h, topk, prefetch, mode);
            }
        }
    }
    check_hybrid(4096, 1024, 1, true, mode);
    check_workspace_ready_region(mode);
    auto legacy = make_tiling(31, 1024, 6, true, mode);
    legacy.topoType = TOPO_TYPE_MTE;
    legacy.dispatchRanksPerNode = 8;
    assert(RailServerSize(&legacy) == 8);
    assert(CalcServerNum(32, RailServerSize(&legacy)) == 4);
    // Frozen EP32 legacy layout: catches accidental size/offset ABI changes.
    assert(CalcPeermemLeastSize(parameters(legacy)) == (mode == GROUPED_MATMUL_MODE_A8W4 ? 531456 : 526848));
    const auto legacy_size = CalcPeermemLeastSize(parameters(legacy));
    legacy.dispatchRanksPerNode = 0;
    assert(CalcPeermemLeastSize(parameters(legacy)) == legacy_size);
    legacy.epWorldSize = 16;
    legacy.dispatchRanksPerNode = 16;
    assert(!UseHierarchicalRoce(&legacy));
    assert(CalcRailTopkIdsRecvSize(&legacy) == CalcRailTopkIdsSlotBytes(&legacy));
    for (uint32_t rank = 0; rank < 32; ++rank) {
        assert(HierarchicalGateway(HierarchicalGateway(rank)) == rank);
        assert(HierarchicalGateway(rank) % 16 == rank % 16);
        assert(HierarchicalHost(HierarchicalGateway(rank)) == HierarchicalSendJob(rank));
    }
}
