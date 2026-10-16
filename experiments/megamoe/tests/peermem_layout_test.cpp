// Copyright (c) 2026, Lu Lu
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../csrc/kernels/mega_moe/common/mega_moe_peermem.hpp"

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

MegaMoeTilingData make_tiling(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch)
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
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    return t;
}

void check_hybrid(uint32_t capacity, uint32_t h, uint32_t topk, bool prefetch)
{
    auto t = make_tiling(capacity, h, topk, prefetch);
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
    for (uint32_t bs : {1U, 17U, 31U}) {
        if (bs > capacity) continue;
        t.bs = bs;
        PeermemInfo next(base, &t);
        assert(CalcPeermemLeastSize(parameters(t)) == size);
        assert(next.relayPayloadPtr == hybrid.relayPayloadPtr);
        assert(next.gatewayCombineReadyPtr == hybrid.gatewayCombineReadyPtr);
        assert(next.hierarchicalControlPtr == hybrid.hierarchicalControlPtr);
    }

    // All additional storage follows the unchanged local MTE layout.
    t.topoType = TOPO_TYPE_MTE;
    PeermemInfo local(base, &t);
    assert(local.maskRecvPtr == hybrid.maskRecvPtr);
    assert(local.expertCountRecvPtr == hybrid.expertCountRecvPtr);
    assert(local.railTopkIdsRecvPtr == hybrid.railTopkIdsRecvPtr);
    assert(local.quantTokenScalePtr == hybrid.quantTokenScalePtr);
    assert(local.combineSendPtr == hybrid.combineSendPtr);
    assert(base + CalcPeermemLeastSize(parameters(t)) == hybrid.relayPayloadPtr);
    assert(local.relayPayloadPtr == nullptr && local.gatewayCombinePtr == nullptr);
    assert(local.gatewayCombineReadyPtr == nullptr && local.hierarchicalControlPtr == nullptr);
}

int main()
{
    for (uint32_t capacity : {1U, 17U, 31U, 64U}) {
        for (uint32_t h : {1024U, 7168U, 8192U}) {
            for (uint32_t topk : {1U, 6U, 32U}) {
                for (bool prefetch : {false, true}) check_hybrid(capacity, h, topk, prefetch);
            }
        }
    }
    check_hybrid(4096, 1024, 1, true);
    auto legacy = make_tiling(31, 1024, 6, true);
    legacy.topoType = TOPO_TYPE_MTE;
    legacy.dispatchRanksPerNode = 8;
    assert(RailServerSize(&legacy) == 8);
    assert(CalcServerNum(32, RailServerSize(&legacy)) == 4);
    // Frozen EP32 legacy layout: catches accidental size/offset ABI changes.
    assert(CalcPeermemLeastSize(parameters(legacy)) == 531456);
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
