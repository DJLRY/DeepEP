// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_GATEWAY_READINESS_H
#define MEGA_MOE_GATEWAY_READINESS_H

#include "../common/mega_moe_utils.hpp"
#include "../common/ascend_timer_v2_device.hpp"

namespace MegaMoeImpl {

// Snapshot dependencies before the count table is replaced by inclusive
// prefixes. The mask describes the whole rank, not only this block's tiles,
// and remains valid after count-ready is published to AIC/AIV0.
template <typename CountTensor>
__aicore__ inline uint32_t BuildHierarchicalGatewayNeededMask(const MoeStageCommonConfig& common,
                                                              const BlockWorkspaceContext& workspace,
                                                              const Params& params, CountTensor& counts)
{
    if constexpr (g_coreType == AIC) {
        return 0U;
    }
    if (GetSubBlockIdx() != 1U || !UseHierarchicalRoce(params.tilingData) || workspace.blockNum == 0U ||
        workspace.blockIdx >= workspace.blockNum) {
        return 0U;
    }
    const uint32_t sourceFirst = HierarchicalHost(HierarchicalGateway(common.rankId)) * HIERARCHICAL_RANKS_PER_HOST;
    uint32_t neededMask = 0U;
    for (uint32_t slot = 0U; slot < HIERARCHICAL_RANKS_PER_HOST; ++slot) {
        const uint32_t source = sourceFirst + slot;
        for (uint32_t expert = 0U; expert < common.moeExpertPerRank; ++expert) {
            if (counts.GetValue(expert * common.worldSize + source) != 0) {
                neededMask |= uint32_t(1) << slot;
                break;
            }
        }
    }
    return neededMask;
}

// One checker per source acquires the gateway flag and publishes a local epoch.
// The local slots are reset BEFORE the existing input barrier. Never clear them
// here: another checker may have published. AIV1 must finish this acquire before
// dispatching payload, even though AIC/AIV0 may already consume the count table.
__aicore__ inline void AcquireHierarchicalGatewayPayload(const MoeStageCommonConfig& common,
                                                         const BlockWorkspaceContext& workspace, const Params& params,
                                                         GM_ADDR* winRankAddr, uint32_t neededMask,
                                                         AscendTimerDevice* timer = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || !UseHierarchicalRoce(params.tilingData) || workspace.blockNum == 0U ||
        workspace.blockIdx >= workspace.blockNum) {
        return;
    }
    const uint32_t sourceFirst = HierarchicalHost(HierarchicalGateway(common.rankId)) * HIERARCHICAL_RANKS_PER_HOST;
    const int32_t sequence = static_cast<int32_t>(params.tilingData->hierarchicalSequence);
    const uint64_t readyOffset = params.peermemInfo.hierarchicalControlPtr - params.peermemInfo.rankSyncInWorldPtr +
                                 HierarchicalSignalOffset(HierarchicalSignal::NetPayloadCommit);
    // All checkers publish before consuming anyone else's local notification.
    // Interleaving local waits with this loop would serialize the distributed
    // gateway checks. Striding also covers devices with fewer than 16 blocks.
    for (uint32_t slot = workspace.blockIdx; slot < HIERARCHICAL_RANKS_PER_HOST; slot += workspace.blockNum) {
        if ((neededMask & (uint32_t(1) << slot)) == 0U) {
            continue;
        }
        const uint32_t source = sourceFirst + slot;
        const uint32_t gateway = HierarchicalGateway(source);
        {
            AscendTimerGatewayWaitScope wait(timer, source, gateway, params.tilingData->hierarchicalSequence);
            auto* ready = reinterpret_cast<__gm__ int32_t*>(winRankAddr[gateway] + readyOffset);
            // Acquire all eight independent QP commits directly on this local
            // gateway. No pre-barrier gateway publisher waits for payload.
            for (uint32_t lane = 0U; lane < HIERARCHICAL_PAYLOAD_LANES; ++lane) {
                WaitUntilGmFlagEquals(ready + lane * (HIERARCHICAL_FLAG_BYTES / sizeof(int32_t)), sequence);
            }
            // Order the acquired GM notification before forwarding it to local
            // consumers. Payload completion is certified by the gateway's
            // per-QP ordered commits; this hop does not copy payload.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        auto* localReady = reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.gatewayPayloadReadyPtr +
                                                             static_cast<uint64_t>(slot) * HIERARCHICAL_FLAG_BYTES);
        WriteGmByPassDCache(localReady, sequence);
    }
    for (uint32_t slot = 0U; slot < HIERARCHICAL_RANKS_PER_HOST; ++slot) {
        if ((neededMask & (uint32_t(1) << slot)) == 0U) {
            continue;
        }
        auto* localReady = reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.gatewayPayloadReadyPtr +
                                                             static_cast<uint64_t>(slot) * HIERARCHICAL_FLAG_BYTES);
        WaitUntilGmFlagEquals(localReady, sequence);
    }
    // Preserve each consumer's scalar-ready -> payload DMA ordering.
    SyncFuncStatic<HardEvent::S_MTE2, 7U>();
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_GATEWAY_READINESS_H
