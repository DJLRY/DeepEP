// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_DISPATCH_PLAN_H
#define MEGA_MOE_DISPATCH_PLAN_H

#include "mega_moe_peermem.hpp"

namespace MegaMoeImpl {

// One mask per owner retains original token offsets. Adjacent bits are one Put.
// The packer computes this once; senders never rescan top-k metadata.
struct HierarchicalDispatchPlan {
    uint32_t tokens[HIERARCHICAL_PAYLOAD_LANES]{};
};

// Inspect the rail packer's already loaded UB, skipping the marker word in
// each 512B packet. Short-circuit tokens already known to need the other host.
template <typename PackedTensor>
HOST_DEVICE void AccumulateHierarchicalRemoteTokens(PackedTensor& packed, uint32_t firstItem, uint32_t itemCount,
                                                    uint32_t topK, uint32_t expertsPerHost, uint32_t remoteHost,
                                                    uint32_t& tokens)
{
    for (uint32_t i = 0U; i < itemCount; ++i) {
        const uint32_t token = (firstItem + i) / topK;
        if ((tokens & (uint32_t(1) << token)) != 0U) {
            continue;
        }
        const int32_t expert = packed.GetValue(i + i / RAIL_PACKET_DATA_WORDS);
        if (expert >= 0 && static_cast<uint32_t>(expert) / expertsPerHost == remoteHost) {
            tokens |= uint32_t(1) << token;
        }
    }
}

HOST_DEVICE HierarchicalDispatchPlan BuildHierarchicalDispatchPlan(uint32_t validTokens)
{
    HierarchicalDispatchPlan plan;
    uint32_t total = 0U;
    for (uint32_t bits = validTokens; bits != 0U; bits &= bits - 1U) {
        ++total;
    }
    uint32_t assigned[HIERARCHICAL_PAYLOAD_LANES]{};
    uint32_t posts[HIERARCHICAL_PAYLOAD_LANES]{};
    uint32_t capacity[HIERARCHICAL_PAYLOAD_LANES]{};
    for (uint32_t lane = 0U; lane < HIERARCHICAL_PAYLOAD_LANES; ++lane) {
        capacity[lane] = total / HIERARCHICAL_PAYLOAD_LANES + (lane < total % HIERARCHICAL_PAYLOAD_LANES);
    }
    for (uint32_t token = 0U; token < HIERARCHICAL_MAX_TOKENS;) {
        if ((validTokens & (uint32_t(1) << token)) == 0U) {
            ++token;
            continue;
        }
        uint32_t end = token + 1U;
        while (end < HIERARCHICAL_MAX_TOKENS && (validTokens & (uint32_t(1) << end)) != 0U) {
            ++end;
        }
        while (token < end) {
            uint32_t selected = HIERARCHICAL_PAYLOAD_LANES;
            for (uint32_t lane = 0U; lane < HIERARCHICAL_PAYLOAD_LANES; ++lane) {
                if (assigned[lane] == capacity[lane]) {
                    continue;
                }
                if (selected == HIERARCHICAL_PAYLOAD_LANES || posts[lane] < posts[selected] ||
                    (posts[lane] == posts[selected] && assigned[lane] < assigned[selected])) {
                    selected = lane;
                }
            }
            uint32_t count = capacity[selected] - assigned[selected];
            count = count < end - token ? count : end - token;
            plan.tokens[selected] |= (0xffffffffU >> (32U - count)) << token;
            assigned[selected] += count;
            ++posts[selected];
            token += count;
        }
    }
    return plan;
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_DISPATCH_PLAN_H
