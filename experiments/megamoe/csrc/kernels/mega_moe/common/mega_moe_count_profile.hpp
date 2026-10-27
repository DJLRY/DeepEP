// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_COUNT_PROFILE_H
#define MEGA_MOE_COUNT_PROFILE_H

#include "ascend_timer_v2.hpp"

namespace MegaMoeImpl {

// Decode the preserved failed snapshot only after the acquisition succeeds.
// It is an observation of one poll, not the physical last-arriving source.
template <typename Snapshot>
__aicore__ inline void CaptureFailedCountSnapshot(const Snapshot& snapshot, uint32_t sources, uint32_t experts,
                                                  uint32_t stride, uint32_t expected, uint64_t* values)
{
    using namespace AscendTimer;
    values[COUNT_LAST_STALE_SOURCE_MASK] = 0;
    values[COUNT_LAST_STALE_SOURCE_PLUS_ONE] = 0;
    values[COUNT_LAST_STALE_EXPERT_PLUS_ONE] = 0;
    values[COUNT_LAST_OBSERVED_EPOCH] = 0;
    values[COUNT_SOURCE_MASK_OVERFLOW] = 0;
    for (uint32_t source = 0; source < sources; ++source) {
        for (uint32_t expert = 0; expert < experts; ++expert) {
            const uint32_t epoch = static_cast<uint32_t>(snapshot.GetValue(source * stride + expert)) >> 24;
            if (epoch == expected) {
                continue;
            }
            if (source < 64U) {
                values[COUNT_LAST_STALE_SOURCE_MASK] |= uint64_t(1) << source;
            } else {
                values[COUNT_SOURCE_MASK_OVERFLOW] = 1U;
            }
            if (values[COUNT_LAST_STALE_SOURCE_PLUS_ONE] == 0U) {
                values[COUNT_LAST_STALE_SOURCE_PLUS_ONE] = source + 1U;
                values[COUNT_LAST_STALE_EXPERT_PLUS_ONE] = expert + 1U;
                values[COUNT_LAST_OBSERVED_EPOCH] = epoch;
            }
            // The VF already counted stale words. One mismatch identifies
            // this source; avoid decoding its remaining expert slots.
            break;
        }
    }
}

}  // namespace MegaMoeImpl
#endif
