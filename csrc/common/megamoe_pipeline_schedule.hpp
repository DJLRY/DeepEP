// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGAMOE_SCHEDULE_INLINE __aicore__ inline
#else
#define MEGAMOE_SCHEDULE_INLINE inline
#endif

struct CombineRowRange {
    uint32_t start;
    uint32_t count;
};

// Owners receive one weight unit, other AIV1 workers receive two. Rotate the
// remainder in weight space, then concatenate each physical worker's rows.
// This retains one contiguous range per worker and exact coverage even for
// slices smaller than the worker count. Rotation never changes QP ownership.
MEGAMOE_SCHEDULE_INLINE constexpr uint32_t WeightedRowBoundary(uint32_t rows, uint32_t boundary, uint32_t units,
                                                               uint32_t shift)
{
    const uint32_t extra = rows % units;
    const uint32_t end = shift + extra;
    const uint32_t firstEnd = end < units ? end : units;
    const uint32_t clipped = boundary < firstEnd ? boundary : firstEnd;
    const uint32_t first = clipped > shift ? clipped - shift : 0U;
    const uint32_t wrap = end > units ? end - units : 0U;
    return rows / units * boundary + first + (boundary < wrap ? boundary : wrap);
}

MEGAMOE_SCHEDULE_INLINE constexpr CombineRowRange HalfOwnerCombineRange(uint32_t rows, uint32_t worker,
                                                                        uint32_t workers, uint32_t owners,
                                                                        uint64_t rowPrefix, uint64_t wave)
{
    if (workers == 0U || owners > workers || worker >= workers) {
        return {0U, 0U};
    }
    const uint32_t units = 2U * workers - owners;
    const uint32_t first = worker < owners ? worker : owners + 2U * (worker - owners);
    const uint32_t width = worker < owners ? 1U : 2U;
    const uint32_t shift = static_cast<uint32_t>((rowPrefix + wave) % units);
    const uint32_t start = WeightedRowBoundary(rows, first, units, shift);
    return {start, WeightedRowBoundary(rows, first + width, units, shift) - start};
}

}  // namespace deepep::megamoe_config

#undef MEGAMOE_SCHEDULE_INLINE
