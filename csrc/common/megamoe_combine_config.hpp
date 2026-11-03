// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {
inline constexpr uint32_t REDUCE_LANE_INIT = 2U;
inline constexpr uint32_t COMBINE_SPARSE_SCAN = 4U;
inline constexpr uint32_t COMBINE_OWNER_HALF = 8U;
inline constexpr uint32_t COMBINE_ASYNC = 16U;
inline constexpr uint32_t COMBINE_SYNC_PROGRESS = 32U;
inline constexpr uint32_t COMBINE_READY_TRACE = 64U;
inline constexpr uint32_t PIPELINE_FLAGS = REDUCE_LANE_INIT | COMBINE_SPARSE_SCAN | COMBINE_OWNER_HALF | COMBINE_ASYNC |
                                           COMBINE_SYNC_PROGRESS | COMBINE_READY_TRACE;

// Zero keeps the original forwarding path. Reduction uses one fixed AIV per QP.
inline constexpr bool ValidReduceQps(uint32_t count)
{
    return count == 0U || count == 4U || count == 8U;
}

}  // namespace deepep::megamoe_config
