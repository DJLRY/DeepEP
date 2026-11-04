// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {
inline constexpr uint32_t REDUCE_LANE_INIT = 2U;
inline constexpr uint32_t COMBINE_SPARSE_SCAN = 4U;
inline constexpr uint32_t COMBINE_READY_TRACE = 64U;
inline constexpr uint32_t COMBINE_SINGLE_ROW = 128U;
inline constexpr uint32_t REDUCE_DOUBLE_BUFFER = 256U;
inline constexpr uint32_t DISPATCH_SKIP_EMPTY = 512U;
inline constexpr uint32_t DISPATCH_LOOKAHEAD = 1024U;
inline constexpr uint32_t DISPATCH_SINGLE_ROW = 2048U;
inline constexpr uint32_t PIPELINE_FLAGS = REDUCE_LANE_INIT | COMBINE_SPARSE_SCAN | COMBINE_READY_TRACE |
                                           COMBINE_SINGLE_ROW | REDUCE_DOUBLE_BUFFER | DISPATCH_SKIP_EMPTY |
                                           DISPATCH_LOOKAHEAD | DISPATCH_SINGLE_ROW;

// Zero keeps the original forwarding path. Reduction uses one fixed AIV per QP.
inline constexpr bool ValidReduceQps(uint32_t count)
{
    return count == 0U || count == 4U || count == 8U;
}

}  // namespace deepep::megamoe_config
