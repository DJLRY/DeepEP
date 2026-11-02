// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {
inline constexpr uint32_t COMBINE_GATE = 1U;
inline constexpr uint32_t REDUCE_LANE_INIT = 2U;
inline constexpr uint32_t PIPELINE_FLAGS = COMBINE_GATE | REDUCE_LANE_INIT;

// Zero keeps the original forwarding path. Reduction uses one fixed AIV per QP.
inline constexpr bool ValidReduceQps(uint32_t count)
{
    return count == 0U || count == 4U || count == 8U;
}

}  // namespace deepep::megamoe_config
