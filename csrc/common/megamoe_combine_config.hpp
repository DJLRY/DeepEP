// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

namespace deepep::megamoe_config {
// Zero keeps the original forwarding path. Reduction uses one fixed AIV per QP.
inline constexpr bool ValidReduceQps(uint32_t count)
{
    return count == 0U || count == 4U || count == 8U;
}

}  // namespace deepep::megamoe_config
