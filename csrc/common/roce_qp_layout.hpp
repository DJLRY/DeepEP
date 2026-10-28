// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>

namespace ascend_deepep::roce_layout {
// QP0 is reserved for SDK operations. Each application QP has one AIV owner.
inline constexpr uint32_t kPayloadQps = 8U;
inline constexpr uint32_t kFirstPayloadQp = 1U;
inline constexpr uint32_t kMetadataQp = kFirstPayloadQp + kPayloadQps;
inline constexpr uint32_t kQpCount = kMetadataQp + 1U;
}  // namespace ascend_deepep::roce_layout
