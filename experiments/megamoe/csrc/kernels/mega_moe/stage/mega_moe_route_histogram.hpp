// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_ROUTE_HISTOGRAM_H
#define MEGA_MOE_ROUTE_HISTOGRAM_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// For a 32-bit route ID, only its low byte should count. Map out-of-range
// IDs to bin 127, and all three unused bytes to 255 (outside BIN0).
__simd_vf__ inline void CountRouteSourceVF(__ubuf__ uint32_t* ids, __ubuf__ uint16_t* output, uint32_t count,
                                           uint32_t firstExpert)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = CreateMask<uint32_t, MaskPattern::ALL>();
    auto binsMask = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> bins;
    Duplicate(bins, uint16_t(0), binsMask);
    for (uint32_t start = 0; start < count; start += lanes) {
        RegTensor<uint32_t> values;
        LoadAlign(values, ids + start);
        Adds(values, values, 0U - firstExpert, full);
        Mins(values, values, 127U, full);
        Adds(values, values, 0xffffff00U, full);
        uint32_t activeBytes = (count - start) * 4U;
        auto active = UpdateMask<uint8_t>(activeBytes);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(
            bins, reinterpret_cast<RegTensor<uint8_t>&>(values), active);
    }
    StoreAlign(output, bins, binsMask);
}

}  // namespace MegaMoeImpl
#endif
