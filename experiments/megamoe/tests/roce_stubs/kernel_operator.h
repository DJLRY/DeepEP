// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*

inline void trap()
{
    throw std::runtime_error("unsupported backend");
}

namespace RoceTest {
inline bool signalObserved = false;
inline bool scalarEventSet = false;
inline bool mteCanRead = false;
inline std::vector<int> operations;
}  // namespace RoceTest

namespace AscendC {
using TEventID = uint32_t;
enum class HardEvent { S_MTE2 };

template <HardEvent Event>
void SetFlag(TEventID event)
{
    static_assert(Event == HardEvent::S_MTE2);
    if (event != 7U || !RoceTest::signalObserved || RoceTest::scalarEventSet) {
        throw std::runtime_error("MTE acquire was issued before observing the commit");
    }
    RoceTest::scalarEventSet = true;
}

template <HardEvent Event>
void WaitFlag(TEventID event)
{
    static_assert(Event == HardEvent::S_MTE2);
    if (event != 7U || !RoceTest::scalarEventSet) {
        throw std::runtime_error("MTE acquire used an unowned event");
    }
    RoceTest::scalarEventSet = false;
    RoceTest::mteCanRead = true;
}
}  // namespace AscendC
