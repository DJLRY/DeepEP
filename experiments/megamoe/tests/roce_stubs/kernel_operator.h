// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*
#define ASCEND_IS_AIV (RoceTest::isAiv)
constexpr int PIPE_ALL = 0;

inline void trap()
{
    throw std::runtime_error("unsupported backend");
}

namespace RoceTest {
inline bool isAiv = true;
inline int blockIndex = 0;
inline int taskRatio = 2;
inline int subBlockIndex = 0;
inline bool signalObserved = false;
inline bool scalarEventSet = false;
inline bool mteCanRead = false;
inline std::vector<int> operations;
inline int pipelineBarriers = 0;
inline int64_t cycle = 100;
}  // namespace RoceTest

namespace AscendC {
inline int GetBlockIdx()
{
    return RoceTest::blockIndex;
}
inline int GetTaskRation()
{
    return RoceTest::taskRatio;
}
inline int GetSubBlockIdx()
{
    return RoceTest::subBlockIndex;
}
inline int64_t GetSystemCycle()
{
    return ++RoceTest::cycle;
}
template <int Pipe>
void PipeBarrier()
{
    ++RoceTest::pipelineBarriers;
}
template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T* data)
    {
        data_ = data;
    }
    T& operator()(int index)
    {
        return data_[index];
    }

private:
    T* data_ = nullptr;
};

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
