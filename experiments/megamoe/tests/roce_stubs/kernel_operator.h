// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*
#define ASCEND_IS_AIV (RoceTest::isAiv)
#if !defined(MEGAMOE_TEST_HOST_PASS) && !defined(__CCE_KT_TEST__)
#define __CCE_KT_TEST__ 1
#endif
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
inline int64_t cycleStep = 1;
inline uint32_t timeoutDiagnostics = 0U;
inline uint32_t lastWaitKind = 0U;
inline uint32_t lastWaitPeer = 0U;
inline uint32_t lastWaitQp = 0U;
inline uint32_t lastWaitExpected = 0U;
inline uint32_t lastWaitObserved = 0U;
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
    RoceTest::cycle += RoceTest::cycleStep;
    return RoceTest::cycle;
}
inline void printf(const char*, uint32_t kind, uint32_t peer, uint32_t qp, uint32_t expected, uint32_t observed)
{
    ++RoceTest::timeoutDiagnostics;
    RoceTest::lastWaitKind = kind;
    RoceTest::lastWaitPeer = peer;
    RoceTest::lastWaitQp = qp;
    RoceTest::lastWaitExpected = expected;
    RoceTest::lastWaitObserved = observed;
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
    T GetValue(uint32_t index) const
    {
        return data_[index];
    }
    T* operator[](uint32_t index) const
    {
        return data_ + index;
    }

private:
    T* data_ = nullptr;
};

using TEventID = uint32_t;
enum class HardEvent { S_MTE2, MTE2_MTE3, MTE3_MTE2, MTE3_S };
inline std::array<bool, 4> packEvents{};

template <HardEvent Event>
void SetFlag(TEventID event)
{
    if constexpr (Event != HardEvent::S_MTE2) {
        if (event != 7U || packEvents[static_cast<size_t>(Event)]) throw std::runtime_error("packer event reused");
        packEvents[static_cast<size_t>(Event)] = true;
        return;
    }
    if (event != 7U || !RoceTest::signalObserved || RoceTest::scalarEventSet) {
        throw std::runtime_error("MTE acquire was issued before observing the commit");
    }
    RoceTest::scalarEventSet = true;
}

template <HardEvent Event>
void WaitFlag(TEventID event)
{
    if constexpr (Event != HardEvent::S_MTE2) {
        if (event != 7U || !packEvents[static_cast<size_t>(Event)]) throw std::runtime_error("packer event unset");
        packEvents[static_cast<size_t>(Event)] = false;
        return;
    }
    if (event != 7U || !RoceTest::scalarEventSet) {
        throw std::runtime_error("MTE acquire used an unowned event");
    }
    RoceTest::scalarEventSet = false;
    RoceTest::mteCanRead = true;
}

enum class TPosition { VECCALC };
template <typename T>
struct LocalTensor {
    std::vector<T> data;
    LocalTensor(TPosition, uint32_t, uint32_t count) : data(count) {}
};
struct CopyParams {
    uint32_t blocks, bytes, sourceStride, destinationStride, reserved;
};
struct CopyPad {
    bool enabled;
    uint32_t left, right, value;
};
template <typename T>
void DataCopyPad(LocalTensor<T>& dst, const T* src, CopyParams p, CopyPad)
{
    if (p.blocks != 1 || p.bytes > dst.data.size() * sizeof(T)) throw std::runtime_error("bad weight copy");
    std::memcpy(dst.data.data(), src, p.bytes);
}
template <typename T>
void DataCopyPad(T* dst, const LocalTensor<T>& src, CopyParams p)
{
    if (p.blocks != 1 || p.bytes > src.data.size() * sizeof(T)) throw std::runtime_error("bad packed weight copy");
    std::memcpy(dst, src.data.data(), p.bytes);
}
}  // namespace AscendC
