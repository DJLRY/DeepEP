// Copyright (c) 2026, Lu Lu
// Production row submit/retire code with deferred MTE3 writes. This checks UB,
// event and ready publication lifetimes; it does not emulate NPU coherence.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#define __aicore__
#define __gm__
using bfloat16_t = uint16_t;
using GM_ADDR = uint8_t*;
constexpr uint32_t META_INFO_SIZE = 8U, RANK_ID = 0U, TOKEN_ID = 1U, TOPK_INDEX = 2U;
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512U;
enum class TPosition { VECCALC, VECIN };
enum class HardEvent { S_MTE2, MTE2_S, MTE2_MTE3, MTE3_S };
using TEventID = uint32_t;
alignas(512) std::array<uint8_t, 256U * 1024U> ub{};
std::array<std::array<bool, 8>, 4> events{};
template <typename T>
struct LocalTensor {
    T* data = nullptr;
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t count)
    {
        assert(offset % 32U == 0U && offset + count * sizeof(T) <= ub.size());
        data = reinterpret_cast<T*>(ub.data() + offset);
    }
    T GetValue(uint32_t index) const
    {
        return data[index];
    }
};
template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* address)
    {
        data = address;
    }
    GlobalTensor operator[](uint64_t offset) const
    {
        return {data + offset};
    }
};
struct DataCopyExtParams {
    uint32_t count, bytes, srcStride, dstStride, reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool enabled;
    uint32_t left, right, value;
};
struct PendingWrite {
    void* target;
    const void* source;
    uint32_t bytes;
};
std::vector<PendingWrite> pending;
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    std::memcpy(dst.data, src.data, copy.bytes);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    pending.push_back({dst.data, src.data, copy.bytes});
}
template <HardEvent E>
void SetFlag(TEventID id)
{
    assert(id == 6U && !events[static_cast<size_t>(E)][id]);
    events[static_cast<size_t>(E)][id] = true;
}
template <HardEvent E>
void WaitFlag(TEventID id)
{
    assert(id == 6U && events[static_cast<size_t>(E)][id]);
    if constexpr (E == HardEvent::MTE3_S) {
        for (auto write : pending) std::memcpy(write.target, write.source, write.bytes);
        pending.clear();
    }
    events[static_cast<size_t>(E)][id] = false;
}
template <HardEvent E, uint32_t Id>
void SyncFuncStatic()
{
    SetFlag<E>(Id);
    WaitFlag<E>(Id);
}
void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    assert(pending.empty());  // No ready store before payload visibility.
    *address = value;
}
int64_t GetSystemCycle()
{
    static int64_t cycle = 0;
    return ++cycle;
}
namespace AscendTimer {
constexpr int64_t NO_WAVE = -1;
enum DynamicTimingType { AIV_COMBINE_ASYNC };
}  // namespace AscendTimer
struct AscendTimerDevice {
    uint32_t rows = 0U;
    int64_t wave = -1;
    bool Enabled() const
    {
        return true;
    }
    void SetWave(int64_t value)
    {
        wave = value;
    }
    void RecordDetail(AscendTimer::DynamicTimingType, int64_t start, int64_t end, uint64_t first, uint64_t second)
    {
        assert(end > start && first >> 32 == 4U && second == 1U && wave == 2);
        ++rows;
    }
};
struct WorkRange {
    uint32_t start, count;
};
struct ExpertLoopState {
    uint64_t globalTokenStartIndex;
    uint32_t expertIdx;
};
struct MoeStageCommonConfig {
    uint32_t tokenHiddenDim, rankId;
};
struct Tiling {
    uint32_t topK, hierarchicalSequence;
};
struct Params {
    Tiling* tilingData;
    struct {
        uint8_t* metaInfoPtr;
        uint8_t* gmm2MmadResPtr;
    } workspaceInfo;
    struct {
        uint8_t* rankSyncInWorldPtr;
        uint8_t* combineSendPtr;
        uint8_t* gatewayCombinePtr;
        uint8_t* gatewayCombineReadyPtr;
    } peermemInfo;
};
uint8_t* windows[32]{};
uint8_t* GetRankWinAddrWithOffset(uint32_t rank, uint64_t offset)
{
    assert(windows[rank]);
    return windows[rank] + offset;
}
uint32_t HierarchicalHost(uint32_t rank)
{
    return rank / 16U;
}
uint32_t HierarchicalGateway(uint32_t rank)
{
    return rank ^ 16U;
}
namespace CombineImpl {
/* PRODUCTION_ROUTES */
}
/* PRODUCTION_ASYNC */

int main()
{
    for (uint32_t hidden : {1024U, 7168U, 8192U}) {
        for (uint32_t sequence : {1U, 129U, 0xffffffffU}) {
            ub.fill(0xcc);
            events = {};
            std::vector<uint8_t> local(768U * 1024U), peer(local.size());
            windows[0] = local.data();
            windows[7] = peer.data();
            Tiling tiling{4U, sequence};
            MoeStageCommonConfig common{hidden, 0U};
            std::vector<int32_t> metadata(12U * META_INFO_SIZE);
            std::vector<uint16_t> source(12U * hidden);
            Params params{&tiling,
                          {reinterpret_cast<uint8_t*>(metadata.data()), reinterpret_cast<uint8_t*>(source.data())},
                          {local.data(), local.data(), local.data() + 256U * 1024U, local.data() + 512U * 1024U}};
            // Source rows 7..9, destination rows 1,6,7; row 8 stays on this host.
            for (uint32_t i = 0U; i < 3U; ++i) {
                const uint32_t record = (7U + i) * META_INFO_SIZE;
                metadata[record] = i == 1U ? 7 : 16;
                metadata[record + 1] = i == 0U ? 0 : 1;
                metadata[record + 2] = i + 1U;
                for (uint32_t n = 0; n < hidden; ++n) source[(7U + i) * hidden + n] = (i + 1U) * 19U + n;
            }
            WaveCombineScratch scratch;
            scratch.asyncWave = 2;
            AscendTimerDevice timer;
            ExpertLoopState state{3U, 3U};
            WorkRange owned{2U, 3U};
            assert(!ProgressAsyncCombineSlice(common, params, state, 2U, owned, timer, scratch));
            assert(scratch.asyncInFlight && scratch.asyncRowsDone == 0U && pending.size() == 1U);
            auto* flag =
                reinterpret_cast<int32_t*>(params.peermemInfo.gatewayCombineReadyPtr + HIERARCHICAL_FLAG_BYTES);
            assert(*flag == 0);
            // Simulate a complete small activation between submit and retire.
            std::fill(ub.begin(), ub.begin() + 192U * 1024U, uint8_t{0x55});
            assert(!ProgressAsyncCombineSlice(common, params, state, 2U, owned, timer, scratch));
            assert(static_cast<uint32_t>(*flag) == sequence && scratch.asyncRowsDone == 1U);
            // Large activation fallback explicitly retires before overwriting UB.
            RetireAsyncCombineRow(params, scratch, timer);
            assert(!scratch.asyncInFlight && scratch.asyncRowsDone == 2U);
            ub.fill(0xaa);
            assert(!ProgressAsyncCombineSlice(common, params, state, 2U, owned, timer, scratch));
            assert(ProgressAsyncCombineSlice(common, params, state, 2U, owned, timer, scratch));
            assert(!scratch.asyncInFlight && scratch.asyncRowsDone == 0U && pending.empty() && timer.rows == 3U);
            for (uint32_t i = 0; i < 3U; ++i) {
                const uint32_t row = i == 0U ? 1U : 5U + i;
                auto* target = i == 1U ? peer.data() : params.peermemInfo.gatewayCombinePtr;
                assert(std::memcmp(target + row * hidden * 2U, source.data() + (7U + i) * hidden, hidden * 2U) == 0);
            }
            for (auto type : events)
                for (bool live : type) assert(!live);
        }
    }
}
