// Copyright (c) 2026, Lu Lu
// Production synchronous single-row code with deferred DMA writes. This checks UB,
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
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512U, EVENT_ID0 = 0U;
constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
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
std::vector<PendingWrite> pendingLoads;
std::array<std::vector<PendingWrite>, 4> loadEvents;
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    pendingLoads.push_back({dst.data, src.data, static_cast<uint32_t>(count * sizeof(T))});
}
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    pendingLoads.push_back({dst.data, src.data, copy.bytes});
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    assert(pendingLoads.empty());
    for (const auto& loads : loadEvents) assert(loads.empty());
    pending.push_back({dst.data, src.data, copy.bytes});
}
template <HardEvent E>
void SetFlag(TEventID id)
{
    assert(id == 0U && !events[static_cast<size_t>(E)][id]);
    events[static_cast<size_t>(E)][id] = true;
    if constexpr (E == HardEvent::MTE2_S || E == HardEvent::MTE2_MTE3) {
        assert(loadEvents[static_cast<size_t>(E)].empty());
        loadEvents[static_cast<size_t>(E)].swap(pendingLoads);
    }
}
template <HardEvent E>
void WaitFlag(TEventID id)
{
    assert(id == 0U && events[static_cast<size_t>(E)][id]);
    if constexpr (E == HardEvent::MTE2_S || E == HardEvent::MTE2_MTE3) {
        for (auto load : loadEvents[static_cast<size_t>(E)]) std::memcpy(load.target, load.source, load.bytes);
        loadEvents[static_cast<size_t>(E)].clear();
    }
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
/* PRODUCTION_SINGLE_ROW */

int main()
{
    for (uint32_t hidden : {1024U, 3584U, 7168U, 8192U}) {
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
            for (uint32_t i = 0; i < 3U; ++i) {
                RunSingleRowCombine(common, params, 7U + i, scratch);
                assert(pending.empty());
                for (auto type : events)
                    for (bool live : type) assert(!live);
                // Every call fully retires its transfer before activation can
                // reuse UB. Cached addresses remain valid across those stages.
                ub.fill(0xaa);
                const uint32_t destinationRow = i == 0U ? 1U : 5U + i;
                auto* flag = reinterpret_cast<uint32_t*>(params.peermemInfo.gatewayCombineReadyPtr +
                                                         destinationRow * HIERARCHICAL_FLAG_BYTES);
                assert(*flag == (i == 1U ? 0U : sequence));
            }
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
