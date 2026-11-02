// Copyright (c) 2026, Lu Lu
// Execute the production layout, readiness scheduler, FP32 reduction, and final
// unpermute against CPU device primitives. NIC delivery is explicitly deferred.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*
#define MEGA_MOE_ROCE_H
#define MEGA_MOE_UTILS_H
#define MEGA_MOE_MXFP8_UTILS_H
constexpr int AIC = 0, AIV = 1, g_coreType = AIV;
constexpr int PIPE_V = 1;
constexpr uint32_t EVENT_ID0 = 0;

struct bfloat16_t {
    uint16_t bits = 0;
    bfloat16_t() = default;
    bfloat16_t(float value)
    {
        uint32_t word;
        std::memcpy(&word, &value, 4);
        word += 0x7fffU + ((word >> 16) & 1U);
        bits = static_cast<uint16_t>(word >> 16);
    }
    operator float() const
    {
        uint32_t word = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &word, 4);
        return value;
    }
};
struct fp8_e4m3fn_t {};
struct fp8_e5m2_t {};
namespace Std {
template <class A, class B>
using IsSame = std::is_same<A, B>;
}
namespace ReduceTest {
inline void Check(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
alignas(512) inline std::array<uint8_t, 256U * 1024U> ub{};
inline std::array<std::array<int, 8>, 16> events{};
inline uint64_t cycle = 0;
inline uint32_t activeLane = 0;
inline uint32_t expectedPeer = 0;
inline uint8_t* localBase = nullptr;
inline uint8_t* remoteBase = nullptr;
inline size_t heapBytes = 0;
struct Post {
    uint32_t qp;
    uint8_t* source;
    uint64_t destination;
    std::vector<uint8_t> bytes;
};
inline std::vector<Post> posts;
inline void Deliver()
{
    for (auto& post : posts) {
        Check(std::equal(post.bytes.begin(), post.bytes.end(), post.source), "source reused before completion");
        std::copy(post.bytes.begin(), post.bytes.end(), remoteBase + post.destination);
    }
}
}  // namespace ReduceTest
namespace AscendC {
using TEventID = uint32_t;
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class HardEvent { S_MTE2, MTE2_S, MTE2_V, V_MTE2, S_V, V_S, V_MTE3, MTE3_S, MTE3_V, MTE2_MTE3, MTE3_MTE2 };
template <typename T>
struct LocalTensor {
    T* data = nullptr;
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t count)
    {
        ReduceTest::Check(offset % 32 == 0 && offset + count * sizeof(T) <= ReduceTest::ub.size(),
                          "UB bounds/alignment");
        data = reinterpret_cast<T*>(ReduceTest::ub.data() + offset);
    }
    explicit LocalTensor(T* address) : data(address) {}
    LocalTensor operator[](uint64_t offset) const
    {
        return LocalTensor(data + offset);
    }
    T GetValue(uint32_t index) const
    {
        return data[index];
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        return LocalTensor<U>(reinterpret_cast<U*>(data));
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
struct DataCopyParams {
    uint16_t blockCount, blockLen, srcStride, dstStride;
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride, reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool pad;
    uint32_t left, right;
    T value;
};
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    std::memmove(dst.data, src.data, count * sizeof(T));
}
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyParams p)
{
    for (uint32_t i = 0; i < p.blockCount; ++i) {
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data) + i * (p.blockLen + p.dstStride) * 32U,
                    reinterpret_cast<uint8_t*>(src.data) + i * (p.blockLen + p.srcStride) * 32U, p.blockLen * 32U);
    }
}
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams p, DataCopyPadExtParams<T>)
{
    ReduceTest::Check(p.blockCount == 1, "unexpected packed DMA");
    std::memcpy(dst.data, src.data, p.blockLen);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p)
{
    ReduceTest::Check(p.blockCount == 1, "unexpected packed DMA");
    std::memcpy(dst.data, src.data, p.blockLen);
}
template <typename D, typename S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = D(static_cast<float>(src.data[i]));
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> src, float weight, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = src.data[i] * weight;
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) dst.data[i] = a.data[i] + b.data[i];
}
template <HardEvent E>
void SetFlag(TEventID id)
{
    auto& flag = ReduceTest::events[static_cast<size_t>(E)][id];
    ReduceTest::Check(flag == 0, "reusing a live event");
    flag = 1;
}
template <HardEvent E>
void WaitFlag(TEventID id)
{
    auto& flag = ReduceTest::events[static_cast<size_t>(E)][id];
    ReduceTest::Check(flag == 1, "waiting on an unpublished event");
    flag = 0;
}
template <int>
void PipeBarrier()
{
}
inline int64_t GetSystemCycle()
{
    return ++ReduceTest::cycle;
}
}  // namespace AscendC

#include "../csrc/kernels/mega_moe/common/mega_moe_peermem.hpp"
#include "../csrc/kernels/mega_moe/common/ascend_timer_v2.hpp"
class AscendTimerDevice {
public:
    bool Enabled() const
    {
        return true;
    }
    uint32_t reduced = 0;
    void RecordDetail(AscendTimer::DynamicTimingType type, int64_t start, int64_t end, uint64_t first, uint64_t second)
    {
        ReduceTest::Check(end > start, "empty trace interval");
        if (type == AscendTimer::AIV_COMBINE_REDUCE) {
            ReduceTest::Check((first >> 32) == ReduceTest::activeLane + 1U && (second >> 32) != 0U,
                              "wrong reduction trace");
            ++reduced;
        }
    }
};
struct AscendTimerFixedScope {
    AscendTimerFixedScope(AscendTimerDevice*, AscendTimer::FixedTiming) {}
};
struct AscendTimerCombineProgressScope {
    AscendTimerCombineProgressScope(AscendTimerDevice*, AscendTimer::DynamicTimingType) {}
    void StartScan() {}
    void StopScan() {}
    void Finish(uint32_t, uint32_t, uint32_t, uint32_t) {}
};
namespace MegaMoeImpl {
using namespace AscendC;
constexpr uint32_t MEGAMOE_ROCE_SCRATCH_BYTES = 256U, MEGAMOE_ROCE_EVENT_ID = 7U;
struct MoeStageCommonConfig {
    uint32_t rankId, tokenNum, topK, tokenHiddenDim, moeExpertPerRank, sharedExpertNum = 0;
};
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
};
struct WorkRange {
    uint32_t start, count;
};
inline WorkRange TilingByJobContext(uint32_t count, uint32_t job, uint32_t jobs, uint32_t)
{
    const uint32_t chunk = (count + jobs - 1U) / jobs, first = chunk * job;
    return {first, first < count ? std::min(chunk, count - first) : 0U};
}
struct Params {
    MegaMoeTilingData* tilingData;
    PeermemInfo peermemInfo;
    GM_ADDR probsGmAddr = nullptr;
    GM_ADDR expertIdxGmAddr = nullptr;
    GM_ADDR y2GmAddr = nullptr;
    struct {
        GM_ADDR sharedExpertGmm2TileCounterPtr = nullptr;
        GM_ADDR sharedExpertResultPtr = nullptr;
    } workspaceInfo{};
};
template <HardEvent E, uint32_t Id>
void SyncFuncStatic()
{
    SetFlag<E>(Id);
    WaitFlag<E>(Id);
}
struct RoceWaitState {};
enum class RoceWaitKind { CombineRows };
inline void RoceCheckWait(RoceWaitState&, uint32_t, uint8_t*, RoceWaitKind, uint32_t, uint32_t, int32_t* = nullptr,
                          uint32_t = 1U)
{
    throw std::runtime_error("drain waited on an incomplete producer");
}
inline void RocePutBytes(uint8_t* dst, uint8_t* src, uint32_t bytes, uint32_t peer, uint8_t*, AscendTimerDevice*,
                         AscendTimer::RdmaPhase, bool commit = false, uint32_t qp = 1U)
{
    using namespace ReduceTest;
    Check(!commit && qp == activeLane + 1U && peer == expectedPeer, "foreign QP owner/destination");
    Check(src >= localBase && src + bytes <= localBase + heapBytes && dst >= localBase &&
              dst + bytes <= localBase + heapBytes,
          "non-symmetric operand");
    posts.push_back({qp, src, static_cast<uint64_t>(dst - localBase), {src, src + bytes}});
}
namespace Mxfp8 {
template <typename A, typename B, typename... Args>
void DeQuantMxFp8(Args&&...)
{
}
}  // namespace Mxfp8
inline int32_t* GetCombineSyncCounterAddress(int32_t* base, uint32_t group)
{
    return base + group * INT_CACHELINE;
}
inline void WaitUntilGmFlagEquals(int32_t* address, int32_t expected)
{
    ReduceTest::Check(*address == expected, "shared result not ready");
}
}  // namespace MegaMoeImpl
#include "../csrc/kernels/mega_moe/stage/mega_moe_hierarchical_reduce.hpp"
#include "../csrc/kernels/mega_moe/stage/mega_moe_unpermute.hpp"

namespace MegaMoeImpl {
/* PRODUCTION_WEIGHT_PACKER */
}
using namespace MegaMoeImpl;

void CheckCase(uint32_t qps, uint32_t sourceRank, uint32_t bs, uint32_t capacity, uint32_t topk, uint32_t hidden,
               uint32_t pattern, uint32_t sequence, bool shared)
{
    using namespace ReduceTest;
    ub.fill(0);
    events = {};
    posts.clear();
    MegaMoeTilingData t{};
    t.bs = bs;
    t.numMaxTokensPerRank = capacity;
    t.topK = topk;
    t.h = hidden;
    t.epWorldSize = 32U;
    t.dispatchRanksPerNode = 16U;
    t.moeExpertPerRank = 12U;
    t.blockAivNum = 56U;
    t.aicNum = 28U;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    t.combineReduceQps = qps;
    t.roceScratchUbOffset = 256U * 1024U - 512U;
    t.hierarchicalSequence = sequence;
    t.sharedExpertNum = shared ? 1U : 0U;
    const PeermemSizeParams sizes{capacity,           topk, hidden, 12U, 32U, 2U, 1U, false, false,
                                  TOPO_TYPE_MTE_ROCE, true, 16U,    qps};
    heapBytes = CalcPeermemLeastSize(sizes);
    std::vector<uint8_t> sourceHeap(heapBytes, 0xA5), gatewayHeap(heapBytes, 0xA5);
    PeermemInfo source(sourceHeap.data(), &t), gateway(gatewayHeap.data(), &t);
    Check(gateway.combineReduceCommitPtr + qps * 512U == gatewayHeap.data() + heapBytes, "layout budget mismatch");
    localBase = gatewayHeap.data();
    remoteBase = sourceHeap.data();
    expectedPeer = sourceRank;
    const uint32_t gatewayRank = HierarchicalGateway(sourceRank);
    MoeStageCommonConfig srcCommon{sourceRank, bs, topk, hidden, 12U, shared ? 1U : 0U};
    MoeStageCommonConfig gwCommon{gatewayRank, bs, topk, hidden, 12U};
    std::vector<int32_t> ids(bs * topk);
    std::vector<float> weights(bs * topk);
    std::vector<bfloat16_t> expertRows(static_cast<size_t>(bs) * topk * hidden);
    std::vector<bfloat16_t> output(bs * hidden), sharedRows(bs * hidden);
    std::vector<float> expectedPartial(bs * hidden, 0.0f);
    std::vector<bfloat16_t> expectedOutput(bs * hidden);
    std::vector<uint32_t> masks(bs, 0U);
    const uint32_t sourceHost = HierarchicalHost(sourceRank);
    auto* originalLocal = reinterpret_cast<bfloat16_t*>(source.combineSendPtr);
    auto* originalRemote = reinterpret_cast<bfloat16_t*>(gateway.gatewayCombinePtr);
    auto* ready = reinterpret_cast<uint32_t*>(gateway.gatewayCombineReadyPtr);
    const bfloat16_t poison(std::nanf(""));
    std::fill_n(originalLocal, capacity * topk * hidden, poison);
    std::fill_n(originalRemote, capacity * topk * hidden, poison);
    for (uint32_t token = 0; token < bs; ++token) {
        for (uint32_t k = 0; k < topk; ++k) {
            const bool remote = pattern == 1 || (pattern == 2 && (token + k) % 3 == 0) ||
                                (pattern == 3 && (token % 4) >= 2) || (pattern == 4 && token == bs - 1U);
            ids[token * topk + k] = static_cast<int32_t>((remote ? 1U - sourceHost : sourceHost) * 192U + k);
            weights[token * topk + k] = static_cast<float>(static_cast<int>((token * 11U + k * 7U) % 17U) - 8) / 19.0f;
            if (remote) masks[token] |= uint32_t{1} << k;
            for (uint32_t h = 0; h < hidden; ++h) {
                const size_t index = (static_cast<size_t>(token) * topk + k) * hidden + h;
                expertRows[index] =
                    bfloat16_t(static_cast<float>(static_cast<int>((token * 31U + k * 13U + h) % 61U) - 30) / 23.0f);
                (remote ? originalRemote : originalLocal)[index] = expertRows[index];
            }
            ready[(token * topk + k) * 128U] = sequence - 1U;
        }
        for (uint32_t h = 0; h < hidden; ++h) {
            float remoteSum = 0.0f, localSum = 0.0f;
            bool firstRemote = true, firstLocal = true;
            for (uint32_t k = 0; k < topk; ++k) {
                const float value =
                    static_cast<float>(expertRows[(static_cast<size_t>(token) * topk + k) * hidden + h]) *
                    weights[token * topk + k];
                const bool remote = (masks[token] & (uint32_t{1} << k)) != 0U;
                float& sum = remote ? remoteSum : localSum;
                bool& first = remote ? firstRemote : firstLocal;
                sum = first ? value : sum + value;
                first = false;
            }
            expectedPartial[token * hidden + h] = remoteSum;
            float total = firstLocal ? remoteSum : firstRemote ? localSum : localSum + remoteSum;
            sharedRows[token * hidden + h] = bfloat16_t(static_cast<float>(h % 7U) / 13.0f);
            if (shared) total += static_cast<float>(sharedRows[token * hidden + h]);
            expectedOutput[token * hidden + h] = bfloat16_t(total);
        }
    }
    Params sourceParams{&t, source, reinterpret_cast<uint8_t*>(weights.data()), reinterpret_cast<uint8_t*>(ids.data()),
                        reinterpret_cast<uint8_t*>(output.data())};
    // Execute the production weight packer, then model one combined metadata Put.
    PackHierarchicalCombineWeights(srcCommon, sourceParams);
    const uint64_t slot = sourceHost * CalcRailTopkIdsSlotBytes(&t);
    auto* sourceRail = reinterpret_cast<int32_t*>(source.railTopkIdsRecvPtr + slot);
    for (uint32_t row = 0; row < bs * topk; ++row) sourceRail[(row / 127U) * 128U + row % 127U] = ids[row];
    for (uint32_t packet = 0; packet < (bs * topk + 126U) / 127U; ++packet)
        sourceRail[packet * 128U + 127U] = static_cast<int32_t>(sequence);
    const uint64_t prefix = HierarchicalMetadataPrefixBytes(&t);
    const uint64_t bytes = prefix + CalcRailTopkIdsSlotDataBytes(bs, topk);
    std::memcpy(gateway.railTopkIdsRecvPtr + slot - prefix, source.railTopkIdsRecvPtr + slot - prefix, bytes);
    Check(std::memcmp(gateway.railTopkIdsRecvPtr + slot - prefix, weights.data(), weights.size() * 4U) == 0,
          "packed weights differ");
    Params params{&t, gateway};
    std::array<HierarchicalCombineProgress, 8> state{};
    AscendTimerDevice timer;
    auto* scratch = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(t.roceScratchUbOffset));
    for (activeLane = 0; activeLane < qps; ++activeLane) {
        Check(!TryProgressHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer),
              "stale flags released data");
    }
    // Publish expert rows in reverse order. A partial token must never be posted.
    std::vector<uint32_t> published(bs, 0U);
    for (uint32_t row = bs * topk; row-- > 0;) {
        const uint32_t token = row / topk, k = row % topk;
        if ((masks[token] & (uint32_t{1} << k)) == 0) continue;
        ready[row * 128U] = sequence;
        published[token] |= uint32_t{1} << k;
        const size_t before = posts.size();
        for (activeLane = 0; activeLane < qps; ++activeLane)
            TryProgressHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer);
        for (size_t p = before; p < posts.size(); ++p) {
            const auto& post = posts[p];
            const uint32_t sent = static_cast<uint32_t>((post.source - gateway.combinePartialSendPtr) / (hidden * 4U));
            Check(published[sent] == masks[sent], "incomplete token was sent");
        }
    }
    for (activeLane = 0; activeLane < qps; ++activeLane)
        DrainHierarchicalReducedCombine(gwCommon, params, state[activeLane], activeLane, scratch, &timer);
    std::vector<uint32_t> sent(bs, 0U);
    for (const auto& post : posts) {
        const uint32_t token = static_cast<uint32_t>((post.source - gateway.combinePartialSendPtr) / (hidden * 4U));
        Check(post.bytes.size() == hidden * 4U, "partial was not FP32");
        Check(token >= bs * (post.qp - 1U) / qps && token < bs * post.qp / qps, "token owned by wrong QP");
        ++sent[token];
        Check(std::memcmp(post.bytes.data(), expectedPartial.data() + token * hidden, post.bytes.size()) == 0,
              "FP32 weighted sum differs");
    }
    for (uint32_t token = 0; token < bs; ++token)
        Check(sent[token] == static_cast<uint32_t>(masks[token] != 0), "missing or duplicate token");
    Check(timer.reduced == posts.size(), "missing reduction timing record");
    // UB can be reused while the NIC owns immutable symmetric GM sources.
    ub.fill(0xCC);
    Deliver();
    if (!posts.empty()) {
        posts.front().source[0] ^= 1U;
        bool failed = false;
        try {
            Deliver();
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Check(failed, "source lifetime negative control failed");
        posts.front().source[0] ^= 1U;
    }
    // Production output consumption must skip poisoned remote BF16 slots and
    // include the FP32 partial once, then the optional shared result once.
    const uint32_t sharedN = SelectSharedExpertPolicy(true, shared ? 1U : 0U, bs).gmm2TileN;
    std::vector<int32_t> sharedReady(INT_CACHELINE, static_cast<int32_t>((hidden + sharedN - 1U) / sharedN));
    sourceParams.workspaceInfo.sharedExpertGmm2TileCounterPtr = reinterpret_cast<uint8_t*>(sharedReady.data());
    sourceParams.workspaceInfo.sharedExpertResultPtr = reinterpret_cast<uint8_t*>(sharedRows.data());
    for (uint32_t physical = 0; physical < t.blockAivNum; ++physical) {
        const uint32_t logical = UnpermuteJobIndex(&t, physical);
        if (logical >= UnpermuteJobCount(&t)) continue;
        events = {};
        MegaMoeUnpermuteBufferConfig buffer{};
        buffer.tokensPerBatch = 2;
        buffer.inputBufferCount = 2;
        buffer.bf16SlotElementCount = hidden;
        buffer.fp32SlotElementCount = hidden;
        buffer.topKWeightsBufferBytes = Ops::Base::CeilAlign(topk * 2U * 4U, 32U);
        TokenUnpermuteConfig context{{logical, UnpermuteJobCount(&t)}, 0U, UnpermuteJobCount(&t), buffer, {}};
        TokenUnpermuteScratch buffers;
        auto actual = CreateTokenUnpermuteBuffers<float, COMBINE_NO_QUANT>(context, hidden, buffers);
        UnpermuteTokens<COMBINE_NO_QUANT, float, false, 256U>(context, srcCommon, sourceParams, buffers, actual);
        for (const auto& type : events)
            for (int live : type) Check(live == 0, "unpermute leaked an event");
    }
    for (size_t i = 0; i < output.size(); ++i)
        Check(output[i].bits == expectedOutput[i].bits, "final output differs / remote weighted twice");
}

int main()
{
    try {
        for (uint32_t qps : {4U, 8U})
            for (uint32_t rank : {0U, 21U})
                for (uint32_t pattern = 0; pattern < 5; ++pattern) {
                    CheckCase(qps, rank, 32U, 64U, 8U, 3584U, pattern, 0x80000101U, false);
                    CheckCase(qps, rank, 7U, 32U, 1U, 1024U, pattern, 0xffffffffU, true);
                    CheckCase(qps, rank, 1U, 8U, 32U, 8192U, pattern, 129U, false);
                }
        CheckCase(4U, 0U, 256U, 256U, 32U, 1024U, 2U, 257U, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
