// Copyright (c) 2026, Lu Lu

// Compile the production progress/initialization/drain code, replacing only
// device primitives and heavyweight type dependencies. The independent oracle
// is a set of original route coordinates, not another progress-loop algorithm.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define GM_ADDR uint8_t*
#define MEGA_MOE_ROCE_H
#define MEGA_MOE_UTILS_H

constexpr int AIC = 0;
constexpr int AIV = 1;
constexpr int g_coreType = AIV;
using bfloat16_t = uint16_t;
// CATLASS also declares a global CacheMode. Keep that name collision visible
// when compiling the real progress header, even without the heavy dependency.
enum class CacheMode { CATLASS_TEST_VALUE };
namespace AscendTimer {
enum class RdmaPhase { Combine };
enum FixedTiming { AIV_COMBINE_INIT };
enum DynamicTimingType { AIV_COMBINE_PROGRESS, AIV_COMBINE_DRAIN, AIV_COMBINE_RUN };
}  // namespace AscendTimer

// The timer itself has a separate production-header test. This double checks
// that the production scheduler brackets scans/Put and reports actual counters.
class AscendTimerDevice {
public:
    struct Record {
        AscendTimer::DynamicTimingType type;
        uint32_t before;
        uint32_t after;
        uint32_t posts;
        uint32_t scanned;
    };
    std::vector<Record> records;
    struct RunRecord {
        uint64_t first;
        uint64_t second;
    };
    std::vector<RunRecord> runs;
    bool Enabled() const
    {
        return true;
    }
    void RecordDetail(AscendTimer::DynamicTimingType type, int64_t start, int64_t end, uint64_t first, uint64_t second)
    {
        if (type != AscendTimer::AIV_COMBINE_RUN || end <= start) std::abort();
        runs.push_back({first, second});
    }
    uint32_t initializations = 0U;
    uint32_t scanStarts = 0U;
    uint32_t scanStops = 0U;
    bool scanActive = false;
};

class AscendTimerFixedScope {
public:
    AscendTimerFixedScope(AscendTimerDevice* timer, AscendTimer::FixedTiming)
    {
        if (timer != nullptr) {
            ++timer->initializations;
        }
    }
};

class AscendTimerCombineProgressScope {
public:
    AscendTimerCombineProgressScope(AscendTimerDevice* timer, AscendTimer::DynamicTimingType type)
        : timer_(timer), type_(type)
    {
    }
    void StartScan()
    {
        if (timer_ != nullptr) {
            if (timer_->scanActive) {
                std::fputs("scan timer started twice\n", stderr);
                std::abort();
            }
            timer_->scanActive = true;
            ++timer_->scanStarts;
        }
    }
    void StopScan()
    {
        if (timer_ != nullptr && timer_->scanActive) {
            timer_->scanActive = false;
            ++timer_->scanStops;
        }
    }
    void Finish(uint32_t before, uint32_t after, uint32_t posts, uint32_t scanned)
    {
        StopScan();
        if (timer_ != nullptr) {
            timer_->records.push_back({type_, before, after, posts, scanned});
        }
    }

private:
    AscendTimerDevice* timer_;
    AscendTimer::DynamicTimingType type_;
};

namespace ProgressTest {
inline void Check(bool condition, const char* message)
{
    if (!condition) {
        std::fputs(message, stderr);
        std::fputc('\n', stderr);
        std::abort();
    }
}

struct Post {
    uint32_t firstRow;
    uint32_t rowCount;
    uint8_t* source;
    uint8_t* destination;
};
struct Copy {
    uint32_t sourceWord;
    uint32_t words;
};
struct State {
    alignas(32) std::array<uint8_t, 256> ub{};
    std::vector<int32_t> rail;
    std::vector<int32_t> flags;
    std::vector<uint8_t> source;
    std::vector<uint8_t> destination;
    std::vector<bool> expected;
    std::vector<uint32_t> sent;
    std::vector<uint32_t> probes;
    std::vector<uint32_t> probeOrder;
    std::vector<Post> posts;
    std::vector<Copy> copies;
    const int32_t* pairedRail = nullptr;
    uint32_t railSlotWords = 0U;
    uint32_t rowBytes = 0U;
    uint32_t rows = 0U;
    uint32_t topK = 0U;
    const uint32_t* pendingMasks = nullptr;
    uint32_t peer = 0U;
    uint32_t sequence = 0U;
    uint32_t flagReads = 0U;
    uint32_t scalarUbReads = 0U;
    uint32_t metadataScalarReads = 0U;
    uint64_t cycles = 0U;
    uint64_t releaseAtCycle = 0U;
    bool overwriteOrdered = false;
    const int32_t* pendingDma = nullptr;
    uint32_t pendingDmaWords = 0U;
    uint32_t pendingDmaStrideWords = 0U;
    uint32_t pendingDmaBlocks = 0U;
    uint32_t flagDmas = 0U;
    uint32_t releaseAfterSnapshot = 0xffffffffU;
    uint32_t waitChecks = 0U;
    uint32_t failAfterWaitChecks = 0U;
};
inline State* state = nullptr;
constexpr uint32_t kUbOffset = 4096U;
constexpr uint32_t kFlagWords = 512U / sizeof(int32_t);

inline void ReleaseAll()
{
    for (uint32_t row = 0U; row < state->rows; ++row) {
        if (state->expected[row]) {
            state->flags[row * kFlagWords] = static_cast<int32_t>(state->sequence);
        }
    }
}
}  // namespace ProgressTest

namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { S_MTE2, MTE2_S };
enum class CacheMode { CACHE_MODE_DISABLE, CACHE_MODE_NORMAL };
struct DataCopyParams {
    uint16_t blockCount;
    uint16_t blockLen;
    uint16_t srcStride;
    uint16_t dstStride;
};

template <typename T>
class LocalTensor {
public:
    LocalTensor(TPosition position, uint32_t offset, uint32_t size)
    {
        ProgressTest::Check(position == TPosition::VECCALC && offset == ProgressTest::kUbOffset,
                            "metadata load used an unowned UB range");
        ProgressTest::Check(size * sizeof(T) <= ProgressTest::state->ub.size(), "UB tensor exceeds scratch");
        data_ = reinterpret_cast<T*>(ProgressTest::state->ub.data());
        size_ = size;
    }
    T GetValue(uint32_t index) const
    {
        auto& s = *ProgressTest::state;
        ProgressTest::Check(s.pendingDma == nullptr, "metadata read preceded MTE2 completion");
        ProgressTest::Check(index < size_, "metadata scalar read exceeded UB tensor");
        ++s.scalarUbReads;
        s.overwriteOrdered = false;
        return data_[index];
    }
    T* Data() const
    {
        return data_;
    }

private:
    T* data_ = nullptr;
    uint32_t size_ = 0U;
};

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T* address)
    {
        data_ = address;
    }
    GlobalTensor operator[](uint32_t offset) const
    {
        GlobalTensor result;
        result.data_ = data_ + offset;
        return result;
    }
    T* Data() const
    {
        return data_;
    }

    void SetL2CacheHint(CacheMode mode)
    {
        cacheMode_ = mode;
    }
    CacheMode GetL2CacheHint() const
    {
        return cacheMode_;
    }

private:
    T* data_ = nullptr;
    CacheMode cacheMode_ = CacheMode::CACHE_MODE_NORMAL;
};

template <typename T>
void DataCopy(const LocalTensor<T>& destination, const GlobalTensor<T>& source, uint32_t count)
{
    using namespace ProgressTest;
    auto& s = *state;
    static_assert(sizeof(T) == sizeof(int32_t));
    Check(s.overwriteOrdered && s.pendingDma == nullptr, "metadata DMA reused live scratch");
    Check(destination.Data() == reinterpret_cast<int32_t*>(s.ub.data()), "DMA destination is not owned scratch");
    const auto address = reinterpret_cast<uintptr_t>(source.Data());
    const auto begin = reinterpret_cast<uintptr_t>(s.pairedRail);
    const auto bytes = static_cast<uintptr_t>(s.railSlotWords) * sizeof(int32_t);
    Check(address >= begin && address <= begin + bytes, "metadata DMA used wrong host's slot");
    Check(count == 64U && count * sizeof(T) <= begin + bytes - address, "metadata DMA exceeds source slot");
    const uint32_t offset = static_cast<uint32_t>((address - begin) / sizeof(int32_t));
    Check(offset % 64U == 0U, "metadata DMA is not a scratch-sized aligned block");
    s.copies.push_back({offset, count});
    s.pendingDma = source.Data();
    s.pendingDmaWords = count;
    s.overwriteOrdered = false;
}

template <typename T>
void DataCopy(const LocalTensor<T>& destination, const GlobalTensor<T>& source, DataCopyParams copy)
{
    using namespace ProgressTest;
    auto& s = *state;
    Check(s.overwriteOrdered && s.pendingDma == nullptr, "ready DMA reused live scratch");
    Check(destination.Data() == reinterpret_cast<T*>(s.ub.data()), "ready DMA used unowned scratch");
    Check(source.GetL2CacheHint() == CacheMode::CACHE_MODE_NORMAL, "ready DMA disabled the default L2 cache");
    Check(copy.blockCount > 0U && copy.blockCount <= 8U && copy.blockLen == 1U && copy.srcStride == 15U &&
              copy.dstStride == 0U,
          "invalid ready gather geometry");
    const auto offset = source.Data() - s.flags.data();
    Check(offset >= 0 && offset % kFlagWords == 0, "ready gather is not flag-aligned");
    const auto first = static_cast<uint32_t>(offset / kFlagWords);
    Check(first + copy.blockCount <= s.rows, "ready gather exceeded row tensor");
    for (uint32_t row = first; row < first + copy.blockCount; ++row) {
        ++s.flagReads;
        ++s.probes[row];
        s.probeOrder.push_back(row);
    }
    ++s.flagDmas;
    s.pendingDma = source.Data();
    s.pendingDmaWords = 8U;
    s.pendingDmaStrideWords = kFlagWords;
    s.pendingDmaBlocks = copy.blockCount;
    s.overwriteOrdered = false;
}

inline int64_t GetSystemCycle()
{
    auto& s = *ProgressTest::state;
    ProgressTest::Check(++s.cycles < 1000000U, "drain failed to make bounded test progress");
    if (s.releaseAtCycle != 0U && s.cycles >= s.releaseAtCycle) {
        ProgressTest::ReleaseAll();
        s.releaseAtCycle = 0U;
    }
    return static_cast<int64_t>(s.cycles);
}
}  // namespace AscendC

namespace MegaMoeImpl {
using namespace AscendC;
constexpr uint32_t HIERARCHICAL_MAX_TOKENS = 144U;
constexpr uint32_t HIERARCHICAL_RANKS_PER_HOST = 16U;
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512U;
constexpr uint32_t MEGAMOE_ROCE_SCRATCH_BYTES = 256U;
constexpr uint32_t MEGAMOE_ROCE_EVENT_ID = 7U;
constexpr uint32_t RAIL_PACKET_WORDS = 128U;
constexpr uint32_t RAIL_PACKET_DATA_WORDS = 127U;
constexpr uint32_t GM_FLAG_POLL_BACKOFF_CYCLES = 8U;
struct MoeStageCommonConfig {
    uint32_t rankId = 0U;
    uint32_t moeExpertPerRank = 32U;
    uint32_t tokenNum = 0U;
    uint32_t topK = 0U;
    uint32_t tokenHiddenDim = 16U;
};
struct MegaMoeTilingData {
    uint32_t numMaxTokensPerRank = 32U;
    uint32_t topK = 0U;
    uint32_t hierarchicalSequence = 0U;
    uint32_t roceScratchUbOffset = ProgressTest::kUbOffset;
    bool hierarchical = true;
};
struct Params {
    MegaMoeTilingData* tilingData = nullptr;
    struct {
        uint8_t* railTopkIdsRecvPtr = nullptr;
        uint8_t* gatewayCombineReadyPtr = nullptr;
        uint8_t* combineSendPtr = nullptr;
        uint8_t* gatewayCombinePtr = nullptr;
    } peermemInfo;
};
inline uint32_t HierarchicalHost(uint32_t rank)
{
    return rank / HIERARCHICAL_RANKS_PER_HOST;
}
inline uint32_t HierarchicalGateway(uint32_t rank)
{
    return rank ^ HIERARCHICAL_RANKS_PER_HOST;
}
struct RoceWaitState {};
enum class RoceWaitKind { CombineRows };
inline void RoceCheckWait(RoceWaitState&, uint32_t peer, uint8_t* scratch, RoceWaitKind kind, uint32_t expected,
                          uint32_t observed)
{
    auto& s = *ProgressTest::state;
    ProgressTest::Check(peer == s.peer && scratch == s.ub.data(), "drain wait used another owner's QP/scratch");
    ProgressTest::Check(kind == RoceWaitKind::CombineRows && observed < expected,
                        "drain wait lost its incomplete-row diagnostic");
    ++s.waitChecks;
    if (s.failAfterWaitChecks != 0U && s.waitChecks >= s.failAfterWaitChecks) {
        throw std::runtime_error("stalled producer");
    }
}
inline bool UseHierarchicalRoce(const MegaMoeTilingData* tiling)
{
    return tiling->hierarchical;
}
inline uint64_t CalcRailTopkIdsSlotBytes(const MegaMoeTilingData* tiling)
{
    const auto rows = tiling->numMaxTokensPerRank * tiling->topK;
    return static_cast<uint64_t>((std::max(rows, 1U) + 126U) / 127U) * 512U;
}
template <HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    static_assert(Id == MEGAMOE_ROCE_EVENT_ID);
    auto& s = *ProgressTest::state;
    if constexpr (Event == HardEvent::S_MTE2) {
        ProgressTest::Check(s.pendingDma == nullptr, "scratch overwrite ordering before pending DMA completes");
        s.overwriteOrdered = true;
    } else {
        static_assert(Event == HardEvent::MTE2_S);
        ProgressTest::Check(s.pendingDma != nullptr, "MTE2 completion without a metadata DMA");
        if (s.pendingDmaBlocks != 0U) {
            for (uint32_t i = 0U; i < s.pendingDmaBlocks; ++i) {
                std::memcpy(s.ub.data() + i * 32U, s.pendingDma + i * s.pendingDmaStrideWords, 32U);
            }
            // Publish only after the DMA image has been captured.
            if (s.releaseAfterSnapshot < s.rows) {
                s.flags[s.releaseAfterSnapshot * ProgressTest::kFlagWords] = static_cast<int32_t>(s.sequence);
                s.releaseAfterSnapshot = 0xffffffffU;
            }
        } else {
            std::memcpy(s.ub.data(), s.pendingDma, s.pendingDmaWords * sizeof(int32_t));
        }
        s.pendingDma = nullptr;
        s.pendingDmaWords = 0U;
        s.pendingDmaBlocks = 0U;
    }
}
inline int32_t ReadGmByPassDCache(const int32_t* address)
{
    using namespace ProgressTest;
    auto& s = *state;
    const auto raw = reinterpret_cast<uintptr_t>(address);
    const auto begin = reinterpret_cast<uintptr_t>(s.flags.data());
    const auto size = static_cast<uintptr_t>(s.flags.size()) * sizeof(int32_t);
    if (raw < begin || raw >= begin + size) {
        ++s.metadataScalarReads;
        Check(false, "progress made a scalar GM load outside the ready flags");
    }
    const auto offset = raw - begin;
    Check(offset % HIERARCHICAL_FLAG_BYTES == 0U, "flag probe did not target a row epoch");
    const auto row = static_cast<uint32_t>(offset / HIERARCHICAL_FLAG_BYTES);
    Check(row < s.rows && s.expected[row] && s.sent[row] == 0U, "probed an unused or already-submitted row");
    Check(++s.flagReads < 1000000U, "flag polling did not terminate");
    ++s.probes[row];
    s.probeOrder.push_back(row);
    return *address;
}
inline void RocePutBytes(uint8_t* destination, uint8_t* source, uint32_t bytes, uint32_t peer, uint8_t* scratch,
                         AscendTimerDevice* timer, AscendTimer::RdmaPhase phase)
{
    using namespace ProgressTest;
    auto& s = *state;
    Check(timer == nullptr || !timer->scanActive, "scan timer included RDMA submission");
    Check(phase == AscendTimer::RdmaPhase::Combine && peer == s.peer, "Put used wrong phase or paired rank");
    Check(scratch == s.ub.data() && s.pendingDma == nullptr && s.overwriteOrdered,
          "Put reused metadata scratch before its reads completed");
    const auto srcOffset = reinterpret_cast<uintptr_t>(source) - reinterpret_cast<uintptr_t>(s.source.data());
    const auto dstOffset = reinterpret_cast<uintptr_t>(destination) - reinterpret_cast<uintptr_t>(s.destination.data());
    Check(srcOffset == dstOffset && srcOffset <= s.source.size(), "Put changed original row coordinates");
    Check(bytes != 0U && bytes % s.rowBytes == 0U && srcOffset % s.rowBytes == 0U,
          "Put must contain complete BF16 rows");
    Check(bytes <= s.source.size() - srcOffset, "Put exceeded the last source row");
    const auto first = static_cast<uint32_t>(srcOffset / s.rowBytes);
    const auto count = bytes / s.rowBytes;
    for (uint32_t row = first; row < first + count; ++row) {
        Check(s.expected[row], "Put merged across an unused route");
        Check(s.flags[row * kFlagWords] == static_cast<int32_t>(s.sequence), "Put merged across an unready row");
        Check((s.pendingMasks[row / s.topK] & (uint32_t(1) << (row % s.topK))) != 0U,
              "pending row was cleared before Put returned");
        Check(s.sent[row]++ == 0U, "row submitted more than once");
    }
    s.posts.push_back({first, count, source, destination});
    // Submission deliberately leaves the destination untouched. The SDK may
    // overwrite its scratch immediately; cached pending masks must survive it.
    s.ub.fill(0xD7U);
}
}  // namespace MegaMoeImpl

#include "../csrc/kernels/mega_moe/stage/mega_moe_hierarchical_combine.hpp"

namespace ProgressTest {
using namespace MegaMoeImpl;

class Fixture {
public:
    State s;
    MoeStageCommonConfig common;
    MegaMoeTilingData tiling;
    Params params;
    HierarchicalCombineProgress progress{};
    std::vector<int32_t> ids;

    Fixture(uint32_t tokens, uint32_t topk, uint32_t rank = 0U, uint32_t sequence = 129U)
    {
        state = &s;
        common.tokenNum = tokens;
        tiling.numMaxTokensPerRank = std::max(32U, tokens);
        common.topK = tiling.topK = topk;
        common.rankId = rank;
        tiling.hierarchicalSequence = s.sequence = sequence;
        s.rows = tokens * topk;
        s.topK = topk;
        s.pendingMasks = progress.pendingTopkMasks;
        s.rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
        s.peer = HierarchicalGateway(rank);
        s.railSlotWords = static_cast<uint32_t>(CalcRailTopkIdsSlotBytes(&tiling) / sizeof(int32_t));
        s.rail.assign(2U * s.railSlotWords, -1);
        s.flags.assign(std::max(s.rows, 1U) * kFlagWords, static_cast<int32_t>(sequence - 1U));
        s.source.resize(s.rows * s.rowBytes);
        s.destination.assign(s.source.size(), 0xA5U);
        s.expected.assign(s.rows, false);
        s.sent.assign(s.rows, 0U);
        s.probes.assign(s.rows, 0U);
        ids.assign(s.rows, -1);
        auto* paired = s.rail.data() + HierarchicalHost(s.peer) * s.railSlotWords;
        s.pairedRail = paired;
        params.tilingData = &tiling;
        params.peermemInfo.railTopkIdsRecvPtr = reinterpret_cast<uint8_t*>(s.rail.data());
        params.peermemInfo.gatewayCombineReadyPtr = reinterpret_cast<uint8_t*>(s.flags.data());
        params.peermemInfo.gatewayCombinePtr = s.source.data();
        params.peermemInfo.combineSendPtr = s.destination.data();
        for (uint32_t index = 0U; index < s.source.size(); ++index) {
            s.source[index] = static_cast<uint8_t>((index * 37U + index / s.rowBytes) % 251U);
        }
    }
    void Routes(const char* pattern)
    {
        const uint32_t expertsPerHost = 16U * common.moeExpertPerRank;
        const uint32_t localBase = HierarchicalHost(common.rankId) * expertsPerHost;
        const uint32_t otherBase = (1U - HierarchicalHost(common.rankId)) * expertsPerHost;
        // Poison marker/unused slots with a valid local expert: consuming either
        // as a route creates an observable extra expected row.
        std::fill(s.rail.begin(), s.rail.end(), static_cast<int32_t>(localBase + 1U));
        auto* paired = const_cast<int32_t*>(s.pairedRail);
        for (uint32_t row = 0U; row < s.rows; ++row) {
            bool remoteContribution = std::strcmp(pattern, "all") == 0 || std::strcmp(pattern, "duplicate") == 0;
            if (std::strcmp(pattern, "mixed") == 0) {
                remoteContribution = row % 5U == 0U || row % 7U == 2U;
            }
            if (std::strcmp(pattern, "last") == 0) {
                remoteContribution = row + 1U == s.rows;
            }
            const uint32_t expert = remoteContribution
                                        ? localBase + (std::strcmp(pattern, "duplicate") == 0 ? 7U : row % 511U)
                                        : otherBase + row % 511U;
            ids[row] = std::strcmp(pattern, "holes") == 0 || (std::strcmp(pattern, "mixed") == 0 && row % 11U == 3U)
                           ? -1
                           : static_cast<int32_t>(expert);
            s.expected[row] =
                ids[row] >= 0 && static_cast<uint32_t>(ids[row]) / expertsPerHost == HierarchicalHost(common.rankId);
            paired[row / 127U * 128U + row % 127U] = ids[row];
        }
    }
    void Init()
    {
        InitializeHierarchicalCombineProgress(common, params, progress);
        Check(progress.initialized, "initializer did not finish");
        Check(progress.expectedRows == Expected(), "metadata route count differs from independent route set");
        Check(s.pendingDma == nullptr, "initializer left metadata DMA in flight");
    }
    void RouteRows(const std::vector<uint32_t>& rows)
    {
        Routes("holes");
        auto* paired = const_cast<int32_t*>(s.pairedRail);
        for (uint32_t row : rows) {
            Check(row < s.rows, "test attempted to route nonexistent row");
            ids[row] = static_cast<int32_t>(HierarchicalHost(common.rankId) * 16U * common.moeExpertPerRank);
            s.expected[row] = true;
            paired[row / 127U * 128U + row % 127U] = ids[row];
        }
    }
    uint32_t Expected() const
    {
        return static_cast<uint32_t>(std::count(s.expected.begin(), s.expected.end(), true));
    }
    void Publish(uint32_t row, uint32_t epoch)
    {
        Check(row < s.rows, "test attempted to publish nonexistent row");
        s.flags[row * kFlagWords] = static_cast<int32_t>(epoch);
    }
    bool Step(uint32_t posts, uint32_t scans)
    {
        const auto beforePosts = s.posts.size();
        const auto beforeReads = s.flagReads;
        const auto beforeFlags = s.flags;
        const auto beforeRow = progress.nextRow;
        const bool made =
            TryProgressHierarchicalCombine(common, params, progress, s.ub.data(), nullptr, {posts, scans});
        const auto afterPosts = s.posts.size() - beforePosts;
        Check(made == (afterPosts != 0U), "progress result did not describe actual submission");
        Check(afterPosts <= posts && s.flagReads - beforeReads <= scans, "progress exceeded its operation budget");
        Check(beforeFlags == s.flags, "progress modified persistent producer epochs");
        if (s.rows != 0U && scans < s.rows && posts != 0U && scans != 0U) {
            const uint32_t advanced = (progress.nextRow + s.rows - beforeRow) % s.rows;
            Check(advanced <= scans, "cursor exceeded the row scan budget");
        }
        Check(progress.sentRows == static_cast<uint32_t>(std::count(s.sent.begin(), s.sent.end(), 1U)),
              "sentRows accounting mismatch");
        return made;
    }
    void Finish(uint32_t posts = 2U, uint32_t scans = 7U)
    {
        for (uint32_t attempts = 0U; progress.sentRows < Expected() || !progress.initialized; ++attempts) {
            Check(attempts <= s.rows * 4U + 4U, "ready rows were starved by the resumable cursor");
            Step(posts, scans);
        }
    }
    void CheckPayload()
    {
        Check(std::all_of(s.destination.begin(), s.destination.end(), [](uint8_t b) { return b == 0xA5U; }),
              "progress implicitly completed asynchronous Put");
        for (const auto& post : s.posts) {
            std::memcpy(post.destination, post.source, post.rowCount * s.rowBytes);
        }
        for (uint32_t row = 0U; row < s.rows; ++row) {
            Check(s.sent[row] == (s.expected[row] ? 1U : 0U), "final send coverage differs from routing oracle");
            for (uint32_t column = 0U; column < s.rowBytes; ++column) {
                const uint32_t index = row * s.rowBytes + column;
                Check(s.destination[index] == (s.expected[row] ? s.source[index] : 0xA5U),
                      "final payload changed BF16 bits or overwrote an unused row");
            }
        }
    }
};

void Routing()
{
    for (uint32_t tokens : {1U, 2U, 7U, 31U, 32U}) {
        for (uint32_t topk : {1U, 3U, 8U, 31U, 32U}) {
            for (uint32_t rank : {0U, 7U, 16U, 31U}) {
                for (const char* pattern : {"none", "all", "mixed", "duplicate", "holes", "last"}) {
                    Fixture f(tokens, topk, rank);
                    f.Routes(pattern);
                    ReleaseAll();
                    f.Finish();
                    f.CheckPayload();
                }
            }
        }
    }
}
void LateReady()
{
    Fixture f(2U, 8U, 16U);
    f.Routes("all");
    for (uint32_t row : {0U, 1U, 3U, 4U, 6U, 7U, 15U}) {
        f.Publish(row, f.s.sequence);
    }
    for (uint32_t pass = 0U; pass < 8U; ++pass) {
        f.Step(2U, 3U);
    }
    Check(f.progress.sentRows == 7U, "late low-index rows blocked later ready runs");
    const auto posts = f.s.posts.size();
    for (uint32_t pass = 0U; pass < 8U; ++pass) {
        Check(!f.Step(2U, 3U), "old ready row was resubmitted");
    }
    Check(f.s.posts.size() == posts, "late-row polling duplicated a prior Put");
    ReleaseAll();
    f.Finish();
    f.CheckPayload();
}
void Fairness()
{
    Fixture f(32U, 32U);
    f.Routes("all");
    f.Publish(1023U, f.s.sequence);
    for (uint32_t call = 0U; call < 1024U; ++call) {
        f.Step(1U, 1U);
    }
    Check(f.s.sent[1023U] == 1U && f.progress.sentRows == 1U, "late high-index row starved behind unready rows");
    Check(std::all_of(f.s.probes.begin(), f.s.probes.end(), [](uint32_t count) { return count == 1U; }),
          "one-row budget did not make a fair single sweep");
    // Publish behind the cursor after it has already passed this location.
    f.Step(1U, 17U);
    f.Publish(2U, f.s.sequence);
    for (uint32_t call = 0U; call < 65U && f.s.sent[2U] == 0U; ++call) {
        f.Step(1U, 17U);
    }
    Check(f.s.sent[2U] == 1U, "cursor never wrapped to a late publication");
    ReleaseAll();
    f.Finish();
    f.CheckPayload();
}
void Budgets()
{
    for (uint32_t scans : {0U, 1U, 2U, 3U, 47U, 48U, 49U, 63U, 64U, 127U, 128U, 1024U}) {
        for (uint32_t posts : {0U, 1U, 2U, 4U, 8U}) {
            Fixture f(32U, 8U);
            f.Routes("mixed");
            f.Init();
            ReleaseAll();
            for (uint32_t call = 0U; call < 4U; ++call) {
                f.Step(posts, scans);
            }
            if (posts == 0U || scans == 0U) {
                Check(f.s.posts.empty() && f.s.flagReads == 0U, "zero budget performed progress work");
            }
            f.Finish();
            f.CheckPayload();
        }
    }
    Fixture f(32U, 32U);
    f.Routes("all");
    f.Init();
    ReleaseAll();
    f.Step(1U, 63U);
    Check(f.s.posts.size() == 1U && f.s.posts[0].rowCount == 63U && f.progress.nextRow == 63U,
          "contiguous ready run was not split at the row budget");
    f.Finish();
    f.CheckPayload();
    // The default remains an unrestricted sweep for callers not opting in.
    Fixture legacy(32U, 32U);
    legacy.Routes("all");
    ReleaseAll();
    Check(TryProgressHierarchicalCombine(legacy.common, legacy.params, legacy.progress, legacy.s.ub.data()),
          "default progress failed to submit ready rows");
    Check(legacy.progress.sentRows == 1024U && legacy.s.posts.size() == 1U,
          "default progress unexpectedly fragmented the prior full-sweep behavior");
}
void Epochs()
{
    for (uint32_t sequence : {1U, 128U, 129U, 256U, 257U, 0x7FFFFFFEU}) {
        Fixture f(32U, 32U, 31U, sequence);
        f.Routes("last");
        const auto stale = sequence > 128U ? sequence - 128U : sequence + 128U;
        f.Publish(1023U, stale);
        f.Init();
        for (uint32_t call = 0U; call < 20U; ++call) {
            Check(!f.Step(2U, 64U), "old epoch satisfied a current ready flag");
        }
        Check(f.s.flags[1023U * kFlagWords] == static_cast<int32_t>(stale), "old epoch was cleared");
        f.Publish(1023U, sequence);
        const auto before = f.s.flags;
        f.Finish();
        Check(before == f.s.flags, "sender cleared or changed producer epochs");
        f.CheckPayload();
    }
}
void Completed()
{
    for (const char* pattern : {"all", "none", "holes"}) {
        Fixture f(32U, 32U);
        f.Routes(pattern);
        ReleaseAll();
        f.Finish();
        const auto copies = f.s.copies.size();
        const auto reads = f.s.flagReads;
        const auto ubReads = f.s.scalarUbReads;
        const auto posts = f.s.posts.size();
        for (uint32_t call = 0U; call < 100U; ++call) {
            Check(!f.Step(2U, 64U), "completed progress reported more work");
        }
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
        Check(f.s.copies.size() == copies && f.s.flagReads == reads && f.s.scalarUbReads == ubReads &&
                  f.s.posts.size() == posts,
              "completed or zero-contribution progress performed probes/DMA/posts");
    }
}
void Metadata()
{
    for (uint32_t tokens : {1U, 2U, 7U, 15U, 16U, 17U, 31U, 32U}) {
        for (uint32_t topk : {1U, 7U, 8U, 31U, 32U}) {
            for (uint32_t rank : {0U, 16U}) {
                Fixture f(tokens, topk, rank);
                f.Routes("mixed");
                f.Init();
                uint32_t blocks = 0U;
                for (uint32_t remaining = f.s.rows; remaining != 0U;) {
                    const uint32_t valid = std::min(remaining, 127U);
                    blocks += (valid + 63U) / 64U;
                    remaining -= valid;
                }
                Check(f.s.copies.size() == blocks, "metadata DMA copied beyond required packet halves");
                Check(f.s.scalarUbReads == f.s.rows && f.s.metadataScalarReads == 0U,
                      "metadata initializer read marker/padding or used scalar GM reads");
                Check(f.s.overwriteOrdered, "final metadata reads were not ordered before RoCE scratch reuse");
                const auto copies = f.s.copies.size();
                f.Init();
                Check(f.s.copies.size() == copies, "initialized state reloaded metadata");
                ReleaseAll();
                f.Finish();
                f.CheckPayload();
            }
        }
    }
}
void Drain()
{
    Fixture f(32U, 32U, 16U);
    f.Routes("mixed");
    f.Init();
    f.s.releaseAtCycle = 23U;
    DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
    Check(f.s.cycles >= 23U && f.progress.sentRows == f.Expected(), "final drain missed delayed producer rows");
    f.CheckPayload();
    Fixture ready(32U, 32U);
    ready.Routes("all");
    ReleaseAll();
    DrainHierarchicalCombine(ready.common, ready.params, ready.progress, ready.s.ub.data());
    Check(ready.progress.sentRows == 1024U && ready.s.posts.size() == 1U,
          "final drain used steady-state fragmentation instead of unrestricted runs");
    ready.CheckPayload();
}

void DrainStalledProducer()
{
    Fixture f(1U, 8U);
    f.Routes("all");
    f.Init();
    f.s.failAfterWaitChecks = 2U;
    bool failed = false;
    try {
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
    } catch (const std::runtime_error&) {
        failed = true;
    }
    Check(failed && f.s.waitChecks == 2U, "missing producer bypassed bounded wait handling");
    Check(f.s.posts.empty() && f.progress.sentRows == 0U, "missing producer caused an unready row to be sent");
}
void Topology()
{
    Fixture f(32U, 32U);
    f.Routes("all");
    f.tiling.hierarchical = false;
    ReleaseAll();
    Check(!f.Step(2U, 64U), "non-hierarchical topology entered RoCE progress");
    DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
    Check(!f.progress.initialized && f.s.copies.empty() && f.s.posts.empty() && f.s.flagReads == 0U,
          "non-hierarchical topology touched transport state");
}

void Telemetry()
{
    Fixture f(32U, 8U, 16U);
    f.Routes("all");
    ReleaseAll();
    AscendTimerDevice timer;
    Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {4U, 64U}),
          "instrumented progress failed to send ready rows");
    Check(timer.initializations == 1U && timer.records.size() == 1U, "missing initialization/progress observation");
    const auto first = timer.records.front();
    Check(first.type == AscendTimer::AIV_COMBINE_PROGRESS && first.before == 256U && first.after == 192U &&
              first.posts == 1U && first.scanned == 64U,
          "progress counters do not match the bounded production loop");
    Check(timer.scanStarts == 2U && timer.scanStops == 2U && !timer.scanActive,
          "scan time did not exclude one asynchronous Put");
    // An unready sweep must still report its examined slots, with no sends.
    for (uint32_t row = 64U; row < 128U; ++row) {
        f.Publish(row, f.s.sequence - 1U);
    }
    Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {4U, 64U}),
          "unready instrumented sweep submitted a row");
    const auto idle = timer.records.back();
    Check(idle.before == 192U && idle.after == 192U && idle.posts == 0U && idle.scanned == 64U,
          "no-progress scan counters were omitted or changed");
    ReleaseAll();
    DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer);
    const auto drain = timer.records.back();
    Check(drain.type == AscendTimer::AIV_COMBINE_DRAIN && drain.before == 192U && drain.after == 0U &&
              drain.posts == 0U && drain.scanned == 0U,
          "drain did not report its entry/exit pending rows independently of child scans");
    const auto records = timer.records.size();
    Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {4U, 64U}),
          "completed instrumented state reported progress");
    Check(timer.initializations == 1U && timer.records.size() == records && !timer.scanActive,
          "completed fast path added scan events or repeated initialization");
    f.CheckPayload();
}

// This oracle first materializes the bounded circular row sequence, then
// partitions eligible coordinates into contiguous runs. It has no token masks,
// cached cursor, production readiness helper, or mutable scan/send loop.
void OracleStep(Fixture& f, uint32_t maxPosts, uint32_t maxScans)
{
    Check(f.progress.initialized, "oracle requires initialized routing");
    const uint32_t beforeRow = f.progress.nextRow;
    const uint32_t pendingBefore = f.Expected() - f.progress.sentRows;
    const auto beforePosts = f.s.posts.size();
    const auto beforeProbes = f.s.probeOrder.size();
    const auto beforeFlags = f.s.flags;
    std::vector<uint32_t> visits;
    std::vector<bool> pending(f.s.rows), eligible(f.s.rows);
    for (uint32_t row = 0U; row < f.s.rows; ++row) {
        pending[row] = f.s.expected[row] && f.s.sent[row] == 0U;
        eligible[row] = pending[row] && f.s.flags[row * kFlagWords] == static_cast<int32_t>(f.s.sequence);
    }
    if (pendingBefore != 0U && maxPosts != 0U && maxScans != 0U) {
        for (uint32_t i = 0U; i < std::min(f.s.rows, maxScans); ++i) {
            visits.push_back((beforeRow + i) % f.s.rows);
        }
    }
    struct Run {
        uint32_t first;
        uint32_t count;
        uint32_t visitEnd;
    };
    std::vector<Run> runs;
    for (uint32_t i = 0U; i < visits.size(); ++i) {
        const uint32_t row = visits[i];
        if (eligible[row]) {
            if (i != 0U && eligible[visits[i - 1U]] && visits[i - 1U] + 1U == row) {
                ++runs.back().count;
                runs.back().visitEnd = i + 1U;
            } else {
                runs.push_back({row, 1U, i + 1U});
            }
        }
    }
    if (runs.size() > maxPosts) {
        runs.resize(maxPosts);
    }
    uint32_t sent = 0U;
    for (const auto& run : runs) {
        sent += run.count;
    }
    uint32_t scanned = static_cast<uint32_t>(visits.size());
    // Stop at the last accepted run if the POST budget or all outstanding
    // route coordinates are covered. Otherwise every bounded visit is needed.
    if (!runs.empty() && runs.size() == maxPosts) {
        const auto& last = runs.back();
        scanned = last.visitEnd;
        // A false slot that terminates a run is consumed once. The physical
        // tensor end terminates independently, without consuming row zero.
        if (scanned < visits.size() && last.first + last.count < f.s.rows && !eligible[visits[scanned]]) {
            ++scanned;
        }
    }
    std::vector<uint32_t> probes;
    for (uint32_t i = 0U; i < scanned; ++i) {
        if (pending[visits[i]]) {
            probes.push_back(visits[i]);
        }
    }
    AscendTimerDevice timer;
    const bool made =
        TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {maxPosts, maxScans});
    Check(made == !runs.empty() && f.s.posts.size() - beforePosts == runs.size(),
          "POST count differs from contiguous-coordinate oracle");
    for (uint32_t i = 0U; i < runs.size(); ++i) {
        const auto& actual = f.s.posts[beforePosts + i];
        Check(actual.firstRow == runs[i].first && actual.rowCount == runs[i].count,
              "POST order/coalescing differs from contiguous-coordinate oracle");
    }
    // DMA may prefetch holes, sent slots, and up to seven future slots. Every
    // logical pending probe must be covered, and every physical load must stay
    // in the bounded visit set without loading any row twice in one call.
    std::vector<uint32_t> loaded(f.s.probeOrder.begin() + beforeProbes, f.s.probeOrder.end());
    for (uint32_t row : probes) {
        Check(std::count(loaded.begin(), loaded.end(), row) == 1, "snapshot missed a logical pending probe");
    }
    for (uint32_t row : loaded) {
        Check(std::count(loaded.begin(), loaded.end(), row) == 1 &&
                  std::find(visits.begin(), visits.end(), row) != visits.end(),
              "snapshot exceeded scan bounds");
    }
    const uint32_t nextRow = scanned == 0U ? beforeRow : (beforeRow + scanned) % f.s.rows;
    Check(f.progress.nextRow == nextRow, "resumed cursor differs from bounded circular row sequence");
    Check(f.progress.sentRows == f.Expected() - pendingBefore + sent, "oracle sentRows count mismatch");
    Check(f.s.flags == beforeFlags, "progress changed producer epochs");
    for (uint32_t row = 0U; row < f.s.rows; ++row) {
        const bool remains =
            (f.progress.pendingTopkMasks[row / f.common.topK] & (uint32_t(1) << (row % f.common.topK))) != 0U;
        Check(remains == (f.s.expected[row] && f.s.sent[row] == 0U), "pending mask differs from unsent route set");
    }
    if (visits.empty()) {
        Check(timer.records.empty(), "zero-budget/completed oracle path emitted a progress record");
    } else {
        Check(timer.records.size() == 1U && timer.records[0].scanned == scanned &&
                  timer.records[0].posts == runs.size() && timer.records[0].before == pendingBefore &&
                  timer.records[0].after == pendingBefore - sent,
              "telemetry differs from exact oracle slot/send counts");
    }
}

void CompleteInCall()
{
    struct Case {
        uint32_t tokens;
        uint32_t topk;
        uint32_t first;
        std::vector<uint32_t> rows;
        uint32_t posts;
        uint32_t scanned;
        uint32_t next;
    };
    const std::vector<Case> cases = {
        {32U, 8U, 0U, {0U}, 1U, 64U, 64U},
        {32U, 8U, 252U, {0U, 1U}, 1U, 64U, 60U},
        {32U, 8U, 250U, {255U}, 1U, 64U, 58U},
        {32U, 8U, 6U, {6U, 7U, 8U}, 1U, 64U, 70U},
        {32U, 8U, 254U, {255U, 0U, 1U}, 2U, 5U, 3U},
        {32U, 31U, 30U, {30U, 31U}, 1U, 64U, 94U},
        {4U, 32U, 31U, {31U, 32U, 33U}, 1U, 64U, 95U},
        {1U, 1U, 0U, {0U}, 1U, 1U, 0U},
    };
    for (const auto& c : cases) {
        Fixture f(c.tokens, c.topk, 16U);
        f.RouteRows(c.rows);
        ReleaseAll();
        AscendTimerDevice timer;
        // Even immediately ready sparse routes must not initialize or submit
        // anything when either budget is zero.
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {0U, 64U}) &&
                  !TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {2U, 0U}),
              "zero budget submitted sparse routes");
        Check(!f.progress.initialized && f.s.copies.empty() && f.s.posts.empty() && f.s.flagReads == 0U &&
                  timer.initializations == 0U && timer.records.empty() && timer.scanStarts == 0U,
              "zero budget touched progress state");
        f.progress.nextRow = c.first;
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {2U, 64U}),
              "sparse completion failed to submit ready routes");
        Check(f.progress.sentRows == c.rows.size() && f.progress.sentRows == f.progress.expectedRows &&
                  f.progress.nextRow == c.next && f.s.posts.size() == c.posts,
              "sparse completion lost routes, cursor position, or probe order");
        Check(timer.records.size() == 1U && timer.records[0].before == c.rows.size() && timer.records[0].after == 0U &&
                  timer.records[0].posts == c.posts && timer.records[0].scanned == c.scanned &&
                  timer.scanStarts == c.posts + 1U && timer.scanStops == timer.scanStarts && !timer.scanActive,
              "completion did not preserve the baseline scan budget and cursor");
        const auto reads = f.s.flagReads;
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {2U, 64U}) &&
                  timer.records.size() == 1U && f.s.flagReads == reads,
              "completed sparse routes were scanned again");
        f.CheckPayload();
    }

    // An unfinished call still scans all 64 slots when its sole ready run is
    // early, and still stops at two Posts when more runs are ready.
    Fixture unfinished(32U, 8U);
    unfinished.RouteRows({0U, 200U});
    unfinished.Init();
    ReleaseAll();
    OracleStep(unfinished, 2U, 64U);
    Check(unfinished.progress.sentRows == 1U && unfinished.progress.nextRow == 64U && unfinished.s.posts.size() == 1U,
          "unfinished progress stopped before its row budget");
    for (uint32_t call = 0U; call < 3U; ++call) {
        OracleStep(unfinished, 2U, 64U);
    }
    unfinished.CheckPayload();

    Fixture capped(32U, 8U);
    capped.RouteRows({0U, 2U, 4U});
    capped.Init();
    ReleaseAll();
    OracleStep(capped, 2U, 64U);
    Check(capped.progress.sentRows == 2U && capped.progress.nextRow == 4U && capped.s.posts.size() == 2U,
          "completion exceeded the two-Post budget");
    OracleStep(capped, 2U, 64U);
    Check(capped.progress.sentRows == 3U && capped.progress.nextRow == 68U && capped.s.posts.size() == 3U,
          "resumed final run did not preserve the baseline scan budget");
    capped.CheckPayload();
}

void ReadySnapshot()
{
    Fixture dense(32U, 8U);
    dense.Routes("all");
    dense.Init();
    ReleaseAll();
    dense.Step(2U, 64U);
    Check(dense.s.flagDmas == 8U && dense.s.flagReads == 64U && dense.progress.sentRows == 64U,
          "dense scan did not batch eight flags per DMA");
    dense.Finish();
    dense.CheckPayload();

    Fixture late(1U, 8U);
    late.Routes("all");
    late.Init();
    late.Publish(0U, late.s.sequence);
    late.s.releaseAfterSnapshot = 3U;
    Check(TryProgressHierarchicalCombine(late.common, late.params, late.progress, late.s.ub.data(), nullptr, {2U, 8U}),
          "initial ready row was not submitted");
    Check(late.s.sent[0U] == 1U && late.s.sent[3U] == 0U, "snapshot read a post-DMA publication");
    late.Step(2U, 8U);
    Check(late.s.sent[3U] == 1U, "negative snapshot survived across progress calls");
    ReleaseAll();
    late.Finish();
    late.CheckPayload();

    Fixture reuse(1U, 8U);
    reuse.RouteRows({0U, 2U, 4U, 6U});
    reuse.Init();
    ReleaseAll();
    reuse.Step(4U, 8U);
    Check(reuse.s.posts.size() == 4U && reuse.s.flagDmas == 1U, "Put scratch overwrite destroyed cached ready bits");
    reuse.CheckPayload();
}

void CursorOracle()
{
    std::mt19937 random(20261020U);
    for (uint32_t topk : {1U, 3U, 8U, 31U, 32U}) {
        for (uint32_t tokens : {1U, 2U, 3U, 7U, 32U}) {
            for (const char* pattern : {"all", "mixed", "duplicate", "holes", "last"}) {
                Fixture f(tokens, topk, 16U);
                f.Routes(pattern);
                f.Init();
                f.progress.nextRow = f.s.rows - 1U;
                for (uint32_t call = 0U; call < 40U; ++call) {
                    // Publications arrive in arbitrary order; already-sent flags
                    // remain ready so accidental rereads/resubmissions fail.
                    for (uint32_t row = 0U; row < f.s.rows; ++row) {
                        if (random() % 17U == 0U) {
                            f.Publish(row, f.s.sequence);
                        }
                    }
                    const std::array<uint32_t, 11> scans = {0U,  1U,  2U,  topk, topk + 1U,  47U,
                                                            48U, 49U, 63U, 64U,  0xffffffffU};
                    OracleStep(f, random() % 6U, scans[random() % scans.size()]);
                }
                ReleaseAll();
                for (uint32_t call = 0U; f.progress.sentRows != f.Expected(); ++call) {
                    Check(call <= f.s.rows + 1U, "oracle completion failed to visit all pending rows");
                    OracleStep(f, 2U, 7U);
                }
                OracleStep(f, 4U, 64U);
                f.CheckPayload();
            }
        }
        // Start at the last slot of a token and cut a contiguous ready run
        // inside a later token. This also exercises a full32-bit token clear.
        Fixture crossing(4U, topk);
        crossing.Routes("all");
        crossing.Init();
        crossing.progress.nextRow = topk - 1U;
        ReleaseAll();
        OracleStep(crossing, 1U, topk + 2U);
        Check(crossing.s.posts.size() == 1U && crossing.s.posts[0].rowCount == topk + 2U,
              "cross-token run was fragmented before its scan budget");
        OracleStep(crossing, 1U, 0xffffffffU);
        OracleStep(crossing, 0xffffffffU, 0xffffffffU);
        crossing.CheckPayload();
    }
}
void LargeBatch()
{
    // DSV41's 144*6 routes and the maximum supported top-k exercise private
    // masks beyond token 31, metadata packet boundaries and the final row.
    for (uint32_t topk : {6U, 32U}) {
        Fixture f(144U, topk, 16U);
        f.Routes("all");
        f.Init();
        f.progress.nextRow = 31U * topk + topk - 1U;
        for (uint32_t row = 0; row + 1U < f.s.rows; ++row) f.Publish(row, f.s.sequence);
        OracleStep(f, 1U, 64U);
        while (f.progress.sentRows + 1U < f.Expected()) OracleStep(f, 2U, 64U);
        Check(f.s.sent.back() == 0U, "last row sent before its producer published");
        f.Publish(f.s.rows - 1U, f.s.sequence);
        f.Finish(2U, 64U);
        f.CheckPayload();
    }
}

void Coalescing()
{
    {
        // Reproduce the observed 17..20 and 22..31 fragments inside one
        // natural 16..31 range. Both exceed the old short-run threshold.
        Fixture f(4U, 8U);
        std::vector<uint32_t> rows;
        for (uint32_t row = 16U; row < 32U; ++row) rows.push_back(row);
        f.RouteRows(rows);
        for (uint32_t row : rows) {
            if (row != 16U && row != 21U) f.Publish(row, f.s.sequence);
        }
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                              {8U, 32U, true, true}),
              "medium fragments were posted before their bounded coalescing opportunity");
        const auto reads = f.s.flagReads;
        ReleaseAll();
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                              {8U, 32U, true, true}) &&
                  f.s.flagReads == reads,
              "empty sweep immediately repeated its flag reads");
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                             {8U, 32U, true, true}),
              "newly complete range did not progress after one skipped visit");
        Check(f.s.posts.size() == 1U && f.s.posts[0].firstRow == 16U && f.s.posts[0].rowCount == 16U,
              "medium fragments did not reunite into their natural range");
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        for (uint32_t row = 2U; row < 14U; ++row) f.Publish(row, f.s.sequence);
        for (uint32_t call = 0U; call < 2U; ++call) {
            Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                                  {2U, 16U, true, true}),
                  "partial region ignored its bounded deferral or retry spacing");
        }
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                             {2U, 16U, true, true}) &&
                  f.progress.sentRows == 12U,
              "unfinished neighbours indefinitely held an already-deferred medium run");
        ReleaseAll();
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.RouteRows({0U, 1U, 2U, 3U, 8U, 9U, 10U, 11U});
        ReleaseAll();
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                             {8U, 16U, true, true}),
              "complete regions were deferred across a route gap");
        Check(f.s.posts.size() == 2U && f.s.posts[0].rowCount == 4U && f.s.posts[1].rowCount == 4U,
              "coalescing crossed a non-routed gap");
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        f.Publish(0U, f.s.sequence);
        AscendTimerDevice timer;
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {2U, 16U, true}),
              "short growable run was submitted before its bounded deferral");
        Check(f.s.posts.empty() && f.progress.sentRows == 0U && timer.runs.size() == 1U &&
                  (timer.runs[0].second & 0x100U) != 0U && (timer.runs[0].second & 0xffU) == 2U,
              "deferral changed payload or lost its producer-not-ready reason");
        ReleaseAll();
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer, {8U, 16U, true}),
              "coalesced run did not progress");
        Check(f.s.posts.size() == 1U && f.s.posts[0].rowCount == 16U, "late neighbouring rows were not coalesced");
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        f.Publish(0U, f.s.sequence);
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr, {2U, 16U, true}),
              "first short-run visit should defer");
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr, {2U, 16U, true}),
              "an unfinished neighbour starved an already-deferred row");
        Check(f.s.sent[0] == 1U && f.progress.sentRows == 1U, "bounded retry submitted an unready row");
        ReleaseAll();
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
        f.CheckPayload();
    }
    for (uint32_t topk : {1U, 8U, 32U}) {
        Fixture f(32U, topk);
        f.Routes("all");
        ReleaseAll();
        // Even a one-row scan budget must terminate: each start may be held
        // only once, with no wait on a producer inside the progress call.
        for (uint32_t calls = 0; f.progress.sentRows != f.Expected(); ++calls) {
            Check(calls <= 2U * f.s.rows, "coalescing exceeded its per-start retry bound");
            TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr, {1U, 1U, true});
        }
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        f.Publish(0U, f.s.sequence);
        TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr, {1U, 16U, true});
        ReleaseAll();
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
        Check(f.s.posts.size() == 1U && f.s.posts[0].rowCount == 16U,
              "final drain inherited foreground coalescing or fragmented a ready run");
        f.CheckPayload();
    }
}
void RetrySpacing()
{
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                              {2U, 16U, true, true}),
              "empty sweep submitted unready rows");
        const auto reads = f.s.flagReads;
        const auto next = f.progress.nextRow;
        AscendTimerDevice timer;
        ReleaseAll();
        Check(!TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), &timer,
                                              {2U, 16U, true, true}) &&
                  f.s.flagReads == reads && f.progress.nextRow == next && timer.records.empty(),
              "skipped retry performed flag DMA or recorded a nonexistent scan");
        Check(TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                             {2U, 16U, true, true}),
              "retry spacing suppressed the next eligible visit");
        f.CheckPayload();
    }
    {
        // A partial empty sweep must continue to late rows at the next visit.
        Fixture f(4U, 8U);
        f.RouteRows({31U});
        f.Publish(31U, f.s.sequence);
        for (uint32_t call = 0U; call < 4U; ++call) {
            const bool made = TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr,
                                                             {2U, 8U, true, true});
            Check(made == (call == 3U), "retry spacing stalled a partial sweep before a late ready row");
        }
        f.CheckPayload();
    }
    {
        Fixture f(2U, 8U);
        f.Routes("all");
        f.Publish(3U, f.s.sequence);
        TryProgressHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data(), nullptr, {2U, 16U, true, true});
        Check(f.progress.retryDeferred, "fixture did not arm foreground retry spacing");
        // All producers have finished: drain must ignore the armed retry and
        // pending coalescing, and issue the complete range in one Put.
        ReleaseAll();
        DrainHierarchicalCombine(f.common, f.params, f.progress, f.s.ub.data());
        Check(f.s.posts.size() == 1U && f.s.posts[0].rowCount == 16U,
              "final drain inherited foreground deferral or retry spacing");
        f.CheckPayload();
    }
}
}  // namespace ProgressTest

int main(int argc, char** argv)
{
    using namespace ProgressTest;
    Check(argc == 2, "expected one test scenario");
    const char* scenario = argv[1];
    if (std::strcmp(scenario, "routing") == 0)
        Routing();
    else if (std::strcmp(scenario, "late-ready") == 0)
        LateReady();
    else if (std::strcmp(scenario, "fairness") == 0)
        Fairness();
    else if (std::strcmp(scenario, "budgets") == 0)
        Budgets();
    else if (std::strcmp(scenario, "epochs") == 0)
        Epochs();
    else if (std::strcmp(scenario, "completed") == 0)
        Completed();
    else if (std::strcmp(scenario, "complete-in-call") == 0)
        CompleteInCall();
    else if (std::strcmp(scenario, "metadata") == 0)
        Metadata();
    else if (std::strcmp(scenario, "drain") == 0)
        Drain();
    else if (std::strcmp(scenario, "drain-stalled") == 0)
        DrainStalledProducer();
    else if (std::strcmp(scenario, "topology") == 0)
        Topology();
    else if (std::strcmp(scenario, "telemetry") == 0)
        Telemetry();
    else if (std::strcmp(scenario, "cursor-oracle") == 0)
        CursorOracle();
    else if (std::strcmp(scenario, "large-batch") == 0)
        LargeBatch();
    else if (std::strcmp(scenario, "ready-snapshot") == 0)
        ReadySnapshot();
    else if (std::strcmp(scenario, "coalescing") == 0)
        Coalescing();
    else if (std::strcmp(scenario, "retry-spacing") == 0)
        RetrySpacing();
    else
        Check(false, "unknown test scenario");
    return 0;
}
