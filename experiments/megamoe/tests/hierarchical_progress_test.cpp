// Copyright (c) 2026, Lu Lu

// Compile the production progress/initialization/drain code, replacing only
// device primitives and heavyweight type dependencies. The independent oracle
// is a set of original route coordinates, not another progress-loop algorithm.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
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
class AscendTimerDevice {};
namespace AscendTimer {
enum class RdmaPhase { Combine };
}

namespace ProgressTest {
inline void Check(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
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
    std::vector<Post> posts;
    std::vector<Copy> copies;
    const int32_t* pairedRail = nullptr;
    uint32_t railSlotWords = 0U;
    uint32_t rowBytes = 0U;
    uint32_t rows = 0U;
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
    T* Data() const { return data_; }

private:
    T* data_ = nullptr;
    uint32_t size_ = 0U;
};

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T* address) { data_ = address; }
    GlobalTensor operator[](uint32_t offset) const
    {
        GlobalTensor result;
        result.data_ = data_ + offset;
        return result;
    }
    T* Data() const { return data_; }

private:
    T* data_ = nullptr;
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
constexpr uint32_t HIERARCHICAL_MAX_TOKENS = 32U;
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
inline uint32_t HierarchicalHost(uint32_t rank) { return rank / HIERARCHICAL_RANKS_PER_HOST; }
inline uint32_t HierarchicalGateway(uint32_t rank) { return rank ^ HIERARCHICAL_RANKS_PER_HOST; }
inline bool UseHierarchicalRoce(const MegaMoeTilingData* tiling) { return tiling->hierarchical; }
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
        std::memcpy(s.ub.data(), s.pendingDma, s.pendingDmaWords * sizeof(int32_t));
        s.pendingDma = nullptr;
        s.pendingDmaWords = 0U;
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
        throw std::runtime_error("progress made a scalar GM load outside the ready flags");
    }
    const auto offset = raw - begin;
    Check(offset % HIERARCHICAL_FLAG_BYTES == 0U, "flag probe did not target a row epoch");
    const auto row = static_cast<uint32_t>(offset / HIERARCHICAL_FLAG_BYTES);
    Check(row < s.rows && s.expected[row] && s.sent[row] == 0U, "probed an unused or already-submitted row");
    Check(++s.flagReads < 1000000U, "flag polling did not terminate");
    ++s.probes[row];
    return *address;
}
inline void RocePutBytes(uint8_t* destination, uint8_t* source, uint32_t bytes, uint32_t peer, uint8_t* scratch,
                         AscendTimerDevice*, AscendTimer::RdmaPhase phase)
{
    using namespace ProgressTest;
    auto& s = *state;
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
        common.topK = tiling.topK = topk;
        common.rankId = rank;
        tiling.hierarchicalSequence = s.sequence = sequence;
        s.rows = tokens * topk;
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
    void Routes(const std::string& pattern)
    {
        const uint32_t expertsPerHost = 16U * common.moeExpertPerRank;
        const uint32_t localBase = HierarchicalHost(common.rankId) * expertsPerHost;
        const uint32_t otherBase = (1U - HierarchicalHost(common.rankId)) * expertsPerHost;
        // Poison marker/unused slots with a valid local expert: consuming either
        // as a route creates an observable extra expected row.
        std::fill(s.rail.begin(), s.rail.end(), static_cast<int32_t>(localBase + 1U));
        auto* paired = const_cast<int32_t*>(s.pairedRail);
        for (uint32_t row = 0U; row < s.rows; ++row) {
            bool remoteContribution = pattern == "all" || pattern == "duplicate";
            if (pattern == "mixed") {
                remoteContribution = row % 5U == 0U || row % 7U == 2U;
            }
            if (pattern == "last") {
                remoteContribution = row + 1U == s.rows;
            }
            const uint32_t expert = remoteContribution ? localBase + (pattern == "duplicate" ? 7U : row % 511U)
                                                       : otherBase + row % 511U;
            ids[row] = pattern == "holes" || (pattern == "mixed" && row % 11U == 3U)
                           ? -1
                           : static_cast<int32_t>(expert);
            s.expected[row] = ids[row] >= 0 && static_cast<uint32_t>(ids[row]) / expertsPerHost ==
                                                       HierarchicalHost(common.rankId);
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
        const bool made = TryProgressHierarchicalCombine(common, params, progress, s.ub.data(), nullptr, {posts, scans});
        const auto afterPosts = s.posts.size() - beforePosts;
        Check(made == (afterPosts != 0U), "progress result did not describe actual submission");
        Check(afterPosts <= posts && s.flagReads - beforeReads <= scans, "progress exceeded its operation budget");
        Check(beforeFlags == s.flags, "progress modified persistent producer epochs");
        if (s.rows != 0U && scans < s.rows && posts != 0U && scans != 0U) {
            const uint32_t advanced = (progress.nextRow + s.rows - beforeRow) % s.rows;
            Check(advanced <= scans, "cursor exceeded the row scan budget");
        }
        Check(progress.sentRows == std::count(s.sent.begin(), s.sent.end(), 1U), "sentRows accounting mismatch");
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
        for (uint32_t topk : {1U, 8U, 32U}) {
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
    for (uint32_t scans : {0U, 1U, 2U, 3U, 63U, 64U, 127U, 128U, 1024U}) {
        for (uint32_t posts : {0U, 1U, 2U, 8U}) {
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
}  // namespace ProgressTest

int main(int argc, char** argv)
{
    using namespace ProgressTest;
    try {
        Check(argc == 2, "expected one test scenario");
        const std::string scenario = argv[1];
        if (scenario == "routing") Routing();
        else if (scenario == "late-ready") LateReady();
        else if (scenario == "fairness") Fairness();
        else if (scenario == "budgets") Budgets();
        else if (scenario == "epochs") Epochs();
        else if (scenario == "completed") Completed();
        else if (scenario == "metadata") Metadata();
        else if (scenario == "drain") Drain();
        else if (scenario == "topology") Topology();
        else throw std::runtime_error("unknown test scenario");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
