// Copyright (c) 2026, Lu Lu

// The mapper, readiness probe, scheduler and range/count helpers below are
// extracted from production. Only device IO and the other stages are doubled.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

constexpr int AIC = 0, AIV = 1, g_coreType = AIV;
uint32_t GetSubBlockIdx()
{
    return 1U;
}
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 4;
/* TILE_CONSTANTS */
namespace Ops::Base
{
uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}
} // namespace Ops::Base
namespace AscendTimer
{
constexpr int64_t NO_WAVE = -1;
}
namespace AscendC
{
uint32_t reads = 0U;
int32_t ReadGmByPassDCache(const int32_t* flag)
{
    ++reads;
    return *flag;
}
int64_t GetSystemCycle()
{
    static int64_t cycle = 0;
    return ++cycle;
}
} // namespace AscendC
struct AivJobContext
{
    uint32_t jobIndex = 0U, totalJobs = 28U;
};
struct WorkRange
{
    uint32_t start = 0U, count = 0U;
};
struct MoeStageCommonConfig
{
    uint32_t tokenHiddenDim = 3584U, moeExpertPerRank = 0U;
};
struct TilingData
{
    bool hierarchical = true;
};
bool UseHierarchicalRoce(const TilingData* tiling)
{
    return tiling->hierarchical;
}
uint32_t HierarchicalSendJob()
{
    return 1U;
}
struct Params
{
    TilingData* tilingData;
    std::array<int32_t, 128> ready{};
    std::vector<uint32_t>* writes = nullptr;
    uint32_t sendCalls = 0U;
    uint32_t expectedProducers = 0U;
    uint64_t expectedPrefix = 0U;
    AivJobContext lastConsumer{};
};
int32_t* GetTokenCombineReadyCounter(Params& params, uint32_t expert, uint32_t rowStart)
{
    auto slot = expert * 8U + rowStart / L1_TILE_M_256;
    assert(slot < params.ready.size());
    return &params.ready[slot];
}
struct ExpertLoopState
{
    uint32_t expertIdx = 0U;
    uint64_t globalTokenStartIndex = 0U;
};
ExpertLoopState CreateExpertLoopState(const MoeStageCommonConfig&)
{
    return {};
}
struct Timer
{
    int64_t wave = 999;
    uint32_t clears = 0U;
    void SetWave(int64_t value)
    {
        wave = value;
    }
    void ClearSlice()
    {
        ++clears;
    }
};
uint32_t GetMGroupCountForRows(uint64_t rowCount, uint32_t rowsPerMGroup)
{ /* M_GROUP_COUNT */
}
WorkRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount)
{ /* BALANCED_RANGE */
}
WorkRange GetRotatedBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount,
                                       uint64_t globalRowPrefix)
{ /* ROTATED_RANGE */
}
template <bool IncludeAiv0 = false>
WorkRange GetWaveCombineOwnedRange(const AivJobContext& job, uint32_t tokenCount, uint64_t expertRowPrefix)
{ /* OWNED_RANGE */
}
uint32_t GetTokenCombineProducerCount(const MoeStageCommonConfig& common, const AivJobContext& job, uint32_t rowCount)
{ /* PRODUCER_COUNT */
}
template <bool InputReadyChecked>
void RunTokenCombineSlice(const MoeStageCommonConfig&, const AivJobContext& job, Params& params,
                          const ExpertLoopState& state, uint32_t rowStart, uint32_t rowCount, Timer&, int&)
{
    static_assert(InputReadyChecked);
    assert(*GetTokenCombineReadyCounter(params, state.expertIdx, rowStart) >=
           static_cast<int32_t>(params.expectedProducers));
    assert(state.globalTokenStartIndex + rowStart == params.expectedPrefix);
    WorkRange range = GetWaveCombineOwnedRange(job, rowCount, state.globalTokenStartIndex + rowStart);
    assert(range.count != 0U && range.start + range.count <= rowCount);
    assert(params.writes && params.writes->size() == rowCount);
    for (uint32_t row = range.start; row < range.start + range.count; ++row)
    {
        ++params.writes->at(row);
    }
    ++params.sendCalls;
    params.lastConsumer = job;
}

// Keep this member in a dependent base, as in the production kernel. Declaring
// it directly in the fixture would hide missing this-> qualification errors.
template <typename Tag> struct FixtureBase
{
    AivJobContext aivJob_{1U, 56U};
};
template <typename Tag> struct FixtureImpl : FixtureBase<Tag>
{
    struct ExpertTokenPosition
    {
        uint32_t expertIdx = 0U;
        uint32_t tokenIndexInExpert = 0U;
        uint64_t globalTokenIndex = 0U;
    };
    struct AivTaskCursor
    {
        struct
        {
            ExpertTokenPosition begin{}, end{};
        } slice;
        uint64_t wave = 0U;
        uint32_t nextTile = 0U, tileCount = 1U, nextSlice = 0U;
    };
    struct SharedActivationCursor
    {
        bool done = false;
    };
    TilingData tiling{};
    Params params_{&tiling};
    AivJobContext waveCombineJob_{};
    struct
    {
        AivJobContext blockJob{};
    } gmmExecutionConfig_;
    MoeStageCommonConfig commonConfig_{};
    Timer timer_{};
    int waveCombineScratch_ = 0;
    uint32_t sharedExpertNum_ = 0U, mGroupsPerWave_ = 1U;
    std::vector<AivTaskCursor> slices;
    bool activationReady = false, activationDone = false;
    uint32_t currentSlice = 0U, retired = 0U, remoteCalls = 0U;

    explicit FixtureImpl(uint32_t block = 0U, uint32_t blocks = 28U, bool hierarchical = true)
    {
        tiling.hierarchical = hierarchical;
        waveCombineJob_ = {block, blocks};
        gmmExecutionConfig_.blockJob = waveCombineJob_;
        this->aivJob_ = {2U * block + 1U, 2U * blocks};
    }
    static AivTaskCursor Slice(uint32_t rows, uint64_t prefix, uint64_t wave, uint32_t expert = 0U,
                               uint32_t rowStart = 0U)
    {
        AivTaskCursor cursor;
        cursor.slice.begin = {expert, rowStart, prefix};
        cursor.slice.end = {expert, rowStart + rows, prefix + rows};
        cursor.wave = wave;
        return cursor;
    }
    uint32_t Producers(const AivTaskCursor& cursor) const
    {
        return GetTokenCombineProducerCount(commonConfig_, waveCombineJob_,
                                            cursor.slice.end.globalTokenIndex - cursor.slice.begin.globalTokenIndex);
    }
    int32_t& Ready(const AivTaskCursor& cursor)
    {
        return *GetTokenCombineReadyCounter(params_, cursor.slice.begin.expertIdx,
                                            cursor.slice.begin.tokenIndexInExpert);
    }
    bool TryRunReadyCombine(const AivTaskCursor& cursor)
    { /* READY_COMBINE */
    }
    void ProcessAiv1PriorityStages(int32_t& readySequence)
    { /* SCHEDULER */
    }

    bool NextAivSlice(AivTaskCursor& cursor, bool activation)
    {
        if (activation)
        {
            if (cursor.nextSlice++ != 0U)
            {
                assert(activationDone);
                return false;
            }
            return true;
        }
        uint32_t next = cursor.nextSlice;
        if (next != 0U)
        {
            // The owner cannot retire an unfinished slice just because it moves no rows.
            assert(Ready(slices[next - 1U]) >= static_cast<int32_t>(Producers(slices[next - 1U])));
            ++retired;
        }
        currentSlice = next;
        if (next == slices.size())
        {
            return false;
        }
        cursor = slices[next];
        cursor.nextSlice = next + 1U;
        return true;
    }
    bool TryRunReadyActivation(const AivTaskCursor&, int32_t& readySequence)
    {
        if (!activationReady)
        {
            return false;
        }
        assert(!activationDone);
        activationDone = true;
        ++readySequence;
        return true;
    }
    void DispatchNextExpertSlice(ExpertTokenPosition&, uint32_t&, uint64_t)
    {
        assert(false);
    }
    bool TryRunSharedExpertActivation(SharedActivationCursor&)
    {
        assert(false);
        return false;
    }
    bool TryProgressRemoteCombine()
    {
        assert(this->aivJob_.jobIndex == HierarchicalSendJob() && tiling.hierarchical);
        assert(++remoteCalls < 100U);
        if (remoteCalls == 1U)
        {
            activationReady = true;
        }
        else
        {
            assert(activationDone);
            if (currentSlice < slices.size())
            {
                ++Ready(slices[currentSlice]);
            }
        }
        // Exercise both the successful progress and idle/backoff paths.
        return remoteCalls % 2U != 0U;
    }
};

using Fixture = FixtureImpl<void>;

void Coverage()
{
    for (bool hierarchical : {false, true})
    {
        for (uint32_t blocks : {1U, 2U, 4U, 28U})
        {
            for (uint32_t rows : {1U, 7U, 21U, 22U, 27U, 28U, 29U, 32U, 255U, 256U, 257U, 1024U})
            {
                for (uint64_t prefix : {0ULL, 1ULL, 27ULL, 257ULL, (1ULL << 40U) + 17ULL})
                {
                    for (uint64_t wave : {0ULL, 1ULL, 27ULL, 28ULL, (1ULL << 35U) + 3ULL})
                    {
                        std::vector<uint32_t> writes(rows, 0U), loads;
                        for (uint32_t block = 0U; block < blocks; ++block)
                        {
                            Fixture f(block, blocks, hierarchical);
                            // H8192 requires all 28 producers, exposing accidental use of 27 consumers.
                            f.commonConfig_.tokenHiddenDim = 8192U;
                            auto cursor = Fixture::Slice(rows, prefix + 256U, wave, 3U, 256U);
                            f.Ready(cursor) = f.Producers(cursor);
                            f.params_.writes = &writes;
                            f.params_.expectedPrefix = prefix + 256U;
                            f.params_.expectedProducers = f.Producers(cursor);
                            uint32_t before = 0U;
                            for (auto count : writes)
                            {
                                before += count;
                            }
                            assert(f.TryRunReadyCombine(cursor));
                            uint32_t after = 0U;
                            for (auto count : writes)
                            {
                                after += count;
                            }
                            if (hierarchical && blocks > 1U && block == 0U)
                            {
                                assert(f.params_.sendCalls == 0U && after == before);
                            }
                            else
                            {
                                loads.push_back(after - before);
                                uint32_t consumers = blocks - static_cast<uint32_t>(hierarchical && blocks > 1U);
                                uint32_t consumer = block - static_cast<uint32_t>(hierarchical && blocks > 1U);
                                uint32_t first =
                                    (cursor.slice.begin.globalTokenIndex % consumers + wave % consumers) % consumers;
                                uint32_t logical = (consumer + consumers - first) % consumers;
                                assert(after - before ==
                                       rows / consumers + static_cast<uint32_t>(logical < rows % consumers));
                            }
                            assert(f.waveCombineJob_.jobIndex == block && f.waveCombineJob_.totalJobs == blocks);
                            assert(f.gmmExecutionConfig_.blockJob.jobIndex == block);
                            assert(f.aivJob_.jobIndex == 2U * block + 1U);
                        }
                        for (auto count : writes)
                        {
                            assert(count == 1U);
                        }
                        auto limits = std::minmax_element(loads.begin(), loads.end());
                        assert(*limits.second - *limits.first <= 1U);
                    }
                }
            }
        }
    }
}

void Readiness()
{
    for (uint32_t hidden : {1024U, 3584U, 8192U})
    {
        for (uint32_t block : {0U, 1U, 27U})
        {
            Fixture f(block);
            f.commonConfig_.tokenHiddenDim = hidden;
            auto cursor = Fixture::Slice(64U, 256U, 28U, 3U, 256U);
            std::vector<uint32_t> writes(64U, 0U);
            f.params_.writes = &writes;
            f.params_.expectedPrefix = 256U;
            f.params_.expectedProducers = f.Producers(cursor);
            f.params_.ready.fill(1000); // Other slices must not release this slice.
            f.Ready(cursor) = f.Producers(cursor) - 1U;
            AscendC::reads = 0U;
            assert(!f.TryRunReadyCombine(cursor));
            assert(AscendC::reads == 1U && f.params_.sendCalls == 0U);
            assert(f.timer_.wave == 999);
            ++f.Ready(cursor);
            assert(f.TryRunReadyCombine(cursor));
            assert(f.timer_.wave == 28);
            assert(f.params_.sendCalls == (block == 0U ? 0U : 1U));
        }
    }
    // Empty non-owner consumers still skip without polling unrelated producers.
    Fixture empty(27U);
    AscendC::reads = 0U;
    assert(empty.TryRunReadyCombine(Fixture::Slice(1U, 0U, 0U)));
    assert(AscendC::reads == 0U && empty.params_.sendCalls == 0U);
}

void Scheduler()
{
    Fixture f;
    f.commonConfig_.tokenHiddenDim = 8192U;
    for (uint32_t expert = 0U; expert < 12U; ++expert)
    {
        f.slices.push_back(Fixture::Slice(21U, 21U * expert, expert / 4U, expert));
        f.Ready(f.slices.back()) = f.Producers(f.slices.back()) - 2U;
    }
    int32_t readySequence = 40;
    f.ProcessAiv1PriorityStages(readySequence);
    assert(readySequence == 41 && f.activationDone);
    assert(f.retired == f.slices.size() && f.currentSlice == f.slices.size());
    assert(f.remoteCalls > f.slices.size());
    assert(f.params_.sendCalls == 0U);
    assert(f.timer_.clears == 1U && f.timer_.wave == AscendTimer::NO_WAVE);
    assert(f.waveCombineJob_.totalJobs == 28U && f.gmmExecutionConfig_.blockJob.totalJobs == 28U);
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    const char* scenario = argv[1];
    if (std::strcmp(scenario, "coverage") == 0)
    {
        Coverage();
    }
    else if (std::strcmp(scenario, "readiness") == 0)
    {
        Readiness();
    }
    else
    {
        assert(std::strcmp(scenario, "scheduler") == 0);
        Scheduler();
    }
}
