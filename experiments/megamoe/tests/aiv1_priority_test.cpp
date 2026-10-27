// Copyright (c) 2026, Lu Lu

// CPU stage doubles around the extracted production scheduler. Readiness changes
// during local Combine, remote progress, or polling; no scheduler is reimplemented.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 4;
namespace AscendTimer {
constexpr int64_t NO_WAVE = -1;
}
namespace AscendC {
std::function<int64_t()> cycle;
int64_t GetSystemCycle()
{
    return cycle();
}
}  // namespace AscendC

struct TilingData {
    bool hierarchicalRoce = true;
};
bool UseHierarchicalRoce(const TilingData* tiling)
{
    return tiling->hierarchicalRoce;
}
uint32_t HierarchicalSendJob()
{
    return 1U;
}

struct Fixture {
    struct AivTaskCursor {
        uint32_t nextTile = 0;
        uint32_t tileCount = 0;
        uint32_t sliceIndex = 0;
        uint32_t nextSlice = 0;
    };
    struct SharedActivationCursor {
        bool done = false;
        uint32_t tile = 0;
    };
    struct ExpertTokenPosition {
        uint32_t expertIdx = 0;
    };
    struct Job {
        uint32_t jobIndex = 1;
        uint32_t totalJobs = 2;
    } aivJob_;
    struct {
        Job blockJob;
    } gmmExecutionConfig_;
    struct {
        uint32_t moeExpertPerRank = 0;
    } commonConfig_;
    TilingData tiling;
    struct {
        TilingData* tilingData;
    } params_{&tiling};
    struct {
        uint32_t clears = 0;
        int64_t finalWave = 999;
        void ClearSlice()
        {
            ++clears;
        }
        void SetWave(int64_t wave)
        {
            finalWave = wave;
        }
    } timer_;
    uint32_t sharedExpertNum_ = 0;
    uint32_t mGroupsPerWave_ = 2;

    // Each activation plan is {first physical tile, exclusive tile bound}.
    std::vector<std::pair<uint32_t, uint32_t>> activationPlans;
    std::set<std::pair<uint32_t, uint32_t>> readyTiles;
    std::set<std::pair<uint32_t, uint32_t>> consumedTiles;
    uint32_t combineSlices = 0;
    std::set<uint32_t> readyCombines;
    std::set<uint32_t> consumedCombines;
    bool sharedReady = false;
    uint32_t sharedTiles = 2;
    uint32_t consumedShared = 0;
    uint32_t activationAdvances = 0;
    uint32_t combineAdvances = 0;
    uint32_t remoteCalls = 0;
    std::vector<bool> remoteForeground;
    uint32_t operationCount = 0;
    int64_t clock = 0;
    std::vector<std::string> events;
    std::function<void(Fixture&, uint32_t)> afterCombine;
    std::function<void(Fixture&, uint32_t, uint32_t)> afterActivation;
    std::function<bool(Fixture&)> remoteProgress;
    std::function<void(Fixture&)> onClock;

    void Guard()
    {
        assert(++operationCount < 1000);
    }
    void ReadyAllActivationTiles()
    {
        for (uint32_t slice = 0; slice < activationPlans.size(); ++slice) {
            for (uint32_t tile = activationPlans[slice].first; tile < activationPlans[slice].second;
                 tile += gmmExecutionConfig_.blockJob.totalJobs) {
                readyTiles.emplace(slice, tile);
            }
        }
    }
    void ReadyAllCombines()
    {
        for (uint32_t slice = 0; slice < combineSlices; ++slice) {
            readyCombines.insert(slice);
        }
    }
    bool NextAivSlice(AivTaskCursor& cursor, bool activation)
    {
        Guard();
        if (activation) {
            ++activationAdvances;
            if (cursor.nextSlice != 0) {
                assert(cursor.nextTile >= cursor.tileCount);
            }
            if (cursor.nextSlice == activationPlans.size()) {
                return false;
            }
            cursor.sliceIndex = cursor.nextSlice++;
            cursor.nextTile = activationPlans[cursor.sliceIndex].first;
            cursor.tileCount = activationPlans[cursor.sliceIndex].second;
        } else {
            ++combineAdvances;
            if (cursor.nextSlice != 0) {
                assert(consumedCombines.count(cursor.sliceIndex) == 1);
            }
            if (cursor.nextSlice == combineSlices) {
                return false;
            }
            cursor.sliceIndex = cursor.nextSlice++;
        }
        return true;
    }
    bool TryRunReadyActivation(const AivTaskCursor& cursor, int32_t& readySequence)
    {
        Guard();
        auto key = std::make_pair(cursor.sliceIndex, cursor.nextTile);
        assert(cursor.nextTile < cursor.tileCount);
        if (readyTiles.count(key) == 0) {
            return false;
        }
        assert(consumedTiles.insert(key).second);
        events.push_back("A" + std::to_string(key.first) + ":" + std::to_string(key.second));
        ++readySequence;
        if (afterActivation) {
            afterActivation(*this, key.first, key.second);
        }
        return true;
    }
    bool TryRunReadyCombine(const AivTaskCursor& cursor)
    {
        Guard();
        if (readyCombines.count(cursor.sliceIndex) == 0) {
            return false;
        }
        assert(consumedCombines.insert(cursor.sliceIndex).second);
        events.push_back("C" + std::to_string(cursor.sliceIndex));
        if (afterCombine) {
            afterCombine(*this, cursor.sliceIndex);
        }
        return true;
    }
    void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& groups, uint64_t wave)
    {
        Guard();
        events.push_back("D" + std::to_string(position.expertIdx) + ":" + std::to_string(wave));
        ++position.expertIdx;
        ++groups;
    }
    bool TryRunSharedExpertActivation(SharedActivationCursor& cursor)
    {
        Guard();
        if (!sharedReady) {
            return false;
        }
        assert(cursor.tile == consumedShared);
        events.push_back("S" + std::to_string(cursor.tile));
        ++consumedShared;
        cursor.done = ++cursor.tile == sharedTiles;
        return true;
    }
    bool TryProgressRemoteCombine(bool foregroundPending = true)
    {
        Guard();
        ++remoteCalls;
        remoteForeground.push_back(foregroundPending);
        if (!UseHierarchicalRoce(params_.tilingData) || aivJob_.jobIndex != HierarchicalSendJob()) {
            events.push_back("R:noop");
            return false;
        }
        events.push_back(remoteCalls == 1 ? "R:init" : "R:scan");
        // Every completed local Combine offers progress, even with no ready payload.
        return remoteProgress ? remoteProgress(*this) : false;
    }

    void ProcessAiv1PriorityStages(int32_t& readySequence)
        /* PRODUCTION_SCHEDULER_BODY */

        void Run(int32_t expectedActivations)
    {
        AscendC::cycle = [this]() {
            Guard();
            ++clock;
            if (onClock) {
                onClock(*this);
            }
            return clock;
        };
        int32_t sequence = 40;
        ProcessAiv1PriorityStages(sequence);
        AscendC::cycle = {};
        assert(sequence == 40 + expectedActivations);
        assert(consumedTiles.size() == static_cast<std::size_t>(expectedActivations));
        assert(consumedCombines.size() == combineSlices);
        assert(activationAdvances == activationPlans.size() + 1);
        assert(combineAdvances == combineSlices + 1);
        assert(consumedShared == (sharedExpertNum_ == 0 ? 0 : sharedTiles));
        assert(timer_.clears == 1);
        assert(timer_.finalWave == AscendTimer::NO_WAVE);
        assert(gmmExecutionConfig_.blockJob.totalJobs == 2);
    }
    void Expect(std::initializer_list<std::string> expected) const
    {
        assert(events == std::vector<std::string>(expected));
    }
};

void CombineReleasesMultipleTilesAndSlices()
{
    Fixture f;
    f.activationPlans = {{1, 6}, {0, 3}};
    f.combineSlices = 2;
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t slice) {
        if (slice == 0) {
            fixture.ReadyAllActivationTiles();
        }
    };
    f.Run(5);
    f.Expect({"C0", "R:init", "A0:1", "A0:3", "A0:5", "A1:0", "A1:2", "C1", "R:scan"});
    assert(f.remoteForeground == std::vector<bool>({true, false}));
    assert(f.clock == 0);
}

void RemoteProgressRunsAfterEveryCombine()
{
    for (uint32_t slices : {1U, 3U}) {
        Fixture f;
        f.combineSlices = slices;
        f.ReadyAllCombines();
        f.Run(0);
        if (slices == 1) {
            f.Expect({"C0", "R:init"});
        } else {
            f.Expect({"C0", "R:init", "C1", "R:scan", "C2", "R:scan"});
        }
        assert(f.remoteCalls == slices);
        assert(f.remoteForeground == std::vector<bool>(slices, false));
        assert(f.clock == 0);
    }
}

void SharedActivationReleasedByCombineFollowsRemoteProgress()
{
    Fixture f;
    f.sharedExpertNum_ = 1;
    f.combineSlices = 2;
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t) { fixture.sharedReady = true; };
    f.Run(0);
    f.Expect({"C0", "R:init", "S0", "S1", "C1", "R:scan"});
    assert(f.remoteForeground == std::vector<bool>({true, false}));
}

void ExistingActivationDispatchSharedPriorityIsPreserved()
{
    Fixture f;
    f.activationPlans = {{1, 4}};
    f.ReadyAllActivationTiles();
    f.commonConfig_.moeExpertPerRank = 3;
    f.sharedExpertNum_ = 1;
    f.sharedReady = true;
    f.combineSlices = 1;
    f.ReadyAllCombines();
    f.Run(2);
    f.Expect({"A0:1", "A0:3", "D0:0", "D1:0", "D2:1", "S0", "S1", "C0", "R:init"});
}

void UnreadyActivationAllowsRemoteProgressThenResumes()
{
    Fixture f;
    f.activationPlans = {{0, 3}};
    f.combineSlices = 2;
    f.ReadyAllCombines();
    f.afterCombine = [](Fixture& fixture, uint32_t slice) {
        if (slice == 0) {
            fixture.readyTiles.emplace(0, 0);
        }
    };
    f.remoteProgress = [](Fixture& fixture) {
        fixture.readyTiles.emplace(0, 2);
        return true;
    };
    f.Run(2);
    f.Expect({"C0", "R:init", "A0:0", "A0:2", "C1", "R:scan"});
}

void FallbackRemoteProgressCanReleaseActivation()
{
    Fixture f;
    f.activationPlans = {{1, 2}};
    f.remoteProgress = [](Fixture& fixture) {
        fixture.ReadyAllActivationTiles();
        return true;
    };
    f.Run(1);
    f.Expect({"R:init", "A0:1"});
    assert(f.remoteForeground == std::vector<bool>({true}));
    assert(f.clock == 0);
}

void BackoffCanReleaseActivationAndCombine()
{
    Fixture f;
    f.activationPlans = {{1, 2}, {0, 1}};
    f.combineSlices = 1;
    f.onClock = [](Fixture& fixture) {
        if (fixture.clock == 3) {
            fixture.readyTiles.emplace(0, 1);
            fixture.ReadyAllCombines();
        }
    };
    f.afterActivation = [](Fixture& fixture, uint32_t slice, uint32_t) {
        if (slice == 0) {
            fixture.readyTiles.emplace(1, 0);
        }
    };
    f.Run(2);
    f.Expect({"R:init", "A0:1", "A1:0", "C0", "R:scan"});
    assert(f.clock >= GM_FLAG_POLL_BACKOFF_CYCLES);
}

void NonOwnersLeaveRemoteProgressToTheOwner()
{
    for (auto setting : {std::make_pair(true, 3U), std::make_pair(false, 1U), std::make_pair(false, 3U)}) {
        Fixture f;
        f.tiling.hierarchicalRoce = setting.first;
        f.aivJob_.jobIndex = setting.second;
        f.combineSlices = 2;
        f.ReadyAllCombines();
        f.Run(0);
        f.Expect({"C0", "R:noop", "C1", "R:noop"});
        assert(f.remoteCalls == 2);
        assert(f.clock == 0);
    }
}

int main()
{
    CombineReleasesMultipleTilesAndSlices();
    RemoteProgressRunsAfterEveryCombine();
    SharedActivationReleasedByCombineFollowsRemoteProgress();
    ExistingActivationDispatchSharedPriorityIsPreserved();
    UnreadyActivationAllowsRemoteProgressThenResumes();
    FallbackRemoteProgressCanReleaseActivation();
    BackoffCanReleaseActivationAndCombine();
    NonOwnersLeaveRemoteProgressToTheOwner();
}
