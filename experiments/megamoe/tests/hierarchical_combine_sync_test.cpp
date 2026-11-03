// Copyright (c) 2026, Lu Lu
// Real producer/host rendezvous with concurrent CPU workers. Only GM access,
// hardware SyncAll and one-token reduction are device doubles.
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
#define MEGA_MOE_HIERARCHICAL_REDUCE_H
using GM_ADDR = uint8_t*;
constexpr int PIPE_ALL = 0;
constexpr uint32_t EVENT_ID0 = 0, HIERARCHICAL_FLAG_BYTES = 512, HIERARCHICAL_MAX_AIV_WORKERS = 96;
constexpr uint32_t HIERARCHICAL_RANKS_PER_HOST = 16, INT_CACHELINE = 16;
constexpr uint32_t RANK_SYNC_COUNTER_OFFSET_BYTES = 48U * 1024U, RANK_SYNC_COUNTER_SLOT_BYTES = 64;
enum class HardEvent { MTE3_S };
enum class MemDsbT { DDR };
template <int>
void PipeBarrier()
{
}
template <HardEvent, uint32_t>
void SyncFuncStatic()
{
}
template <MemDsbT>
void DataSyncBarrier()
{
}
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
};
struct MoeStageCommonConfig {
    uint32_t rankId = 0;
};
struct Tiling {
    uint32_t combineReduceQps, hierarchicalSequence;
};
struct Params {
    Tiling* tilingData;
    struct {
        uint8_t* rankSyncInWorldPtr;
        uint8_t* combineProducerDonePtr;
    } peermemInfo;
};
struct HierarchicalCombineProgress {
    uint32_t sent = 0;
};
struct AscendTimerDevice {};
struct RoceWaitState {
    uint32_t checks = 0;
};
enum class RoceWaitKind { HostProducerBarrier };
uint8_t* g_winRankAddr_[32]{};
uint32_t HierarchicalHost(uint32_t rank)
{
    return rank / 16U;
}
uint32_t HierarchicalGateway(uint32_t rank)
{
    return rank ^ 16U;
}
uint32_t HierarchicalCombineLane(const Tiling* tiling, uint32_t job)
{
    return (job & 1U) && job / 2U < tiling->combineReduceQps ? job / 2U : tiling->combineReduceQps;
}
bool IsHierarchicalCombineOwner(const Tiling* t, uint32_t job)
{
    return HierarchicalCombineLane(t, job) < t->combineReduceQps;
}
uint8_t* HierarchicalRoceScratch(const Params&)
{
    static uint8_t scratch[1];
    return scratch;
}
std::mutex mutex;
std::condition_variable changed;
thread_local uint32_t activeJob = 0, activePhase = 0;
uint32_t arrivals = 0, generation = 0, workerCount = 56, localPosts = 0, hostPosts = 0;
uint32_t expectedSequence = 0;
int32_t expectedEpoch = 0;
uint8_t* producerFlags = nullptr;
uint8_t* rankFlags = nullptr;
int32_t ReadGmByPassDCache(const int32_t* address)
{
    std::lock_guard<std::mutex> lock(mutex);
    return *address;
}
void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    std::lock_guard<std::mutex> lock(mutex);
    *address = value;
    changed.notify_all();
}
template <bool>
void SyncAll()
{
    std::unique_lock<std::mutex> lock(mutex);
    const auto old = generation;
    if (++arrivals == workerCount) {
        // No worker can leave either hardware barrier before every producer /
        // handshake worker publishes this invocation, including empty lanes.
        for (uint32_t i = 0; i < workerCount; ++i) {
            auto* flag = reinterpret_cast<uint32_t*>(producerFlags + (activePhase * HIERARCHICAL_MAX_AIV_WORKERS + i) *
                                                                         HIERARCHICAL_FLAG_BYTES);
            assert(*flag == expectedSequence);
        }
        if (activePhase == 1) {
            for (uint32_t rank = 0; rank < 16U; ++rank)
                assert(reinterpret_cast<int32_t*>(rankFlags)[rank * INT_CACHELINE] == expectedEpoch);
        }
        arrivals = 0;
        ++generation;
        changed.notify_all();
    } else {
        changed.wait(lock, [&] { return generation != old; });
    }
}
void RoceCheckWait(RoceWaitState& state, uint32_t, uint8_t* scratch, RoceWaitKind, uint32_t, uint32_t, int32_t*,
                   uint32_t qp)
{
    assert(scratch == nullptr || activeJob == (qp - 1U) * 2U + 1U);
    assert(++state.checks < 10000000U);
    std::this_thread::yield();
}
bool TryProgressHierarchicalReducedCombine(const MoeStageCommonConfig&, const Params& params,
                                           HierarchicalCombineProgress& progress, uint32_t lane, uint8_t*,
                                           AscendTimerDevice*, uint32_t maxPosts)
{
    assert(maxPosts == 1U && activeJob == 2U * lane + 1U && lane < params.tilingData->combineReduceQps);
    std::lock_guard<std::mutex> lock(mutex);
    if (lane + 1U == params.tilingData->combineReduceQps) return false;  // Empty lane still joins.
    if (progress.sent == 0U) {
        ++progress.sent;
        ++localPosts;
        changed.notify_all();
        return true;
    }
    if (activePhase == 1U && progress.sent == 1U && lane != 1U) {
        ++progress.sent;
        ++hostPosts;
        // A remote producer finishes after progress has run in the host wait.
        reinterpret_cast<int32_t*>(rankFlags)[INT_CACHELINE] = expectedEpoch;
        changed.notify_all();
        return true;
    }
    return false;  // Lane 1's second token is NOT ready until after the barrier.
}

#include "../csrc/kernels/mega_moe/stage/mega_moe_hierarchical_combine_sync.hpp"

int main()
{
    std::array<std::vector<uint8_t>, 16> windows;
    for (uint32_t i = 0; i < 16; ++i) {
        windows[i].resize(60U * 1024U);
        g_winRankAddr_[i] = windows[i].data();
    }
    std::vector<uint8_t> flags(2U * HIERARCHICAL_MAX_AIV_WORKERS * HIERARCHICAL_FLAG_BYTES);
    producerFlags = flags.data();
    rankFlags = windows[0].data();
    for (uint32_t qps : {4U, 8U}) {
        for (uint32_t sequence : {129U, 0xffffffffU, 0x80000101U}) {
            Tiling tiling{qps, sequence};
            Params params{&tiling, {rankFlags, producerFlags}};
            MoeStageCommonConfig common;
            localPosts = hostPosts = 0U;
            expectedSequence = sequence;
            // Emulate the input handshake advancing the same per-AIV epoch.
            for (uint32_t i = 0; i < workerCount; ++i) {
                auto* counter = reinterpret_cast<int32_t*>(rankFlags + RANK_SYNC_COUNTER_OFFSET_BYTES + i * 64U);
                ++*counter;
                expectedEpoch = *counter + 1;
            }
            for (uint32_t rank = 0; rank < 16U; ++rank)
                reinterpret_cast<int32_t*>(rankFlags)[rank * INT_CACHELINE] = expectedEpoch;
            reinterpret_cast<int32_t*>(rankFlags)[INT_CACHELINE] = expectedEpoch - 1;
            std::vector<std::thread> workers;
            std::vector<uint32_t> sent(workerCount);
            for (uint32_t job = 0; job < workerCount; ++job) {
                workers.emplace_back([&, job] {
                    activeJob = job;
                    activePhase = 0;
                    if (job == 2U) {
                        // Delay a nonowner until an owner posts while waiting.
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [] { return localPosts != 0U; });
                    }
                    AivJobContext context{job, workerCount};
                    HierarchicalCombineProgress progress;
                    MegaMoeImpl::JoinHierarchicalCombineProducers(context, common, params, progress, nullptr, 0U);
                    activePhase = 1;
                    MegaMoeImpl::CrossRankSyncInHierarchicalHostWithProgress(context, common, params, progress,
                                                                             nullptr);
                    sent[job] = progress.sent;
                });
            }
            for (auto& thread : workers) thread.join();
            assert(localPosts > 0 && hostPosts > 0);
            for (uint32_t job = 0; job < workerCount; ++job) {
                assert(*reinterpret_cast<int32_t*>(rankFlags + RANK_SYNC_COUNTER_OFFSET_BYTES + job * 64U) ==
                       expectedEpoch);
                if (!IsHierarchicalCombineOwner(&tiling, job)) assert(sent[job] == 0U);
            }
            assert(sent[3] <= 1U);  // Barrier never blocks draining an unready token.
            assert(sent[2U * (qps - 1U) + 1U] == 0U);
        }
    }
}
