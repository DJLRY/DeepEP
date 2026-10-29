// Copyright (c) 2026, Lu Lu

// Execute the production acquisition helper concurrently. Only GM flag access,
// device synchronization, timers and heavyweight type dependencies are doubled.
// These are host control-flow/order checks, not NPU cache-coherence validation.
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define __aicore__
#define __gm__
#define GM_ADDR uint8_t*
#define MEGA_MOE_UTILS_H
#define ASCEND_DEEPEP_MEGA_MOE_ASCEND_TIMER_V2_DEVICE_H

constexpr int AIC = 0;
#ifdef GATEWAY_TEST_AIC
constexpr int g_coreType = AIC;
#else
constexpr int g_coreType = 1;
#endif

namespace GatewayTest {
constexpr uint32_t kSlots = 16U;
constexpr uint32_t kWorld = 32U;
constexpr uint32_t kExperts = 3U;
constexpr uint32_t kLineBytes = 512U;
constexpr uint32_t kLineWords = kLineBytes / sizeof(int32_t);
constexpr uint32_t kWindowBytes = 16384U;
constexpr uint32_t kControlOffset = 4096U;
constexpr uint32_t kCommitOffset = kControlOffset + 4U * kLineBytes;
constexpr uint32_t kReadyOffset = kControlOffset + 20U * kLineBytes;
constexpr uint32_t kMaxGroups = 32U;
constexpr uint32_t kSparse = 0x8429U;  // Sources 0, 3, 5, 10 and 15 of the other host.
thread_local uint32_t currentGroup = 0U;
thread_local uint32_t currentLane = 1U;

inline void Check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct State {
    std::mutex mutex;
    std::condition_variable changed;
    // One guard line on both ends of the local notification range.
    alignas(512) std::array<int32_t, (kSlots + 2U) * kLineWords> local{};
    alignas(512) std::array<std::array<int32_t, kWindowBytes / sizeof(int32_t)>, kWorld> windows{};
    uint32_t rank = 0U;
    uint32_t groups = 1U;
    uint32_t needed = 0U;
    int32_t sequence = 0;
    bool aborted = false;
    std::exception_ptr failure;
    uint64_t order = 0U;
    uint32_t commitWaits = 0U;
    std::array<uint64_t, 8U> commitObservedAt{};
    uint64_t commitBarrierAt = 0U;
    uint64_t aggregatePublishedAt = 0U;
    std::array<uint32_t, kSlots> remoteWaits{};
    std::array<uint32_t, kSlots> remoteBarriers{};
    std::array<uint32_t, kSlots> publications{};
    std::array<uint32_t, kSlots> timerStarts{};
    std::array<uint32_t, kSlots> timerEnds{};
    std::array<uint64_t, kSlots> publishedAt{};
    std::array<std::array<uint64_t, kSlots>, kMaxGroups> remoteObservedAt{};
    std::array<std::array<uint64_t, kSlots>, kMaxGroups> remoteBarrierAt{};
    std::array<uint32_t, kMaxGroups> lastRemoteSlot{};
    std::array<std::array<uint64_t, kSlots>, kMaxGroups> localObservedAt{};
    std::array<uint32_t, kMaxGroups> localWaits{};
    std::array<uint64_t, kMaxGroups> acquiredAt{};
    std::array<uint64_t, kMaxGroups> finishedAt{};
    std::array<uint32_t, kMaxGroups> countReads{};

    int32_t* Local(uint32_t slot)
    {
        return local.data() + (slot + 1U) * kLineWords;
    }
    int32_t* Remote(uint32_t slot)
    {
        // The source is on the other host; its gateway is on this host.
        const uint32_t gateway = (rank < kSlots ? 0U : kSlots) + slot;
        return windows[gateway].data() + kReadyOffset / sizeof(int32_t);
    }
    int32_t* Commit(uint32_t lane)
    {
        return windows[rank].data() + kCommitOffset / sizeof(int32_t) + lane * kLineWords;
    }
    bool Needed(uint32_t slot) const
    {
        return (needed & (uint32_t(1) << slot)) != 0U;
    }
};
inline State* state = nullptr;

struct Address {
    bool local;
    uint32_t slot;
    uint32_t qpLane = 0U;
    bool commit = false;
};

// Bounds/line checks do not emulate the production work distribution.
inline Address Decode(const int32_t* pointer)
{
    auto& s = *state;
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const auto localBegin = reinterpret_cast<uintptr_t>(s.Local(0U));
    if (address >= localBegin && address < localBegin + kSlots * kLineBytes) {
        Check((address - localBegin) % kLineBytes == 0U, "local flag did not own an aligned line");
        return {true, static_cast<uint32_t>((address - localBegin) / kLineBytes)};
    }
    for (uint32_t rank = 0U; rank < kWorld; ++rank) {
        const auto begin = reinterpret_cast<uintptr_t>(s.windows[rank].data());
        if (address >= begin && address < begin + kWindowBytes) {
            Check(rank / kSlots == s.rank / kSlots, "gateway read addressed the source host");
            if (address == begin + kReadyOffset) return {false, rank % kSlots};
            Check(rank == s.rank && address >= begin + kCommitOffset &&
                      address < begin + kCommitOffset + 8U * kLineBytes &&
                      (address - begin - kCommitOffset) % kLineBytes == 0U,
                  "payload commits must be acquired only on the gateway's own card");
            return {false, rank % kSlots, static_cast<uint32_t>((address - begin - kCommitOffset) / kLineBytes), true};
        }
    }
    throw std::runtime_error("flag access exceeded the declared workspace/window ranges");
}

inline void WaitFlag(int32_t* pointer, int32_t expected)
{
    auto& s = *state;
    std::unique_lock<std::mutex> lock(s.mutex);
    const auto address = Decode(pointer);
    Check(currentGroup < s.groups && (address.commit || s.Needed(address.slot)),
          "wait accessed an unused source or invalid group");
    Check(expected == s.sequence, "wait did not compare the complete invocation epoch");
    if (address.local) {
        ++s.localWaits[currentGroup];
    } else if (address.commit) {
        Check(currentGroup == 0U, "more than one local aggregator joined payload commits");
        Check(++s.commitWaits == address.qpLane + 1U, "duplicate or skipped local QP acquire");
    } else {
        Check(currentGroup == address.slot % s.groups, "gateway had a duplicate or incorrect checker");
        ++s.remoteWaits[address.slot];
        Check(s.remoteWaits[address.slot] == 1U, "duplicate aggregate acquire");
    }
    s.changed.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (*pointer != expected && !s.aborted) {
        if (s.changed.wait_until(lock, deadline) == std::cv_status::timeout) {
            throw std::runtime_error("gateway acquisition deadlocked or accepted no late arrival");
        }
    }
    Check(!s.aborted, "another worker failed");
    if (address.local) {
        Check(s.publishedAt[address.slot] != 0U, "consumer accepted a stale/unpublished local epoch");
        s.localObservedAt[currentGroup][address.slot] = ++s.order;
    } else if (address.commit) {
        s.commitObservedAt[address.qpLane] = ++s.order;
    } else {
        s.remoteObservedAt[currentGroup][address.slot] = ++s.order;
        s.lastRemoteSlot[currentGroup] = address.slot;
    }
}

inline void RemoteAcquire()
{
    auto& s = *state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint32_t slot = s.lastRemoteSlot[currentGroup];
    if (slot == kSlots) {
        Check(currentGroup == 0U && s.commitWaits == 8U && s.commitBarrierAt == 0U,
              "invalid local aggregate DDR ordering");
        for (auto observed : s.commitObservedAt) Check(observed != 0U, "aggregate omitted an independent QP");
        s.commitBarrierAt = ++s.order;
        return;
    }
    Check(slot < kSlots && s.remoteObservedAt[currentGroup][slot] != 0U,
          "DDR ordering preceded a successful gateway epoch observation");
    Check(++s.remoteBarriers[slot] == 1U, "DDR ordering repeated without another gateway dependency");
    s.remoteBarrierAt[currentGroup][slot] = ++s.order;
}

inline void Publish(int32_t* pointer, int32_t value)
{
    auto& s = *state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto address = Decode(pointer);
    if (!address.local) {
        Check(!address.commit && address.slot == s.rank % kSlots && currentGroup == 0U,
              "aggregate publisher wrote another gateway or a network commit");
        Check(value == s.sequence && s.aggregatePublishedAt == 0U, "aggregate epoch was cleared or duplicated");
        Check(s.commitBarrierAt > s.commitObservedAt[7U] && s.commitObservedAt[7U] != 0U,
              "aggregate publication preceded acquisition of all eight commits");
        *pointer = value;
        s.aggregatePublishedAt = ++s.order;
        s.changed.notify_all();
        return;
    }
    Check(address.local && s.Needed(address.slot), "helper wrote a remote or unused flag");
    Check(value == s.sequence, "helper cleared or truncated a notification epoch");
    Check(currentGroup == address.slot % s.groups, "non-owner published a gateway notification");
    Check(s.remoteObservedAt[currentGroup][address.slot] != 0U, "publication preceded remote readiness");
    Check(s.remoteBarrierAt[currentGroup][address.slot] > s.remoteObservedAt[currentGroup][address.slot],
          "local publication preceded the remote-read DDR ordering boundary");
    Check(s.timerStarts[address.slot] == 0U || s.timerEnds[address.slot] == 1U,
          "gateway timer included local publication/consumer work");
    Check(++s.publications[address.slot] == 1U, "gateway notification published more than once");
    *pointer = value;
    s.publishedAt[address.slot] = ++s.order;
    s.lastRemoteSlot[currentGroup] = kSlots;
    s.changed.notify_all();
}

inline void AcquirePayload()
{
    auto& s = *state;
    std::lock_guard<std::mutex> lock(s.mutex);
    for (uint32_t slot = 0U; slot < kSlots; ++slot) {
        if (s.Needed(slot)) {
            Check(s.localObservedAt[currentGroup][slot] > s.publishedAt[slot] && s.publishedAt[slot] != 0U,
                  "payload DMA ordering preceded a local notification");
        }
    }
    Check(s.acquiredAt[currentGroup] == 0U, "consumer issued duplicate payload acquire");
    s.acquiredAt[currentGroup] = ++s.order;
}
}  // namespace GatewayTest

namespace AscendC {
enum class MemDsbT { DDR };
template <MemDsbT Kind>
inline void DataSyncBarrier()
{
    static_assert(Kind == MemDsbT::DDR);
    GatewayTest::RemoteAcquire();
}
}  // namespace AscendC

class AscendTimerDevice {};

class AscendTimerGatewayWaitScope {
public:
    AscendTimerGatewayWaitScope(AscendTimerDevice* timer, uint32_t source, uint32_t gateway, uint32_t sequence)
        : enabled_(timer != nullptr), slot_(source % GatewayTest::kSlots)
    {
        if (!enabled_) return;
        using namespace GatewayTest;
        auto& s = *state;
        std::lock_guard<std::mutex> lock(s.mutex);
        const uint32_t sourceFirst = s.rank < kSlots ? kSlots : 0U;
        const uint32_t gatewayFirst = s.rank < kSlots ? 0U : kSlots;
        Check(source == sourceFirst + slot_ && gateway == gatewayFirst + slot_, "timer lost source/gateway mapping");
        Check(sequence == static_cast<uint32_t>(s.sequence), "timer recorded the wrong epoch");
        Check(s.Needed(slot_) && currentGroup == slot_ % s.groups, "local wait was mislabeled as a gateway wait");
        Check(++s.timerStarts[slot_] == 1U, "gateway timer duplicated the remote dependency");
    }
    ~AscendTimerGatewayWaitScope()
    {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(GatewayTest::state->mutex);
        ++GatewayTest::state->timerEnds[slot_];
    }

private:
    bool enabled_;
    uint32_t slot_;
};

namespace MegaMoeImpl {
constexpr uint32_t HIERARCHICAL_RANKS_PER_HOST = 16U;
constexpr uint32_t HIERARCHICAL_FLAG_BYTES = 512U;
constexpr uint32_t HIERARCHICAL_PAYLOAD_LANES = 8U;
enum class HierarchicalSignal : uint32_t { NetPayloadCommit = 4U, RelayPayloadReady = 20U };
enum class HardEvent { S_MTE2 };
struct MoeStageCommonConfig {
    uint32_t rankId = 0U;
    uint32_t worldSize = 32U;
    uint32_t moeExpertPerRank = 3U;
};
struct BlockWorkspaceContext {
    uint32_t blockIdx = 0U;
    uint32_t blockNum = 1U;
};
struct MegaMoeTilingData {
    uint32_t hierarchicalSequence = 1U;
    bool hierarchical = true;
};
struct Params {
    MegaMoeTilingData* tilingData = nullptr;
    struct {
        GM_ADDR rankSyncInWorldPtr = nullptr;
        GM_ADDR hierarchicalControlPtr = nullptr;
    } peermemInfo;
    struct {
        GM_ADDR gatewayPayloadReadyPtr = nullptr;
    } workspaceInfo;
};
inline uint32_t HierarchicalHost(uint32_t rank)
{
    return rank / HIERARCHICAL_RANKS_PER_HOST;
}
inline uint32_t HierarchicalGateway(uint32_t source)
{
    return source ^ HIERARCHICAL_RANKS_PER_HOST;
}
inline uint64_t HierarchicalSignalOffset(HierarchicalSignal signal)
{
    return static_cast<uint64_t>(signal) * HIERARCHICAL_FLAG_BYTES;
}
inline uint32_t GetSubBlockIdx()
{
    return GatewayTest::currentLane;
}
inline bool UseHierarchicalRoce(const MegaMoeTilingData* tiling)
{
    return tiling->hierarchical;
}
inline void WaitUntilGmFlagEquals(int32_t* pointer, int32_t value)
{
    GatewayTest::WaitFlag(pointer, value);
}
inline void WriteGmByPassDCache(int32_t* pointer, int32_t value)
{
    GatewayTest::Publish(pointer, value);
}
template <HardEvent Event, uint32_t Id>
inline void SyncFuncStatic()
{
    static_assert(Event == HardEvent::S_MTE2 && Id == 7U);
    GatewayTest::AcquirePayload();
}
}  // namespace MegaMoeImpl

#include "../csrc/kernels/mega_moe/stage/mega_moe_gateway_readiness.hpp"

namespace GatewayTest {
using namespace MegaMoeImpl;

class CountTensor {
public:
    explicit CountTensor(const State& s)
    {
        // Local sources deliberately have nonzero counts; they must not cause a
        // gateway dependency. Remote nonzero entries vary by expert.
        const uint32_t localFirst = s.rank < kSlots ? 0U : kSlots;
        const uint32_t sourceFirst = s.rank < kSlots ? kSlots : 0U;
        for (uint32_t slot = 0U; slot < kSlots; ++slot) {
            data_[localFirst + slot] = 123;
            if (s.Needed(slot)) data_[(slot % kExperts) * kWorld + sourceFirst + slot] = slot + 1U;
        }
    }
    int32_t GetValue(uint32_t index) const
    {
        Check(index < data_.size(), "count-table read exceeded complete snapshot");
        std::lock_guard<std::mutex> lock(state->mutex);
        ++state->countReads[currentGroup];
        return data_[index];
    }
    void OverwriteWithPrefixes()
    {
        // Zero counts may become nonzero inclusive prefixes. Acquiring after
        // publication must use the saved dependency mask, never rescan this.
        data_.fill(123);
    }

private:
    std::array<int32_t, kWorld * kExperts> data_{};
};

struct Fixture {
    State s;
    MoeStageCommonConfig common;
    MegaMoeTilingData tiling;
    Params params;
    std::array<GM_ADDR, kWorld> addresses{};
    std::vector<std::thread> workers;
    bool timed = false;

    Fixture(uint32_t rank, uint32_t groups) : common{rank, kWorld, kExperts}
    {
        state = &s;
        s.rank = rank;
        s.groups = groups;
        params.tilingData = &tiling;
        params.peermemInfo.rankSyncInWorldPtr = reinterpret_cast<GM_ADDR>(s.windows[rank].data());
        params.peermemInfo.hierarchicalControlPtr = params.peermemInfo.rankSyncInWorldPtr + kControlOffset;
        params.workspaceInfo.gatewayPayloadReadyPtr = reinterpret_cast<GM_ADDR>(s.Local(0U));
        for (uint32_t rankIndex = 0U; rankIndex < kWorld; ++rankIndex) {
            addresses[rankIndex] = reinterpret_cast<GM_ADDR>(s.windows[rankIndex].data());
        }
    }
    ~Fixture()
    {
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.aborted = true;
            s.changed.notify_all();
        }
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
        state = nullptr;
    }

    void Begin(uint32_t sequence, uint32_t needed, bool reset = false)
    {
        Check(workers.empty(), "previous round was not joined");
        s.sequence = static_cast<int32_t>(sequence);
        s.needed = needed;
        tiling.hierarchicalSequence = sequence;
        s.order = 0U;
        s.commitWaits = 0U;
        s.commitObservedAt = {};
        s.commitBarrierAt = 0U;
        s.aggregatePublishedAt = 0U;
        s.remoteWaits = {};
        s.remoteBarriers = {};
        s.publications = {};
        s.timerStarts = {};
        s.timerEnds = {};
        s.publishedAt = {};
        s.remoteObservedAt = {};
        s.remoteBarrierAt = {};
        s.lastRemoteSlot.fill(kSlots);
        s.localObservedAt = {};
        s.localWaits = {};
        s.acquiredAt = {};
        s.finishedAt = {};
        s.countReads = {};
        if (reset) {
            // Models the caller's reset BEFORE its existing input barrier.
            s.local.fill(0);
            for (auto& window : s.windows) window.fill(0);
        }
    }
    void PublishRemote(uint32_t slot)
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        Check(s.Needed(slot) || slot == s.rank % kSlots, "test publisher selected an unused gateway");
        if (slot == s.rank % kSlots) {
            for (uint32_t lane = 0U; lane < 8U; ++lane) *s.Commit(lane) = s.sequence;
        } else {
            // Other gateways execute the same aggregation independently.
            *s.Remote(slot) = s.sequence;
        }
        s.changed.notify_all();
    }
    void PublishAllExcept(uint32_t held = kSlots)
    {
        for (uint32_t slot = 0U; slot < kSlots; ++slot) {
            if ((s.Needed(slot) || slot == s.rank % kSlots) && slot != held) PublishRemote(slot);
        }
    }
    void Start(uint32_t group)
    {
        workers.emplace_back([this, group] {
            currentGroup = group;
            currentLane = 1U;
            try {
                CountTensor counts(s);
                BlockWorkspaceContext workspace{group, s.groups};
                AscendTimerDevice timer;
                const uint32_t needed = BuildHierarchicalGatewayNeededMask(common, workspace, params, counts);
                Check(needed == s.needed, "dependency snapshot lost or invented a remote source");
                counts.OverwriteWithPrefixes();
                AcquireHierarchicalGatewayPayload(common, workspace, params, addresses.data(), needed,
                                                  timed ? &timer : nullptr);
                std::lock_guard<std::mutex> lock(s.mutex);
                Check(s.acquiredAt[group] != 0U, "consumer returned before its payload DMA acquire");
                for (uint32_t slot = 0U; slot < kSlots; ++slot) {
                    if (s.Needed(slot)) {
                        Check(*s.Local(slot) == s.sequence && *s.Remote(slot) == s.sequence,
                              "consumer completed before the current payload epoch");
                    }
                }
                s.finishedAt[group] = ++s.order;
                s.changed.notify_all();
            } catch (...) {
                std::lock_guard<std::mutex> lock(s.mutex);
                if (!s.failure) s.failure = std::current_exception();
                s.aborted = true;
                s.changed.notify_all();
            }
        });
    }
    void StartAll()
    {
        for (uint32_t group = 0U; group < s.groups; ++group) Start(group);
    }
    template <typename Predicate>
    void Await(Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        const bool done = s.changed.wait_for(lock, std::chrono::seconds(3), [&] { return s.aborted || predicate(); });
        Check(done && !s.aborted, "workers did not reach the requested protocol boundary");
    }
    void JoinAndCheck()
    {
        for (auto& worker : workers) worker.join();
        workers.clear();
        if (s.failure) std::rethrow_exception(s.failure);
        Check(s.commitWaits == 8U && s.aggregatePublishedAt > s.commitBarrierAt && s.commitBarrierAt != 0U,
              "gateway failed to publish its aggregate independently of local demand");
        for (uint32_t group = 0U; group < s.groups; ++group) {
            Check(s.finishedAt[group] > s.acquiredAt[group] && s.acquiredAt[group] != 0U,
                  "missing consumer completion/acquire");
        }
        for (uint32_t slot = 0U; slot < kSlots; ++slot) {
            const uint32_t expected = s.Needed(slot) ? 1U : 0U;
            Check(s.remoteWaits[slot] == expected && s.remoteBarriers[slot] == expected &&
                      s.publications[slot] == expected,
                  "needed gateways did not each have exactly one checker/publication");
            Check(s.timerStarts[slot] == (timed ? expected : 0U) && s.timerEnds[slot] == (timed ? expected : 0U),
                  "gateway timing included local waits or missed a remote checker");
            if (!s.Needed(slot)) continue;
            for (uint32_t group = 0U; group < s.groups; ++group) {
                Check(s.localObservedAt[group][slot] > s.publishedAt[slot] &&
                          s.acquiredAt[group] > s.localObservedAt[group][slot],
                      "consumer did not observe publication before acquiring payload");
            }
        }
        for (uint32_t i = 0U; i < kLineWords; ++i) {
            Check(s.local[i] == 0 && s.local[(kSlots + 1U) * kLineWords + i] == 0,
                  "notification access changed workspace guard lines");
        }
    }
};

void Distribution()
{
    for (uint32_t groups : {1U, 3U, 7U, 8U, 16U, 28U, 32U}) {
        for (uint32_t rank : {0U, 16U}) {
            for (uint32_t mask : {0U, kSparse, 0xffffU}) {
                Fixture f(rank, groups);
                f.Begin(137U, mask);
                f.PublishAllExcept();
                f.StartAll();
                f.JoinAndCheck();
            }
        }
    }
}

void LateArrival()
{
    for (uint32_t groups : {1U, 8U, 28U, 32U}) {
        Fixture f(16U, groups);
        f.Begin(143U, kSparse);
        f.PublishAllExcept(0U);
        f.StartAll();
        {
            std::lock_guard<std::mutex> lock(f.s.mutex);
            for (uint32_t lane = 0U; lane < 7U; ++lane) *f.s.Commit(lane) = f.s.sequence;
            f.s.changed.notify_all();
        }
        f.Await([&] { return f.s.commitWaits == 8U; });
        {
            std::lock_guard<std::mutex> lock(f.s.mutex);
            for (uint32_t group = 0U; group < groups; ++group) {
                Check(f.s.finishedAt[group] == 0U, "late gateway failed to hold every consumer");
            }
            Check(f.s.publications[0] == 0U, "late gateway was published before its payload");
            Check(f.s.aggregatePublishedAt == 0U, "aggregate ignored the last QP's late commit");
        }
        f.PublishRemote(0U);
        f.JoinAndCheck();
    }
}

void Epochs()
{
    for (uint32_t rank : {7U, 31U}) {
        Fixture f(rank, 28U);
        f.Begin(0x80000011U, kSparse);
        f.PublishAllExcept();
        f.StartAll();
        f.JoinAndCheck();
        // Keep the previous local AND remote values. The full new epoch must
        // hold consumers until the new remote payload is actually published.
        f.Begin(0xffffffffU, kSparse);
        f.PublishAllExcept(15U);
        f.StartAll();
        f.Await([&] { return rank % kSlots == 15U ? f.s.commitWaits != 0U : f.s.remoteWaits[15] != 0U; });
        {
            std::lock_guard<std::mutex> lock(f.s.mutex);
            Check(f.s.publications[15] == 0U, "checker accepted an old nonzero remote epoch");
            for (uint32_t group = 0U; group < f.s.groups; ++group) {
                Check(f.s.finishedAt[group] == 0U, "consumer accepted an old nonzero local epoch");
            }
        }
        f.PublishRemote(15U);
        f.JoinAndCheck();
        // Graph replay may reuse a sequence only after the caller's reset.
        f.Begin(0xffffffffU, 0xffffU, true);
        f.PublishAllExcept(3U);
        f.StartAll();
        f.Await([&] { return f.s.remoteWaits[3] != 0U; });
        f.PublishRemote(3U);
        f.JoinAndCheck();
    }
}

void NoLocalConsumers()
{
    Fixture f(5U, 28U);
    f.Begin(144U, 0U);
    f.StartAll();
    f.Await([&] { return f.s.commitWaits == 1U; });
    {
        std::lock_guard<std::mutex> lock(f.s.mutex);
        Check(f.s.aggregatePublishedAt == 0U && *f.s.Remote(5U) == 0,
              "zero local demand incorrectly released other ranks before payload arrival");
    }
    f.PublishRemote(5U);
    f.JoinAndCheck();
    Check(*f.s.Remote(5U) == f.s.sequence, "zero local demand suppressed the aggregate needed by other ranks");
}

void LateConsumer()
{
    Fixture f(0U, 32U);
    f.Begin(149U, 0xffffU);
    f.PublishAllExcept();
    // The first 16 groups own all notifications; later consumers arrive after
    // publication and must never clear a ready flag from the same invocation.
    for (uint32_t group = 0U; group < kSlots; ++group) f.Start(group);
    f.Await([&] {
        for (uint32_t slot = 0U; slot < kSlots; ++slot)
            if (!f.s.publications[slot]) return false;
        return true;
    });
    for (uint32_t group = kSlots; group < f.s.groups; ++group) f.Start(group);
    f.JoinAndCheck();
}

void Telemetry()
{
    for (uint32_t rank : {0U, 16U}) {
        Fixture f(rank, 7U);
        f.timed = true;
        f.Begin(151U, 0xffffU);
        f.PublishAllExcept();
        f.StartAll();
        f.JoinAndCheck();
    }
}

void Guards()
{
    Fixture f(0U, 1U);
    f.Begin(157U, 0xffffU);
    CountTensor counts(f.s);
    currentGroup = 0U;
    const auto guardedAcquire = [&](BlockWorkspaceContext workspace) {
        const uint32_t mask = BuildHierarchicalGatewayNeededMask(f.common, workspace, f.params, counts);
        Check(mask == 0U, "guarded path created payload dependencies");
        AcquireHierarchicalGatewayPayload(f.common, workspace, f.params, f.addresses.data(), mask);
    };
    currentLane = 0U;
    guardedAcquire({0U, 1U});
    currentLane = 1U;
    f.tiling.hierarchical = false;
    guardedAcquire({0U, 1U});
    f.tiling.hierarchical = true;
    guardedAcquire({0U, 0U});
    guardedAcquire({1U, 1U});
#ifdef GATEWAY_TEST_AIC
    guardedAcquire({0U, 1U});
#endif
    Check(f.s.countReads[0] == 0U && f.s.order == 0U, "guarded path touched counts, flags or synchronization");
}
}  // namespace GatewayTest

int main(int argc, char** argv)
{
    using namespace GatewayTest;
    try {
        Check(argc == 2, "expected one scenario name");
        const std::string scenario = argv[1];
        if (scenario == "distribution")
            Distribution();
        else if (scenario == "late-arrival")
            LateArrival();
        else if (scenario == "epochs")
            Epochs();
        else if (scenario == "no-local-consumers")
            NoLocalConsumers();
        else if (scenario == "late-consumer")
            LateConsumer();
        else if (scenario == "telemetry")
            Telemetry();
        else if (scenario == "guards")
            Guards();
        else
            throw std::runtime_error("unknown scenario");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
