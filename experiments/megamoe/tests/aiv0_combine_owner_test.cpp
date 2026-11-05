// Copyright (c) 2026, Lu Lu
// Actual ProcessWave output orchestration, progress gate and host barrier.
// Stage doubles enforce prologue/UB lifetime and per-QP ownership; numerical
// readiness and asynchronous transport are covered by their production tests.
#include <array>
#include <cassert>
#include <cstdint>
#include "../csrc/kernels/mega_moe/common/mega_moe_peermem.hpp"

#define __gm__
#define __ubuf__
constexpr int AIC = 0, AIV = 1, g_coreType = AIV, PIPE_ALL = 0;

namespace AscendTimer {
enum Event { AIV_EXPORT_SYNC, AIV_FINAL_CROSS_RANK_WAIT, AIV_HOST_COMBINE_BARRIER_WAIT, AIV_UNPERMUTE };
enum class RdmaPhase { Combine };
}  // namespace AscendTimer

namespace MegaMoeImpl {
struct State {
    uint32_t physical = 0, qps = 0, preDrains = 0, payloads = 0, commits = 0, waits = 0, quiets = 0;
    uint32_t hostCqCalls = 0, unpermutes = 0, expectedRows = 0;
    uint32_t singleSweeps = 0;
    bool scanInScheduler = false;
    bool routedDone = false, sharedDone = false, drained = false, hostDone = false, exported = false;
    bool timer = false;
} current;
alignas(512) std::array<uint8_t, 64U * 1024U> window{};
std::array<uint8_t, 512> scratch{};
GM_ADDR g_winRankAddr_[32]{};
struct AivJobContext {
    uint32_t jobIndex, totalJobs;
};
struct MoeStageCommonConfig {
    uint32_t rankId = 0, topK = 8;
};
struct Params {
    MegaMoeTilingData* tilingData;
    struct {
        GM_ADDR rankSyncInWorldPtr = window.data();
        GM_ADDR combineReduceCommitPtr = window.data() + 4096;
    } peermemInfo;
};
struct Progress {
    uint32_t sentRows = 0;
    bool initialized = false;
};
struct AscendTimerDevice {
    bool Enabled() const
    {
        return current.timer;
    }
    void Start(AscendTimer::Event event)
    {
        if (event == AscendTimer::AIV_EXPORT_SYNC) current.exported = true;
    }
    void End(AscendTimer::Event) {}
};
struct AscendTimerFixedScope {
    AscendTimerFixedScope(AscendTimerDevice*, AscendTimer::Event) {}
};
enum class Stage { SHARED_EXPERT_GMM2, CROSS_RANK_SYNC_OUTPUT, UNPERMUTE };
struct Dump {
    void UpdateStage(Stage) {}
};
struct MegaMoeUnpermuteBufferConfig {};
struct HierarchicalCombineProgressBudget {
    uint32_t posts = 0xffffffffU, rows = 0xffffffffU;
    bool bounded = false, scan = false;
};
enum class RoceWaitKind { HostProducerBarrier };

bool ExpectedOwner()
{
    return current.physical % 2U == (current.qps == 8U ? 0U : 1U) &&
           current.physical / 2U < (current.qps == 0U ? 1U : current.qps);
}
void CheckOwner(uint32_t qp)
{
    assert(ExpectedOwner());
    assert(qp == current.physical / 2U + 1U);
    assert(current.routedDone && current.sharedDone);
}
template <int>
void PipeBarrier()
{
    current.drained = true;
}
template <bool>
void SyncAll()
{
    assert(current.routedDone && current.sharedDone);
    assert(current.exported);  // No extra all-core barrier before the AIV0 drain.
}
int32_t ReadGmByPassDCache(int32_t* address)
{
    return *address;
}
void WriteGmByPassDCache(int32_t* address, int32_t value)
{
    *address = value;
}
uint8_t* HierarchicalRoceScratch(const Params&)
{
    return scratch.data();
}
int32_t* HierarchicalSignalAddress(const decltype(Params::peermemInfo)&, HierarchicalSignal signal)
{
    return reinterpret_cast<int32_t*>(window.data() + 8192 + static_cast<uint32_t>(signal) * 512U);
}
void RoceWaitFlagEquals(int32_t*, uint32_t, uint32_t, uint8_t* ub, RoceWaitKind)
{
    if (ub != nullptr) {
        CheckOwner(1U);
        assert(current.physical == (current.qps == 8U ? 0U : 1U));
        ++current.hostCqCalls;
    }
}
void GmSignalWaitBarrier(int32_t*, int32_t) {}
void ProductionHostBarrier(const AivJobContext& job, const MoeStageCommonConfig& common, const Params& params)
{
    /* PRODUCTION_HOST_BARRIER */
}
void CrossRankSyncInHierarchicalHost(const AivJobContext& job, const MoeStageCommonConfig& common, const Params& params)
{
    ProductionHostBarrier(job, common, params);
    current.hostDone = true;
}

bool TryProgressHierarchicalReducedCombine(const MoeStageCommonConfig&, const Params&, Progress& progress,
                                           uint32_t lane, uint8_t*, AscendTimerDevice*)
{
    CheckOwner(lane + 1U);
    assert(current.qps == 4U);  // QP=8 must never enter the inter-wave path.
    progress.sentRows = current.expectedRows;
    return true;
}
bool TryProgressHierarchicalCombine(const MoeStageCommonConfig&, const Params&, Progress& progress, uint8_t*,
                                    AscendTimerDevice*, HierarchicalCombineProgressBudget budget, bool waitForAll)
{
    CheckOwner(1U);
    assert(current.qps == 0U && !current.hostDone);
    assert(!progress.initialized && ++current.singleSweeps == 1U);
    assert(budget.posts == 0xffffffffU && budget.rows == 0xffffffffU && !budget.bounded && !budget.scan);
    assert(waitForAll);
    progress.initialized = true;
    progress.sentRows = current.expectedRows;
    current.payloads += current.expectedRows;
    return current.expectedRows != 0U;
}
void DrainHierarchicalReducedCombine(const MoeStageCommonConfig&, const Params&, Progress& progress, uint32_t lane,
                                     uint8_t*, AscendTimerDevice*)
{
    CheckOwner(lane + 1U);
    assert(current.drained);
    if (!current.exported) {
        assert(current.qps == 8U && !current.hostDone);
        ++current.preDrains;
    } else {
        assert(current.hostDone);  // AIV1 still cannot block before its producer barrier.
    }
    current.payloads += current.expectedRows - progress.sentRows;
    progress.sentRows = current.expectedRows;
}
void TraceHierarchicalCombineBoundary(const MoeStageCommonConfig&, const Params&, Progress&, uint32_t,
                                      AscendTimerDevice*, uint32_t)
{
}
void RocePublishCommit(GM_ADDR dst, GM_ADDR src, uint32_t, uint8_t*, AscendTimerDevice*, AscendTimer::RdmaPhase phase,
                       uint32_t qp = 1U)
{
    CheckOwner(qp);
    assert(current.hostDone && phase == AscendTimer::RdmaPhase::Combine);
    assert(src == reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress({}, HierarchicalSignal::Sequence)));
    if (current.qps == 0U) {
        assert(dst == reinterpret_cast<GM_ADDR>(HierarchicalSignalAddress({}, HierarchicalSignal::CombineCommit)));
        assert(current.singleSweeps == 1U && current.payloads == current.expectedRows);
    }
    ++current.commits;
}
void RoceWaitCommit(int32_t* flag, uint32_t, AscendTimerDevice*, AscendTimer::RdmaPhase phase, uint32_t, uint8_t*,
                    uint32_t qp = 1U)
{
    CheckOwner(qp);
    assert(current.commits == 1U && current.hostDone);
    assert(phase == AscendTimer::RdmaPhase::Combine);
    if (current.qps == 0U) {
        assert(flag == HierarchicalSignalAddress({}, HierarchicalSignal::CombineCommit));
    }
    ++current.waits;
}
void RoceFinishOutstanding(uint32_t, uint8_t*, AscendTimerDevice*, AscendTimer::RdmaPhase, uint32_t qp)
{
    CheckOwner(qp);
    assert(current.waits == 1U);
    ++current.quiets;
}
void PublishHierarchicalLocalSignal(const Params&, HierarchicalSignal)
{
    assert(current.quiets == 1U);
}
template <typename... T>
void ExportCompactExpertTokenCounts(T&&...)
{
}
template <typename... T>
void CrossRankSyncInWorldSize(T&&...)
{
    assert(false);
}
template <int, typename, bool, uint32_t, typename... T>
void UnpermuteTokens(T&&...)
{
    assert(!ExpectedOwner());
    ++current.unpermutes;
}

struct Kernel {
    static constexpr bool ENABLE_A8W4 = true, TopkWeightsPrefetch = false;
    static constexpr uint32_t GMM1_TILE_M = 256;
    static constexpr int CombineQuantMode = COMBINE_NO_QUANT;
    using TopkWeightsType = float;
    MegaMoeTilingData tiling{};
    Params params_{&tiling, {}};
    AivJobContext aivJob_{current.physical, 56U};
    MoeStageCommonConfig commonConfig_{};
    Progress hierarchicalCombine_{};
    AscendTimerDevice timer_{};
    Dump exceptionDump_{};
    uint32_t sharedExpertNum_ = 0, rankId_ = 0, worldSize_ = 32;
    int countWorkspace_ = 0, tokenDispatchScratch_ = 0, tokenUnpermuteConfig_ = 0, tokenUnpermuteScratch_ = 0;
    bool TryProgressRemoteCombine(bool foregroundPending = true)
    {
        /* PRODUCTION_PROGRESS */
    }
    void ProcessMoeExpertStages(int32_t&)
    {
        assert(current.preDrains == 0U);
        if (current.qps == 0U) {
            for (uint32_t i = 0U; i < 4U; ++i) assert(!TryProgressRemoteCombine(true));
            assert(!hierarchicalCombine_.initialized && current.singleSweeps == 0U);
        }
        if (current.qps == 8U) {
            assert(!TryProgressRemoteCombine(true));
            assert(!TryProgressRemoteCombine(false));
        }
        current.routedDone = true;
        current.sharedDone = sharedExpertNum_ == 0U;
        current.drained = true;
        if (current.qps == 0U && !current.sharedDone) assert(!TryProgressRemoteCombine(true));
        if (current.qps == 0U && current.sharedDone && current.scanInScheduler) CheckSingleSweep();
    }
    void CheckSingleSweep()
    {
        TryProgressRemoteCombine(false);
        for (uint32_t i = 0U; i < 4U; ++i) assert(!TryProgressRemoteCombine(false));
    }
    void ProcessMoeExpertStages()
    {
        assert(false);
    }
    void ProcessSharedExpertTail()
    {
        assert(current.routedDone && !current.sharedDone && current.preDrains == 0U);
        current.drained = false;  // Shared prologue still owns UB/events here.
        if (current.qps == 8U) assert(!TryProgressRemoteCombine(false));
        if (current.qps == 0U) assert(!TryProgressRemoteCombine(true));
        current.sharedDone = current.drained = true;
        if (current.qps == 0U && current.scanInScheduler) CheckSingleSweep();
    }
    void ProcessSharedExpertGmm2()
    {
        assert(false);
    }
    MegaMoeUnpermuteBufferConfig InitTokenUnpermuteBuffers()
    {
        return {};
    }
    void Run()
    {
        auto& derived = *this;
        int32_t gmm1TileReadySequence = 0;
        /* PRODUCTION_STAGES */
    }
};
}  // namespace MegaMoeImpl

int main()
{
    using namespace MegaMoeImpl;
    for (auto& address : g_winRankAddr_) address = window.data();
    for (uint32_t qps : {0U, 4U, 8U})
        for (bool shared : {false, true})
            for (bool timer : {false, true})
                for (uint32_t rank : {0U, 30U})
                    for (uint32_t rows : {0U, 16U})
                        for (bool scanInScheduler : {false, true})
                            for (uint32_t physical = 0; physical < 56U; ++physical) {
                                current = {};
                                current.physical = physical;
                                current.qps = qps;
                                current.expectedRows = rows;
                                current.scanInScheduler = scanInScheduler;
                                current.timer = timer;
                                Kernel kernel;
                                kernel.rankId_ = kernel.commonConfig_.rankId = rank;
                                kernel.tiling.topoType = TOPO_TYPE_MTE_ROCE;
                                kernel.tiling.combineReduceQps = qps;
                                kernel.tiling.blockAivNum = 56U;
                                kernel.sharedExpertNum_ = shared ? 1U : 0U;
                                kernel.Run();
                                const bool owner = ExpectedOwner();
                                assert(current.preDrains == static_cast<uint32_t>(qps == 8U && owner));
                                if ((qps == 0U || qps == 8U) && owner) assert(current.payloads == rows);
                                assert(current.singleSweeps == static_cast<uint32_t>(qps == 0U && owner));
                                assert(current.commits == static_cast<uint32_t>(owner));
                                assert(current.waits == static_cast<uint32_t>(owner));
                                assert(current.quiets == static_cast<uint32_t>(owner));
                                assert(current.unpermutes == static_cast<uint32_t>(!owner));
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
                                assert(current.hostCqCalls == static_cast<uint32_t>(physical == (qps == 8U ? 0U : 1U)));
#else
                                assert(current.hostCqCalls == 0U);
#endif
                            }
}
