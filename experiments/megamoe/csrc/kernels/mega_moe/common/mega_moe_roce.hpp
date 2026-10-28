// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_ROCE_H
#define MEGA_MOE_ROCE_H

#include <cstdint>
#include "../../../../../../csrc/common/roce_qp_layout.hpp"
#include "kernel_operator.h"
#include "shmem.h"
#include "ascend_timer_v2_device.hpp"
#include "mega_moe_roce_progress.hpp"

namespace MegaMoeImpl {

// Each owner exclusively submits and polls its QP throughout the invocation.
// QP0 belongs to SHMEM; payload uses QP1..8 and metadata uses QP9.
constexpr uint32_t MEGAMOE_ROCE_QP_INDEX = ascend_deepep::roce_layout::kFirstPayloadQp;
constexpr uint32_t MEGAMOE_ROCE_EVENT_ID = 7U;
constexpr uint32_t MEGAMOE_ROCE_SCRATCH_BYTES = 256U;

// Match the pinned HNS backend's hardware clock conversion, including its host
// compilation pass. The supported Ascend950 path uses 1000 cycles/us.
#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
constexpr uint64_t MEGAMOE_ROCE_CYCLES_PER_US = 50U;
#else
constexpr uint64_t MEGAMOE_ROCE_CYCLES_PER_US = 1000U;
#endif
constexpr uint64_t MEGAMOE_ROCE_PROGRESS_INTERVAL_CYCLES = 1000U * MEGAMOE_ROCE_CYCLES_PER_US;
constexpr uint64_t MEGAMOE_ROCE_WAIT_TIMEOUT_CYCLES = 60ULL * 1000U * 1000U * MEGAMOE_ROCE_CYCLES_PER_US;

__aicore__ inline uint32_t RoceTryProgress(uint32_t peer, __ubuf__ uint8_t* scratch,
                                           uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
    return RoceProgressReady(peer, qpIndex, scratch, MEGAMOE_ROCE_EVENT_ID);
}

struct RoceWaitState {
    uint64_t startCycle = 0U;
    uint64_t progressCycle = 0U;
    uint32_t checks = 0U;
    bool started = false;
};

// Numeric diagnostics avoid passing device string pointers through printf.
enum class RoceWaitKind : uint32_t {
    Commit = 0U,
    LocalReadiness = 1U,
    HostProducerBarrier = 2U,
    CombineRows = 3U,
};

// Only failed waits call this helper. Sample the clock once per 64 attempts and
// poll ready CQEs at most once per millisecond; a successful receive adds no CQ
// DMA. A null scratch identifies a nonowner, which must never consume the CQ.
// Callers may reset the state after useful producer progress to bound inactivity.
__aicore__ inline void RoceCheckWait(RoceWaitState& state, uint32_t peer, __ubuf__ uint8_t* scratch, RoceWaitKind kind,
                                     uint32_t expected, uint32_t observed, __gm__ int32_t* flag = nullptr,
                                     uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825) && (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__))
    if (!state.started) {
        state.startCycle = state.progressCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
        state.started = true;
        return;
    }
    if ((++state.checks & 63U) != 0U) {
        return;
    }
    const uint64_t now = static_cast<uint64_t>(AscendC::GetSystemCycle());
    if (scratch != nullptr && now - state.progressCycle >= MEGAMOE_ROCE_PROGRESS_INTERVAL_CYCLES) {
        RoceTryProgress(peer, scratch, qpIndex);
        state.progressCycle = now;
    }
    if (now - state.startCycle >= MEGAMOE_ROCE_WAIT_TIMEOUT_CYCLES) {
        if (flag != nullptr) {
            // The failed public SHMEM test above invalidated this flag's cache
            // line. Read its value only for diagnosis, not on the polling path.
            AscendC::GlobalTensor<int32_t> value;
            value.SetGlobalBuffer(flag);
            observed = static_cast<uint32_t>(value.GetValue(0));
        }
        AscendC::printf("MegaMoE RoCE wait timeout: kind=%u peer=%u qp=%u expected=%u observed=%u\n",
                        static_cast<uint32_t>(kind), peer, qpIndex, expected, observed);
        trap();
    }
#else
    (void)state;
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    (void)kind;
    (void)expected;
    (void)observed;
    (void)flag;
#endif
}

__aicore__ inline void RoceWaitFlagEquals(__gm__ int32_t* flag, uint32_t sequence, uint32_t peer = 0U,
                                          __ubuf__ uint8_t* scratch = nullptr, RoceWaitKind kind = RoceWaitKind::Commit,
                                          uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825) && (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__))
    RoceWaitState state;
    while (aclshmem_int32_test(flag, ACLSHMEM_CMP_EQ, static_cast<int32_t>(sequence)) == 0) {
        RoceCheckWait(state, peer, scratch, kind, sequence, 0U, flag, qpIndex);
    }
#else
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    (void)kind;
    // The SDK host pass exposes a different four-argument test API. Keep its
    // compatible wait declaration here; only the device pass executes above.
    aclshmem_signal_wait_until(flag, ACLSHMEM_CMP_EQ, static_cast<int32_t>(sequence));
#endif
}

// Both operands are LOCAL symmetric-heap addresses, including the destination.
// SHMEM applies the destination PE translation. A mapped peer GM address is invalid here.
// Producers must drain their MTE3 writes before publishing readiness to this owner;
// scalar producers must write through to GM. The source stays immutable until Quiet.
// Scratch is private to this AIV, 32B-aligned, and at least 256B; no default SDK UB is used.
__aicore__ inline void RocePutBytes(GM_ADDR dstLocalSym, GM_ADDR srcLocalSym, uint32_t bytes, uint32_t peer,
                                    __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                    AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                    bool isCommit = false, uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
    if (bytes == 0U) {
        return;
    }
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_POST, phase, peer, bytes, isCommit);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemx_roce_qp_put_nbi<uint8_t>(dstLocalSym, srcLocalSym, scratch, bytes, static_cast<int>(peer), qpIndex,
                                       MEGAMOE_ROCE_EVENT_ID);
#else
    // Host rejects this topology without a matching SDK; fail closed if that guard regresses.
    (void)dstLocalSym;
    (void)srcLocalSym;
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    trap();
#endif
}

__aicore__ inline void RoceQuiet(uint32_t peer, __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                 AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                 bool isCommit = false, uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_QUIET_WAIT, phase, peer, isCommit ? sizeof(uint32_t) : 0U,
                                isCommit);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemx_roce_qp_quiet<uint8_t>(peer, qpIndex, scratch, MEGAMOE_ROCE_EVENT_ID);
#else
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    trap();
#endif
}

// HNS_1825 hardware puts SO on the final 2KiB fragment automatically. Its contract
// orders all preceding WQEs on this QP into remote GM before that fragment becomes
// visible; this 4B commit occupies a single fragment. No SHMEM SO bit is needed.
// Posting therefore publishes readiness without waiting for local completion.
// This hardware contract is not assumed for XSCALE, which retains both Quiets.
// HNS sources, including Sequence, stay immutable until RoceFinishOutstanding.
// Each sequence/commit owns a distinct 512B line; the receiver never clears an inbox.
__aicore__ inline void RocePublishCommit(GM_ADDR dstFlagLocalSym, GM_ADDR srcSeqLocalSym, uint32_t peer,
                                         __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                         AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                         uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceQuiet(peer, scratch, timer, phase, false, qpIndex);
#endif
    RocePutBytes(dstFlagLocalSym, srcSeqLocalSym, sizeof(uint32_t), peer, scratch, timer, phase, true, qpIndex);
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceQuiet(peer, scratch, timer, phase, true, qpIndex);
#endif
}

// Each QP owner calls this once alongside Unpermute and before kernel return.
// It closes the source/Sequence lifetime and retires all remaining SQ/CQ entries.
// The reciprocal receive commit does not replace this local completion boundary.
__aicore__ inline void RoceFinishOutstanding(uint32_t peer, __ubuf__ uint8_t* scratch,
                                             AscendTimerDevice* timer = nullptr,
                                             AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::Combine,
                                             uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    RoceQuiet(peer, scratch, timer, phase, false, qpIndex);
#else
    (void)peer;
    (void)qpIndex;
    (void)scratch;
    (void)timer;
    (void)phase;
#endif
}

// Quiet alone is sender-side completion. Receivers wait for the ordered commit in
// GM through SHMEM's DCache-invalidating signal APIs, then order subsequent MTE2 loads
// after that scalar observation. Payload is only read through DMA, not scalar cache.
// This event does not flush a NIC: NIC -> GM visibility comes from the SHMEM ordered
// signal protocol. In particular, an old MTE packet-tail flag is not an RDMA commit.
__aicore__ inline void RoceWaitCommit(__gm__ int32_t* localFlag, uint32_t sequence, AscendTimerDevice* timer = nullptr,
                                      AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                      uint32_t peer = 0U, __ubuf__ uint8_t* scratch = nullptr,
                                      uint32_t qpIndex = MEGAMOE_ROCE_QP_INDEX)
{
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_RECV_WAIT, phase, peer, sizeof(uint32_t), true);
    RoceWaitFlagEquals(localFlag, sequence, peer, scratch, RoceWaitKind::Commit, qpIndex);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<AscendC::TEventID>(MEGAMOE_ROCE_EVENT_ID));
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<AscendC::TEventID>(MEGAMOE_ROCE_EVENT_ID));
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_ROCE_H
