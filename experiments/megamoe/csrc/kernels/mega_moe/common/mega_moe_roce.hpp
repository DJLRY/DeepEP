// Copyright (c) 2026, Lu Lu

#ifndef MEGA_MOE_ROCE_H
#define MEGA_MOE_ROCE_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#include "ascend_timer_v2_device.hpp"

namespace MegaMoeImpl {

// QP0 belongs to SHMEM's default collectives. Physical AIV1 owns QP1 for
// payload/Combine; the last physical AIV1 owns QP2 for Dispatch metadata.
constexpr uint32_t MEGAMOE_ROCE_QP_INDEX = 1U;
constexpr uint32_t MEGAMOE_ROCE_META_QP_INDEX = 2U;
constexpr uint32_t MEGAMOE_ROCE_EVENT_ID = 7U;
constexpr uint32_t MEGAMOE_ROCE_SCRATCH_BYTES = 256U;

// Both operands are LOCAL symmetric-heap addresses, including the destination.
// SHMEM applies the destination PE translation. A mapped peer GM address is invalid here.
// Producers must drain their MTE3 writes before publishing readiness to this owner;
// scalar producers must write through to GM. The source stays immutable until Quiet.
// Scratch is private to this AIV, 32B-aligned, and at least 256B; no default SDK UB is used.
template <uint32_t QpIndex = MEGAMOE_ROCE_QP_INDEX>
__aicore__ inline void RocePutBytes(GM_ADDR dstLocalSym, GM_ADDR srcLocalSym, uint32_t bytes, uint32_t peer,
                                    __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                    AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                    bool isCommit = false)
{
    if (bytes == 0U) {
        return;
    }
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_POST, phase, peer, bytes, isCommit);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemx_roce_qp_put_nbi<uint8_t>(dstLocalSym, srcLocalSym, scratch, bytes, static_cast<int>(peer), QpIndex,
                                       MEGAMOE_ROCE_EVENT_ID);
#else
    // Host rejects this topology without a matching SDK; fail closed if that guard regresses.
    (void)dstLocalSym;
    (void)srcLocalSym;
    (void)peer;
    (void)scratch;
    trap();
#endif
}

template <uint32_t QpIndex = MEGAMOE_ROCE_QP_INDEX>
__aicore__ inline void RoceQuiet(uint32_t peer, __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                 AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                 bool isCommit = false)
{
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_QUIET_WAIT, phase, peer, isCommit ? sizeof(uint32_t) : 0U,
                                isCommit);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemx_roce_qp_quiet<uint8_t>(peer, QpIndex, scratch, MEGAMOE_ROCE_EVENT_ID);
#else
    (void)peer;
    (void)scratch;
    trap();
#endif
}

// A commit is a separate WQE after all preceding payload WQEs complete. The final
// Quiet also protects the symmetric source sequence against reuse by the next call.
// This is the same quiet -> signal write -> quiet ordering used by the pinned SHMEM
// RoCE barrier, restricted to our owned QP instead of a collective or the default QP0.
// Each sequence/commit owns a distinct 512B line; the receiver never clears an inbox.
template <uint32_t QpIndex = MEGAMOE_ROCE_QP_INDEX>
__aicore__ inline void RocePublishCommit(GM_ADDR dstFlagLocalSym, GM_ADDR srcSeqLocalSym, uint32_t peer,
                                         __ubuf__ uint8_t* scratch, AscendTimerDevice* timer = nullptr,
                                         AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload)
{
    RoceQuiet<QpIndex>(peer, scratch, timer, phase);
    RocePutBytes<QpIndex>(dstFlagLocalSym, srcSeqLocalSym, sizeof(uint32_t), peer, scratch, timer, phase, true);
    RoceQuiet<QpIndex>(peer, scratch, timer, phase, true);
}

// Quiet alone is sender-side completion. Receivers wait for the ordered commit in
// GM through SHMEM's DCache-invalidating signal API, then order subsequent MTE2 loads
// after that scalar observation. Payload is only read through DMA, not scalar cache.
// This event does not flush a NIC: NIC -> GM visibility comes from the SHMEM ordered
// signal protocol. In particular, an old MTE packet-tail flag is not an RDMA commit.
__aicore__ inline void RoceWaitCommit(__gm__ int32_t* localFlag, uint32_t sequence, AscendTimerDevice* timer = nullptr,
                                      AscendTimer::RdmaPhase phase = AscendTimer::RdmaPhase::DispatchPayload,
                                      uint32_t peer = 0U)
{
    AscendTimerRdmaScope timing(timer, AscendTimer::AIV_RDMA_RECV_WAIT, phase, peer, sizeof(uint32_t), true);
    aclshmem_signal_wait_until(localFlag, ACLSHMEM_CMP_EQ, static_cast<int32_t>(sequence));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<AscendC::TEventID>(MEGAMOE_ROCE_EVENT_ID));
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<AscendC::TEventID>(MEGAMOE_ROCE_EVENT_ID));
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_ROCE_H
