# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Host mirror of kernels/mega_moe/common/ascend_timer_v2.hpp."""

STARTUP_FIXED_NAMES = [
    "AIC_MOE_STARTUP",
    "AIV_MOE_STARTUP",
    "AIC_DISPATCH_BUFFER_INIT",
    "AIV_DISPATCH_BUFFER_INIT",
    "AIV_INPUT_PREP_BARRIER_WAIT",
    "AIV_COUNT_TABLE_PREPARE",
    "AIV_COUNT_METADATA_WAIT",
    "AIV_GATEWAY_PAYLOAD_CHECK",
    "AIV_COUNT_PREFIX_BUILD",
    "AIV_COUNT_TABLE_PUBLISH",
    "AIC_COUNT_READY_WAIT",
    "AIV_COUNT_READY_WAIT",
]

COMBINE_PROGRESS_FIXED_NAMES = [
    "AIV_COMBINE_INIT",
    "AIV_HOST_COMBINE_BARRIER_WAIT",
]

FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INPUT_BUFFER_INIT",
    "AIV_QUANT",
    "AIV_ROUTE_MASK",
    "AIV_RESET",
    "AIV_INPUT_CROSS_RANK_SYNC",
    "AIV_EXPORT_SYNC",
    "AIV_FINAL_CROSS_RANK_WAIT",
    "AIV_UNPERMUTE",
    "AIC_WAIT_INPUT_PREP",
    *STARTUP_FIXED_NAMES,
    *COMBINE_PROGRESS_FIXED_NAMES,
]

DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIV_GMM1_PROLOGUE",
    "AIV_GMM2_PROLOGUE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
    "AIC_SHARED_GMM1",
    "AIV_SHARED_GMM1_PROLOGUE",
    "AIC_SHARED_GMM2",
    "AIV_SHARED_GMM2_PROLOGUE",
    "AIV_SHARED_SWIGLU",
    "AIV_RDMA_POST",
    "AIV_RDMA_QUIET_WAIT",
    "AIV_RDMA_RECV_WAIT",
    "AIV_GATEWAY_PAYLOAD_WAIT",
    "AIV_COMBINE_PROGRESS",
    "AIV_COMBINE_DRAIN",
]

RDMA_PHASES = ("dispatch_meta", "dispatch_payload", "combine")

# W4 动态槽保存 tile/slice；其他路径保持原有区间协议。
MAX_DYNAMIC_ITER = 4096
N_CORE_COUNT = 96
CYCLE_TO_US = 1000.0

COUNTERS_PER_CORE = ((len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER) * 2 + 15) // 16 * 16
N_TIMING_COUNTER = COUNTERS_PER_CORE * N_CORE_COUNT
DYNAMIC_ITER_PER_CORE = (len(DYNAMIC_NAMES) + 15) // 16 * 16
TASK_METADATA_OFFSET = N_TIMING_COUNTER + DYNAMIC_ITER_PER_CORE * N_CORE_COUNT
TASK_METADATA_SIZE = N_CORE_COUNT * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER * 2
NO_WAVE = -1
WAVE_ID_OFFSET = TASK_METADATA_OFFSET + TASK_METADATA_SIZE
WAVE_ID_PER_CORE = (len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + 15) // 16 * 16
TOTAL_BUFFER_SIZE = WAVE_ID_OFFSET + WAVE_ID_PER_CORE * N_CORE_COUNT


def task_metadata(raw, core_id, logical_index):
    """坐标直接来自设备，不根据 shape 或耗时猜测任务身份。"""
    dynamic_index = logical_index - len(FIXED_NAMES)
    if dynamic_index < 0:
        return {}
    offset = TASK_METADATA_OFFSET + (core_id * len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER + dynamic_index) * 2
    first, second = raw[offset : offset + 2]
    family = DYNAMIC_NAMES[dynamic_index // MAX_DYNAMIC_ITER]
    mask = (1 << 32) - 1
    if family in ("AIV_COMBINE_PROGRESS", "AIV_COMBINE_DRAIN"):
        # Raw storage is signed int64; these two metadata words are uint64.
        first = int(first) & ((1 << 64) - 1)
        result = dict(
            granularity="combine_progress" if family == "AIV_COMBINE_PROGRESS" else "combine_drain",
            pending_before=first & 0xFFFF,
            pending_after=(first >> 16) & 0xFFFF,
        )
        before, after = result["pending_before"], result["pending_after"]
        if not 0 <= after <= before <= 1024:
            raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, pending counts")
        if family == "AIV_COMBINE_PROGRESS":
            scan_cycles = int(second) & ((1 << 64) - 1)
            posts, scanned = (first >> 32) & 0xFFFF, (first >> 48) & 0xFFFF
            sent = before - after
            if not (posts <= sent <= scanned <= 1024 and (posts == 0) == (sent == 0)):
                raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, progress counts")
            result.update(
                posts=posts,
                scanned_rows=scanned,
                scan_control_cycles=scan_cycles,
                scan_control_us=scan_cycles / CYCLE_TO_US,
            )
        elif (first >> 32) != 0 or int(second) != 0:
            raise ValueError(f"invalid combine metadata: core={core_id}, slot={dynamic_index}, unused drain fields")
        # Drain is an enclosing interval, not a second scan measurement. Its
        # unused metadata fields must not become measured zeroes in reports.
        return result
    if family == "AIV_GATEWAY_PAYLOAD_WAIT":
        source = ((first >> 32) & mask) - 1
        gateway = (first & mask) - 1
        if not (0 <= source < 32 and gateway == (source + 16) % 32 and 0 < second <= mask):
            raise ValueError(f"invalid gateway wait metadata: core={core_id}, slot={dynamic_index}")
        return dict(granularity="startup_dependency", source_rank=source, gateway_rank=gateway, sequence=second)
    if family.startswith("AIV_RDMA_"):
        peer = ((first >> 32) & mask) - 1
        phase = (first & mask) - 1
        kind = (second >> 32) & mask
        if not (0 <= peer < 32 and 0 <= phase < len(RDMA_PHASES) and kind in (0, 1)):
            raise ValueError(f"invalid RDMA metadata: core={core_id}, slot={dynamic_index}")
        return dict(
            granularity="rdma",
            peer_rank=peer,
            phase=RDMA_PHASES[phase],
            kind="commit" if kind else "data",
            bytes=second & mask,
        )
    if first == second == 0:
        return {}
    expert = ((first >> 32) & mask) - 1
    row_start = first & mask
    n_tile = ((second >> 32) & mask) - 1
    row_count = second & mask
    if expert < 0 or row_count == 0:
        raise ValueError(f"invalid task metadata: core={core_id}, slot={dynamic_index}")
    result = dict(
        expert_id=expert,
        row_start=row_start,
        row_count=row_count,
        m_tile=row_start // 256,
        granularity="tile" if n_tile >= 0 else "slice",
    )
    if "_SHARED_" in family:
        result["expert_kind"] = "shared"
    if n_tile >= 0:
        result.update(n_tile=n_tile, tile_id=dynamic_index % MAX_DYNAMIC_ITER)
    return result
