# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Per-round device rendezvous and device-kernel measurements."""

import csv
from datetime import timedelta
import math
from pathlib import Path
import statistics


# 整批开始前的计算，让 Host 有时间排入后续算子；不计入 kernel Duration。
DEFAULT_ALIGNMENT = dict(rows=100000, cols=30000, repeats=20)
ROUND_SYNC = "hccl_all_reduce_before_each_iteration"


class DeviceRoundBarrier:
    """Benchmark-only HCCL rendezvous, ordered on the caller's current stream.

    MegaMoE waits for this stream before launching on its runtime stream and
    makes this stream wait for completion afterwards. Thus the next collective
    cannot overtake the previous MegaMoE. Do not host-synchronize between them.
    """

    def __init__(self, torch, group):
        import torch.distributed as dist

        if dist.is_initialized():
            raise RuntimeError("MegaMoE benchmark requires its own HCCL profiling process group")
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=group.rank(),
            world_size=group.size(),
            timeout=timedelta(seconds=300),
        )
        self.dist, self.torch = dist, torch
        # Zero stays finite across arbitrarily many SUM reductions.
        self.dummy = torch.zeros(1, dtype=torch.float32, device="npu")
        self()
        torch.npu.synchronize()  # Communicator creation stays outside profiling.
        group.barrier("profiling device barrier ready")

    def __call__(self):
        # async_op=False inserts the completion dependency on the current stream.
        self.dist.all_reduce(self.dummy, op=self.dist.ReduceOp.SUM, async_op=False)

    def close(self):
        self.torch.npu.synchronize()
        self.dist.destroy_process_group()


def alignment_config(config):
    value = dict(config.get("alignment", DEFAULT_ALIGNMENT))
    if set(value) != set(DEFAULT_ALIGNMENT) or any(type(n) is not int or n < 1 for n in value.values()):
        raise ValueError("alignment requires positive integer rows, cols, repeats")
    return value


def enqueue_batch(call, torch, prefix, alignment, iterations, timer=None, *, barrier):
    """Queue busy work once per batch, then device rendezvous -> MegaMoE each round."""
    for _ in range(alignment["repeats"]):
        torch.exp(prefix, out=prefix)
    for iteration in range(iterations):
        barrier()
        # 所有 rank 排入同样次数；只在最后一轮、选中的 rank 上打点。
        call(timer if iteration + 1 == iterations else None)


def read_kernel_samples(directory, iterations):
    """Read only this rank/case's fresh profiler export; never subtract device clocks."""
    samples = []
    paths = sorted(Path(directory).rglob("kernel_details*.csv"))
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for index, row in enumerate(reader):
                row = {key.strip(): value for key, value in row.items() if key is not None}
                name = " ".join(row.get(key, "") for key in ("Name", "Type", "Kernel Name"))
                if "megamoe" not in name.replace("_", "").lower():
                    continue
                try:
                    start, duration = (float(row[key]) for key in ("Start Time(us)", "Duration(us)"))
                except (KeyError, ValueError) as error:
                    raise ValueError(f"Invalid MegaMoE profiler row {path}:{index + 2}: {row}") from error
                if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Nonfinite/nonpositive MegaMoE timing: {path}:{index + 2}")
                samples.append(
                    dict(
                        start_us=start,
                        duration_us=duration,
                        name=name,
                        stream_id=row.get("Stream ID"),
                        device_id=row.get("Device_id"),
                        file=str(path),
                        row=index + 2,
                    )
                )
    samples.sort(key=lambda sample: sample["start_us"])
    keys = [(s["device_id"], s["stream_id"], s["start_us"]) for s in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate MegaMoE kernel samples under {directory}; refuse misaligned rounds")
    if len(samples) != iterations:
        raise ValueError(
            f"{directory}: expected {iterations} MegaMoE kernels, found {len(samples)} "
            f"in {len(paths)} kernel_details CSV files; inspect the raw profiler export"
        )
    return samples


def summarize_rounds(per_rank, ranks_per_node=None):
    """Aggregate only the last ceil(formal rounds / 2) samples."""
    if not per_rank or not per_rank[0] or any(len(row) != len(per_rank[0]) for row in per_rank):
        raise ValueError("Missing ranks or mismatched profiling rounds")
    if any(not 0 < value < float("inf") for row in per_rank for value in row):
        raise ValueError("Non-finite/nonpositive timing sample")
    # Validate the complete capture before selection; malformed early samples
    # must not silently pass. Raw kernel_samples.json keeps every formal round.
    recorded_rounds = len(per_rank[0])
    first_selected_iteration = recorded_rounds // 2
    per_rank = [row[first_selected_iteration:] for row in per_rank]
    minimum = [min(values) for values in zip(*per_rank)]
    maximum = [max(values) for values in zip(*per_rank)]
    report = dict(
        metric="megamoe_kernel_duration_us",
        aggregation="mean_of_per_round_min_rank",
        iteration_selection="tail_half_ceil",
        recorded_rounds=recorded_rounds,
        first_selected_iteration=first_selected_iteration,
        rounds=len(minimum),
        mean_us=statistics.mean(minimum),
        median_us=statistics.median(minimum),
        min_us=min(minimum),
        max_us=max(minimum),
        rank_max_mean_us=statistics.mean(maximum),
        rank_min_rounds_us=minimum,
        rank_max_rounds_us=maximum,
        per_rank_rounds_us=per_rank,
        timer_pass=False,
        scope="all_ep_ranks",
    )
    if ranks_per_node is not None:
        if ranks_per_node < 1 or len(per_rank) % ranks_per_node:
            raise ValueError("ranks_per_node must divide the sampled world size")
        report["per_node"] = [
            dict(
                node=node,
                rank_start=offset,
                mean_us=statistics.mean(min(values) for values in zip(*per_rank[offset : offset + ranks_per_node])),
            )
            for node, offset in enumerate(range(0, len(per_rank), ranks_per_node))
        ]
    return report


def collect_profile(call, group, torch, config, prefix, directory, bs, *, barrier):
    import torch_npu

    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        export_type=profiler.ExportType.Text,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental,
    ) as prof:
        # profiler 初始化后先 Host 会合，再逐轮排入设备 barrier，不打断设备队列。
        torch.npu.synchronize()
        group.barrier(f"profiler ready BS={bs}")
        enqueue_batch(call, torch, prefix, config["alignment"], config["iters"], barrier=barrier)
        torch.npu.synchronize()
        print(f"[PROFILE] BS={bs} batch complete; exporting {directory}", flush=True)
        prof.step()
    return read_kernel_samples(directory, config["iters"])
