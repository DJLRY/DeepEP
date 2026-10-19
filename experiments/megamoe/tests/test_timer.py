# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

from pathlib import Path
import re
import struct
import csv

import pytest

from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode, load_raw
from deep_ep_experimental.megamoe.timer.trace_parser import (
    first_start_cycle,
    raw_events_for_rank,
    synchronized_rank_bases,
    trace_events,
    write_startup_breakdown,
)

ROOT = Path(__file__).resolve().parents[3]


def test_device_schema_and_extended_size():
    header = (ROOT / "experiments/megamoe/csrc/kernels/mega_moe/common/ascend_timer_v2.hpp").read_text(encoding="utf-8")

    def names(enum, sentinel):
        body = re.search(r"enum\s+" + enum + r"\s*\{(.*?)\};", header, re.S)[1]
        return [item.split("=")[0].strip() for item in body.split(",") if item.strip() != sentinel]

    assert names("FixedTiming", "FIXED_TIMING_COUNT") == p.FIXED_NAMES
    assert names("DynamicTimingType", "DYNAMIC_TYPE_COUNT_ENUM") == p.DYNAMIC_NAMES
    for name in ("MAX_DYNAMIC_ITER", "N_CORE_COUNT"):
        assert int(re.search(rf"{name}\s*=\s*(\d+)", header)[1]) == getattr(p, name)
    assert p.TOTAL_BUFFER_SIZE == p.WAVE_ID_OFFSET + p.WAVE_ID_PER_CORE * p.N_CORE_COUNT
    assert p.TASK_METADATA_SIZE == p.N_CORE_COUNT * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER * 2


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "rank_0_npu_time.bin"
    with path.open("wb") as stream:
        stream.truncate(p.TOTAL_BUFFER_SIZE * 8)

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for core, name, start, end in (
            (0, "AIC_GMM1", 102000, 107000),
            (0, "AIC_GMM1_WAIT", 100000, 102000),
            (1, "AIV_GMM1_PROLOGUE", 101000, 105000),
            (2, "AIV_DISPATCH", 100000, 103000),
        ):
            family = p.DYNAMIC_NAMES.index(name)
            logical = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER
            put(core * p.COUNTERS_PER_CORE + logical * 2, start, end)
            put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 1)
            metadata = (
                p.TASK_METADATA_OFFSET
                + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + family * p.MAX_DYNAMIC_ITER) * 2
            )
            put(metadata, (3 << 32) | 256, (2 << 32) | 96)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, 4)
    return path


def test_binary_to_csv_to_trace_wait_metadata(raw_file):
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    events = raw_events_for_rank(parsed, 0)
    assert len(events) == 4
    assert {event["group_id"] for event in events} == {0}
    trace = trace_events(events, "keep", first_start_cycle(events), include_wait=False, time_origin="rank")
    assert len(trace) == 3
    compute = next(event for event in trace if event["args"]["event_family"] == "AIC_GMM1")
    assert compute["dur"] == 5 and compute["ts"] == 2
    assert compute["args"]["expert_id"] == 2
    assert compute["args"]["n_tile"] == 1
    assert compute["args"]["wave_id"] == 4
    assert len(trace_events(events, "keep", first_start_cycle(events), include_wait=True)) == 4


def test_truncated_and_overflow_rejected(raw_file):
    with raw_file.open("r+b") as stream:
        stream.seek(p.N_TIMING_COUNTER * 8)
        stream.write(struct.pack("<q", p.MAX_DYNAMIC_ITER + 1))
    with pytest.raises(ValueError, match="count"):
        decode(raw_file, raw_file.with_suffix(".csv"))
    with raw_file.open("r+b") as stream:
        stream.truncate(8)
    with pytest.raises(ValueError, match="size"):
        load_raw(raw_file)


def test_sync_alignment_is_explicit_and_requires_anchor():
    with pytest.raises(ValueError, match="missing"):
        synchronized_rank_bases({0: [{"name": "AIC_GMM1", "start_cycle": 100, "end_cycle": 200}]})


def test_rdma_binary_metadata_and_wait_visibility(raw_file):
    core = 1
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for phase in range(3):
            for name, commit, size in (
                ("AIV_RDMA_POST", 0, 3584),
                ("AIV_RDMA_QUIET_WAIT", 0, 0),
                ("AIV_RDMA_RECV_WAIT", 1, 4),
            ):
                family = p.DYNAMIC_NAMES.index(name)
                slot = family * p.MAX_DYNAMIC_ITER + phase
                logical = len(p.FIXED_NAMES) + slot
                put(core * p.COUNTERS_PER_CORE + logical * 2, 101000 + phase * 1000, 101500 + phase * 1000)
                put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 3)
                put(
                    p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + slot) * 2,
                    (17 << 32) | (phase + 1),
                    (commit << 32) | size,
                )
                put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical, p.NO_WAVE)
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    base = first_start_cycle(raw)
    hidden = trace_events(raw, "keep", base)
    shown = trace_events(raw, "keep", base, include_wait=True)
    assert len(hidden) == 6 and len(shown) == 13
    rdma = [e for e in shown if e["args"].get("granularity") == "rdma"]
    assert len(rdma) == 9
    assert {e["args"]["phase"] for e in rdma} == set(p.RDMA_PHASES)
    for event in rdma:
        args = event["args"]
        assert event["tid"] == "group_0_AIV_0/RDMA"
        assert args["peer_rank"] == 16 and args["wave_id"] is None
        assert "expert_id" not in args and "n_tile" not in args
        if args["event_family"] == "AIV_RDMA_POST":
            assert args["kind"] == "data" and args["bytes"] == 3584
            assert event["cat"] == "rdma_post"
        else:
            assert event["cat"] == "rdma_wait"
    for event in hidden:
        counterpart = next(e for e in shown if e["name"] == event["name"] and e["tid"] == event["tid"])
        assert (event["ts"], event["dur"]) == (counterpart["ts"], counterpart["dur"])


@pytest.mark.parametrize("first,second", [(0, 0), (33 << 32 | 1, 4), (1 << 32 | 4, 4), (1 << 32 | 1, 2 << 32)])
def test_rdma_invalid_metadata_rejected(first, second):
    class Metadata:
        def __getitem__(self, key):
            return first, second

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index("AIV_RDMA_POST") * p.MAX_DYNAMIC_ITER
    with pytest.raises(ValueError, match="invalid RDMA metadata"):
        p.task_metadata(Metadata(), 1, logical)


def test_startup_roundtrip_keeps_waits_in_diagnostic_csv(raw_file):
    phases = [
        (3, "AIC_MOE_STARTUP", 100000, 110000),
        (3, "AIC_COUNT_READY_WAIT", 101000, 109000),
        (5, "AIV_MOE_STARTUP", 100000, 110000),
        (5, "AIV_COUNT_TABLE_PREPARE", 100500, 109500),
        (5, "AIV_COUNT_METADATA_WAIT", 101000, 104000),
        (5, "AIV_GATEWAY_PAYLOAD_CHECK", 104000, 107000),
        (5, "AIV_COUNT_PREFIX_BUILD", 107000, 108000),
        (5, "AIV_COUNT_TABLE_PUBLISH", 108000, 109000),
    ]
    with raw_file.open("r+b") as stream:

        def put(index, *values):
            stream.seek(index * 8)
            stream.write(struct.pack("<" + "q" * len(values), *values))

        for core, name, start, end in phases:
            slot = p.FIXED_NAMES.index(name)
            put(core * p.COUNTERS_PER_CORE + slot * 2, start, end)
            put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + slot, p.NO_WAVE)
        core = 5
        family = p.DYNAMIC_NAMES.index("AIV_GATEWAY_PAYLOAD_WAIT")
        slot = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER
        put(core * p.COUNTERS_PER_CORE + slot * 2, 104500, 106500)
        put(p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + slot, p.NO_WAVE)
        put(p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family, 1)
        put(
            p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) + family) * p.MAX_DYNAMIC_ITER * 2,
            (18 << 32) | 2,
            130,
        )
    parsed = raw_file.with_suffix(".parsed.csv")
    decode(raw_file, parsed)
    raw = raw_events_for_rank(parsed, 0)
    base = first_start_cycle(raw)
    shown = trace_events(raw, "keep", base, include_wait=True)
    startup = [e for e in shown if e["tid"].endswith("/STARTUP")]
    assert len(startup) == 9
    gateway = next(e for e in startup if e["args"].get("granularity") == "startup_dependency")
    assert gateway["name"] == "AIV_GATEWAY_PAYLOAD_WAIT_source17_gateway1_seq130"
    assert gateway["args"]["source_rank"] == 17 and gateway["args"]["gateway_rank"] == 1
    assert gateway["args"]["wave_id"] is None and "expert_id" not in gateway["args"]
    assert gateway["ts"] == 4.5 and gateway["dur"] == 2
    hidden = trace_events(raw, "keep", base)
    assert not any("WAIT" in e["name"] for e in hidden)
    for event in hidden:
        counterpart = next(e for e in shown if e["name"] == event["name"] and e["tid"] == event["tid"])
        assert (event["ts"], event["dur"]) == (counterpart["ts"], counterpart["dur"])
    report = raw_file.with_name("startup_breakdown.csv")
    write_startup_breakdown({0: raw}, report)
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert sum("WAIT" in row["event_family"] for row in rows) == 3
    assert sum(row["scope"] == "envelope" for row in rows) == 4
    dependency = next(row for row in rows if row["event_family"] == "AIV_GATEWAY_PAYLOAD_WAIT")
    assert (dependency["source_rank"], dependency["gateway_rank"], dependency["sequence"]) == ("17", "1", "130")
    assert float(dependency["duration_us"]) == 2


@pytest.mark.parametrize("first,second", [(0, 0), (18 << 32 | 3, 130), (18 << 32 | 2, 0), (18 << 32 | 2, 1 << 32)])
def test_gateway_invalid_metadata_rejected(first, second):
    class Metadata:
        def __getitem__(self, key):
            return first, second

    logical = len(p.FIXED_NAMES) + p.DYNAMIC_NAMES.index("AIV_GATEWAY_PAYLOAD_WAIT") * p.MAX_DYNAMIC_ITER
    with pytest.raises(ValueError, match="invalid gateway wait metadata"):
        p.task_metadata(Metadata(), 2, logical)
