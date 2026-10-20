# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Exercise real clipboard boundaries and lossless offline trace reconstruction."""

import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from deep_ep_experimental.megamoe.timer import protocol as p
from deep_ep_experimental.megamoe.timer import text_transfer as transfer
from deep_ep_experimental.megamoe.timer.parse_timer_csv import core_info
from deep_ep_experimental.megamoe.timer.trace_parser import first_start_cycle, raw_events_for_rank, trace_events


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/megamoe/scripts/trace_text.py"


def csv_bytes(base):
    events = [
        (0, "KERNEL_TIMING", 1, 501, {}),
        (0, "AIC_WAIT_INPUT_PREP", 2, 101, {}),
        (0, "AIC_COUNT_READY_WAIT", 101, 151, {}),
        (0, "AIC_GMM1_WAIT_0", 151, 201, {}),
        (0, "AIC_GMM1_0", 201, 301, {"expert_id": 2, "row_start": 0, "row_count": 21, "n_tile": 0}),
        (0, "AIC_GMM2_0", 350, 450, {"expert_id": 2, "row_start": 0, "row_count": 21, "n_tile": 1}),
        (
            2,
            "AIV_GATEWAY_PAYLOAD_WAIT_0",
            101,
            141,
            {"granularity": "startup_dependency", "source_rank": 17, "gateway_rank": 1, "sequence": 120},
        ),
        (2, "AIV_DISPATCH_0", 151, 200, {"expert_id": 2, "row_start": 0, "row_count": 21}),
        (
            2,
            "AIV_RDMA_POST_0",
            10,
            50,
            {"granularity": "rdma", "peer_rank": 16, "phase": "dispatch_meta", "kind": "data", "bytes": 1536},
        ),
        (2, "AIV_UNPERMUTE", 450, 480, {}),
    ]
    # Use the complete fixed/dynamic naming convention, with all idle cores.
    names = list(dict.fromkeys(event[1] for event in events))
    families = [name for name in p.DYNAMIC_NAMES if any(event[1] == name + "_0" for event in events)]
    fields = ["core_id", "group_id", "core_type", "sub_id"]
    for name in names:
        fields.extend(f"{name}_{suffix}" for suffix in ("start_cycle", "end_cycle", "duration(us)", "task", "wave_id"))
    fields.extend(f"{name}_task_count" for name in families)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for core in range(p.N_CORE_COUNT):
        group, kind, sub = core_info(core)
        row = {field: "0" for field in fields}
        row.update(core_id=core, group_id=group, core_type=kind, sub_id=sub)
        for name in names:
            row[f"{name}_task"] = ""
            row[f"{name}_wave_id"] = -1
        for owner, name, start, end, task in events:
            if owner != core:
                continue
            row[f"{name}_start_cycle"] = base + start
            row[f"{name}_end_cycle"] = base + end
            row[f"{name}_duration(us)"] = (end - start) / 1000
            row[f"{name}_task"] = json.dumps(task) if task else ""
            if name.endswith("_0"):
                row[f"{name[:-2]}_task_count"] = 1
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


@pytest.fixture
def capture(tmp_path):
    run = tmp_path / "trace_node_0_capture"
    timer = run / "bs_32/timer"
    timer.mkdir(parents=True)
    # Integers above 2**53 catch accidental float conversion during transfer.
    for rank in (0, 16):
        (timer / f"rank_{rank}_npu_time_parsed.csv").write_bytes(csv_bytes(2**55 + rank * 1000000))
        rank_dir = run / f"rank_{rank}"
        rank_dir.mkdir()
        (rank_dir / "manifest.json").write_text(
            json.dumps(dict(rank=rank, build="fixture", note="中文元数据")), encoding="utf-8"
        )
    (run / "case.json").write_text('{"warmup":20,"iters":100}\n', encoding="utf-8")
    (run / "run.json").write_text('{"action":"trace","timer_ranks":[0,16]}\n', encoding="utf-8")
    (run / "launch.json").write_text('{"machine_local":"must not be collected"}', encoding="utf-8")
    return run, timer


@pytest.mark.parametrize("limit", [1024, 4096, 60000, 65535])
def test_complete_block_fits_even_after_crlf_and_reassembles(limit):
    data = random.Random(123).randbytes(150000).hex()
    bundle = {"format": transfer.FORMAT, "uncompressible_fixture": data}
    _, _, parts = transfer.make_parts(bundle, limit)
    assert len(parts) > 1
    for text in parts:
        assert text.isascii()
        assert len(text.encode("ascii")) <= len(text.replace("\n", "\r\n").encode("ascii")) <= limit
        assert max(map(len, text.splitlines())) < 80
    status, compressed = transfer.parse_parts([text.replace("\n", "\r\n") for text in reversed(parts)])
    assert not status["missing"]
    import lzma

    assert lzma.decompress(compressed) == transfer.json_bytes(bundle)


def test_export_restore_exact_files_and_trace_with_waits(capture, tmp_path):
    run, timer = capture
    perf = tmp_path / "perf.json"
    perf.write_text('{"mean_us":590}\n', encoding="utf-8")
    output, index = transfer.export_text(
        run, tmp_path / "export", attachments=[perf], max_bytes=1024, label="EP32 流水"
    )
    parts = [transfer.show_part(output, number) for number in range(1, len(index["parts"]) + 1)]
    assert len(parts) > 1
    assert not any("launch.json" in record["name"] for record in index["files"])
    # Model copied messages, including Markdown fences, CRLF, reordered blocks
    # and an accidentally repeated message. Identical duplicates are harmless.
    pasted = ["```text\n" + part.replace("\n", "\r\n") + "```\n" for part in reversed(parts)] + [parts[0]]
    restored = tmp_path / "restored"
    transfer.restore_text(pasted, restored)
    for rank in (0, 16):
        original = timer / f"rank_{rank}_npu_time_parsed.csv"
        recovered = restored / "timer" / original.name
        assert recovered.read_bytes() == original.read_bytes()
        raw = raw_events_for_rank(original, rank)
        assert max(event["start_cycle"] for event in raw) > 2**53
        expected = trace_events(raw, "keep", first_start_cycle(raw), include_wait=True, time_origin="rank")
        actual = json.loads((restored / "timer" / f"rank_{rank}_npu_time.json").read_text(encoding="utf-8"))
        assert actual == expected
        assert any(event["args"]["event_family"] == "AIV_GATEWAY_PAYLOAD_WAIT" for event in actual)
        manifest = Path(f"rank_{rank}/manifest.json")
        assert (restored / "context" / manifest).read_bytes() == (run / manifest).read_bytes()
    assert (restored / "attachments/001_perf.json").read_bytes() == perf.read_bytes()
    assert (restored / "timer/startup_breakdown.csv").exists()
    assert (restored / "timer/all_rank_trace.json").exists()


def test_missing_corrupt_truncated_mixed_and_duplicate_parts(capture, tmp_path):
    run, _ = capture
    output, index = transfer.export_text(run, tmp_path / "export", max_bytes=1024)
    parts = [transfer.show_part(output, number) for number in range(1, len(index["parts"]) + 1)]
    assert len(parts) > 2
    status, payload = transfer.parse_parts(parts[1:])
    assert status["missing"] == [1] and payload is None
    restored = tmp_path / "restored"
    with pytest.raises(ValueError, match="Missing parts"):
        transfer.restore_text(parts[1:], restored)
    assert not restored.exists()
    header, body = parts[0].split("\n\n", 1)
    bad_char = "A" if body[0] != "A" else "B"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        transfer.parse_parts([header + "\n\n" + bad_char + body[1:], *parts[1:]])
    with pytest.raises(ValueError, match="truncated"):
        transfer.parse_parts([parts[0].replace(transfer.END, "")])
    _, _, different = transfer.make_parts(transfer.collect_bundle(run, label="different capture"), 1024)
    with pytest.raises(ValueError, match="Mixed bundles"):
        transfer.parse_parts([parts[0], different[0]])
    # Duplicate part with valid individual checksum but different contents is rejected.
    decoded = transfer.parse_parts(parts)[1]
    conflict = transfer.render_part(index["bundle_sha256"], len(decoded), 1, len(parts), b"conflicting content")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        transfer.parse_parts([parts[0], conflict])


def test_clipboard_payload_line_wrapping_is_harmless(capture):
    run, _ = capture
    _, _, parts = transfer.make_parts(transfer.collect_bundle(run), 1024)
    wrapped = []
    for part in parts:
        header, body = part.split("\n\n", 1)
        encoded, footer = body.split(transfer.END)
        encoded = "".join(encoded.split())
        wrapped.append(
            header
            + "\n\n"
            + "\n".join(encoded[n : n + 41] for n in range(0, len(encoded), 41))
            + "\n"
            + transfer.END
            + footer
        )
    assert transfer.parse_parts(wrapped) == transfer.parse_parts(parts)


@pytest.mark.parametrize("mutation", ["missing_core", "cut_row", "duplicate_core", "missing_slots"])
def test_incomplete_csv_is_rejected(capture, tmp_path, mutation):
    run, timer = capture
    path = timer / "rank_0_npu_time_parsed.csv"
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if mutation == "missing_core":
        rows.pop()
    elif mutation == "cut_row":
        rows[-1].pop()
    elif mutation == "duplicate_core":
        rows[-1] = rows[1]
    else:
        rows[1][rows[0].index("AIC_GMM1_task_count")] = "2"
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        transfer.export_text(run, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "path",
    ["../escape.txt", "/absolute.txt", "context/../../escape.txt", "context\\escape.txt", "context/C:escape.txt"],
)
def test_restore_rejects_archive_paths_before_writing(capture, tmp_path, path):
    bundle = transfer.collect_bundle(capture[0])
    bundle["files"].append(dict(name=path, text="bad", bytes=3, sha256=transfer.digest(b"bad")))
    _, _, parts = transfer.make_parts(bundle)
    output = tmp_path / "restored"
    with pytest.raises(ValueError, match="path"):
        transfer.restore_text(parts, output)
    assert not output.exists()


def test_restore_checks_file_hash_and_never_overwrites(capture, tmp_path):
    bundle = transfer.collect_bundle(capture[0])
    bundle["files"][0]["text"] += "corrupted"
    _, _, parts = transfer.make_parts(bundle)
    with pytest.raises(ValueError, match="File size/SHA-256"):
        transfer.restore_text(parts, tmp_path / "bad")
    _, _, parts = transfer.make_parts(transfer.collect_bundle(capture[0]))
    output = tmp_path / "already-exists"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        transfer.restore_text(parts, output)
    assert list(output.iterdir()) == [output / "keep.txt"]


def test_source_entrypoint_show_and_restore_via_stdin_without_device(capture, tmp_path):
    command = [sys.executable, "-I", str(SCRIPT)]
    output = tmp_path / "export"
    subprocess.run(
        [*command, "export", str(capture[0]), "--output-dir", str(output), "--ranks", "16"],
        capture_output=True,
        check=True,
    )
    index = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(index["parts"]) == 1 and index["ranks"] == [16]
    shown = subprocess.run([*command, "show", str(output), "--part", "1"], capture_output=True, check=True)
    assert shown.stdout.decode("ascii").replace("\r\n", "\n") == transfer.show_part(output, 1)
    assert not shown.stderr and len(shown.stdout) <= transfer.DEFAULT_MAX_BYTES
    inspected = subprocess.run([*command, "inspect", "-"], input=shown.stdout, capture_output=True, check=True)
    assert json.loads(inspected.stdout)["missing"] == []
    restored = tmp_path / "stdin-restored"
    subprocess.run(
        [*command, "restore", "-", "--output-dir", str(restored), "--no-render"],
        input=shown.stdout,
        capture_output=True,
        check=True,
    )
    assert (restored / "timer/rank_16_npu_time_parsed.csv").read_bytes() == (
        capture[1] / "rank_16_npu_time_parsed.csv"
    ).read_bytes()


@pytest.mark.parametrize("limit", [0, 1023, 65536])
def test_invalid_clipboard_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="1024 and 65535"):
        transfer.make_parts({}, limit)
