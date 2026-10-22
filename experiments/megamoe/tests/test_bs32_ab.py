# Copyright (c) 2026, Lu Lu

"""Exercise two real control connections with simulated builds/device reports."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/megamoe/scripts/bs32_ab.py"


def load():
    spec = importlib.util.spec_from_file_location("bs32_ab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = load()
CANN_PREFIX = "/home/z00880305/CANN_9_2_B100/cann-9.2.0"
CMAKE_PREFIX = f"{CANN_PREFIX}/toolkit/tools/tikicpulib/lib/cmake:{CANN_PREFIX}/lib64/cmake"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"{CMAKE_PREFIX}:{CMAKE_PREFIX}", CMAKE_PREFIX),
        ("/second:/first:/second", "/second:/first"),
        ("/path with spaces:/other:/path with spaces", "/path with spaces:/other"),
        ("/sdk:/sdk/", "/sdk:/sdk/"),
        (":/sdk::/sdk:", ":/sdk::"),
        ("", ""),
    ],
)
def test_cmake_prefix_dedup_preserves_search_order_and_other_spelling(value, expected):
    assert ab.normalize_cmake_prefix_path(value) == expected
    assert ab.normalize_cmake_prefix_path(expected) == expected


@pytest.fixture
def peers():
    # Use TCP, not socketpair, to cover SO_KEEPALIVE on Windows too.
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(server.getsockname())
        accepted, _ = server.accept()
    pair = ab.Peer(accepted, 3), ab.Peer(client, 3)
    try:
        yield pair
    finally:
        for peer in pair:
            peer.close()


def report_for(label="A", iters=4):
    spec = importlib.util.spec_from_file_location(
        "performance", ROOT / "experiments/megamoe/python/deep_ep_experimental/megamoe/performance.py"
    )
    performance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(performance)
    value = {"A": 300, "B": 310, "C": 298, "D": 309}[label]
    return performance.summarize_rounds([[value + rank + i / 10 for i in range(iters)] for rank in range(32)], 16)


def test_real_control_barrier_waits_for_peer_and_rejects_identity(peers):
    first, second = peers
    with ThreadPoolExecutor(2) as pool:
        task = pool.submit(first.exchange, "build-A", {"source": "A"}, same=True)
        time.sleep(0.1)
        assert not task.done()
        assert second.exchange("build-A", {"source": "A"}, same=True) == {"source": "A"}
        task.result(timeout=2)
        task = pool.submit(first.exchange, "install-B", "A", same=True)
        with pytest.raises(RuntimeError, match="disagree"):
            second.exchange("install-B", "B", same=True)
        with pytest.raises(RuntimeError, match="disagree"):
            task.result(timeout=2)


def test_peer_failure_stops_running_child_and_preserves_log(tmp_path, peers):
    log = tmp_path / "child.log"
    first, second = peers
    command = [sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(30)"]
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(ab.run_process, command, dict(os.environ), log, peer=first, cwd=tmp_path, timeout=10)
        deadline = time.monotonic() + 5
        while not log.exists() or "started" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        second.abort("compile failed")
        with pytest.raises(RuntimeError, match="compile failed"):
            task.result(timeout=3)
    assert "started" in log.read_text()


def test_disconnect_and_barrier_timeout_are_failures(peers):
    peers[0].timeout = 0.1
    with pytest.raises(TimeoutError, match="waiting"):
        peers[0].exchange("no-peer")
    peers[1].close()
    deadline = time.monotonic() + 2
    while not peers[0].failure:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    with pytest.raises(RuntimeError, match="disconnected"):
        peers[0].check()


def test_process_timeout(tmp_path):
    with pytest.raises(TimeoutError, match="exceeded"):
        ab.run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            dict(os.environ),
            tmp_path / "timeout.log",
            timeout=0.2,
            cwd=tmp_path,
        )


@pytest.mark.parametrize("first_node", [0, 1])
def test_connect_supports_either_launch_order(first_node):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    connections = []
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(ab.Peer.connect, dict(node_rank=first_node, master_addr="127.0.0.1"), port, 4, 2)
        time.sleep(0.1)
        second = pool.submit(ab.Peer.connect, dict(node_rank=1 - first_node, master_addr="127.0.0.1"), port, 4, 2)
        try:
            connections = [first.result(timeout=5), second.result(timeout=5)]
            barrier = pool.submit(connections[0].exchange, "hello", "same", same=True)
            connections[1].exchange("hello", "same", same=True)
            barrier.result(timeout=2)
        finally:
            for peer in connections:
                peer.close()


def test_report_requires_complete_32_rank_tail_half_and_correct_aggregates(tmp_path):
    path = tmp_path / "profiling.json"
    good = report_for(iters=5)
    ab.write_json(path, good)
    assert ab.read_performance(path, 5) == good
    bad_reports = [
        dict(good, timer_pass=True),
        dict(good, recorded_rounds=6),
        dict(good, per_rank_rounds_us=good["per_rank_rounds_us"][:-1]),
        dict(good, mean_us=999),
        dict(good, rank_max_mean_us=999),
        dict(good, scope="local_ranks"),
        dict(good, first_selected_iteration=0),
    ]
    bad = json.loads(json.dumps(good))
    bad["per_rank_rounds_us"][0][0] = float("nan")
    bad_reports.append(bad)
    for report in bad_reports:
        path.write_text(json.dumps(report))
        with pytest.raises(RuntimeError):
            ab.read_performance(path, 5)


def test_summary_uses_independent_runs_and_paired_blocks(tmp_path):
    records = []
    for index, label in enumerate("ABCDDCBA" * 3):
        value = {"A": 300, "B": 310, "C": 298, "D": 309}[label] + (index // 4) * 10
        records.append(
            dict(
                step=index + 1,
                block=index // 4,
                variant=label,
                mean_us=value,
                rank_max_mean_us=value + 30,
                source_sha256=f"sha-{label}",
                report=f"run_{index}/profiling.json",
            )
        )
    summary = ab.summarize(records, 3)
    assert summary["complete"]
    assert summary["variants"]["A"]["mean_us"]["n"] == 6
    for name, expected in (("B-A", 10), ("C-A", -2), ("D-A", 9), ("D-B-C+A", 1)):
        assert summary["paired_deltas"][name]["mean_us"]["values"] == [expected] * 6
    assert not ab.summarize(records[:-1], 3)["complete"]
    ab.save_results(tmp_path, records, 3)
    assert len((tmp_path / "results.csv").read_text().splitlines()) == 25


def test_archive_uses_committed_tree_and_does_not_copy_local_outputs(tmp_path):
    # Read the real repository; exporting does not touch its branch or working tree.
    destination = tmp_path / "source"
    head = ab.git(ROOT, "rev-parse", "HEAD").decode().strip()
    ab.snapshot(ROOT, destination, head)
    assert (destination / ab.HEADERS[0]).read_bytes() == ab.git(ROOT, "show", f"{head}:{ab.HEADERS[0]}")
    assert not (destination / ".git").exists()
    assert not (destination / "benchmark-runs").exists()
    assert not (destination / ".tmp-roce-checks").exists()
    assert (destination / "build_support.py").is_file()


@pytest.mark.parametrize("cycles,order", [(1, "AB"), (2, "ABBA"), (6, "ABBA" * 3)])
def test_dispatch_summary_pairs_forward_and_reverse_runs(tmp_path, cycles, order):
    assert ab.measurement_order("dispatch", cycles) == order
    records = []
    for index, label in enumerate(order):
        value = {"A": 300, "B": 280}[label] + (index // 2) * 5
        records.append(
            dict(
                step=index + 1,
                block=index // 2,
                variant=label,
                mean_us=value,
                rank_max_mean_us=value + 15,
                source_sha256=f"sha-{label}",
                report=f"run_{index}/profiling.json",
            )
        )
    summary = ab.summarize(records, cycles, experiment="dispatch")
    assert summary["complete"] and set(summary["variants"]) == {"A", "B"}
    assert summary["variants"]["B"]["mean_us"]["n"] == cycles
    assert set(summary["paired_deltas"]) == {"B-A"}
    assert summary["paired_deltas"]["B-A"]["mean_us"]["values"] == [-20] * cycles
    assert not ab.summarize(records[:-1], cycles, experiment="dispatch")["complete"]
    text = ab.save_results(tmp_path, records, cycles, experiment="dispatch")
    assert f"{len(order)}/{len(order)} perf runs" in text and "one Dispatch commit" in text
    assert ("one run per variant" in text) is (cycles == 1)
    assert "scan early stop" not in text and "D-B-C+A" not in text


def test_dirty_tracked_sources_stop_before_snapshot(monkeypatch):
    monkeypatch.setattr(ab, "git", lambda *args: b" M kernel.hpp\n")
    with pytest.raises(RuntimeError, match="local edits"):
        ab.source_plan(ROOT)


def unpack_text(parts):
    bodies, report_hash = [], None
    for index, raw in enumerate(parts, 1):
        # Clipboard conversion must not change the payload identity.
        text = raw.replace("\r\n", "\n")
        header, body = text.split("\n\n", 1)
        assert header.startswith("-----BEGIN BS32 AB TEXT v1-----\n")
        fields = dict(line.split("=", 1) for line in header.splitlines()[1:])
        assert fields["part"] == f"{index:06d}/{len(parts):06d}"
        assert body.endswith("-----END BS32 AB TEXT-----\n")
        body = body.removesuffix("-----END BS32 AB TEXT-----\n")
        assert ab.digest(body.encode("utf-8")) == fields["payload_sha256"]
        if report_hash is not None:
            assert report_hash == fields["report_sha256"]
        report_hash = fields["report_sha256"]
        bodies.append(body)
    payload = "".join(bodies)
    assert ab.digest(payload.encode("utf-8")) == report_hash
    return payload


def test_text_limit_includes_headers_unicode_crlf_and_exact_boundary():
    payload = "结果检查 αβγ\n" * 3000
    parts = ab.text_parts(payload, 0, max_bytes=1024)
    assert len(parts) > 1
    assert all(len(part.replace("\n", "\r\n").encode("utf-8")) <= 1024 for part in parts)
    assert unpack_text([part.replace("\n", "\r\n") for part in parts]) == payload
    overhead = len(ab.text_parts("", 0, max_bytes=1024)[0].replace("\n", "\r\n").encode("utf-8"))
    exact = "a" * (1024 - overhead - 2) + "\n"
    assert len(ab.text_parts(exact, 0, max_bytes=1024)[0].replace("\n", "\r\n").encode()) == 1024
    with pytest.raises(ValueError, match="record exceeds"):
        ab.text_parts("a" + exact, 0, max_bytes=1024)
    with pytest.raises(ValueError, match="60000"):
        ab.text_parts("test\n", 0, max_bytes=65535)


@pytest.fixture
def completed_ab_results(tmp_path):
    directory = tmp_path / "ab_node_0_existing"
    directory.mkdir()
    ab.write_json(
        directory / "plan.json",
        dict(
            source=dict(head="a" * 40, baseline="b" * 40, changed="c" * 40),
            case=dict(ab.load_launcher().CASE, warmup=20, iters=100),
            order="ABCDDCBA" * 3,
            backend="HNS_1825",
            controller="d" * 64,
            launcher="e" * 64,
        ),
    )
    ab.write_json(directory / "host.json", dict(node_rank=0, hostname="test-node-0"))
    ab.write_json(
        directory / "status.json",
        dict(complete=True, error=None, restore_error=None, installed_after_run="D (timer OFF)"),
    )
    records = []
    for index, label in enumerate("ABCDDCBA" * 3):
        report = report_for(label, iters=100)
        path = directory / f"run_{index}"
        path.mkdir()
        ab.write_json(path / "profiling.json", report)
        records.append(
            dict(
                step=index + 1,
                block=index // 4,
                variant=label,
                source_sha256=label.lower() * 64,
                mean_us=report["mean_us"],
                rank_max_mean_us=report["rank_max_mean_us"],
                report=f"run_{index}/profiling.json",
            )
        )
    ab.save_results(directory, records, 3)
    return directory


def test_default_24_runs_fit_one_readable_text_block_and_reexport_is_offline(completed_ab_results, capsys):
    directory = completed_ab_results
    originals = {path: path.read_bytes() for path in directory.rglob("*.json")}
    paths = ab.export_text(directory)
    assert len(paths) == 1
    text = paths[0].read_text(encoding="utf-8")
    assert len(text.replace("\n", "\r\n").encode("utf-8")) <= 60000 < 65535
    payload = unpack_text([text])
    records = [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1])) for line in payload.splitlines()]
    runs = [row for kind, row in records if kind == "RUN"]
    assert "".join(row["variant"] for row in runs) == "ABCDDCBA" * 3
    assert all(len(row["rank_mean_us"]) == 32 for row in runs)
    assert all(row["slowest_mean_rank"] == 31 for row in runs)
    assert len([row for kind, row in records if kind == "DELTA"]) == 8
    assert "cat " in capsys.readouterr().out
    result = subprocess.run([sys.executable, "-I", SCRIPT, "--export-text", directory], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Copy this block: cat " in result.stdout
    assert len(list(directory.glob("copy_text_*/part_001.txt"))) == 2
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_failed_text_is_bounded_and_keeps_status_log_excerpt_and_missing_evidence(completed_ab_results):
    directory = completed_ab_results
    error = "错误\n" * 20000
    ab.write_json(directory / "status.json", dict(complete=False, error=error, restore_error=None))
    log = directory / "builds/A/build.log"
    log.parent.mkdir(parents=True)
    log.write_text("lots of output\n" * 5000 + "[ERROR] compiler failed to build kernel\n", encoding="utf-8")
    (directory / "run_0/profiling.json").unlink()
    paths = ab.export_text(directory)
    assert len(paths) == 1
    payload = unpack_text([path.read_text(encoding="utf-8") for path in paths])
    entries = [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1])) for line in payload.splitlines()]
    status = next(value for kind, value in entries if kind == "STATUS")
    assert status["complete"] is False
    assert "truncated" in status["error"]
    assert ab.digest(error.encode("utf-8")) in status["error"]
    assert any("compiler failed" in row["excerpt"] for kind, row in entries if kind == "ERROR_LOG")
    first = next(row for kind, row in entries if kind == "RUN")
    assert first["step"] == 1 and "rank_detail_error" in first
    assert all(len(path.read_bytes().replace(b"\n", b"\r\n")) < 65535 for path in paths)


@pytest.mark.parametrize(
    "fail_at", [None, "perf-B", "build-last", "restore-1", "settings-order", "settings-flag", "source-mismatch"]
)
@pytest.mark.parametrize("experiment", ["scheduling", "dispatch"])
def test_full_two_host_experiment(tmp_path, peers, monkeypatch, fail_at, experiment):
    spec = ab.EXPERIMENTS[experiment]
    succeeds = fail_at in {None, "settings-order", "settings-flag"}
    calls = []
    modules = []
    configurations = []
    launchers = []
    directories = []
    # Windows test runners need not have permission to create symlinks. The
    # simulated Golden runner never reads this link; real servers are Linux.
    if os.name != "posix":
        monkeypatch.setattr(Path, "symlink_to", lambda *a, **kw: None)
    for node in range(2):
        module = load()
        root = tmp_path / f"checkout_{node}"
        root.mkdir()
        (root / "third-party/shmem/install/shmem").mkdir(parents=True)
        (root / "third-party/catlass").mkdir(parents=True)
        launcher_path = root / "bs32.py"
        launcher_path.write_text("# same launcher\n")
        config = dict(
            node_rank=node,
            case_dir=str(tmp_path / f"output_{node}"),
            rdma_backend="HNS_1825",
            master_addr="127.0.0.1",
            master_port=29654,
            shmem_port=26733,
        )
        directory = Path(config["case_dir"]) / "ab"
        directory.mkdir(parents=True)
        blobs = {
            ref: {path: f"{ref}:{path}".encode() for path in spec["headers"]}
            for ref in (spec["baseline"], spec["changed"])
        }
        plan = dict(
            head="same-head",
            variants={
                label: {path: module.digest(blobs[ref][path]) for ref, path in zip(refs, spec["headers"])}
                for label, refs in spec["variants"].items()
            },
        )
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(module, "source_plan", lambda root, p=plan, b=blobs, **kw: (p, b))
        monkeypatch.setattr(module, "git", lambda *args: b"")

        def snapshot(root, tree, head):
            for header in spec["headers"]:
                (tree / header).parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(module, "snapshot", snapshot)
        installed = {}

        def run(command, env, log, *, peer=None, timeout=7200, cwd="/tmp", n=node, state=installed):
            command = list(map(str, command))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("simulated process\n")
            if peer:
                peer.check()
            if command[1].endswith("build.py"):
                expected_prefix = (
                    ":".join(reversed(CMAKE_PREFIX.split(":")))
                    if n == 1 and fail_at == "settings-order"
                    else CMAKE_PREFIX
                )
                assert env["CMAKE_PREFIX_PATH"] == expected_prefix
                if n == 1 and fail_at == "settings-flag":
                    assert env["CXXFLAGS"] == "-O0"
                label = Path(cwd).name
                for header, ref in zip(spec["headers"], spec["variants"][label]):
                    assert (Path(cwd) / header).read_bytes() == f"{ref}:{header}".encode()
                if n == 1 and fail_at == "build-last" and label == spec["build_order"][-1]:
                    raise RuntimeError("simulated build failure")
                info = dict(
                    source_sha256=f"sha-{label}",
                    megamoe=dict(enabled=True, timer=False),
                    cmake_version=["cmake version 3.25.1", "cmake version 3.31.6"][n],
                    shmem=dict(sha256={"lib/libshmem.so": f"host-{n}"}),
                    build_environment=dict(CXXFLAGS=env.get("CXXFLAGS", ""), CMAKE_PREFIX_PATH=expected_prefix),
                )
                if n == 1 and fail_at == "source-mismatch":
                    info["source_sha256"] += "-wrong"
                wheel = Path(cwd) / "dist/build-unique/ascend_deepep-test.whl"
                wheel.parent.mkdir(parents=True)
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("deep_ep/_build_info.json", json.dumps(info))
                state["info"] = info
                kind = f"build-{label}"
            elif "pip" in command:
                if n == 1 and fail_at == "restore-1" and log.parent.name == f"restore_{spec['restore']}":
                    raise RuntimeError("simulated restore failure")
                state["info"] = ab.wheel_metadata(command[-1])
                kind = "install-" + state["info"]["source_sha256"][-1]
            elif "-c" in command:
                ab.write_json(command[-1], state["info"])
                return
            else:
                cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
                step = Path(cfg["case_dir"])
                label = step.name[-1]
                kind = ("perf-" if "perf" in command else "accuracy-") + label
                if "perf" in command:
                    if n == 1 and fail_at == f"perf-{label}":
                        raise RuntimeError("simulated profiler failure")
                    output = step / f"perf_node_{n}_unique"
                    for rank in range(n * 16, n * 16 + 16):
                        rank_dir = output / f"rank_{rank}"
                        (rank_dir / "bs_32").mkdir(parents=True)
                        ab.write_json(
                            rank_dir / "manifest.json", dict(build=state["info"], mode="perf", timer_ranks=[0])
                        )
                        ab.write_json(rank_dir / "bs_32/profiling.json", report_for(label))
                        ab.write_json(rank_dir / "bs_32/kernel_samples.json", [300, 301, 302, 303])
            calls.append((n, kind))

        monkeypatch.setattr(module, "run_process", run)
        modules.append(module)
        configurations.append(config)
        build_env = dict(
            PYTHON=sys.executable, CMAKE_PREFIX_PATH=f"{CMAKE_PREFIX}:{CMAKE_PREFIX}" if node == 0 else CMAKE_PREFIX
        )
        if node == 1 and fail_at == "settings-order":
            build_env["CMAKE_PREFIX_PATH"] = ":".join(reversed(CMAKE_PREFIX.split(":")))
        if node == 1 and fail_at == "settings-flag":
            build_env["CXXFLAGS"] = "-O0"
        launchers.append(
            SimpleNamespace(
                __file__=str(launcher_path),
                CASE=ab.load_launcher().CASE,
                environment=lambda _, env=build_env: dict(env),
            )
        )
        directories.append(directory)
    args = SimpleNamespace(cycles=1, warmup=2, iters=4, stage_timeout=3, experiment=experiment)
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(modules[n].experiment, args, configurations[n], launchers[n], peers[n], directories[n])
            for n in range(2)
        ]
        for task in tasks:
            if succeeds:
                task.result(timeout=20)
            else:
                with pytest.raises(RuntimeError):
                    task.result(timeout=20)
    for node, directory in enumerate(directories):
        status = json.loads((directory / "status.json").read_text())
        assert status["complete"] is succeeds
        assert json.loads((directory / "comparison.json").read_text())["complete"] is succeeds
        raw_settings = json.loads((directory / "build_settings_raw.json").read_text())
        settings = json.loads((directory / "build_settings.json").read_text())
        if node == 0:
            assert raw_settings["CMAKE_PREFIX_PATH"] == f"{CMAKE_PREFIX}:{CMAKE_PREFIX}"
            assert settings["CMAKE_PREFIX_PATH"] == CMAKE_PREFIX
        assert status["installed_after_run"] == (
            "not verified" if fail_at == "restore-1" and node == 1 else f"{spec['restore']} (timer OFF)"
        )
        with zipfile.ZipFile(directory / "results.zip") as archive:
            assert "plan.json" in archive.namelist()
            assert "summary.txt" in archive.namelist()
            assert not any(name.startswith("sources/") for name in archive.namelist())
        parts = sorted(directory.glob("copy_text_*/part_*.txt"))
        assert len(parts) == 1
        payload = unpack_text([path.read_text(encoding="utf-8") for path in parts])
        assert '"complete":' + ("true" if succeeds else "false") in payload
        entries = [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1])) for line in payload.splitlines()]
        meta = next(value for kind, value in entries if kind == "META")
        assert meta["experiment"] == experiment
        assert meta["order_per_cycle"] == spec["order"]
        variants = [value for kind, value in entries if kind == "VARIANT"]
        assert [value["label"] for value in variants] == list(spec["variants"])
        if experiment == "dispatch":
            assert [value["dispatch_commits"] for value in variants] == [2, 1]
            assert all(not value["activation_priority"] and not value["scan_early_stop"] for value in variants)
        if succeeds:
            build_info = json.loads((directory / f"builds/{spec['restore']}/wheel.json").read_text())["build"]
            assert build_info["cmake_version"] == ["cmake version 3.25.1", "cmake version 3.31.6"][node]
            records = json.loads((directory / "results.json").read_text())
            assert "".join(row["variant"] for row in records) == spec["order"]
            local_calls = [name for n, name in calls if n == node]
            assert [name for name in local_calls if name.startswith("accuracy-")] == [
                f"accuracy-{v}" for v in spec["variants"]
            ]
            assert local_calls[-1] == f"install-{spec['restore']}"
    if succeeds:
        for label in spec["variants"]:
            first_perf = next(i for i, (_, name) in enumerate(calls) if name == f"perf-{label}")
            for node in range(2):
                assert (node, f"accuracy-{label}") in calls[:first_perf]
                assert (node, f"install-{label}") in calls[:first_perf]
    if fail_at == "source-mismatch":
        assert not any(name.startswith(("accuracy-", "perf-")) for _, name in calls)


def test_cli_help_isolated():
    result = subprocess.run([sys.executable, "-I", SCRIPT, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--control-port" in result.stdout
    assert "Dispatch: AB pairs" in " ".join(result.stdout.split())
    assert "--experiment {scheduling,dispatch}" in result.stdout
