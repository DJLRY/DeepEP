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


def test_dirty_tracked_sources_stop_before_snapshot(monkeypatch):
    monkeypatch.setattr(ab, "git", lambda *args: b" M kernel.hpp\n")
    with pytest.raises(RuntimeError, match="local edits"):
        ab.source_plan(ROOT)


@pytest.mark.parametrize("fail_at", [None, "perf-B", "build-C", "restore-1"])
def test_full_two_host_experiment(tmp_path, peers, monkeypatch, fail_at):
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
            ref: {path: f"{ref}:{path}".encode() for path in module.HEADERS} for ref in (module.BASE, module.CHANGED)
        }
        plan = dict(
            head="same-head",
            variants={
                label: {path: module.digest(blobs[ref][path]) for ref, path in zip(refs, module.HEADERS)}
                for label, refs in module.VARIANTS.items()
            },
        )
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(module, "source_plan", lambda root, p=plan, b=blobs: (p, b))
        monkeypatch.setattr(module, "git", lambda *args: b"")

        def snapshot(root, tree, head):
            (tree / module.HEADERS[1]).parent.mkdir(parents=True)

        monkeypatch.setattr(module, "snapshot", snapshot)
        installed = {}

        def run(command, env, log, *, peer=None, timeout=7200, cwd="/tmp", n=node, state=installed):
            command = list(map(str, command))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("simulated process\n")
            if peer:
                peer.check()
            if command[1].endswith("build.py"):
                label = Path(cwd).name
                assert (Path(cwd) / ab.HEADERS[0]).read_bytes().startswith(ab.VARIANTS[label][0].encode())
                assert (Path(cwd) / ab.HEADERS[1]).read_bytes().startswith(ab.VARIANTS[label][1].encode())
                if n == 1 and fail_at == f"build-{label}":
                    raise RuntimeError("simulated build failure")
                info = dict(source_sha256=f"sha-{label}", megamoe=dict(enabled=True, timer=False))
                wheel = Path(cwd) / "dist/build-unique/ascend_deepep-test.whl"
                wheel.parent.mkdir(parents=True)
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("deep_ep/_build_info.json", json.dumps(info))
                state["info"] = info
                kind = f"build-{label}"
            elif "pip" in command:
                if n == 1 and fail_at == "restore-1" and log.parent.name == "restore_D":
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
        launchers.append(
            SimpleNamespace(
                __file__=str(launcher_path),
                CASE=ab.load_launcher().CASE,
                environment=lambda _: {"PYTHON": sys.executable},
            )
        )
        directories.append(directory)
    args = SimpleNamespace(cycles=1, warmup=2, iters=4, stage_timeout=3)
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(modules[n].experiment, args, configurations[n], launchers[n], peers[n], directories[n])
            for n in range(2)
        ]
        for task in tasks:
            if fail_at is None:
                task.result(timeout=20)
            else:
                with pytest.raises(RuntimeError):
                    task.result(timeout=20)
    for node, directory in enumerate(directories):
        status = json.loads((directory / "status.json").read_text())
        assert status["complete"] is (fail_at is None)
        assert json.loads((directory / "comparison.json").read_text())["complete"] is (fail_at is None)
        assert status["installed_after_run"] == (
            "not verified" if fail_at == "restore-1" and node == 1 else "D (timer OFF)"
        )
        with zipfile.ZipFile(directory / "results.zip") as archive:
            assert "plan.json" in archive.namelist()
            assert "summary.txt" in archive.namelist()
            assert not any(name.startswith("sources/") for name in archive.namelist())
        if fail_at is None:
            records = json.loads((directory / "results.json").read_text())
            assert "".join(row["variant"] for row in records) == "ABCDDCBA"
            local_calls = [name for n, name in calls if n == node]
            assert [name for name in local_calls if name.startswith("accuracy-")] == [f"accuracy-{v}" for v in "ABCD"]
            assert local_calls[-1] == "install-D"
    if fail_at is None:
        for label in "ABCD":
            first_perf = next(i for i, (_, name) in enumerate(calls) if name == f"perf-{label}")
            for node in range(2):
                assert (node, f"accuracy-{label}") in calls[:first_perf]
                assert (node, f"install-{label}") in calls[:first_perf]


def test_cli_help_isolated():
    result = subprocess.run([sys.executable, "-I", SCRIPT, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--control-port" in result.stdout
    assert "six runs per variant" in result.stdout
