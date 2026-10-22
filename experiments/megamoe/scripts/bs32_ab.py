# Copyright (c) 2026, Lu Lu

"""Run the four BS32 regression variants; launch this command once on each host."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = "732bb3f"
CHANGED = "bfb6fe8"
KERNEL = "experiments/megamoe/csrc/kernels/mega_moe/"
HEADERS = (KERNEL + "mega_moe_wave_a8w4.hpp", KERNEL + "stage/mega_moe_hierarchical_combine.hpp")
VARIANTS = {"A": (BASE, BASE), "B": (CHANGED, BASE), "C": (BASE, CHANGED), "D": (CHANGED, CHANGED)}
ORDER = "ABCDDCBA"


def load_launcher():
    spec = importlib.util.spec_from_file_location("bs32", Path(__file__).with_name("bs32.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def git(root, *args):
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    return subprocess.check_output(["git", "-C", str(root), *args], env=env, stderr=subprocess.PIPE)


def source_plan(root):
    # Ignore unrelated untracked reports, but never silently omit tracked edits.
    if git(root, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all").strip():
        raise RuntimeError("Tracked files have local edits; commit them before the A/B source snapshot")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    changed_paths = set(
        git(root, "diff", "--name-only", BASE, CHANGED, "--", "csrc", "experiments/megamoe/csrc").decode().splitlines()
    )
    if changed_paths != set(HEADERS):
        raise RuntimeError("The reference commits no longer identify exactly the two expected kernel changes")
    if git(root, "diff", CHANGED, head, "--", "csrc", "experiments/megamoe/csrc").strip():
        raise RuntimeError("Current native sources differ from bfb6fe8; this experiment requires its kernel baseline")
    blobs = {ref: {path: git(root, "show", f"{ref}:{path}") for path in HEADERS} for ref in (BASE, CHANGED)}
    variants = {
        label: {path: digest(blobs[ref][path]) for path, ref in zip(HEADERS, refs)} for label, refs in VARIANTS.items()
    }
    return dict(
        head=head,
        baseline=git(root, "rev-parse", BASE).decode().strip(),
        changed=git(root, "rev-parse", CHANGED).decode().strip(),
        variants=variants,
    ), blobs


def snapshot(root, destination, head):
    """Export a committed tree, excluding submodule contents and all local outputs."""
    destination.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "-C", str(root), "archive", head], stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            # Git trees can contain symlinks. These are unnecessary for this build;
            # refuse them instead of extracting links outside the source snapshot.
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"Unsupported archive member: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as output:
                        output.write(source.read())
                    target.chmod(member.mode & 0o777)


class Peer:
    """Persistent JSON connection: barriers, identity checks and prompt failure propagation."""

    def __init__(self, connection, timeout):
        self.connection = connection
        self.timeout = timeout
        self.messages = queue.Queue()
        self.failure = None
        self.closed = False
        self.writer = threading.Lock()
        connection.settimeout(None)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    @classmethod
    def connect(cls, config, port, connect_timeout, stage_timeout):
        address = (config["master_addr"], port)
        print(f"[AB] Waiting for node {1 - config['node_rank']} at {address[0]}:{port}", flush=True)
        if config["node_rank"] == 0:
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(address)
                server.listen(1)
                server.settimeout(connect_timeout)
                connection, _ = server.accept()
        else:
            deadline = time.monotonic() + connect_timeout
            while True:
                try:
                    connection = socket.create_connection(
                        address, timeout=min(5, max(0.1, deadline - time.monotonic()))
                    )
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Node 0 did not listen on {address} within {connect_timeout}s") from None
                    time.sleep(1)
        return cls(connection, stage_timeout)

    def _read(self):
        try:
            with self.connection.makefile("rb") as stream:
                while not self.closed:
                    line = stream.readline(2 * 1024 * 1024 + 1)
                    if not line:
                        raise RuntimeError("Peer disconnected")
                    if len(line) > 2 * 1024 * 1024:
                        raise RuntimeError("Peer message too large")
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise RuntimeError("Invalid peer message")
                    if "error" in message:
                        raise RuntimeError(f"Peer failed: {message['error']}")
                    self.messages.put(message)
        except (OSError, ValueError, RuntimeError) as error:
            self.failure = str(error)

    def send(self, value):
        with self.writer:
            self.connection.sendall(json.dumps(value, allow_nan=False).encode() + b"\n")

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)

    def exchange(self, stage, payload=None, *, same=False):
        print(f"[AB] Synchronizing: {stage}", flush=True)
        self.send(dict(stage=stage, payload=payload))
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                message = self.messages.get(timeout=0.2)
                break
            except queue.Empty:
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for peer at {stage}")
        if message.get("stage") != stage or (same and message.get("payload") != payload):
            raise RuntimeError(f"Nodes disagree at {stage}: local={payload!r}, peer={message!r}")
        return message.get("payload")

    def abort(self, error):
        try:
            self.send(dict(error=str(error)))
        except OSError:
            pass

    def close(self):
        self.closed = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        self.reader.join(timeout=2)


@contextmanager
def local_lock(root):
    # flock is released by the OS even after SIGKILL. One lock per physical host
    # also permits two hosts sharing a checkout, without permitting local overlap.
    import fcntl

    path = root / ".megamoe-bs32" / f"{socket.gethostname()}.ab.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another BS32 A/B experiment is running on this host") from None
        yield


def stop_process(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            process.kill()
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def run_process(command, env, log, *, peer=None, timeout=7200, cwd="/tmp"):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[AB] Running {log.name}; full log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            env=env,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        start = last_notice = time.monotonic()
        try:
            while process.poll() is None:
                if peer:
                    peer.check()
                now = time.monotonic()
                if now - start > timeout:
                    raise TimeoutError(f"Stage exceeded {timeout}s: {log}")
                if now - last_notice >= 60:
                    print(f"[AB] Still running {log.name} ({now - start:.0f}s); log: {log}", flush=True)
                    last_notice = now
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"Command exited {process.returncode}: {log}")
        except BaseException:
            stop_process(process)
            raise


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("deep_ep/_build_info.json"))


def install_wheel(wheel, expected, env, directory, peer, timeout):
    run_process(
        [env["PYTHON"], "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel],
        env,
        directory / "install.log",
        peer=peer,
        timeout=timeout,
    )
    output = directory / "installed.json"
    probe = """import json, sys
from importlib.resources import files
import torch, torch_npu
from deep_ep._native import require_native
import deep_ep_experimental.megamoe._C as megamoe
require_native('BS32 A/B')
info = json.loads(files('deep_ep').joinpath('_build_info.json').read_text())
if megamoe.timer_enabled or info['megamoe']['timer']:
    raise RuntimeError('A/B requires a timer-OFF wheel')
with open(sys.argv[1], 'w') as stream:
    json.dump(info, stream)
"""
    run_process([env["PYTHON"], "-I", "-c", probe, output], env, directory / "verify.log", peer=peer, timeout=timeout)
    if json.loads(output.read_text()) != expected:
        raise RuntimeError(f"Installed package differs from the selected wheel: {directory}")


def read_performance(path, iters):
    report = json.loads(path.read_text())
    rows = report.get("per_rank_rounds_us", [])
    rounds = (iters + 1) // 2
    if (
        report.get("metric") != "megamoe_kernel_duration_us"
        or report.get("scope") != "all_ep_ranks"
        or report.get("timer_pass") is not False
        or report.get("recorded_rounds") != iters
        or report.get("rounds") != rounds
        or report.get("first_selected_iteration") != iters // 2
        or report.get("iteration_selection") != "tail_half_ceil"
        or len(rows) != 32
        or any(len(row) != rounds for row in rows)
        or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for row in rows for x in row)
    ):
        raise RuntimeError(f"Invalid EP32 timer-off tail-half report: {path}")
    for key, reducer in (("mean_us", min), ("rank_max_mean_us", max)):
        expected = statistics.mean(reducer(column) for column in zip(*rows))
        if not math.isclose(report.get(key, -1), expected, rel_tol=1e-9, abs_tol=1e-6):
            raise RuntimeError(f"Inconsistent {key}: {path}")
    return report


def summarize(records, cycles):
    metrics = ("mean_us", "rank_max_mean_us")
    summary = {"complete": len(records) == 8 * cycles, "variants": {}, "paired_deltas": {}}
    for label in VARIANTS:
        rows = [row for row in records if row["variant"] == label]
        if rows:
            summary["variants"][label] = {
                key: dict(
                    n=len(rows),
                    median=statistics.median(row[key] for row in rows),
                    min=min(row[key] for row in rows),
                    max=max(row[key] for row in rows),
                )
                for key in metrics
            }
    # Each forward/reverse half is one block containing exactly A/B/C/D.
    # Never pair the two hosts' duplicate all-rank reports as independent runs.
    for name, coefficients in {
        "B-A": {"B": 1, "A": -1},
        "C-A": {"C": 1, "A": -1},
        "D-A": {"D": 1, "A": -1},
        "D-B-C+A": {"D": 1, "B": -1, "C": -1, "A": 1},
    }.items():
        deltas = {key: [] for key in metrics}
        for block in sorted({row["block"] for row in records}):
            group = {row["variant"]: row for row in records if row["block"] == block}
            if set(group) == set(VARIANTS):
                for key in metrics:
                    deltas[key].append(sum(group[label][key] * factor for label, factor in coefficients.items()))
        if deltas[metrics[0]]:
            summary["paired_deltas"][name] = {
                key: dict(
                    values=values, median=statistics.median(values), positive=sum(x > 0 for x in values), n=len(values)
                )
                for key, values in deltas.items()
            }
    return summary


def save_results(directory, records, cycles, *, error=None):
    write_json(directory / "results.json", records)
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["step", "block", "variant", "mean_us", "rank_max_mean_us", "source_sha256", "report"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    summary = summarize(records, cycles)
    summary["error"] = error
    if error is not None:
        summary["complete"] = False
    write_json(directory / "comparison.json", summary)
    lines = [
        f"BS32 A/B: {'COMPLETE' if summary['complete'] else 'PARTIAL'}; {len(records)}/{cycles * 8} perf runs",
        "EP32 BS32 H3584 I1280 E384 K8 S0; FP8/FP4; balanced; timer OFF",
        "A=baseline; B=activation priority; C=scan early stop; D=both",
        "Kernel Duration only; rank-local clocks; each perf run uses the tail half of its samples.",
        "Variant | runs | rank-min mean median [min,max] us | rank-max mean median [min,max] us",
    ]
    for label, data in summary["variants"].items():
        cells = [f"{v['median']:.3f} [{v['min']:.3f},{v['max']:.3f}]" for v in data.values()]
        lines.append(f"{label} | {data['mean_us']['n']} | " + " | ".join(cells))
    lines.append(
        "Paired block deltas: positive = slower; interaction = D-B-C+A (descriptive, not a significance test)."
    )
    for name, data in summary["paired_deltas"].items():
        cells = [f"{key}: {v['median']:+.3f} us, positive {v['positive']}/{v['n']}" for key, v in data.items()]
        lines.append(name + " | " + " | ".join(cells))
    lines.append("Repeat variation and both metrics must be inspected before attributing a regression.")
    if error is not None:
        lines.append(f"Experiment failed: {error}; inspect status.json and logs.")
    text = "\n".join(lines) + "\n"
    (directory / "summary.txt").write_text(text, encoding="utf-8")
    return text


def package_results(directory):
    output = directory / "results.zip"
    # Keep full raw profiler exports and build trees locally; bundle the evidence
    # needed for attribution, including all-rank timings and local raw samples.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for parent, subdirs, files in os.walk(directory):
            subdirs[:] = sorted(name for name in subdirs if name not in {"sources", "profiler", "golden"})
            for name in sorted(files):
                path = Path(parent) / name
                if path != output and path.suffix in {".json", ".csv", ".txt", ".log", ".diff"}:
                    archive.write(path, path.relative_to(directory).as_posix())
    return output


def experiment(args, config, launcher, peer, directory):
    records = []
    wheels = {}
    env = None
    error = None
    try:
        source, blobs = source_plan(ROOT)
        plan = dict(
            schema_version=1,
            source=source,
            order=ORDER * args.cycles,
            case=dict(launcher.CASE, warmup=args.warmup, iters=args.iters),
            backend=config["rdma_backend"],
            master_addr=config["master_addr"],
            master_port=config["master_port"],
            shmem_port=config["shmem_port"],
            controller=digest(Path(__file__).read_bytes()),
            launcher=digest(Path(launcher.__file__).read_bytes()),
        )
        write_json(directory / "plan.json", plan)
        write_json(directory / "host.json", dict(config, hostname=socket.gethostname(), root=str(ROOT)))
        peer.exchange("plan", plan, same=True)
        env = launcher.environment(config)
        env.update(
            SHMEM_ROOT=str(ROOT / "third-party/shmem/install/shmem"),
            CATLASS_ROOT=str(ROOT / "third-party/catlass"),
            DEEPEP_BUILD_EP_GMM_FUSED="OFF",
            DEEPEP_MEGAMOE_TIMER="OFF",
            PIP_VERBOSE="1",
        )
        flags = {
            name: env.get(name, "")
            for name in (
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "CMAKE_GENERATOR",
                "CMAKE_GENERATOR_PLATFORM",
                "CMAKE_GENERATOR_TOOLSET",
                "CMAKE_PREFIX_PATH",
                "DEEPEP_NPU_ARCH",
                "DEEPEP_LINEINFO",
            )
        }
        write_json(directory / "build_settings.json", flags)
        peer.exchange("build-settings", flags, same=True)
        # Existing validated SDKs are reused via the supported external SDK build
        # path; neither source checkout nor third-party trees are rewritten.
        for name in ("SHMEM_ROOT", "CATLASS_ROOT"):
            if not Path(env[name]).is_dir():
                raise RuntimeError(f"Missing {env[name]}; run the ordinary BS32 build once before A/B")
        catlass_status = git(Path(env["CATLASS_ROOT"]), "status", "--porcelain", "--ignore-submodules=all")
        if catlass_status.strip():
            raise RuntimeError("CATLASS has local changes; use the pinned clean dependency")
        sources = directory / "sources"
        sources.mkdir()
        for label in "DABC":
            peer.exchange(f"build-start-{label}")
            tree = sources / label
            snapshot(ROOT, tree, source["head"])
            for path, ref in zip(HEADERS, VARIANTS[label]):
                (tree / path).write_bytes(blobs[ref][path])
            build_dir = directory / "builds" / label
            build_dir.mkdir(parents=True)
            write_json(build_dir / "headers.json", source["variants"][label])
            run_process(
                [
                    env["PYTHON"],
                    tree / "experiments/megamoe/scripts/build.py",
                    "--rdma-backend",
                    config["rdma_backend"],
                ],
                env,
                build_dir / "build.log",
                peer=peer,
                timeout=args.stage_timeout,
                cwd=tree,
            )
            candidates = list((tree / "dist").glob("build-*/ascend_deepep-*.whl"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one freshly built wheel for {label}: {candidates}")
            wheel = candidates[0]
            info = wheel_metadata(wheel)
            if info["megamoe"]["timer"] or not info["megamoe"]["enabled"]:
                raise RuntimeError("Build did not produce a timer-OFF MegaMoE wheel")
            wheels[label] = (wheel, info)
            write_json(build_dir / "wheel.json", dict(path=str(wheel), sha256=digest(wheel.read_bytes()), build=info))
            peer.exchange(f"build-done-{label}", info, same=True)
        if len({info["source_sha256"] for _, info in wheels.values()}) != 4:
            raise RuntimeError("Four variants did not produce four distinct source fingerprints")
        checked = set()
        installed = None
        for index, label in enumerate(plan["order"]):
            step = f"{index + 1:02d}_{label}"
            print(f"[AB] Measurement {index + 1}/{len(plan['order'])}: {label}", flush=True)
            step_dir = directory / "runs" / step
            step_dir.mkdir(parents=True)
            golden = Path(config["case_dir"]) / "golden"
            golden.mkdir(parents=True, exist_ok=True)
            (step_dir / "golden").symlink_to(golden, target_is_directory=True)
            wheel, info = wheels[label]
            peer.exchange(f"select-{step}", dict(variant=label, source_sha256=info["source_sha256"]), same=True)
            if installed != label:
                install_wheel(wheel, info, env, step_dir, peer, args.stage_timeout)
                installed = label
            peer.exchange(f"installed-{step}")
            # Config snapshots point to a fresh step directory. The existing BS32
            # launcher owns Golden validation, profiler collection and rank setup.
            run_config = dict(config, case_dir=str(step_dir))
            config_path = step_dir / "config.json"
            write_json(config_path, run_config)
            command = [env["PYTHON"], "-I", Path(launcher.__file__), "--config", config_path]
            if label not in checked:
                peer.exchange(f"accuracy-start-{step}")
                run_process(command + ["run"], env, step_dir / "accuracy.log", peer=peer, timeout=args.stage_timeout)
                peer.exchange(f"accuracy-done-{step}")
                checked.add(label)
            peer.exchange(f"perf-start-{step}")
            run_process(
                command + ["perf", "--warmup", args.warmup, "--iters", args.iters],
                env,
                step_dir / "perf.log",
                peer=peer,
                timeout=args.stage_timeout,
            )
            reports = list(step_dir.glob(f"perf_node_*/rank_{config['node_rank'] * 16}/bs_32/profiling.json"))
            if len(reports) != 1:
                raise RuntimeError(f"Expected exactly one fresh local profiling.json: {step_dir}")
            report = read_performance(reports[0], args.iters)
            perf_dir = reports[0].parents[2]
            for rank in range(config["node_rank"] * 16, (config["node_rank"] + 1) * 16):
                manifest = json.loads((perf_dir / f"rank_{rank}" / "manifest.json").read_text())
                if manifest["build"] != info or manifest["mode"] != "perf":
                    raise RuntimeError(f"Rank {rank} ran with the wrong build or timer mode")
            peer.exchange(f"perf-done-{step}", json_digest(report), same=True)
            records.append(
                dict(
                    step=index + 1,
                    block=index // 4,
                    variant=label,
                    mean_us=report["mean_us"],
                    rank_max_mean_us=report["rank_max_mean_us"],
                    report=reports[0].relative_to(directory).as_posix(),
                    source_sha256=info["source_sha256"],
                )
            )
            save_results(directory, records, args.cycles)
    except BaseException as failure:
        error = failure
        peer.abort(failure)
    finally:
        # D is built first so failure in later stages can still leave the host on
        # the original native implementation, with timer support disabled.
        restore_error = None
        if "D" in wheels:
            try:
                restore = directory / "restore_D"
                restore.mkdir()
                install_wheel(*wheels["D"], env, restore, None, args.stage_timeout)
            except BaseException as failure:
                restore_error = str(failure)
                if error is None:
                    error = failure
                    peer.abort(failure)
        if error is None:
            try:
                peer.exchange("restored-D")
            except BaseException as failure:
                error = failure
                peer.abort(failure)
        write_json(
            directory / "status.json",
            dict(
                complete=error is None,
                error=str(error) if error else None,
                installed_after_run="D (timer OFF)" if "D" in wheels and restore_error is None else "not verified",
                restore_error=restore_error,
            ),
        )
        print(save_results(directory, records, args.cycles, error=str(error) if error else None), flush=True)
        bundle = package_results(directory)
        print(f"[AB] Results: {directory / 'summary.txt'}\n[AB] Send this bundle: {bundle}", flush=True)
    if error:
        raise error


def main(argv=None):
    launcher = load_launcher()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launcher.DEFAULT_CONFIG)
    parser.add_argument("--cycles", type=int, default=3, help="ABCDDCBA cycles; default 3 = six runs per variant")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--control-port", type=int, help="Peer coordination TCP port; default master_port + 1")
    parser.add_argument("--connect-timeout", type=int, default=600)
    parser.add_argument("--stage-timeout", type=int, default=7200, help="Maximum seconds per build/test/barrier")
    args = parser.parse_args(argv)
    if args.cycles < 1 or args.warmup < 0 or args.iters < 2 or min(args.connect_timeout, args.stage_timeout) < 1:
        parser.error("cycles/timeouts must be positive, warmup >= 0, iters >= 2")
    if sys.platform != "linux":
        raise RuntimeError("BS32 A/B requires the two Linux NPU servers")
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    launcher.validate(config)
    port = args.control_port if args.control_port is not None else config["master_port"] + 1
    if not 1024 <= port <= 65535 or port in (config["master_port"], config["shmem_port"]):
        parser.error("control-port must be in [1024,65535] and differ from master/shmem ports")
    with local_lock(ROOT):
        parent = Path(config["case_dir"])
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"ab_node_{config['node_rank']}_", dir=parent))
        print(f"[AB] Output: {directory}", flush=True)
        peer = Peer.connect(config, port, args.connect_timeout, args.stage_timeout)
        try:
            remote = peer.exchange("identity", dict(node_rank=config["node_rank"]))
            if remote != dict(node_rank=1 - config["node_rank"]):
                raise RuntimeError("Expected one node 0 and one node 1")
            experiment(args, config, launcher, peer, directory)
        except BaseException as error:
            peer.abort(error)
            raise
        finally:
            peer.close()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
