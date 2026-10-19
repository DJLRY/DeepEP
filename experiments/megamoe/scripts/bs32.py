# Copyright (c) 2026, Lu Lu

"""Persist per-host settings, build separately, and run the two-host EP32 BS32 accuracy case."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / ".megamoe-bs32" / f"{socket.gethostname()}.json"
CASE = {
    "schema_version": 1,
    "model": {"hidden": 3584, "intermediate_hidden": 1280, "num_experts": 384, "num_topk": 8, "num_shared_experts": 0},
    "bs_values": [32],
    "warmup": 0,
    "iters": 1,
    "seed": 20260922,
}
FIELDS = (
    "node_rank",
    "master_addr",
    "cann_env",
    "rdma_backend",
    "python",
    "case_dir",
    "master_port",
    "shmem_port",
    "golden_workers",
)


def defaults():
    return dict(
        schema_version=1,
        node_rank=None,
        master_addr="",
        cann_env="",
        rdma_backend="",
        python=sys.executable,
        case_dir=str(ROOT / "benchmark-runs/bs32"),
        master_port=29654,
        shmem_port=26733,
        golden_workers=4,
    )


def validate(config):
    if set(config) != {"schema_version", *FIELDS} or config["schema_version"] != 1:
        raise ValueError("Unsupported BS32 config fields/schema; use configure")
    if type(config["node_rank"]) is not int or config["node_rank"] not in (0, 1):
        raise ValueError("node_rank must be 0 or 1")
    address = ipaddress.IPv4Address(config["master_addr"])
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("master_addr must be node 0's IPv4 address reachable from both hosts")
    for name in ("master_port", "shmem_port"):
        if type(config[name]) is not int or not 1024 <= config[name] <= 65535:
            raise ValueError(f"{name} must be an integer in [1024, 65535]")
    if config["master_port"] == config["shmem_port"]:
        raise ValueError("master_port and shmem_port must differ")
    if type(config["golden_workers"]) is not int or config["golden_workers"] < 1:
        raise ValueError("golden_workers must be positive")
    if config["rdma_backend"] not in ("XSCALE", "HNS_1825"):
        raise ValueError("rdma_backend must be XSCALE or HNS_1825")
    for name in ("cann_env", "python", "case_dir"):
        if not isinstance(config[name], str) or not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    for name in ("cann_env", "python"):
        if not Path(config[name]).is_file():
            raise ValueError(f"{name} does not exist: {config[name]}")


def save_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure(path, args):
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else defaults()
    overrides = {name: getattr(args, name) for name in FIELDS if getattr(args, name) is not None}
    config.update(overrides)
    if not overrides:
        if not sys.stdin.isatty():
            raise ValueError(
                "No interactive terminal; use configure --node-rank ... --master-addr ... --cann-env ... --rdma-backend ..."
            )
        for name, label in (
            ("node_rank", "本机节点编号（0 或 1）"),
            ("master_addr", "第一台机器的 IP"),
            ("cann_env", "CANN set_env.sh 完整路径"),
            ("rdma_backend", "RoCE 后端（XSCALE 或 HNS_1825）"),
        ):
            value = input(f"{label} [{config[name] if config[name] is not None else ''}]: ").strip()
            if value:
                config[name] = int(value) if name == "node_rank" else value
    # Keep the venv/Conda executable path, even when it is a symlink.
    executable = shutil.which(os.path.expanduser(str(config["python"])))
    if executable is None:
        raise ValueError(f"Python executable not found: {config['python']}")
    config["python"] = os.path.abspath(executable)
    for name in ("cann_env", "case_dir"):
        if config[name]:
            config[name] = str(Path(config[name]).expanduser().absolute())
    validate(config)
    save_config(path, config)
    print(f"[CONFIG] Saved: {path}\n{json.dumps(config, indent=2, ensure_ascii=False)}", flush=True)
    return config


def environment(config):
    env = dict(os.environ)
    # Never inherit an external SDK or Python import override into this workflow.
    for name in ("SHMEM_ROOT", "CATLASS_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC", "CONDA_PREFIX"):
        env.pop(name, None)
    prefix = Path(config["python"]).parent.parent
    env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # Pass paths as positional arguments, never interpolate configuration into shell code.
    probe = (
        "import json, os, pathlib, sys; "
        "env = dict(os.environ); "
        "env.update(CONDA_PREFIX=sys.prefix) if (pathlib.Path(sys.prefix) / 'conda-meta').is_dir() else None; "
        "print(json.dumps(env))"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >&2 || exit $?; exec "$2" -I -c "$3"',
            "bs32-env",
            config["cann_env"],
            config["python"],
            probe,
        ],
        env=env,
        cwd="/tmp",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    env = json.loads(result.stdout)
    env.update(
        PYTHON=config["python"],
        DEEPEP_BUILD_MEGAMOE="ON",
        DEEPEP_MEGAMOE_TIMER="OFF",
        MASTER_ADDR=config["master_addr"],
        MASTER_PORT=str(config["master_port"]),
        DEEPEP_SHMEM_ENDPOINT=f"tcp://{config['master_addr']}:{config['shmem_port']}",
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(16))),
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONSAFEPATH="1",
    )
    env["LD_LIBRARY_PATH"] = f"{ROOT}/third-party/shmem/install/shmem/lib:{prefix}/lib:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    for name in ("SHMEM_ROOT", "CATLASS_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC"):
        env.pop(name, None)
    return env


def execute(command, env, log=None, cwd="/tmp"):
    print("[EXEC] " + shlex.join(map(str, command)), flush=True)
    if log is None:
        subprocess.run(command, env=env, cwd=cwd, check=True)
        return
    with Path(log).open("w", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, env=env, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def preflight(config, env):
    library = ROOT / "third-party/shmem/install/shmem/lib/libshmem.so"
    if not library.is_file():
        raise RuntimeError(f"Missing {library}; run 'bash experiments/megamoe/scripts/run_bs32.sh build' first")
    code = """import json, sys
from pathlib import Path
from importlib.resources import files
import torch, torch_npu, deep_ep
from deep_ep._native import require_native
import deep_ep_experimental.megamoe._C
require_native('BS32 preflight')
info = json.loads(files('deep_ep').joinpath('_build_info.json').read_text())
if Path(deep_ep.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise RuntimeError('deep_ep resolves to source; run build first')
if not info['megamoe']['enabled'] or info['shmem']['rdma_backend'] != sys.argv[2]:
    raise RuntimeError('Installed wheel does not match MegaMoE/RoCE configuration; run build first')
if torch.npu.device_count() != 16:
    raise RuntimeError('BS32 requires sixteen visible NPUs on each host')
print('[CHECK] Python:', sys.executable, 'deep_ep:', deep_ep.__file__, flush=True)
"""
    execute([config["python"], "-I", "-c", code, str(ROOT), config["rdma_backend"]], env)


def test_commands(config, run_dir):
    python = [config["python"], "-I", "-m"]
    golden_dir = str(Path(config["case_dir"]) / "golden")
    common = [
        "--config",
        str(run_dir / "case.json"),
        "--input",
        "fp8",
        "--routing",
        "balanced",
        "--check-tokens",
        "0",
        "--golden-dir",
        golden_dir,
    ]
    rank = config["node_rank"]
    golden = [
        *python,
        "deep_ep_experimental.megamoe.golden",
        *common,
        "--ep-size",
        "32",
        "--ranks",
        f"{rank * 16}-{rank * 16 + 15}",
        "--workers",
        str(config["golden_workers"]),
    ]
    benchmark = [
        *python,
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--nproc-per-node",
        "16",
        "--node-rank",
        str(rank),
        "--master-addr",
        config["master_addr"],
        "--master-port",
        str(config["master_port"]),
        "--rdzv-conf",
        "timeout=7200",
        "--max-restarts",
        "0",
        "--module",
        "deep_ep_experimental.megamoe.benchmark",
        *common,
        "--mode",
        "accuracy",
        "--output-dir",
        str(run_dir),
    ]
    return golden, benchmark


def dispatch(action, config, *, rebuild_shmem=False):
    if platform.system() != "Linux":
        raise RuntimeError("Build/run require Linux; configure/show-config can run without devices")
    env = environment(config)
    if action == "build":
        command = ["bash", str(ROOT / "experiments/megamoe/scripts/build.sh"), "--rdma-backend", config["rdma_backend"]]
        if rebuild_shmem:
            command.append("--rebuild-shmem")
        execute(command, env, cwd=ROOT)
        return
    preflight(config, env)
    directory = Path(config["case_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"node_{config['node_rank']}_", dir=directory))
    (run_dir / "case.json").write_text(json.dumps(CASE, indent=2) + "\n", encoding="utf-8")
    (run_dir / "launch.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[RUN] node={config['node_rank']} master={config['master_addr']} output={run_dir}", flush=True)
    golden, benchmark = test_commands(config, run_dir)
    execute(golden, env, run_dir / "golden.log")
    if action == "run":
        execute(benchmark, env, run_dir / "console.log")
    print(f"[DONE] {action}: {run_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="run", choices=("configure", "show-config", "build", "golden", "run")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Machine-local JSON settings file")
    parser.add_argument(
        "--rebuild-shmem", action="store_true", help="build only: rebuild the bundled SDK after stopping its users"
    )
    for name in FIELDS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name in ("node_rank", "master_port", "shmem_port", "golden_workers") else str,
            help="configure only: persist this setting",
        )
    args = parser.parse_args(argv)
    if args.action != "configure" and any(getattr(args, name) is not None for name in FIELDS):
        parser.error("Settings may only be changed with configure")
    if args.rebuild_shmem and args.action != "build":
        parser.error("--rebuild-shmem requires build")
    path = args.config.expanduser().absolute()
    if args.action == "configure":
        configure(path, args)
        return
    if not path.is_file():
        raise ValueError(
            f"No saved settings at {path}; run 'bash experiments/megamoe/scripts/run_bs32.sh configure' once on each host"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    validate(config)
    print(f"[CONFIG] Using: {path}", flush=True)
    if args.action == "show-config":
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    dispatch(args.action, config, rebuild_shmem=args.rebuild_shmem)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
