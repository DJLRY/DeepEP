# Copyright (c) 2026, Lu Lu
"""Run independent BS32 pipeline experiments in the same order on both hosts."""

import argparse
from pathlib import Path
import runpy
import shlex


CASES = {
    "q0_base": ("--combine-reduce-qps", "0", "--dispatch-workers", "14"),
    "q0_scan": ("--combine-reduce-qps", "0", "--dispatch-workers", "14", "--combine-sparse-scan"),
    "q0_async": ("--combine-reduce-qps", "0", "--dispatch-workers", "14", "--combine-async"),
    "q8_base": ("--combine-reduce-qps", "8", "--dispatch-workers", "28", "--reduce-lane-init"),
    "q8_dispatch14": ("--combine-reduce-qps", "8", "--dispatch-workers", "14", "--reduce-lane-init"),
    "q8_owner_half": (
        "--combine-reduce-qps",
        "8",
        "--dispatch-workers",
        "28",
        "--reduce-lane-init",
        "--combine-owner-half",
    ),
    "q8_sync": (
        "--combine-reduce-qps",
        "8",
        "--dispatch-workers",
        "28",
        "--reduce-lane-init",
        "--combine-sync-progress",
    ),
    "q0_all": ("--combine-reduce-qps", "0", "--dispatch-workers", "14", "--combine-sparse-scan", "--combine-async"),
    "q8_all": (
        "--combine-reduce-qps",
        "8",
        "--dispatch-workers",
        "14",
        "--reduce-lane-init",
        "--combine-owner-half",
        "--combine-sync-progress",
    ),
}
INDEPENDENT_CASES = tuple(name for name in CASES if not name.endswith("_all"))


def case_arguments(action, name, config=None, warmup=None, iters=None):
    args = [action, *CASES[name]]
    if config is not None:
        args.extend(("--config", str(config)))
    if action in ("perf", "trace"):
        args.extend(("--warmup", str(20 if warmup is None else warmup)))
        args.extend(("--iters", str((100 if action == "perf" else 30) if iters is None else iters)))
    if action == "trace":
        args.extend(("--include-wait", "--timer-ranks", "0,14,16,30"))
        if name.startswith("q8_"):
            # Match diagnostic overhead on the baseline and the modified route.
            args.append("--combine-ready-trace")
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "perf", "trace"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=("all", "q0", "q8"), default="all")
    selection.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact per-case commands without launching")
    args = parser.parse_args(argv)
    if args.action == "run" and (args.warmup is not None or args.iters is not None):
        parser.error("--warmup and --iters require perf or trace")
    if args.warmup is not None and args.warmup < 0 or args.iters is not None and args.iters < 1:
        parser.error("warmup must be >= 0 and iters must be >= 1")
    selected = (
        (args.case,)
        if args.case
        else tuple(name for name in INDEPENDENT_CASES if args.suite == "all" or name.startswith(args.suite + "_"))
    )
    launcher = None if args.dry_run else runpy.run_path(str(Path(__file__).with_name("bs32.py")))["main"]
    for name in selected:
        command = case_arguments(args.action, name, args.config, args.warmup, args.iters)
        print(f"[CASE] {name}", flush=True)
        print(shlex.join(["bash", "experiments/megamoe/scripts/run_bs32.sh", *command]), flush=True)
        if launcher is not None:
            # Exceptions propagate: stop at the first failed accuracy/run case.
            # Every underlying run keeps its own run.json and result directory.
            launcher(command)


if __name__ == "__main__":
    main()
