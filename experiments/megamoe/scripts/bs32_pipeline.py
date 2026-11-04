# Copyright (c) 2026, Lu Lu
"""Run independent BS32 pipeline experiments in the same order on both hosts."""

import argparse
import csv
import json
import math
from pathlib import Path
import runpy
import shlex
import tempfile


CASES = {
    "q0_scan": ("--combine-reduce-qps", "0", "--dispatch-workers", "14", "--combine-sparse-scan"),
    "q0_single_row": (
        "--combine-reduce-qps",
        "0",
        "--dispatch-workers",
        "14",
        "--combine-sparse-scan",
        "--combine-single-row",
    ),
    "q8_dispatch14": ("--combine-reduce-qps", "8", "--dispatch-workers", "14", "--reduce-lane-init"),
    "q8_double_buffer": (
        "--combine-reduce-qps",
        "8",
        "--dispatch-workers",
        "14",
        "--reduce-lane-init",
        "--reduce-double-buffer",
    ),
}
INDEPENDENT_CASES = []
for prefix, baseline in (("q0", "q0_scan"), ("q8", "q8_double_buffer")):
    INDEPENDENT_CASES.append(baseline)
    for suffix, switch in (("skip", "skip-empty"), ("lookahead", "lookahead"), ("row", "single-row")):
        name = f"{prefix}_dispatch_{suffix}"
        CASES[name] = (*CASES[baseline], "--dispatch-" + switch)
        INDEPENDENT_CASES.append(name)
    CASES[f"{prefix}_dispatch_all"] = (
        *CASES[baseline],
        "--dispatch-skip-empty",
        "--dispatch-lookahead",
        "--dispatch-single-row",
    )
INDEPENDENT_CASES = tuple(INDEPENDENT_CASES)
SWITCHES = {
    "reduce_lane_init": "lane",
    "combine_sparse_scan": "scan",
    "combine_single_row": "single_row",
    "reduce_double_buffer": "double_buffer",
    "combine_ready_trace": "ready_trace",
    "dispatch_skip_empty": "skip",
    "dispatch_lookahead": "lookahead",
    "dispatch_single_row": "row",
}


def settings(options):
    return {
        "combine_reduce_qps": options.get("combine_reduce_qps", 0),
        "dispatch_workers": options.get("dispatch_workers", 0),
        **{key: options.get(key, False) for key in SWITCHES},
        # Preserve retired/unknown enabled switches when summarizing old results.
        # Otherwise an old strategy could be mistaken for a current baseline.
        **{
            key: value
            for key, value in options.items()
            if key not in SWITCHES and key.startswith(("combine_", "reduce_", "dispatch_")) and value is True
        },
    }


def expected_settings(name, action):
    args = case_arguments(action, name)
    return {
        "combine_reduce_qps": int(args[args.index("--combine-reduce-qps") + 1]),
        "dispatch_workers": int(args[args.index("--dispatch-workers") + 1]),
        **{key: "--" + key.replace("_", "-") in args for key in SWITCHES},
    }


def read_result(directory, expected_name=None):
    directory = Path(directory).resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    launch = json.loads((directory / "launch.json").read_text(encoding="utf-8"))
    actual = settings(run)
    action = run["action"]
    names = [name for name in CASES if actual == expected_settings(name, action)]
    if expected_name is not None and expected_name not in names:
        raise ValueError(f"Result configuration does not match {expected_name}: {directory}")
    row = dict(
        case=expected_name or (names[0] if names else "custom"),
        status="OK",
        action=action,
        node=launch["node_rank"],
        warmup=run["warmup"],
        iters=run["iters"],
        method=run.get("perf_method", "legacy"),
        run_dir=str(directory),
        **actual,
    )
    if action == "perf":
        with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
            samples = [sample for sample in csv.DictReader(stream) if int(sample["bs"]) == 32]
        if len(samples) != 1:
            raise ValueError(f"Expected exactly one BS32 result: {directory}")
        for field in ("mean_us", "rank_max_mean_us"):
            value = float(samples[0][field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {field}: {directory}")
            row[field] = value
    return row


def save_summary(directory, rows, *, display=False):
    lines = [
        "BEGIN MEGAMOE PIPELINE SUMMARY",
        "BS=32; units=us; min_us/max_us = mean of per-round minimum/maximum rank duration",
        "# case             status   node action warmup iters method qps dw flags                    min_us    max_us",
    ]
    for index, row in enumerate(rows, 1):
        labels = [label for key, label in SWITCHES.items() if row.get(key)]
        labels.extend(
            key
            for key, value in row.items()
            if key not in SWITCHES and key.startswith(("combine_", "reduce_", "dispatch_")) and value is True
        )
        flags = "+".join(labels) or "-"
        low = f"{row['mean_us']:.3f}" if "mean_us" in row else "-"
        high = f"{row['rank_max_mean_us']:.3f}" if "rank_max_mean_us" in row else "-"
        lines.append(
            f"{index} {row['case']:<16} {row['status']:<8} {str(row.get('node', '-')):>4} "
            f"{row['action']:<6} {str(row.get('warmup', '-')):>6} {str(row.get('iters', '-')):>5} "
            f"{row.get('method', '-'):<6} {row['combine_reduce_qps']:>3} {row['dispatch_workers']:>2} "
            f"{flags:<24} {low:>9} {high:>9}"
        )
    lines.append("Result directories:")
    for index, row in enumerate(rows, 1):
        lines.append(f"{index} {row['case']}: {row.get('run_dir', '-')}")
        if row.get("error"):
            lines.append("  error: " + " ".join(row["error"].splitlines()))
    lines.append("END MEGAMOE PIPELINE SUMMARY")
    text = "\n".join(lines) + "\n"
    (directory / "summary.txt").write_text(text, encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if display:
        print(text, end="", flush=True)
        print(f"[SUMMARY] {directory / 'summary.txt'}", flush=True)
        print(f"[COPY] cat {shlex.quote(str(directory / 'summary.txt'))}", flush=True)


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
    parser.add_argument("action", choices=("run", "perf", "trace", "summary"))
    parser.add_argument("result_dirs", nargs="*", type=Path, help="summary only: existing perf run directories")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=("all", "q0", "q8"), help="Default: q0 for trace, all for run/perf")
    selection.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact per-case commands without launching")
    args = parser.parse_args(argv)
    if args.action == "summary":
        if (
            not args.result_dirs
            or args.dry_run
            or args.config
            or args.case
            or args.suite not in (None, "all")
            or args.warmup is not None
            or args.iters is not None
        ):
            parser.error("summary requires only existing perf run directories")
        rows = []
        for path in args.result_dirs:
            row = read_result(path)
            if row["action"] != "perf":
                parser.error("summary accepts perf results only")
            rows.append(row)
        directory = Path(tempfile.mkdtemp(prefix="pipeline_summary_", dir=Path(rows[0]["run_dir"]).parent))
        save_summary(directory, rows, display=True)
        return directory
    if args.result_dirs:
        parser.error("Result directories require the summary action")
    if args.action == "run" and (args.warmup is not None or args.iters is not None):
        parser.error("--warmup and --iters require perf or trace")
    if args.warmup is not None and args.warmup < 0 or args.iters is not None and args.iters < 1:
        parser.error("warmup must be >= 0 and iters must be >= 1")
    suite = args.suite or ("q0" if args.action == "trace" else "all")
    selected = (
        (args.case,)
        if args.case
        else tuple(name for name in INDEPENDENT_CASES if suite == "all" or name.startswith(suite + "_"))
    )
    if args.dry_run:
        for name in selected:
            command = case_arguments(args.action, name, args.config, args.warmup, args.iters)
            print(f"[CASE] {name}", flush=True)
            print(shlex.join(["bash", "experiments/megamoe/scripts/run_bs32.sh", *command]), flush=True)
        return
    launcher = runpy.run_path(str(Path(__file__).with_name("bs32.py")))
    config_path = (args.config or launcher["DEFAULT_CONFIG"]).expanduser().absolute()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    launcher["validate"](config)
    parent = Path(config["case_dir"])
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"pipeline_{args.action}_node_{config['node_rank']}_", dir=parent))
    print(f"[PIPELINE] output={directory}", flush=True)
    rows = [
        dict(
            case=name,
            status="NOT_RUN",
            action=args.action,
            node=config["node_rank"],
            **expected_settings(name, args.action),
        )
        for name in selected
    ]
    try:
        for index, name in enumerate(selected):
            command = case_arguments(args.action, name, args.config, args.warmup, args.iters)
            print(f"[CASE] {name}", flush=True)
            print(shlex.join(["bash", "experiments/megamoe/scripts/run_bs32.sh", *command]), flush=True)
            try:
                # Use the exact returned directory, never glob for the latest run.
                result_dir = launcher["main"](command, **({"print_text": False} if args.action == "trace" else {}))
                rows[index]["run_dir"] = str(result_dir)
                rows[index] = read_result(result_dir, name)
            except BaseException as error:
                rows[index].update(status="FAILED", error=str(error) or type(error).__name__)
                raise
            save_summary(directory, rows)
    finally:
        save_summary(directory, rows, display=True)
    if args.action == "trace":
        launcher["print_trace_parts"](directory, config, pipeline=True)
    return directory


if __name__ == "__main__":
    main()
