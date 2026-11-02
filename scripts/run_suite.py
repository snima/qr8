#!/usr/bin/env python3
"""Run the declared benchmark matrix and preserve commands and environment.

Each executable invocation writes a separate immutable raw CSV. This runner does
not aggregate or alter measurements; scripts/summarize.py performs that step.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT).stdout
    except FileNotFoundError:
        return "command unavailable\n"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("./qr8_benchmark"))
    parser.add_argument("--synthetic-grid", type=pathlib.Path, default=pathlib.Path("config/synthetic_grid.json"))
    parser.add_argument("--dataset-lock", type=pathlib.Path, default=pathlib.Path("data/datasets.lock.json"))
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--maximum-refinements", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--output", choices=("q", "qr", "both"), default="both")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--kernel-timing", action="store_true",
                        help="instrument phases; do not use these totals as primary performance numbers")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--include-rank-deficient", action="store_true",
                        help="run separately classified breakdown tests that violate the full-rank assumption")
    parser.add_argument("--l2-shift", type=float, default=0.0,
                        help="diagonal shift magnitude for L2 shift/retry (default 0.0)")
    parser.add_argument("--l2-shift-retries", type=int, default=0,
                        help="number of L2 shift-retry attempts (default 0)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.synthetic_only and args.real_only:
        parser.error("--synthetic-only and --real-only are mutually exclusive")
    args.results.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results / run_id
    run_dir.mkdir()
    binary = args.binary.resolve()
    configuration = vars(args).copy()
    configuration.update({"binary": str(args.binary), "synthetic_grid": str(args.synthetic_grid),
                          "dataset_lock": str(args.dataset_lock), "results": str(args.results)})
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source_sha256": sha256(pathlib.Path("src/qr8_benchmark.cu")),
        "nvcc": command_output(["nvcc", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "-q"]),
        "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "configuration": configuration
    }
    (run_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    command_log = (run_dir / "commands.jsonl").open("w", encoding="utf-8")
    outputs = ("q", "qr") if args.output == "both" else (args.output,)

    def run_case(case_id: str, arguments: list[str]) -> None:
        safe_id = case_id.replace("/", "_").replace(".", "_")
        for output in outputs:
            raw_csv = run_dir / f"{safe_id}_{output}.csv"
            command = [str(binary), *arguments, "--matrix-id", case_id, "--output", output,
                       "--warmups", str(args.warmups), "--trials", str(args.trials),
                       "--maximum-refinements", str(args.maximum_refinements),
                       "--tolerance", repr(args.tolerance), "--raw-csv", str(raw_csv)]
            if args.equilibrate:
                command.append("--equilibrate")
            if args.kernel_timing:
                command.append("--kernel-timing")
            if args.l2_shift != 0.0:
                command.extend(["--l2-shift", repr(args.l2_shift)])
            if args.l2_shift_retries > 0:
                command.extend(["--l2-shift-retries", str(args.l2_shift_retries)])
            command_log.write(json.dumps({"case": case_id, "output": output, "command": command}) + "\n")
            command_log.flush()
            print(" ".join(command))
            if not args.dry_run:
                with (run_dir / f"{safe_id}_{output}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    if not args.real_only:
        grid = json.loads(args.synthetic_grid.read_text(encoding="utf-8"))
        for dimensions in grid["cases"]:
            for condition in grid["condition_numbers"]:
                for seed in grid["seeds"]:
                    case_id = f"synthetic_m{dimensions['m']}_n{dimensions['n']}_k{condition:g}_s{seed}"
                    run_case(case_id, ["--m", str(dimensions["m"]), "--n", str(dimensions["n"]),
                                       "--condition", repr(condition), "--seed", str(seed)])
    if not args.synthetic_only:
        lock = json.loads(args.dataset_lock.read_text(encoding="utf-8"))
        for matrix in lock["matrices"]:
            measured_rank = matrix.get("measured_rank_fp64")
            if measured_rank is not None and measured_rank < matrix["columns"] and not args.include_rank_deficient:
                print(f"skip rank-deficient matrix {matrix['id']}: rank {measured_rank} < {matrix['columns']}")
                continue
            path = args.dataset_lock.parent / f"{matrix['id']}.qr8"
            if sha256(path) != matrix["output_sha256"]:
                raise RuntimeError(f"dataset checksum mismatch: {path}")
            run_case(matrix["id"], ["--input", str(path.resolve())])
    command_log.close()
    if not args.dry_run:
        raw_files = sorted(run_dir.glob("*.csv"))
        combined = run_dir / "raw_all.csv"
        header = None
        with combined.open("w", encoding="utf-8") as output_stream:
            for path in raw_files:
                with path.open("r", encoding="utf-8") as input_stream:
                    current_header = input_stream.readline()
                    if header is None:
                        header = current_header
                        output_stream.write(header)
                    elif current_header != header:
                        raise RuntimeError(f"CSV schema mismatch in {path}")
                    for line in input_stream:
                        output_stream.write(line)
        subprocess.run([sys.executable, "scripts/summarize.py", str(combined), str(run_dir / "summary.csv")], check=True)
        subprocess.run([sys.executable, "scripts/paired_speedups.py", str(combined),
                        str(run_dir / "paired_speedups.csv")], check=True)


if __name__ == "__main__":
    main()
