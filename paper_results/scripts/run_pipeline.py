"""Reproduce the four-fold IVF experiments."""

from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from prepare import fetch, prepare

HERE = Path(__file__).resolve().parent


def command(root, script, *extra):
    return [sys.executable, "-u", str(HERE / script), *map(str, extra)]


def execute(root, label, cmd):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.monotonic()
    with (logs / (label + ".log")).open("w") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    record = dict(
        job=label,
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        seconds=time.monotonic() - start,
    )
    (logs / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise RuntimeError("Job failed; inspect logs/" + label + ".log")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["prepare", "baselines", "evaluate", "analyze", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--hma-sample-workers",
        type=int,
        default=4,
        help="Independent fixed-seed sampling processes per BWH HMA fold",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.hma_sample_workers < 1:
        parser.error("Worker counts must be positive")
    if args.stage in ["prepare", "all"]:
        fetch(root / "raw")
        prepare(root / "raw", root / "preparation")
    if args.stage in ["baselines", "all"]:
        jobs = []
        for cohort in ["antalya", "bwh"]:
            for fold in range(4):
                for method in ["hma", "copula", "tvae300"]:
                    jobs.append(
                        (
                            f"{cohort}_{fold}_{method}",
                            command(
                                root,
                                "baselines.py",
                                "--prepared",
                                root / "preparation",
                                "--output",
                                root / "baselines",
                                "--cohort",
                                cohort,
                                "--fold",
                                fold,
                                "--method",
                                method,
                                "--sample-workers",
                                args.hma_sample_workers,
                            ),
                        )
                    )
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, root, label, cmd) for label, cmd in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failures.append(str(e))
        if failures:
            raise RuntimeError("; ".join(failures))
    if args.stage in ["evaluate", "all"]:
        common = [
            "--prepared",
            root / "preparation",
            "--baselines",
            root / "baselines",
            "--output",
            root / "evaluation",
        ]
        execute(
            root, "bind_evaluation", command(root, "evaluate.py", *common, "--bind")
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, 4)
        ) as pool:
            futures = [
                pool.submit(
                    execute,
                    root,
                    f"evaluate_{c}_{f}",
                    command(root, "evaluate.py", *common, "--cohort", c, "--fold", f),
                )
                for c in ["antalya", "bwh"]
                for f in range(4)
                if not (root / "evaluation" / c / f"fold_{f}" / "DONE.json").exists()
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    if args.stage in ["analyze", "all"]:
        common = [
            "--prepared",
            root / "preparation",
            "--baselines",
            root / "baselines",
            "--output",
            root / "evaluation",
        ]
        if not (root / "evaluation/aggregate/ANALYSIS_DONE.json").exists():
            execute(
                root,
                "analyze_evaluation",
                command(root, "evaluate.py", *common, "--analyze"),
            )


if __name__ == "__main__":
    main()
