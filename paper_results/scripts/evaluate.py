"""Fit SC-IVF and evaluate prediction tasks."""

from pathlib import Path
import argparse
import importlib.util
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
from prepare import sha


def setup(source):
    sys.path.insert(0, str(HERE.parents[1]))
    sys.path.insert(0, str(source))
    import run_prediction

    return run_prediction


def bind(prepared, baselines, output, source):
    files = {}
    jobs = []

    def file(path):
        files[str(path)] = sha(path)
        return str(path)

    for cohort in ["antalya", "bwh"]:
        for fold in range(4):
            job = dict(cohort=cohort, fold=fold, methods={})
            if cohort == "antalya":
                for name in ["training", "validation"]:
                    job[name] = file(prepared / f"antalya/fold_{fold}/{name}.parquet")
            else:
                job["development"] = {
                    n: file(prepared / "bwh/prediction_restart" / n)
                    for n in [
                        "development_parents.csv",
                        "development_cycles.csv",
                        "development_folds.csv",
                    ]
                }
            for method in ["hma", "copula", "tvae300"]:
                root = baselines / cohort / f"fold_{fold}" / method
                done = json.loads((root / "DONE.json").read_text())
                assert done["status"] == "complete"
                file(root / "DONE.json")
                if cohort == "antalya":
                    job["methods"][method] = dict(frame=file(root / "cohort.parquet"))
                else:
                    job["methods"][method] = dict(
                        batches=[
                            {
                                n: file(root / "batches" / f"batch_{b:02d}" / n)
                                for n in ["parents.parquet", "cycles.parquet"]
                            }
                            for b in range(20)
                        ]
                    )
            jobs.append(job)
    manifest = dict(
        files=files,
        jobs=jobs,
        reserved_partitions_accessed=False,
        fresh_raw_data_and_baseline_reproduction=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "INPUTS.json"
    text = json.dumps(manifest, indent=2) + "\n"
    if dest.exists():
        assert dest.read_text() == text
    else:
        dest.write_text(text)
    print(
        json.dumps(
            {"stage": "new_inputs_bound", "files": len(files), "folds": len(jobs)}
        ),
        flush=True,
    )


def copying_baseline(prepared):
    rows, patients, cycles = ([], [], [])
    for fold in range(4):
        name = f"antalya/fold_{fold}/validation.parquet"
        path = prepared / name
        data = pd.read_parquet(
            path,
            columns=["Patient_Code", "Cycle_ID", "blastocysts", "frozen_blastocysts"],
        )
        for column in ["blastocysts", "frozen_blastocysts"]:
            assert data[column].notna().all()
            assert data[column].ge(0).all() and data[column].mod(1).eq(0).all()
        assert data.Cycle_ID.is_unique
        error = (data.frozen_blastocysts - data.blastocysts).abs()
        rows.append(
            dict(
                fold=fold,
                validation_sha256=sha(path),
                patients=int(data.Patient_Code.nunique()),
                records=len(data),
                equal_counts=int(error.eq(0).sum()),
                absolute_error_sum=float(error.sum()),
                copy_mae=float(error.mean()),
            )
        )
        patients.append(set(data.Patient_Code))
        cycles.append(set(data.Cycle_ID))
    for groups in [patients, cycles]:
        assert sum(map(len, groups)) == len(set.union(*groups))
    assert sum((r["records"] for r in rows)) == 1067
    assert sum((r["patients"] for r in rows)) == 1052
    return dict(
        cohort="antalya",
        folds=rows,
        equal_percentage=100
        * sum((r["equal_counts"] for r in rows))
        / sum((r["records"] for r in rows)),
        copy_mae_equal_fold_mean=sum((r["copy_mae"] for r in rows)) / len(rows),
        exploratory=True,
        uncertainty_intervals_computed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--cohort", choices=["antalya", "bwh"])
    parser.add_argument("--fold", type=int, choices=range(4))
    args = parser.parse_args()
    source = HERE
    prepared, baselines, output = (
        args.prepared.resolve(),
        args.baselines.resolve(),
        args.output.resolve(),
    )
    if args.bind:
        bind(prepared, baselines, output, source)
        return
    runner = setup(source)
    runner.ROOT = output
    runner.METHODS = ["shared_ivf_v1_1", "hma", "copula", "tvae300", "real"]
    if args.analyze:
        runner.setup()
        spec = importlib.util.spec_from_file_location(
            "starting_analysis", source / "analyze_prediction.py"
        )
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        analysis.R, analysis.W, analysis.OUT = (output, source, output / "aggregate")
        if analysis.OUT.exists():
            raise FileExistsError("Analysis output already exists.")
        analysis.OUT.mkdir()
        analysis.inference()
        analysis.quality()
        runner.save(analysis.OUT / "copying_baseline.json", copying_baseline(prepared))
        runner.save(
            analysis.OUT / "ANALYSIS_DONE.json",
            dict(
                status="complete",
                input_manifest_sha256=sha(output / "INPUTS.json"),
                artifacts={p.name: sha(p) for p in analysis.OUT.glob("*.csv")},
            ),
        )
    else:
        if args.cohort is None or args.fold is None:
            parser.error("--cohort and --fold required for evaluation")
        runner.work(args.cohort, args.fold)


if __name__ == "__main__":
    main()
