from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from prepare import sha

HERE = Path(__file__).resolve().parent


def check_fit(prepared, root, fold):
    receipt = json.loads((root / "FIT.json").read_text())
    assert receipt["fold"] == fold and receipt["model_sha256"] == sha(
        root / "model.joblib"
    )
    for name, digest in receipt["training_inputs"].items():
        assert sha(prepared / name) == digest, name
    return receipt


def fit(prepared, root, fold):
    if (root / "FIT.json").exists():
        return check_fit(prepared, root, fold)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Unverified partial HMA fit; use a new directory.")
    root.mkdir(parents=True, exist_ok=True)
    from baselines import bwh_adapter, training_frames
    import joblib
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    p, c = training_frames(prepared, "bwh", fold)
    with threadpool_limits(2):
        model = bwh_adapter(p, c)._model("generic_hma")
    joblib.dump(model, root / "model.joblib")
    names = [
        "development_parents.csv",
        "development_cycles.csv",
        "development_folds.csv",
    ]
    receipt = dict(
        fold=fold,
        fit_seed=202609060,
        fit_blas_threads=2,
        sample_blas_threads_by_batch=[
            8 if fold == 0 and b == 0 else 2 if b < 8 else 1 for b in range(20)
        ],
        model_sha256=sha(root / "model.joblib"),
        training_inputs={
            f"bwh/prediction_restart/{n}": sha(prepared / "bwh/prediction_restart" / n)
            for n in names
        },
    )
    (root / "FIT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def restore_original_missingness_state(model, index):
    from importlib.metadata import version
    from rdt.transformers.numerical import FloatFormatter

    assert version("rdt") == "1.22.0" and version("sdv") == "1.38.2"
    previous = index - 10 if index >= 10 else index - 1 if 1 <= index < 8 else 0
    advanced = {}
    for table, synthesizer in model._table_synthesizers.items():
        rows = round(model._table_sizes[table] * 1.12)
        for (
            column,
            transformer,
        ) in synthesizer._data_processor._hyper_transformer.field_transformers.items():
            if isinstance(transformer, FloatFormatter):
                null = transformer.null_transformer
                assert null._missing_value_generation == "random"
                if null.nulls:
                    transformer.random_states["reverse_transform"].random_sample(
                        rows * previous
                    )
                    advanced[table + "." + column] = rows * previous
    return dict(previous_draws=previous, uniforms_advanced=advanced)


def batch(prepared, root, fold, index):
    check_fit(prepared, root, fold)
    dest = root / "batches" / f"batch_{index:02d}"
    receipt = dest / "BATCH.json"
    if receipt.exists():
        done = json.loads(receipt.read_text())
        for name, digest in done["artifacts"].items():
            assert sha(root / name) == digest, name
        return done
    if dest.exists():
        raise FileExistsError("Unverified partial HMA batch; use a new directory.")
    from baselines import bwh_adapter, training_frames
    from threadpoolctl import threadpool_limits
    import joblib
    import torch

    torch.set_num_threads(1)
    sampling_threads = 8 if fold == 0 and index == 0 else 2 if index < 8 else 1
    threadpool_limits(sampling_threads)
    p, c = training_frames(prepared, "bwh", fold)
    adapter = bwh_adapter(p, c)
    adapter.models["generic_hma"] = joblib.load(root / "model.joblib")
    state_replay = restore_original_missingness_state(
        adapter.models["generic_hma"], index
    )
    seed = 202609060 + index
    start = time.monotonic()
    sp, sc, _ = adapter.sample("generic_hma", 2826, seed)
    prefix = f"cachedpool_generic_hma_f{fold}_s{seed}_"
    sp.Patient_ID = prefix + sp.Patient_ID.astype(str)
    sc.Patient_ID = prefix + sc.Patient_ID.astype(str)
    sc.Cycle_ID = prefix + sc.Cycle_ID.astype(str)
    dest.mkdir(parents=True)
    artifacts = {}
    for name, frame in [("parents", sp), ("cycles", sc)]:
        path = dest / (name + ".parquet")
        frame.to_parquet(path, index=False)
        artifacts[str(path.relative_to(root))] = sha(path)
    done = dict(
        fold=fold,
        batch=index,
        seed=seed,
        missingness_state_replay=state_replay,
        sample_blas_threads=sampling_threads,
        seconds=time.monotonic() - start,
        artifacts=artifacts,
        status="passed",
    )
    receipt.write_text(json.dumps(done, indent=2) + "\n")
    return done


def finalize(prepared, root, fold):
    check_fit(prepared, root, fold)
    artifacts = {"model.joblib": sha(root / "model.joblib")}
    for index in range(20):
        record = json.loads(
            (root / "batches" / f"batch_{index:02d}" / "BATCH.json").read_text()
        )
        assert (
            record["status"] == "passed"
            and record["fold"] == fold
            and (record["batch"] == index)
        )
        for name, digest in record["artifacts"].items():
            assert sha(root / name) == digest, name
        artifacts.update(record["artifacts"])
    done = dict(
        status="complete", cohort="bwh", fold=fold, method="hma", artifacts=artifacts
    )
    (root / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    return done


def run(prepared, root, fold, workers=1):
    fit(prepared, root, fold)

    def execute(index):
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "bwh_hma_batches.py"),
            "--prepared",
            str(prepared),
            "--root",
            str(root),
            "--fold",
            str(fold),
            "--batch",
            str(index),
        ]
        with (logs / f"batch_{index:02d}.log").open("w") as stream:
            subprocess.run(
                cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
        print(
            json.dumps(
                {
                    "stage": "sample_complete",
                    "cohort": "bwh",
                    "method": "hma",
                    "fold": fold,
                    "batch": index,
                }
            ),
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, index) for index in range(20)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return finalize(prepared, root, fold)["artifacts"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=range(4), required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", action="store_true")
    g.add_argument("--batch", type=int, choices=range(20))
    g.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    result = (
        fit(a.prepared, a.root, a.fold)
        if a.fit
        else finalize(a.prepared, a.root, a.fold)
        if a.finalize
        else batch(a.prepared, a.root, a.fold, a.batch)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["artifacts", "training_inputs"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
