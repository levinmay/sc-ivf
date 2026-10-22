from pathlib import Path
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from threadpoolctl import threadpool_limits
import antalya_cohort
from prepare import sha

from bwh_cohort import CohortAdapter, seed_everything as seed_all


def bwh_adapter(parents, cycles):
    return CohortAdapter(parents, cycles)


def training_frames(prepared, cohort, fold):
    if cohort == "antalya":
        return pd.read_parquet(prepared / f"antalya/fold_{fold}/training.parquet")
    root = prepared / "bwh/prediction_restart"
    p, c, f = [
        pd.read_csv(root / n)
        for n in [
            "development_parents.csv",
            "development_cycles.csv",
            "development_folds.csv",
        ]
    ]
    valid = set(f.loc[f.Fold.eq(fold), "Patient_ID"])
    return (
        p.loc[~p.Patient_ID.isin(valid)].reset_index(drop=True),
        c.loc[~c.Patient_ID.isin(valid)].reset_index(drop=True),
    )


def antalya(prepared, root, fold, method):
    train = training_frames(prepared, "antalya", fold)
    data = antalya_cohort
    meta = antalya_cohort
    fitseed = 202609480 + 10 * fold
    sampleseed = 202609580 + 10 * fold
    seed_all(fitseed)
    if method == "hma":
        parents = (
            train.groupby("Patient_Code", sort=False)
            .first()
            .reset_index()[["Patient_Code", "age", "afc", "bmi"]]
            .rename(
                columns={"age": "first_age", "afc": "first_afc", "bmi": "first_bmi"}
            )
        )
        relational = {
            "patients": parents,
            "retrievals": train[["Patient_Code", "Cycle_ID", *data.COLS]],
        }
        model = HMASynthesizer(meta.metadata_for(relational), verbose=False)
        model.fit(relational)
    else:
        flat = train[data.COLS].copy()
        metadata = meta.metadata_for({"flat": flat})
        if method == "copula":
            model = GaussianCopulaSynthesizer(
                metadata, enforce_min_max_values=True, enforce_rounding=True
            )
        else:
            assert torch.cuda.is_available(), "The original TVAE recipe uses CUDA"
            model = TVAESynthesizer(
                metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                epochs=300,
                cuda=True,
                verbose=False,
            )
        model.fit(flat)
    joblib.dump(model, root / "model.joblib")
    if method == "tvae300":
        model.get_loss_values().to_csv(root / "loss.csv", index=False)
    print(
        json.dumps(
            {
                "stage": "fit_complete",
                "cohort": "antalya",
                "fold": fold,
                "method": method,
            }
        ),
        flush=True,
    )
    maxpat = 789 * 20
    seed_all(sampleseed)
    if method == "hma":
        for submodel in model._table_synthesizers.values():
            submodel._set_random_state(sampleseed)
        tables = model.sample(scale=20 * 1.12)
        raw = tables["retrievals"]
        eligible = (
            tables["patients"]
            .loc[tables["patients"].Patient_Code.isin(raw.Patient_Code), "Patient_Code"]
            .to_numpy()
        )
        assert len(eligible) >= maxpat
        chosen = np.random.default_rng(sampleseed).permutation(eligible)[:maxpat]
        raw = raw.loc[raw.Patient_Code.isin(chosen)].copy()
        mapping = {v: f"R{fitseed}P{i}" for i, v in enumerate(chosen)}
        raw.Patient_Code = raw.Patient_Code.map(mapping)
        raw.Cycle_ID = [f"R{fitseed}C{i}" for i in range(len(raw))]
        synthetic, _ = data.repair(raw[["Patient_Code", "Cycle_ID", *data.COLS]], train)
    else:
        counts = np.tile(train.groupby("Patient_Code", sort=True).size().to_numpy(), 20)
        np.random.default_rng(sampleseed).shuffle(counts)
        order = [f"R{fitseed}P{i}" for i in range(maxpat)]
        model._set_random_state(sampleseed)
        sampled = model.sample(num_rows=int(counts.sum()))
        sampled = sampled.iloc[
            np.random.default_rng(sampleseed + 1).permutation(len(sampled))
        ].reset_index(drop=True)
        sampled["Patient_Code"] = np.repeat(order, counts)
        sampled["Cycle_ID"] = [f"R{fitseed}C{i}" for i in range(len(sampled))]
        synthetic, _ = data.repair(sampled, train)
    synthetic.to_parquet(root / "cohort.parquet", index=False)
    return {"cohort.parquet": sha(root / "cohort.parquet")}


def bwh(prepared, root, fold, method):
    parents, cycles = training_frames(prepared, "bwh", fold)
    adapter = bwh_adapter(parents, cycles)
    internal = {"copula": "gaussian_copula", "tvae300": "tvae"}[method]
    with threadpool_limits(1):
        model = adapter._model(internal)
    joblib.dump(model, root / "model.joblib")
    if method == "tvae300":
        model.get_loss_values().to_csv(root / "loss.csv", index=False)
    print(
        json.dumps(
            {"stage": "fit_complete", "cohort": "bwh", "fold": fold, "method": method}
        ),
        flush=True,
    )
    artifacts = {}
    for batch in range(20):
        seed = 202609060 + batch
        sp, sc, _ = adapter.sample(internal, 2826, seed)
        prefix = (
            f"tvae_epochs300_dev_v1_f{fold}_s{seed}_"
            if method == "tvae300"
            else f"cachedpool_{internal}_f{fold}_s{seed}_"
        )
        sp.Patient_ID = prefix + sp.Patient_ID.astype(str)
        sc.Patient_ID = prefix + sc.Patient_ID.astype(str)
        sc.Cycle_ID = prefix + sc.Cycle_ID.astype(str)
        dest = root / "batches" / f"batch_{batch:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for name, frame in [("parents", sp), ("cycles", sc)]:
            path = dest / (name + ".parquet")
            frame.to_parquet(path, index=False)
            artifacts[str(path.relative_to(root))] = sha(path)
        print(
            json.dumps(
                {
                    "stage": "sample_complete",
                    "cohort": "bwh",
                    "fold": fold,
                    "method": method,
                    "batch": batch,
                }
            ),
            flush=True,
        )
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=["antalya", "bwh"], required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--method", choices=["hma", "copula", "tvae300"], required=True)
    parser.add_argument("--sample-workers", type=int, default=1)
    args = parser.parse_args()
    threadpool_limits(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.output / args.cohort / f"fold_{args.fold}" / args.method
    if (root / "DONE.json").exists():
        done = json.loads((root / "DONE.json").read_text())
        for n, h in done["artifacts"].items():
            assert sha(root / n) == h, n
        print(
            json.dumps(
                {
                    "stage": "already_complete",
                    "cohort": args.cohort,
                    "fold": args.fold,
                    "method": args.method,
                }
            )
        )
        return
    if args.cohort == "bwh" and args.method == "hma":
        from bwh_hma_batches import run

        run(args.prepared, root, args.fold, args.sample_workers)
        return
    if root.exists():
        raise FileExistsError("Partial run exists; use a new output directory.")
    root.mkdir(parents=True)
    start = time.monotonic()
    artifacts = globals()[args.cohort](args.prepared, root, args.fold, args.method)
    artifacts["model.joblib"] = sha(root / "model.joblib")
    report = dict(
        status="complete",
        cohort=args.cohort,
        fold=args.fold,
        method=args.method,
        seconds=time.monotonic() - start,
        artifacts=artifacts,
    )
    (root / "DONE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    main()
