"""Additional experiments using the fitted models from the primary pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

SOURCE = Path(__file__).resolve().parents[2]
EXPERIMENT_SOURCE = Path(__file__).resolve().parent
EVALUATION = SOURCE / "output/evaluation"
OUT = SOURCE / "output/robustness"

sys.path.insert(0, str(EXPERIMENT_SOURCE))
sys.path.insert(0, str(SOURCE))

import antalya_tasks
import bwh_tasks
from shared_ivf import engine as core, schemas

HGB_BASE = dict(
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
)
TASKS = {
    "antalya": [
        name for name in antalya_tasks.TASKS if name != "first_fet_recorded_pr"
    ],
    "bwh": [
        "poor_response",
        "high_response",
        "high_maturity",
        "transfer",
        "observation",
        "live_birth",
        "egg_count",
        "mii_count",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def safe_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    forbidden = {"sealed_test", "historical_holdout", "fold_4", "data_full_pathway"}
    if forbidden & set(path.parts):
        raise PermissionError(f"Reserved-data boundary: {path}")
    return path


def plan_and_job(cohort: str, fold: int):
    plan = json.loads((EVALUATION / "INPUTS.json").read_text())
    job = next(j for j in plan["jobs"] if j["cohort"] == cohort and j["fold"] == fold)
    return plan, job


def read_checked(path: str | Path, plan: dict) -> pd.DataFrame:
    manifest_path = safe_path(path)
    expected = plan["files"][str(manifest_path)]
    actual_path = manifest_path
    if digest(actual_path) != expected:
        raise ValueError(f"Input hash mismatch: {actual_path}")
    if actual_path.suffix == ".parquet":
        return pd.read_parquet(actual_path)
    return pd.read_csv(actual_path, float_precision="round_trip")


def load_real(cohort: str, fold: int):
    plan, job = plan_and_job(cohort, fold)
    if cohort == "antalya":
        train = read_checked(job["training"], plan)
        valid = read_checked(job["validation"], plan)
    else:
        dev = job["development"]
        parents = read_checked(dev["development_parents.csv"], plan)
        cycles = read_checked(dev["development_cycles.csv"], plan)
        folds = read_checked(dev["development_folds.csv"], plan)
        full = cycles.merge(parents, on="Patient_ID", validate="many_to_one")
        full = full.merge(folds, on="Patient_ID", validate="many_to_one")
        train = full.loc[full.Fold.ne(fold)].drop(columns="Fold").reset_index(drop=True)
        valid = full.loc[full.Fold.eq(fold)].drop(columns="Fold").reset_index(drop=True)
    return train, valid


def schema_for(cohort: str):
    return schemas.SCHEMAS[cohort]()


def task(frame: pd.DataFrame, cohort: str, name: str):
    if cohort == "antalya":
        selected, y = antalya_tasks.task_frame(frame, name)
        return (
            selected,
            y,
            antalya_tasks.FEATURES[name],
            antalya_tasks.TASK_KIND[name],
            antalya_tasks.CAT,
        )
    if name in {"egg_count", "mii_count"}:
        column = "Sum_Eggs" if name == "egg_count" else "Sum_MII"
        selected = frame.loc[
            frame.Retrieval_Observed.eq(1) & frame[column].notna()
        ].copy()
        features = (
            bwh_tasks.BASE_FEATURES if name == "egg_count" else bwh_tasks.MII_FEATURES
        )
        return (
            selected,
            selected[column].to_numpy(float),
            features,
            "count",
            schemas.BWH_CAT,
        )
    selected, features, target, kind = bwh_tasks.task_frame(frame, name)
    return selected, selected[target].to_numpy(float), features, kind, schemas.BWH_CAT


def score(y, prediction, kind: str) -> float:
    if kind == "binary":
        return float(roc_auc_score(y, prediction))
    return float(mean_absolute_error(y, prediction))


def fixed_predict(train, y, valid, features, categorical, kind, fold):
    if len(np.unique(y)) == 1:
        return np.full(len(valid), float(y[0]))
    encoder = core.Encoder().fit(train, features, categorical)
    x = encoder.transform(train)
    v = encoder.transform(valid)
    kwargs = dict(HGB_BASE, max_iter=180, random_state=202609282 + fold)
    model = (
        HistGradientBoostingClassifier(**kwargs)
        if kind == "binary"
        else HistGradientBoostingRegressor(**kwargs)
    )
    model.fit(x, y)
    prediction = (
        model.predict_proba(v)[:, 1]
        if kind == "binary"
        else np.clip(model.predict(v), 1e-8, None)
    )
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return prediction


def hgb_predict(train, y, valid, features, categorical, kind, fold, max_iter):
    if len(np.unique(y)) == 1:
        return np.full(len(valid), float(y[0]))
    encoder = core.Encoder().fit(train, features, categorical)
    x = encoder.transform(train)
    v = encoder.transform(valid)
    kwargs = dict(HGB_BASE, max_iter=max_iter, random_state=2026100400 + fold)
    model = (
        HistGradientBoostingClassifier(**kwargs)
        if kind == "binary"
        else HistGradientBoostingRegressor(**kwargs)
    )
    model.fit(x, y)
    return (
        model.predict_proba(v)[:, 1]
        if kind == "binary"
        else np.clip(model.predict(v), 1e-8, None)
    )


def tune_real(train, valid, cohort, name, fold, patient):
    selected, y, features, kind, categorical = task(train, cohort, name)
    outer, target, _, _, _ = task(valid, cohort, name)
    groups = selected[patient].astype(str).to_numpy()
    candidates = [60, 120, 180, 300]
    rows = []
    splitter = GroupKFold(n_splits=3)
    for max_iter in candidates:
        values = []
        for inner, (ti, vi) in enumerate(splitter.split(selected, y, groups)):
            prediction = hgb_predict(
                selected.iloc[ti],
                y[ti],
                selected.iloc[vi],
                features,
                categorical,
                kind,
                fold * 10 + inner,
                max_iter,
            )
            if kind == "binary" and len(np.unique(y[vi])) < 2:
                continue
            values.append(score(y[vi], prediction, kind))
        mean = float(np.mean(values)) if values else np.nan
        rows.append(
            dict(
                cohort=cohort,
                fold=fold,
                task=name,
                max_iter=max_iter,
                inner_score=mean,
                inner_folds=len(values),
            )
        )
    finite = [r for r in rows if np.isfinite(r["inner_score"])]
    if not finite:
        chosen = 180
    elif kind == "binary":
        chosen = sorted(finite, key=lambda r: (-r["inner_score"], r["max_iter"]))[0][
            "max_iter"
        ]
    else:
        chosen = sorted(finite, key=lambda r: (r["inner_score"], r["max_iter"]))[0][
            "max_iter"
        ]
    prediction = hgb_predict(
        selected, y, outer, features, categorical, kind, fold, chosen
    )
    return prediction, target, outer, kind, chosen, rows


def sample_count_no_missing(model, frame, draws, seed, repair_total=True):
    rng = np.random.default_rng(seed)
    n_rows = len(frame)
    stage = model.stage
    if stage.kind == "positive_count":
        mu = np.clip(model.mean.predict(frame), 0, None)
        if model.dispersion == 0:
            intensity = np.broadcast_to(mu, (draws, n_rows))
        else:
            intensity = rng.gamma(
                1 / model.dispersion, mu * model.dispersion, size=(draws, n_rows)
            )
        values = 1 + rng.poisson(intensity)
    else:
        total_raw = pd.to_numeric(frame[stage.total], errors="coerce").to_numpy(float)
        total = np.nan_to_num(total_raw, nan=0).round().clip(0).astype(int)
        probability = (
            model.mean.predict(frame)
            if model.mean is not None
            else np.full(n_rows, 0.5)
        )
        rho = model.dispersion if model.mean is not None else 0.002
        concentration = 1 / rho - 1
        beta = rng.beta(
            probability * concentration,
            (1 - probability) * concentration,
            size=(draws, n_rows),
        )
        values = rng.binomial(np.broadcast_to(total, (draws, n_rows)), beta).astype(
            float
        )
        if model.exception_probability and len(model.excess):
            exceptions = rng.random((draws, n_rows)) < model.exception_probability
            excess = rng.choice(model.excess, size=(draws, n_rows))
            replacement = np.minimum(total[None, :] + excess, model.upper)
            values[exceptions] = replacement[exceptions]
        values[:, ~np.isfinite(total_raw)] = np.nan
    values = np.clip(values, model.lower, model.upper)
    if stage.total and not stage.allow_excess and repair_total:
        total = pd.to_numeric(frame[stage.total], errors="coerce").to_numpy(float)
        values = np.minimum(values, total[None, :])
    return values


def direct_antalya(model, valid, name, fold):
    allowed = {
        "poor_response": "eggs",
        "high_response": "eggs",
        "egg_count": "eggs",
        "high_maturation": "mii",
        "mii_count": "mii",
        "fertilized_count": "fertilized",
        "blastocyst_count": "blastocysts",
        "frozen_blastocyst_count": "frozen_blastocysts",
    }
    if name not in allowed:
        return None
    selected, target, features, kind, _ = task(valid, "antalya", name)
    restored = model.restore(selected)
    index = {stage.target: i for i, stage in enumerate(model.schema.stages)}
    stage_name = allowed[name]
    stage = model.schema.stages[index[stage_name]]
    stage_model = model.models[index[stage_name]]
    stage_inputs = tuple(stage.features)
    if tuple(features) != stage_inputs:
        raise ValueError(
            f"Direct-input mismatch for {name}: {features} versus {stage_inputs}"
        )
    values = sample_count_no_missing(
        stage_model, restored, 4000, 2026100600 + 10 * fold + index[stage_name]
    )
    if name == "poor_response":
        prediction = np.nanmean(values <= 3, axis=0)
    elif name == "high_response":
        prediction = np.nanmean(values >= 20, axis=0)
    elif name == "high_maturation":
        eggs = pd.to_numeric(restored.eggs, errors="coerce").to_numpy(float)
        prediction = np.nanmean(values / eggs[None, :] >= 0.75, axis=0)
    else:
        prediction = np.nanmean(values, axis=0)
    return selected, target, prediction, kind, stage_name


def run_core():
    destination = OUT / "core"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = []
    predictions = []
    tuning = []
    task_counts = []
    subset_manifest = []
    for cohort_index, cohort in enumerate(["antalya", "bwh"]):
        schema = schema_for(cohort)
        patient = schema.patient
        cycle = schema.cycle
        for fold in range(4):
            print(
                json.dumps({"phase": "core", "cohort": cohort, "fold": fold}),
                flush=True,
            )
            train, valid = load_real(cohort, fold)
            evaluation_dir = EVALUATION / cohort / f"fold_{fold}"
            synthetic_20x = pd.read_parquet(evaluation_dir / "synthetic.parquet")
            model = joblib.load(evaluation_dir / "model.joblib")
            count = train[patient].nunique()
            seed = 2026100400 + 100 * cohort_index + fold
            rng = np.random.default_rng(seed)
            chosen_patients = rng.choice(
                synthetic_20x[patient].unique(), size=count, replace=False
            )
            synthetic_1x = synthetic_20x.loc[
                synthetic_20x[patient].isin(chosen_patients)
            ].copy()
            if synthetic_1x[patient].nunique() != count:
                raise AssertionError("1x patient count mismatch")
            subset_path = destination / "one_x" / cohort / f"fold_{fold}.parquet"
            subset_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_1x.to_parquet(subset_path, index=False)
            subset_manifest.append(
                dict(
                    cohort=cohort,
                    fold=fold,
                    seed=seed,
                    real_training_patients=count,
                    synthetic_patients=synthetic_1x[patient].nunique(),
                    synthetic_cycles=len(synthetic_1x),
                    sha256=digest(subset_path),
                )
            )

            for name in TASKS[cohort]:
                valid_task, valid_y, _, kind, _ = task(valid, cohort, name)
                row = dict(
                    cohort=cohort,
                    fold=fold,
                    task=name,
                    kind=kind,
                    eligible_records=len(valid_task),
                    eligible_patients=valid_task[patient].nunique(),
                )
                if kind == "binary":
                    row.update(
                        positive=int(valid_y.sum()), prevalence=float(valid_y.mean())
                    )
                else:
                    row.update(positive=np.nan, prevalence=np.nan)
                task_counts.append(row)

                selected, y, features, kind, categorical = task(
                    synthetic_1x, cohort, name
                )
                outer, target, _, _, _ = task(valid, cohort, name)
                started = time.monotonic()
                prediction = fixed_predict(
                    selected, y, outer, features, categorical, kind, fold
                )
                metrics.append(
                    dict(
                        cohort=cohort,
                        fold=fold,
                        condition="sc_ivf_1x_fixed",
                        task=name,
                        family="hgb",
                        kind=kind,
                        primary=score(target, prediction, kind),
                        training_rows=len(selected),
                        validation_rows=len(outer),
                        validation_patients=outer[patient].nunique(),
                        seconds=time.monotonic() - started,
                    )
                )
                pred = pd.DataFrame(
                    dict(
                        patient=outer[patient].astype(str),
                        cycle=outer[cycle].astype(str),
                        y=target,
                        p=prediction,
                    )
                )
                pred = pred.assign(
                    cohort=cohort,
                    fold=fold,
                    condition="sc_ivf_1x_fixed",
                    task=name,
                    family="hgb",
                    kind=kind,
                )
                predictions.append(pred)

                prediction, target, outer, kind, chosen, inner = tune_real(
                    train, valid, cohort, name, fold, patient
                )
                tuning.extend(inner)
                metrics.append(
                    dict(
                        cohort=cohort,
                        fold=fold,
                        condition="real_tuned_iter",
                        task=name,
                        family="hgb",
                        kind=kind,
                        primary=score(target, prediction, kind),
                        training_rows=len(task(train, cohort, name)[0]),
                        validation_rows=len(outer),
                        validation_patients=outer[patient].nunique(),
                        chosen_max_iter=chosen,
                    )
                )
                pred = pd.DataFrame(
                    dict(
                        patient=outer[patient].astype(str),
                        cycle=outer[cycle].astype(str),
                        y=target,
                        p=prediction,
                    )
                )
                pred = pred.assign(
                    cohort=cohort,
                    fold=fold,
                    condition="real_tuned_iter",
                    task=name,
                    family="hgb",
                    kind=kind,
                )
                predictions.append(pred)

                if cohort == "antalya":
                    direct = direct_antalya(model, valid, name, fold)
                    if direct is not None:
                        outer, target, prediction, kind, stage_name = direct
                        metrics.append(
                            dict(
                                cohort=cohort,
                                fold=fold,
                                condition="direct_stage",
                                task=name,
                                family="direct_stage",
                                kind=kind,
                                primary=score(target, prediction, kind),
                                training_rows=np.nan,
                                validation_rows=len(outer),
                                validation_patients=outer[patient].nunique(),
                                stage=stage_name,
                            )
                        )
                        pred = pd.DataFrame(
                            dict(
                                patient=outer[patient].astype(str),
                                cycle=outer[cycle].astype(str),
                                y=target,
                                p=prediction,
                            )
                        )
                        pred = pred.assign(
                            cohort=cohort,
                            fold=fold,
                            condition="direct_stage",
                            task=name,
                            family="direct_stage",
                            kind=kind,
                        )
                        predictions.append(pred)

    pd.DataFrame(metrics).to_csv(destination / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        destination / "predictions.parquet", index=False
    )
    pd.DataFrame(tuning).to_csv(destination / "tuning.csv", index=False)
    pd.DataFrame(task_counts).to_csv(destination / "task_counts.csv", index=False)
    pd.DataFrame(subset_manifest).to_csv(
        destination / "one_x_manifest.csv", index=False
    )
    save_json(
        destination / "DONE.json",
        {
            "status": "complete",
            "reserved_partitions_accessed": False,
            "files": {
                p.name: digest(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "DONE.json"
            },
        },
    )


def run_release_variability():
    destination = OUT / "release_variability"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = []
    manifests = []
    for cohort_index, cohort in enumerate(["antalya", "bwh"]):
        schema = schema_for(cohort)
        patient = schema.patient
        for fold in range(4):
            train, valid = load_real(cohort, fold)
            evaluation_dir = EVALUATION / cohort / f"fold_{fold}"
            scaffold = pd.read_parquet(evaluation_dir / "synthetic.parquet")
            model = joblib.load(evaluation_dir / "model.joblib")
            for repetition in range(5):
                seed = 2026100500 + 100 * cohort_index + 10 * fold + repetition
                print(
                    json.dumps(
                        {
                            "phase": "release",
                            "cohort": cohort,
                            "fold": fold,
                            "repetition": repetition,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )
                generated = model.generate(scaffold, seed)
                path = (
                    destination
                    / "releases"
                    / cohort
                    / f"fold_{fold}"
                    / f"release_{repetition}.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                generated.to_parquet(path, index=False)
                manifests.append(
                    dict(
                        cohort=cohort,
                        fold=fold,
                        repetition=repetition,
                        seed=seed,
                        patients=generated[patient].nunique(),
                        cycles=len(generated),
                        sha256=digest(path),
                    )
                )
                for name in TASKS[cohort]:
                    selected, y, features, kind, categorical = task(
                        generated, cohort, name
                    )
                    outer, target, _, _, _ = task(valid, cohort, name)
                    prediction = fixed_predict(
                        selected, y, outer, features, categorical, kind, fold
                    )
                    metrics.append(
                        dict(
                            cohort=cohort,
                            fold=fold,
                            repetition=repetition,
                            seed=seed,
                            task=name,
                            family="hgb",
                            kind=kind,
                            primary=score(target, prediction, kind),
                            training_rows=len(selected),
                            validation_rows=len(outer),
                        )
                    )
    pd.DataFrame(metrics).to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(manifests).to_csv(destination / "release_manifest.csv", index=False)
    save_json(
        destination / "DONE.json",
        {
            "status": "complete",
            "variability_scope": "conditional-stage sampling with fitted model and HMA scaffold fixed",
            "reserved_partitions_accessed": False,
        },
    )


def patient_summary(frame, schema):
    restored = core.SharedIVF(schema).restore(frame)
    columns = [c for c in restored if c not in [schema.patient, schema.cycle]]
    categorical = [c for c in columns if c in schema.categorical]
    numeric = [c for c in columns if c not in categorical]
    output = restored.groupby(schema.patient, sort=True)[numeric].median()
    for column in categorical:
        output[column] = restored.groupby(schema.patient, sort=True)[column].agg(
            lambda x: sorted(x.fillna("__MISSING__").astype(str).mode())[0]
        )
    output["cycle_count"] = restored.groupby(schema.patient).size()
    return output, numeric + ["cycle_count"], categorical


def encode(train, data, numeric, categorical):
    low = train[numeric].min().fillna(0)
    width = (train[numeric].max() - low).replace(0, 1).fillna(1)
    median = train[numeric].median().fillna(0)
    blocks = [
        0.5 * ((data[numeric].fillna(median) - low) / width).clip(0, 1).to_numpy(float),
        0.5 * data[numeric].isna().to_numpy(float),
    ]
    for column in categorical:
        levels = sorted(train[column].fillna("__MISSING__").astype(str).unique())
        mapping = {value: i for i, value in enumerate(levels)}
        index = (
            data[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(len(levels))
            .to_numpy(int)
        )
        block = np.zeros((len(data), len(levels) + 1))
        block[np.arange(len(data)), index] = 0.5
        blocks.append(block)
    return np.column_stack(blocks) / (len(numeric) + len(categorical))


def bootstrap_auc(labels, scores, rng, draws=2000):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(draws):
        index = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        values.append(roc_auc_score(labels[index], scores[index]))
    return np.asarray(values)


def run_diagnostics():
    destination = OUT / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    membership_fold = []
    membership_draws = {"antalya": [], "bwh": []}
    for cohort_index, cohort in enumerate(["antalya", "bwh"]):
        schema = schema_for(cohort)
        for fold in range(4):
            print(
                json.dumps({"phase": "diagnostics", "cohort": cohort, "fold": fold}),
                flush=True,
            )
            train, valid = load_real(cohort, fold)
            synthetic = pd.read_parquet(
                EVALUATION / cohort / f"fold_{fold}" / "synthetic.parquet"
            )
            training_summary, numeric, categorical = patient_summary(train, schema)
            validation_summary, _, _ = patient_summary(valid, schema)
            synthetic_summary, _, _ = patient_summary(synthetic, schema)
            n = min(263, len(training_summary), len(validation_summary))
            training_candidates = training_summary.sample(
                n, random_state=2026090930 + fold
            )
            validation_candidates = validation_summary.sample(
                n, random_state=2026090930 + fold
            )
            candidates = pd.concat([training_candidates, validation_candidates])
            encoded_candidates = encode(
                training_summary, candidates, numeric, categorical
            )
            encoded_synthetic = encode(
                training_summary, synthetic_summary, numeric, categorical
            )
            nearest = []
            for start in range(0, len(encoded_candidates), 32):
                nearest.extend(
                    cdist(
                        encoded_candidates[start : start + 32],
                        encoded_synthetic,
                        metric="cityblock",
                    ).min(axis=1)
                )
            labels = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
            scores = -np.asarray(nearest)
            auc = float(roc_auc_score(labels, scores))
            draws = bootstrap_auc(
                labels,
                scores,
                np.random.default_rng(2026100700 + 100 * cohort_index + fold),
            )
            membership_draws[cohort].append(draws)
            membership_fold.append(
                dict(
                    cohort=cohort,
                    fold=fold,
                    auc=auc,
                    members=n,
                    nonmembers=n,
                    exact_member_matches=int(np.sum(np.asarray(nearest[:n]) <= 1e-12)),
                    exact_nonmember_matches=int(
                        np.sum(np.asarray(nearest[n:]) <= 1e-12)
                    ),
                    member_median_distance=float(np.median(nearest[:n])),
                    nonmember_median_distance=float(np.median(nearest[n:])),
                )
            )

    aggregate = []
    for cohort, fold_draws in membership_draws.items():
        mean_draws = np.mean(fold_draws, axis=0)
        point = float(
            np.mean([row["auc"] for row in membership_fold if row["cohort"] == cohort])
        )
        low, high = np.quantile(mean_draws, [0.025, 0.975])
        aggregate.append(
            dict(
                cohort=cohort,
                mean_auc=point,
                bootstrap_low=float(low),
                bootstrap_high=float(high),
                draws=len(mean_draws),
                scope="conditional on the fitted release and sampled attack candidates",
            )
        )
    pd.DataFrame(membership_fold).to_csv(
        destination / "membership_fold.csv", index=False
    )
    pd.DataFrame(aggregate).to_csv(
        destination / "membership_aggregate.csv", index=False
    )
    reference_fidelity = pd.read_csv(EVALUATION / "aggregate/fidelity.csv")
    reference_fidelity.to_csv(destination / "reference_fidelity.csv", index=False)
    save_json(
        destination / "DONE.json",
        {"status": "complete", "reserved_partitions_accessed": False},
    )


def main():
    global EVALUATION, OUT
    parser = argparse.ArgumentParser(
        description="Run release-size, tuning, sampling, and disclosure checks."
    )
    parser.add_argument("phase", choices=["core", "release", "diagnostics"])
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    EVALUATION = safe_path(args.evaluation)
    OUT = safe_path(args.output)
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "core":
        run_core()
    elif args.phase == "release":
        run_release_variability()
    else:
        run_diagnostics()


if __name__ == "__main__":
    main()
