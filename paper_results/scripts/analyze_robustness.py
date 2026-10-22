from __future__ import annotations

from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "output/robustness"
EVALUATION = ROOT / "output/evaluation"
SUMMARY = RESULTS / "summary"

PRIMARY = {
    "antalya": [
        "poor_response",
        "high_response",
        "egg_count",
        "high_maturation",
        "mii_count",
        "fertilized_count",
        "blastocyst_count",
        "frozen_blastocyst_count",
        "first_fet_live_birth",
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


def metric(y, p, weights, kind):
    if kind == "count":
        return weights @ np.abs(y - p) / weights.sum(axis=1)
    order = np.argsort(p, kind="stable")
    p = p[order]
    y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(p) != 0) + 1]
    w = weights[:, order]
    positive = np.add.reduceat(w * y, starts, axis=1)
    negative = np.add.reduceat(w * (1 - y), starts, axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (negative.cumsum(axis=1) - 0.5 * negative)).sum(axis=1)
    return np.divide(
        numerator, denominator, out=np.full(len(w), np.nan), where=denominator > 0
    )


def load_predictions():
    primary = []
    for cohort in ["antalya", "bwh"]:
        for fold in range(4):
            copied = EVALUATION / cohort / f"fold_{fold}.parquet"
            original = EVALUATION / cohort / f"fold_{fold}" / "predictions.parquet"
            frame = pd.read_parquet(copied if copied.exists() else original)
            frame = frame.loc[frame.method.isin(["real", "shared_ivf_v1_1"])].copy()
            frame["condition"] = frame.method.map(
                {"real": "real_fixed", "shared_ivf_v1_1": "sc_ivf_20x_fixed"}
            )
            primary.append(frame.drop(columns="method"))
    new = pd.read_parquet(RESULTS / "core" / "predictions.parquet")
    return pd.concat([*primary, new], ignore_index=True)


def paired_comparisons(full):
    comparisons = [
        ("sc_ivf_1x_fixed", "sc_ivf_20x_fixed", "hgb", "hgb"),
        ("real_tuned_iter", "real_fixed", "hgb", "hgb"),
        ("sc_ivf_20x_fixed", "real_tuned_iter", "hgb", "hgb"),
        ("direct_stage", "sc_ivf_20x_fixed", "direct_stage", "hgb"),
    ]
    # Restore each published comparison's bootstrap stream independently.
    states = json.loads(
        (ROOT / "paper_results/tables/bootstrap_states.json").read_text()
    )
    rows = []
    for cohort, tasks in PRIMARY.items():
        for task in tasks:
            for left, right, left_family, right_family in comparisons:
                a = full.loc[
                    (full.cohort == cohort)
                    & (full.task == task)
                    & (full.condition == left)
                    & (full.family == left_family)
                ]
                b = full.loc[
                    (full.cohort == cohort)
                    & (full.task == task)
                    & (full.condition == right)
                    & (full.family == right_family)
                ]
                if a.empty or b.empty:
                    continue
                checkpoint = states["/".join([cohort, task, left, right])]
                rng = np.random.default_rng()
                rng.bit_generator.state = checkpoint["state"]
                kind = a.kind.iloc[0]
                fold_a = []
                fold_b = []
                points_a = []
                points_b = []
                for fold in range(4):
                    x = (
                        a.loc[a.fold == fold]
                        .sort_values("cycle")
                        .reset_index(drop=True)
                    )
                    y = (
                        b.loc[b.fold == fold]
                        .sort_values("cycle")
                        .reset_index(drop=True)
                    )
                    if not x[["patient", "cycle", "y"]].equals(
                        y[["patient", "cycle", "y"]]
                    ):
                        raise AssertionError((cohort, task, left, right, fold))
                    patients = x.patient.unique()
                    if len(patients) != checkpoint["fold_patients"][fold]:
                        raise ValueError(
                            "Patient counts differ from the published bootstrap"
                        )
                    mapping = {patient: i for i, patient in enumerate(patients)}
                    index = x.patient.map(mapping).to_numpy()
                    weights = rng.multinomial(
                        len(patients),
                        np.full(len(patients), 1 / len(patients)),
                        size=2000,
                    )[:, index]
                    target = x.y.to_numpy(float)
                    draw_a = np.concatenate(
                        [
                            metric(
                                target,
                                x.p.to_numpy(float),
                                weights[start : start + 128],
                                kind,
                            )
                            for start in range(0, 2000, 128)
                        ]
                    )
                    draw_b = np.concatenate(
                        [
                            metric(
                                target,
                                y.p.to_numpy(float),
                                weights[start : start + 128],
                                kind,
                            )
                            for start in range(0, 2000, 128)
                        ]
                    )
                    fold_a.append(draw_a)
                    fold_b.append(draw_b)
                    unit = np.ones((1, len(target)))
                    points_a.append(
                        float(metric(target, x.p.to_numpy(float), unit, kind)[0])
                    )
                    points_b.append(
                        float(metric(target, y.p.to_numpy(float), unit, kind)[0])
                    )
                draw_a = np.mean(fold_a, axis=0)
                draw_b = np.mean(fold_b, axis=0)
                complete = np.isfinite(draw_a) & np.isfinite(draw_b)
                sign = 1 if kind == "binary" else -1
                gain = sign * (draw_a - draw_b)
                low, high = np.quantile(gain[complete], [0.025, 0.975])
                rows.append(
                    dict(
                        cohort=cohort,
                        task=task,
                        kind=kind,
                        left=left,
                        right=right,
                        left_family=left_family,
                        right_family=right_family,
                        left_score=float(np.mean(points_a)),
                        right_score=float(np.mean(points_b)),
                        gain=float(sign * (np.mean(points_a) - np.mean(points_b))),
                        low=float(low),
                        high=float(high),
                        retained_draws=int(complete.sum()),
                    )
                )
    return pd.DataFrame(rows)


def release_summary():
    frame = pd.read_csv(RESULTS / "release_variability" / "metrics.csv")
    by_release = (
        frame.groupby(["cohort", "task", "kind", "repetition"], sort=False)
        .primary.mean()
        .reset_index()
    )
    summary = (
        by_release.groupby(["cohort", "task", "kind"], sort=False)
        .primary.agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary["range"] = summary["max"] - summary["min"]
    return by_release, summary


def task_counts():
    frame = pd.read_csv(RESULTS / "core" / "task_counts.csv")
    grouped = frame.groupby(["cohort", "task", "kind"], sort=False)
    output = grouped.agg(
        eligible_records=("eligible_records", "sum"),
        fold_records_min=("eligible_records", "min"),
        fold_records_max=("eligible_records", "max"),
        unique_patients_sum=("eligible_patients", "sum"),
        positive=("positive", "sum"),
    ).reset_index()
    output["prevalence"] = output.positive / output.eligible_records
    output.loc[output.kind == "count", ["positive", "prevalence"]] = np.nan
    return output


def main():
    global RESULTS, EVALUATION, SUMMARY
    parser = argparse.ArgumentParser(
        description="Aggregate the additional experiments."
    )
    parser.add_argument("--evaluation", type=Path, default=EVALUATION)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS, EVALUATION = args.results.resolve(), args.evaluation.resolve()
    SUMMARY = RESULTS / "summary"
    SUMMARY.mkdir(parents=True, exist_ok=True)
    full = load_predictions()
    paired = paired_comparisons(full)
    paired.to_csv(SUMMARY / "paired_comparisons.csv", index=False)
    by_release, release = release_summary()
    by_release.to_csv(SUMMARY / "release_scores.csv", index=False)
    release.to_csv(SUMMARY / "release_variability.csv", index=False)
    task_counts().to_csv(SUMMARY / "task_counts.csv", index=False)
    hashes = {}
    for path in sorted(SUMMARY.glob("*.csv")):
        import hashlib

        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (SUMMARY / "MANIFEST.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
