from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

P = Path(__file__).resolve().parents[2]
R = P / "output/evaluation"
W = Path(__file__).resolve().parent
OUT = R / "aggregate"
sys.path.insert(0, str(W))
from run_prediction import sha
from shared_ivf.engine import SharedIVF
from shared_ivf.schemas import SCHEMAS


def metrics(y, p, weights, kind):
    if kind == "count":
        return weights @ np.abs(y - p) / weights.sum(axis=1)
    order = np.argsort(p, kind="stable")
    pred = p[order]
    y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pred) != 0) + 1]
    w = weights[:, order]
    pos = np.add.reduceat(w * y, starts, axis=1)
    neg = np.add.reduceat(w * (1 - y), starts, axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    num = (pos * (neg.cumsum(axis=1) - 0.5 * neg)).sum(axis=1)
    return np.divide(num, den, out=np.full(len(w), np.nan), where=den > 0)


def inference():
    absolute = []
    paired = []
    data = []
    fold_metrics = []
    for cohort in ["antalya", "bwh"]:
        for fold in range(4):
            root = R / cohort / f"fold_{fold}"
            done = json.loads((root / "DONE.json").read_text())
            assert done["status"] == "complete"
            for name, digest in done["artifacts"].items():
                assert sha(root / name) == digest, name
            data.append(pd.read_parquet(root / "predictions.parquet"))
            fold_metrics.append(pd.read_csv(root / "metrics.csv"))
    full = pd.concat(data, ignore_index=True)
    fm = pd.concat(fold_metrics, ignore_index=True)
    fm.to_csv(OUT / "fold_metrics.csv", index=False)
    means = (
        fm.groupby(["cohort", "task", "family", "method", "kind"], sort=False)[
            ["primary", "proper"]
        ]
        .mean()
        .reset_index()
    )
    means.to_csv(OUT / "means.csv", index=False)
    rng = np.random.default_rng(2026090920)
    for (cohort, task, family), group in full.groupby(
        ["cohort", "task", "family"], sort=True
    ):
        methods = sorted(group.method.unique())
        kind = group.kind.iloc[0]
        draws = {m: [] for m in methods}
        points = {m: [] for m in methods}
        for fold in range(4):
            part = group.loc[group.fold.eq(fold)]
            ref = (
                part.loc[part.method.eq("real")]
                .sort_values("cycle")
                .reset_index(drop=True)
            )
            ids = ref.patient.unique()
            mapping = {v: i for i, v in enumerate(ids)}
            index = ref.patient.map(mapping).to_numpy()
            weights = rng.multinomial(
                len(ids), np.full(len(ids), 1 / len(ids)), size=2000
            )[:, index]
            for method in methods:
                item = (
                    part.loc[part.method.eq(method)]
                    .sort_values("cycle")
                    .reset_index(drop=True)
                )
                assert item[["patient", "cycle", "y"]].equals(
                    ref[["patient", "cycle", "y"]]
                ), (cohort, task, method)
                y = item.y.to_numpy()
                p = item.p.to_numpy()
                draws[method].append(
                    np.concatenate(
                        [
                            metrics(y, p, weights[a : a + 128], kind)
                            for a in range(0, 2000, 128)
                        ]
                    )
                )
                points[method].append(
                    float(metrics(y, p, np.ones((1, len(y))), kind)[0])
                )
        draws = {m: np.mean(draws[m], axis=0) for m in methods}
        points = {m: float(np.mean(points[m])) for m in methods}
        complete = np.logical_and.reduce([np.isfinite(v) for v in draws.values()])
        assert complete.sum() >= 500, (
            cohort,
            task,
            "too few evaluable bootstrap draws",
        )
        for method in methods:
            lo, hi = np.quantile(draws[method][complete], [0.025, 0.975])
            absolute.append(
                dict(
                    cohort=cohort,
                    task=task,
                    family=family,
                    method=method,
                    kind=kind,
                    mean=points[method],
                    low=lo,
                    high=hi,
                    retained_draws=int(complete.sum()),
                    attempted_draws=2000,
                )
            )
            if method == "shared_ivf_v1_1":
                continue
            sign = 1 if kind == "binary" else -1
            delta = sign * (draws["shared_ivf_v1_1"] - draws[method])
            lo, hi = np.quantile(delta[complete], [0.025, 0.975])
            paired.append(
                dict(
                    cohort=cohort,
                    task=task,
                    family=family,
                    comparator=method,
                    kind=kind,
                    gain=sign * (points["shared_ivf_v1_1"] - points[method]),
                    low=lo,
                    high=hi,
                    retained_draws=int(complete.sum()),
                )
            )
        print(
            json.dumps(
                dict(
                    stage="inference",
                    cohort=cohort,
                    task=task,
                    family=family,
                    retained=int(complete.sum()),
                )
            ),
            flush=True,
        )
    pd.DataFrame(absolute).to_csv(OUT / "absolute.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False)


def load_frames(cohort, fold):
    plan = json.loads((R / "INPUTS.json").read_text())
    job = next((j for j in plan["jobs"] if j["cohort"] == cohort and j["fold"] == fold))

    def read(path):
        assert sha(path) == plan["files"][str(path)]
        return (
            pd.read_parquet(path)
            if str(path).endswith(".parquet")
            else pd.read_csv(path, float_precision="round_trip")
        )

    if cohort == "antalya":
        train, valid = (read(job["training"]), read(job["validation"]))
    else:
        d = job["development"]
        p = read(d["development_parents.csv"])
        c = read(d["development_cycles.csv"])
        f = read(d["development_folds.csv"])
        full = c.merge(p, on="Patient_ID", validate="many_to_one").merge(
            f, on="Patient_ID", validate="many_to_one"
        )
        train = full.loc[full.Fold.ne(fold)].drop(columns="Fold")
        valid = full.loc[full.Fold.eq(fold)].drop(columns="Fold")
    info = job["methods"]["hma"]
    if "frame" in info:
        hma = read(info["frame"])
    else:
        hma = pd.concat(
            [
                read(b["cycles.parquet"]).merge(
                    read(b["parents.parquet"]), on="Patient_ID", validate="many_to_one"
                )
                for b in info["batches"]
            ],
            ignore_index=True,
        )
    synthetic = pd.read_parquet(R / cohort / f"fold_{fold}" / "synthetic.parquet")
    return (train, valid, synthetic, hma)


def summary(frame, schema):
    f = SharedIVF(schema).restore(frame)
    columns = [c for c in f if c not in [schema.patient, schema.cycle]]
    cat = [c for c in columns if c in schema.categorical]
    num = [c for c in columns if c not in cat]
    out = f.groupby(schema.patient, sort=True)[num].median()
    for c in cat:
        out[c] = f.groupby(schema.patient, sort=True)[c].agg(
            lambda x: sorted(x.fillna("__MISSING__").astype(str).mode())[0]
        )
    out["cycle_count"] = f.groupby(schema.patient).size()
    return (out, num + ["cycle_count"], cat)


def encode(train, data, num, cat):
    lo = train[num].min().fillna(0)
    width = (train[num].max() - lo).replace(0, 1).fillna(1)
    med = train[num].median().fillna(0)
    blocks = [
        0.5 * ((data[num].fillna(med) - lo) / width).clip(0, 1).to_numpy(float),
        0.5 * data[num].isna().to_numpy(float),
    ]
    for c in cat:
        levels = sorted(train[c].fillna("__MISSING__").astype(str).unique())
        mp = {v: i for i, v in enumerate(levels)}
        ix = (
            data[c]
            .fillna("__MISSING__")
            .astype(str)
            .map(mp)
            .fillna(len(levels))
            .to_numpy(int)
        )
        block = np.zeros((len(data), len(levels) + 1))
        block[np.arange(len(data)), ix] = 0.5
        blocks.append(block)
    return np.column_stack(blocks) / (len(num) + len(cat))


def quality():
    output = []
    fidelity = []
    for cohort in ["antalya", "bwh"]:
        s = SCHEMAS[cohort]()
        for fold in range(4):
            train, valid, syn, hma = load_frames(cohort, fold)
            tr, num, cat = summary(train, s)
            va, _, _ = summary(valid, s)
            n = min(263, len(tr), len(va))
            tr_candidates = tr.sample(n, random_state=2026090930 + fold)
            va_candidates = va.sample(n, random_state=2026090930 + fold)
            candidates = pd.concat([tr_candidates, va_candidates])
            x = encode(tr, candidates, num, cat)
            labels = np.r_[np.ones(n), np.zeros(n)]
            for method, frame in [("shared_ivf_v1_1", syn), ("hma", hma)]:
                su, _, _ = summary(frame, s)
                z = encode(tr, su, num, cat)
                nearest = []
                for i in range(0, len(x), 32):
                    nearest.extend(
                        cdist(x[i : i + 32], z, metric="cityblock").min(axis=1)
                    )
                output.append(
                    dict(
                        cohort=cohort,
                        fold=fold,
                        method=method,
                        membership_auc=roc_auc_score(labels, -np.array(nearest)),
                        members=n,
                        nonmembers=n,
                        synthetic_patients=len(su),
                    )
                )
                real = SharedIVF(s).restore(train)
                generated = SharedIVF(s).restore(frame)
                for stage in s.stages:
                    if stage.kind not in ["positive_count", "bounded_count"]:
                        continue
                    a = pd.to_numeric(real[stage.target], errors="coerce").dropna()
                    b = pd.to_numeric(generated[stage.target], errors="coerce").dropna()
                    distance = (
                        wasserstein_distance(a, b) if len(a) and len(b) else np.nan
                    )
                    fidelity.append(
                        dict(
                            cohort=cohort,
                            fold=fold,
                            method=method,
                            target=stage.target,
                            wasserstein=distance,
                            normalized_wasserstein=distance
                            / max(float(a.std()), 1e-08),
                            training_mean=a.mean(),
                            synthetic_mean=b.mean(),
                        )
                    )
                print(
                    json.dumps(
                        dict(stage="quality", cohort=cohort, fold=fold, method=method)
                    ),
                    flush=True,
                )
    pd.DataFrame(output).to_csv(OUT / "membership.csv", index=False)
    pd.DataFrame(fidelity).to_csv(OUT / "fidelity.csv", index=False)
