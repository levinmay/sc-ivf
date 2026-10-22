from pathlib import Path
import os, sys, json, hashlib, time, traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT / "output/evaluation"
METHODS = ["shared_ivf_v1_1", "hma", "copula", "tvae300", "real"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".tmp")
    q.write_text(
        json.dumps(
            x, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
        + "\n"
    )
    q.replace(p)


def guard(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = Path(os.fsdecode(args[0])).resolve()
        if {"sealed_test", "historical_holdout", "fold_4", "data_full_pathway"} & set(
            p.parts
        ) or p.suffix in [".xlsx", ".xls"]:
            raise PermissionError("Shared method development boundary")


def setup():
    sys.addaudithook(guard)
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def task(frame, cohort, name):

    if cohort == "antalya":
        import antalya_tasks as t

        f, y = t.task_frame(frame, name)
        return (f, y, t.FEATURES[name], t.TASK_KIND[name], t.CAT)
    import bwh_tasks as t
    from shared_ivf.schemas import BWH_CAT

    if name in ["egg_count", "mii_count"]:
        col = "Sum_Eggs" if name == "egg_count" else "Sum_MII"
        f = frame.loc[frame.Retrieval_Observed.eq(1) & frame[col].notna()].copy()
        return (
            f,
            f[col].to_numpy(float),
            t.BASE_FEATURES if name == "egg_count" else t.MII_FEATURES,
            "count",
            BWH_CAT,
        )
    f, features, target, kind = t.task_frame(frame, name)
    return (f, f[target].to_numpy(float), features, kind, BWH_CAT)


def predict(train, y, valid, features, cats, kind, family, fold):
    import numpy as np
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from shared_ivf.engine import Encoder

    if not len(y):
        raise ValueError("No eligible synthetic training records")
    if len(np.unique(y)) == 1:
        return np.full(len(valid), float(y[0]))
    if family == "hgb":
        enc = Encoder().fit(train, features, cats)
        x = enc.transform(train)
        v = enc.transform(valid)
        kw = dict(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=202609282 + fold,
        )
        model = (
            HistGradientBoostingClassifier(**kw)
            if kind == "binary"
            else HistGradientBoostingRegressor(**kw)
        )
    else:
        import pandas as pd

        cat = [c for c in features if c in cats]
        num = [c for c in features if c not in cats]
        x = (
            train[num]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        med = x.median().fillna(0.0)
        v = (
            valid[num]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(med)
        )
        scaler = StandardScaler().fit(x.fillna(med))
        x = scaler.transform(x.fillna(med))
        v = scaler.transform(v)
        if cat:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            x = np.column_stack(
                [x, enc.fit_transform(train[cat].fillna("__MISSING__").astype(str))]
            )
            v = np.column_stack(
                [v, enc.transform(valid[cat].fillna("__MISSING__").astype(str))]
            )
        model = (
            LogisticRegression(C=1.0, max_iter=1000, tol=1e-06)
            if kind == "binary"
            else PoissonRegressor(alpha=0.1, max_iter=1000, tol=1e-06)
        )
    model.fit(x, y)
    p = (
        model.predict_proba(v)[:, 1]
        if kind == "binary"
        else np.clip(model.predict(v), 1e-08, None)
    )
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    return p


def work(cohort, fold, generation_only=False):
    setup()
    import pandas as pd, numpy as np, joblib
    from sklearn.metrics import (
        roc_auc_score,
        mean_absolute_error,
        brier_score_loss,
        mean_poisson_deviance,
    )
    from shared_ivf.engine import SharedIVF
    from shared_ivf.schemas import SCHEMAS

    plan = json.loads((ROOT / "INPUTS.json").read_text())
    job = next((j for j in plan["jobs"] if j["cohort"] == cohort and j["fold"] == fold))
    dest = ROOT / cohort / f"fold_{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "DONE.json").exists() and (not (dest / "FAILED.json").exists())

    def read(path):
        assert sha(path) == plan["files"][str(path)], str(path)
        return (
            pd.read_parquet(path)
            if str(path).endswith(".parquet")
            else pd.read_csv(path, float_precision="round_trip")
        )

    def load_condition(info):
        if "frame" in info:
            return read(info["frame"])
        frames = []
        for batch in info["batches"]:
            p = read(batch["parents.parquet"])
            c = read(batch["cycles.parquet"])
            frames.append(c.merge(p, on="Patient_ID", validate="many_to_one"))
        return pd.concat(frames, ignore_index=True)

    start = time.monotonic()
    rows = []
    preds = []
    try:
        if cohort == "antalya":
            train, valid = (read(job["training"]), read(job["validation"]))
            import antalya_tasks

            tasks = antalya_tasks.TASKS
        else:
            d = job["development"]
            p = read(d["development_parents.csv"])
            c = read(d["development_cycles.csv"])
            f = read(d["development_folds.csv"])
            full = c.merge(p, on="Patient_ID", validate="many_to_one").merge(
                f, on="Patient_ID", validate="many_to_one"
            )
            train = (
                full.loc[full.Fold.ne(fold)].drop(columns="Fold").reset_index(drop=True)
            )
            valid = (
                full.loc[full.Fold.eq(fold)].drop(columns="Fold").reset_index(drop=True)
            )
            tasks = [
                "poor_response",
                "high_response",
                "high_maturity",
                "transfer",
                "observation",
                "live_birth",
                "egg_count",
                "mii_count",
            ]
        schema = SCHEMAS[cohort]()
        pid = schema.patient
        cid = schema.cycle
        assert (
            set(train[pid]).isdisjoint(valid[pid])
            and train[cid].is_unique
            and valid[cid].is_unique
        )
        scaffold = load_condition(job["methods"]["hma"])
        scaffold = scaffold[train.columns]
        assert (
            scaffold[pid].nunique() == 20 * train[pid].nunique()
            and scaffold[cid].is_unique
        )
        model = SharedIVF(schema).fit(train)
        synthetic = model.generate(scaffold, 2026090910 + fold)
        assert synthetic[pid].nunique() == scaffold[pid].nunique()
        assert set(synthetic[pid].astype(str)).isdisjoint(
            set(train[pid].astype(str)) | set(valid[pid].astype(str))
        )
        synthetic.to_parquet(dest / "synthetic.parquet", index=False)
        joblib.dump(model, dest / "model.joblib")
        save(
            dest / "GENERATION.json",
            dict(
                method="shared_ivf_v1_1",
                cohort=cohort,
                fold=fold,
                training_digest=model.training_digest,
                synthetic_sha256=sha(dest / "synthetic.parquet"),
                model_sha256=sha(dest / "model.joblib"),
                model_diagnostics=model.diagnostics,
                sampling=model.last_diagnostics,
                seconds=time.monotonic() - start,
            ),
        )
        print(
            json.dumps(
                dict(
                    stage="generated",
                    cohort=cohort,
                    fold=fold,
                    seconds=time.monotonic() - start,
                )
            ),
            flush=True,
        )
        if generation_only:
            return
        for method in METHODS:
            data = (
                synthetic
                if method == "shared_ivf_v1_1"
                else train
                if method == "real"
                else load_condition(job["methods"][method])
            )
            for name in tasks:
                tr, y, features, kind, cats = task(data, cohort, name)
                va, target, _, _, _ = task(valid, cohort, name)
                for family in ["hgb", "linear"]:
                    tick = time.monotonic()
                    prediction = predict(tr, y, va, features, cats, kind, family, fold)
                    primary = (
                        roc_auc_score(target, prediction)
                        if kind == "binary"
                        else mean_absolute_error(target, prediction)
                    )
                    proper = (
                        brier_score_loss(target, prediction)
                        if kind == "binary"
                        else mean_poisson_deviance(
                            target, np.maximum(prediction, 1e-08)
                        )
                    )
                    row = dict(
                        cohort=cohort,
                        fold=fold,
                        method=method,
                        task=name,
                        family=family,
                        kind=kind,
                        primary=primary,
                        proper=proper,
                        training_rows=len(tr),
                        validation_rows=len(va),
                        validation_patients=va[pid].nunique(),
                        seconds=time.monotonic() - tick,
                    )
                    rows.append(row)
                    pr = pd.DataFrame(
                        dict(
                            patient=va[pid].astype(str).to_numpy(),
                            cycle=va[cid].astype(str).to_numpy(),
                            y=target,
                            p=prediction,
                        )
                    )
                    for k in ["cohort", "fold", "method", "task", "family", "kind"]:
                        pr[k] = row[k]
                    preds.append(pr)
                save(
                    dest / "STATUS.json",
                    dict(
                        method=method,
                        task=name,
                        completed_metrics=len(rows),
                        seconds=time.monotonic() - start,
                    ),
                )
            print(
                json.dumps(
                    dict(
                        stage="method_scored",
                        cohort=cohort,
                        fold=fold,
                        method=method,
                        seconds=time.monotonic() - start,
                    )
                ),
                flush=True,
            )
        pd.DataFrame(rows).to_csv(dest / "metrics.csv", index=False)
        pd.concat(preds, ignore_index=True).to_parquet(
            dest / "predictions.parquet", index=False
        )
        save(
            dest / "DONE.json",
            dict(
                status="complete",
                cohort=cohort,
                fold=fold,
                metric_rows=len(rows),
                seconds=time.monotonic() - start,
                input_manifest_sha256=sha(ROOT / "INPUTS.json"),
                artifacts={
                    n: sha(dest / n)
                    for n in [
                        "metrics.csv",
                        "predictions.parquet",
                        "synthetic.parquet",
                        "model.joblib",
                        "GENERATION.json",
                    ]
                },
                reserved_partitions_accessed=False,
            ),
        )
    except Exception as e:
        save(
            dest / "FAILED.json", dict(error=repr(e), traceback=traceback.format_exc())
        )
        raise
