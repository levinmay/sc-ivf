"""Shared stage-conditional engine. No cohort names, file reads or outcome tuning."""

from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

METHOD_ID = "shared_ivf_v1_1"
SEED = 2026090810
EGG = dict(
    loss="poisson",
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=7,
    min_samples_leaf=50,
    l2_regularization=10.0,
    early_stopping=False,
    random_state=SEED,
)
FRACTION = dict(
    learning_rate=0.055,
    max_iter=160,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=SEED,
)
DISCRETE = dict(
    learning_rate=0.05,
    max_iter=150,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=SEED,
)


@dataclass(frozen=True)
class Stage:
    target: str
    kind: str
    features: tuple
    total: str | None = None
    gates: tuple = ()
    allow_excess: bool = False
    log_total: bool = False
    inactive: float = np.nan


@dataclass(frozen=True)
class Schema:
    name: str
    patient: str
    cycle: str
    categorical: tuple
    stages: tuple
    # Mask pairs transport missing numerical values as dummy zero plus a mask.
    masks: tuple = ()
    derived: tuple = ()
    mask_groups: tuple = ()

    def validate(self):
        targets = [s.target for s in self.stages]
        if len(set(targets)) != len(targets):
            raise ValueError("Duplicate stage targets")
        for i, s in enumerate(self.stages):
            if s.kind not in {
                "positive_count",
                "bounded_count",
                "categorical",
                "continuous",
            }:
                raise ValueError("Unknown stage kind")
            unavailable = set(targets[i:])
            dependencies = set(s.features) | {g[0] for g in s.gates}
            if s.total:
                dependencies.add(s.total)
            if dependencies & unavailable:
                raise ValueError("Stage uses its own or later outcome: " + s.target)
            if self.patient in s.features or self.cycle in s.features:
                raise ValueError("Identifiers cannot be predictive features")
        grouped = []
        for columns, features, gates in self.mask_groups:
            if not set(columns) <= set(targets):
                raise ValueError("Mask group contains an unmodeled target")
            if set(columns) & set(features):
                raise ValueError("Mask model uses a masked outcome")
            grouped.extend(columns)
        if len(grouped) != len(set(grouped)):
            raise ValueError("Overlapping mask groups")
        return self


class Encoder:
    def fit(self, frame, features, categorical, log_total=None):
        self.features = tuple(features)
        self.cats = [c for c in features if c in categorical]
        self.nums = [c for c in features if c not in categorical]
        self.log_total = log_total
        self.medians = self.numeric(frame).median().fillna(0.0)
        self.ordinal = None
        if self.cats:
            self.ordinal = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.ordinal.fit(frame[self.cats].fillna("__MISSING__").astype(str))
        return self

    def numeric(self, frame):
        values = (
            frame[self.nums]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        if self.log_total:
            values = values.copy()
            values["_log_total"] = np.log1p(
                pd.to_numeric(frame[self.log_total], errors="coerce").clip(lower=0)
            )
        return values

    def transform(self, frame):
        blocks = [self.numeric(frame).fillna(self.medians).to_numpy(float)]
        if self.ordinal is not None:
            blocks.append(
                self.ordinal.transform(
                    frame[self.cats].fillna("__MISSING__").astype(str)
                )
            )
        result = np.column_stack(blocks)
        return result if result.shape[1] else np.ones((len(frame), 1))


def probability(raw, offset=0.0):
    return np.clip(expit(logit(np.clip(raw, 1e-8, 1 - 1e-8)) + offset), 1e-4, 1 - 1e-4)


class Mean:
    def fit(self, frame, stage, categorical):
        self.stage = stage
        self.encoder = Encoder().fit(
            frame, stage.features, categorical, stage.total if stage.log_total else None
        )
        x = self.encoder.transform(frame)
        y = pd.to_numeric(frame[stage.target], errors="raise").to_numpy(float)
        self.model = None
        if stage.kind == "positive_count":
            y = y - 1
            self.scale = 0.0
            if y.sum() > 0:
                self.model = HistGradientBoostingRegressor(**EGG).fit(x, y)
                self.scale = float(y.mean() / self.model.predict(x).mean())
        else:
            n = frame[stage.total].to_numpy(float)
            ratio = y / n
            self.constant = float(ratio.mean())
            weights = np.r_[ratio, 1 - ratio]
            labels = np.r_[np.ones(len(x), int), np.zeros(len(x), int)]
            keep = weights > 0
            if len(np.unique(labels[keep])) > 1:
                self.model = HistGradientBoostingClassifier(**FRACTION).fit(
                    np.concatenate([x, x])[keep],
                    labels[keep],
                    sample_weight=weights[keep],
                )
            low, high = -30.0, 30.0
            target = np.clip(ratio.mean(), 1e-4, 1 - 1e-4)
            raw = self.raw(frame)
            for _ in range(100):
                middle = (low + high) / 2
                if probability(raw, middle).mean() < target:
                    low = middle
                else:
                    high = middle
            self.offset = (low + high) / 2
        return self

    def raw(self, frame):
        return (
            self.model.predict_proba(self.encoder.transform(frame))[:, 1]
            if self.model is not None
            else np.full(len(frame), self.constant)
        )

    def predict(self, frame):
        if self.stage.kind == "positive_count":
            return (
                self.scale * self.model.predict(self.encoder.transform(frame))
                if self.model is not None
                else np.zeros(len(frame))
            )
        return probability(self.raw(frame), self.offset)


class Discrete:
    def fit(self, frame, stage, categorical):
        self.encoder = Encoder().fit(frame, stage.features, categorical)
        raw = frame[stage.target]
        self.numeric = bool(pd.to_numeric(raw.dropna(), errors="coerce").notna().all())
        raw = (
            pd.to_numeric(raw, errors="coerce")
            if self.numeric
            else raw.map(lambda x: str(x) if pd.notna(x) else np.nan)
        )
        levels = sorted(raw.dropna().unique().tolist(), key=str)
        self.values = np.array(
            levels + ([np.nan] if raw.isna().any() or not levels else []),
            dtype=float if self.numeric else object,
        )
        y = (
            raw.map({v: i for i, v in enumerate(levels)})
            .fillna(len(levels))
            .to_numpy(int)
        )
        self.model = None
        if len(self.values) > 1:
            self.model = HistGradientBoostingClassifier(**DISCRETE).fit(
                self.encoder.transform(frame), y
            )
        return self

    def sample(self, frame, rng):
        if self.model is None:
            return np.repeat(self.values[0], len(frame))
        p = self.model.predict_proba(self.encoder.transform(frame))
        ix = np.minimum(
            (rng.random(len(frame))[:, None] > p.cumsum(axis=1)).sum(axis=1),
            p.shape[1] - 1,
        )
        return self.values[self.model.classes_[ix]]


class NumericStage:
    def fit(self, frame, stage, categorical, patient):
        self.stage = stage
        y = pd.to_numeric(frame[stage.target], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        obs = y.notna()
        self.missing = Discrete().fit(
            frame.assign(_observed=obs.astype(int)),
            Stage("_observed", "categorical", stage.features),
            categorical,
        )
        self.mean = None
        self.dispersion = 0.0
        self.excess = np.array([], int)
        self.exception_probability = 0.0
        self.diag = dict(
            training_rows=len(frame), observed_rows=int(obs.sum()), oof_folds=0
        )
        if not obs.any():
            return self
        self.lower, self.upper = float(y[obs].min()), float(y[obs].max())
        if stage.kind == "continuous":
            self.encoder = Encoder().fit(frame.loc[obs], stage.features, categorical)
            x = self.encoder.transform(frame.loc[obs])
            self.model = HistGradientBoostingRegressor(**FRACTION).fit(x, y[obs])
            self.residual = y[obs].to_numpy() - self.model.predict(x)
            self.residual -= self.residual.mean()
            return self
        if not ((y[obs] >= 0) & (y[obs] == np.round(y[obs]))).all():
            raise ValueError("Invalid training counts: " + stage.target)
        if stage.kind == "positive_count":
            if not y[obs].ge(1).all():
                raise ValueError("Positive count includes zero: " + stage.target)
            usable = obs
        else:
            n = pd.to_numeric(frame[stage.total], errors="coerce")
            domain = obs & n.notna() & n.ge(0) & n.eq(n.round())
            exceptions = domain & y.gt(n)
            if exceptions.any() and not stage.allow_excess:
                raise ValueError("Undeclared source excess: " + stage.target)
            if stage.allow_excess and exceptions.any():
                self.exception_probability = float(exceptions.sum() / domain.sum())
                self.excess = (y[exceptions] - n[exceptions]).to_numpy(int)
            usable = domain & n.gt(0) & y.le(n)
        r = frame.loc[usable].copy()
        if not len(r):
            return self
        groups = r[patient].astype(str)
        folds = min(4, groups.nunique())
        # Fewer than two independent groups cannot support a fitted count stage.
        if folds < 2:
            raise ValueError("Insufficient grouped count support: " + stage.target)
        oof = np.empty(len(r))
        split_ids = []
        for ti, vi in GroupKFold(n_splits=folds).split(r, groups=groups):
            assert set(groups.iloc[ti]).isdisjoint(groups.iloc[vi])
            oof[vi] = Mean().fit(r.iloc[ti], stage, categorical).predict(r.iloc[vi])
            split_ids.append((len(set(groups.iloc[ti])), len(set(groups.iloc[vi]))))
        z = r[stage.target].to_numpy(float)
        if stage.kind == "positive_count":
            z = z - 1
            self.dispersion = float(
                np.clip(
                    np.sum((z - oof) ** 2 - oof) / max(np.sum(oof**2), 1e-12), 0.0, 1.0
                )
            )
        else:
            n = r[stage.total].to_numpy(float)
            self.dispersion = float(
                np.clip(
                    np.sum((z - n * oof) ** 2 - n * oof * (1 - oof))
                    / max(np.sum(n * (n - 1) * oof * (1 - oof)), 1e-12),
                    0.002,
                    0.30,
                )
            )
        self.mean = Mean().fit(r, stage, categorical)
        self.diag.update(
            oof_folds=folds,
            group_counts=split_ids,
            usable_rows=len(r),
            dispersion=self.dispersion,
            exception_probability=self.exception_probability,
        )
        return self

    def sample(self, frame, rng):
        s = self.stage
        if self.diag["observed_rows"] == 0:
            return np.full(len(frame), np.nan)
        if s.kind == "continuous":
            out = np.clip(
                self.model.predict(self.encoder.transform(frame))
                + rng.choice(self.residual, len(frame)),
                self.lower,
                self.upper,
            )
        elif s.kind == "positive_count":
            mu = self.mean.predict(frame)
            intensity = (
                mu
                if self.dispersion == 0
                else rng.gamma(1 / self.dispersion, mu * self.dispersion)
            )
            out = (1 + rng.poisson(intensity)).astype(float)
        else:
            original_n = pd.to_numeric(frame[s.total], errors="coerce").to_numpy(float)
            n = np.nan_to_num(original_n, nan=0).round().clip(0).astype(int)
            p = self.mean.predict(frame) if self.mean else np.full(len(frame), 0.5)
            rho = self.dispersion if self.mean else 0.002
            scale = 1 / rho - 1
            out = rng.binomial(n, rng.beta(p * scale, (1 - p) * scale)).astype(float)
            if self.exception_probability:
                ix = rng.random(len(frame)) < self.exception_probability
                out[ix] = np.minimum(
                    n[ix] + rng.choice(self.excess, ix.sum()), self.upper
                )
            out[~np.isfinite(original_n)] = np.nan
        out[~self.missing.sample(frame, rng).astype(bool)] = np.nan
        return out


class SharedIVF:
    """One engine for any declared IVF stage schema, fitted to one training cohort.

    ``generate`` regenerates outcomes on a supplied synthetic relational scaffold.
    IVFGenerator in pipeline.py also fits and samples the HMA background tables.
    """

    def __init__(self, schema):
        self.schema = schema.validate()

    def restore(self, frame):
        r = frame.copy()
        for value, mask in self.schema.masks:
            r.loc[r[mask].ne(1), value] = np.nan
        return r

    def fit(self, training):
        s = self.schema
        if (
            training.empty
            or training[s.patient].isna().any()
            or not training[s.cycle].is_unique
        ):
            raise ValueError(
                "Training identifiers must be nonmissing and unique per cycle"
            )
        r = self.restore(training).reset_index(drop=True)
        self.columns = list(training)
        self.models = []
        self.bounds = {}
        self.mask_models = []
        grouped_targets = {c for columns, _, _ in s.mask_groups for c in columns}
        for stage in s.stages:
            eligible = self.eligible(r, stage)
            # Joint-mask fields fit values on observed rows. Their observation
            # pattern is sampled once, after latent values, by the shared mask model.
            if stage.target in grouped_targets:
                eligible &= r[stage.target].notna()
            if not eligible.any():
                raise ValueError("No training support for " + stage.target)
            cls = Discrete if stage.kind == "categorical" else NumericStage
            args = () if cls is Discrete else (s.patient,)
            model = cls().fit(r.loc[eligible], stage, s.categorical, *args)
            self.models.append(model)
            if stage.kind != "categorical":
                x = pd.to_numeric(r.loc[eligible, stage.target], errors="coerce")
                self.bounds[stage.target] = (float(x.min()), float(x.max()))
        for columns, features, gates in s.mask_groups:
            spec = Stage("_pattern", "categorical", features, gates=gates)
            selected = r.loc[self.eligible(r, spec)].copy()
            selected["_pattern"] = sum(
                (1 << i) * selected[c].notna().astype(int)
                for i, c in enumerate(columns)
            )
            self.mask_models.append(Discrete().fit(selected, spec, s.categorical))
        self.training_digest = hashlib.sha256(
            pd.util.hash_pandas_object(training, index=True).to_numpy().tobytes()
        ).hexdigest()
        self.diagnostics = dict(
            method=METHOD_ID,
            schema=asdict(s),
            training_rows=len(training),
            training_patients=training[s.patient].nunique(),
            stages={
                st.target: getattr(m, "diag", {})
                for st, m in zip(s.stages, self.models)
            },
        )
        return self

    @staticmethod
    def eligible(frame, stage):
        use = pd.Series(True, index=frame.index)
        for column, values in stage.gates:
            use &= frame[column].isin(values)
        return use

    def generate(self, scaffold, seed, repair=True):
        if not hasattr(self, "models"):
            raise RuntimeError("Fit first")
        s = self.schema
        if (
            scaffold.empty
            or scaffold[s.patient].isna().any()
            or not scaffold[s.cycle].is_unique
        ):
            raise ValueError("Invalid synthetic scaffold identifiers")
        out = scaffold.copy().reset_index(drop=True)
        changes = {}
        for i, (stage, model) in enumerate(zip(s.stages, self.models)):
            eligible = self.eligible(out, stage)
            out[stage.target] = stage.inactive
            if eligible.any():
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), i]))
                value = model.sample(out.loc[eligible], rng)
                if value.dtype == object:
                    out[stage.target] = out[stage.target].astype(object)
                if repair and stage.kind != "categorical":
                    lo, hi = self.bounds[stage.target]
                    clipped = np.clip(value, lo, hi)
                    # Schema-enforced nonexception counts remain bounded after marginal repair.
                    if stage.total and not stage.allow_excess:
                        clipped = np.minimum(
                            clipped,
                            pd.to_numeric(
                                out.loc[eligible, stage.total], errors="coerce"
                            ),
                        )
                    changes[stage.target] = int(
                        np.sum(np.isfinite(value) & (value != clipped))
                    )
                    value = np.asarray(clipped)
                out.loc[eligible, stage.target] = value
        for i, ((columns, features, gates), model) in enumerate(
            zip(s.mask_groups, self.mask_models)
        ):
            eligible = self.eligible(
                out, Stage("_pattern", "categorical", features, gates=gates)
            )
            patterns = np.zeros(len(out), int)
            if eligible.any():
                patterns[eligible] = model.sample(
                    out.loc[eligible],
                    np.random.default_rng(
                        np.random.SeedSequence([int(seed), 1000 + i])
                    ),
                ).astype(int)
            for bit, column in enumerate(columns):
                out.loc[(patterns & (1 << bit)) == 0, column] = np.nan
        for name, op, columns in s.derived:
            if op == "all_observed":
                out[name] = out[list(columns)].notna().all(axis=1).astype(int)
            elif op == "greater":
                out[name] = out[columns[0]].gt(out[columns[1]]).astype(int)
            elif op == "discordant":
                out[name] = np.logical_or.reduce(
                    [out[a].gt(out[b]) for a, b in zip(columns[1:], columns[:-1])]
                ).astype(int)
            elif op == "positive":
                out[name] = out[columns[0]].gt(0).astype(int)
            else:
                raise ValueError("Unknown derivation")
        for value, mask in s.masks:
            out[mask] = out[value].notna().astype(int)
            out[value] = out[value].fillna(0.0)
        self.last_diagnostics = dict(
            seed=int(seed),
            rows=len(out),
            patients=out[s.patient].nunique(),
            support_changes=changes,
        )
        return out[self.columns]
