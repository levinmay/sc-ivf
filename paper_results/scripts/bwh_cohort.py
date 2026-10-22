"""BWH source preparation, patient splits and baseline sampling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
import random
import torch
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

PROJECT = Path(os.environ.get("IVF_SYNTH_PROJECT", "."))
SOURCE = Path(os.environ.get("IVF_SYNTH_SOURCE", "raw/bwh"))
PREPARED = PROJECT / "data_full_pathway"
PARENT_SOURCE = [
    "Patient_Age",
    "Producer_Age",
    "Producer_BMI",
    "AMH",
    "Day3_FSH",
    "Day3_E2",
    "Producer_Race",
    "Gravidity",
    "Parity",
    "Prior_SAB",
    "Prior_TAB",
    "GC",
    "DX_Dim_Ovar_Rsrv",
    "DX_Endometriosis",
    "DX_MaleFactor",
    "DX_Ovul_dysf",
    "DX_Tubal_disease",
    "DX_Unknown",
    "DX_Uterine_factor",
    "DX_Other",
]
CHILD_SOURCE = [
    "Cycle_Type",
    "CET_cycle_type",
    "Patient_Age",
    "Producer_Age",
    "Fresh_Stim_Protocol",
    "Ovulatory_trigger",
    "FSH",
    "HMG",
    "Trigger_E2",
    "Day_Ovulatory_Trigger",
    "Baseline_Endo",
    "Num>=18",
    "Num17",
    "Num16",
    "Num>=12<16",
    "Sum_Eggs",
    "Sum_MII",
    "Sum_Fert",
    "Transfer",
    "Sum_Transferred",
    "Day_of_transfer",
    "Sacs_Max",
    "FH>12wk",
    "#_Babies_(>24wks)",
]
STAGE = ["Sum_Transferred", "Sacs_Max", "FH>12wk", "#_Babies_(>24wks)"]
CATEGORICAL = [
    "Producer_Race",
    "GC",
    "DX_Dim_Ovar_Rsrv",
    "DX_Endometriosis",
    "DX_MaleFactor",
    "DX_Ovul_dysf",
    "DX_Tubal_disease",
    "DX_Unknown",
    "DX_Uterine_factor",
    "DX_Other",
    "Cycle_Type",
    "CET_cycle_type",
    "Fresh_Stim_Protocol",
    "Ovulatory_trigger",
    "Transfer",
    "Retrieval_Observed",
    "Stage_Observed",
    "Stage_Discordant",
    "Fert_Exceeds_MII",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_observed(series: pd.Series):
    observed = series.dropna()
    return observed.iloc[0] if len(observed) else np.nan


def median_observed(series: pd.Series):
    observed = pd.to_numeric(series, errors="coerce").dropna()
    return float(observed.median()) if len(observed) else np.nan


def load(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        na_values=["*", "***", "", "NA", "N/A"],
        keep_default_na=True,
        low_memory=False,
    )
    frame["split"] = split
    return frame


def prepare_source() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    files = {
        split: SOURCE / f"cycle-{split}-data-cleaned.csv" for split in ["train", "test"]
    }
    raw = pd.concat(
        [load(path, split) for split, path in files.items()], ignore_index=True
    )
    initial_by_split = raw.groupby("split").size().to_dict()
    for column in sorted(
        set(PARENT_SOURCE + CHILD_SOURCE + ["Cycle_Code", "Patient_Code"])
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    duplicate = (
        raw.groupby("split")["Cycle_Code"].transform(
            lambda series: series.duplicated(keep=False)
        )
        & raw.Cycle_Code.notna()
    )
    missing_key = raw.Patient_Code.isna() | raw.Cycle_Code.isna()
    excluded = duplicate | missing_key
    exclusions = (
        raw.assign(duplicate=duplicate, missing_key=missing_key)
        .groupby("split")
        .agg(
            duplicate_rows=("duplicate", "sum"), missing_key_rows=("missing_key", "sum")
        )
        .to_dict(orient="index")
    )
    raw = raw.loc[~excluded].copy()
    raw = raw.sort_values(["split", "Patient_Code", "Cycle_Code"]).reset_index(
        drop=True
    )
    raw["Patient_ID"] = (
        raw.Patient_Code.round().astype(int).map(lambda value: f"BWH{value:06d}")
    )
    raw["Cycle_Order"] = raw.groupby(["split", "Patient_ID"]).cumcount() + 1
    raw["Cycle_ID"] = [
        f"{split[0].upper()}FC{index:07d}" for index, split in enumerate(raw.split)
    ]
    complete_counts = raw[["Sum_Eggs", "Sum_MII", "Sum_Fert"]].notna().all(axis=1)
    integer_counts = np.isclose(
        raw[["Sum_Eggs", "Sum_MII", "Sum_Fert"]],
        np.rint(raw[["Sum_Eggs", "Sum_MII", "Sum_Fert"]]),
        equal_nan=False,
    ).all(axis=1)
    count_support = (
        raw.Sum_Eggs.gt(0)
        & raw.Sum_MII.ge(0)
        & raw.Sum_Fert.ge(0)
        & raw.Sum_MII.le(raw.Sum_Eggs)
        & raw.Sum_Fert.le(raw.Sum_Eggs)
    )
    raw["Retrieval_Observed"] = (
        complete_counts & integer_counts & count_support
    ).astype(int)
    stage_complete = raw[STAGE].notna().all(axis=1)
    stage_ordered = (
        raw.Sacs_Max.le(raw.Sum_Transferred)
        & raw["FH>12wk"].le(raw.Sacs_Max)
        & raw["#_Babies_(>24wks)"].le(raw["FH>12wk"])
    )
    raw["Stage_Observed"] = stage_complete.astype(int)
    raw["Stage_Discordant"] = (stage_complete & ~stage_ordered).astype(int)
    raw["Fert_Exceeds_MII"] = np.where(
        raw.Retrieval_Observed.eq(1), raw.Sum_Fert.gt(raw.Sum_MII).astype(int), np.nan
    )
    parent_rows = []
    rename = {
        "Patient_Age": "Baseline_Patient_Age",
        "Producer_Age": "Baseline_Producer_Age",
    }
    for (split, patient), block in raw.groupby(["split", "Patient_ID"], sort=False):
        ordered = block.sort_values("Cycle_Order")
        row = {"Patient_ID": patient, "split": split}
        for column in PARENT_SOURCE:
            target = rename.get(column, column)
            if column.startswith("DX_") or column == "GC":
                row[target] = ordered[column].max(skipna=True)
            elif column in {"Producer_BMI", "AMH", "Day3_FSH", "Day3_E2"}:
                row[target] = median_observed(ordered[column])
            else:
                row[target] = first_observed(ordered[column])
        parent_rows.append(row)
    parents = pd.DataFrame(parent_rows)
    children = raw[
        [
            "Patient_ID",
            "Cycle_ID",
            "split",
            "Cycle_Order",
            *CHILD_SOURCE,
            "Retrieval_Observed",
            "Stage_Observed",
            "Stage_Discordant",
            "Fert_Exceeds_MII",
        ]
    ].copy()
    audit = {
        "source": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in files.items()
        },
        "initial_rows": {key: int(value) for key, value in initial_by_split.items()},
        "exclusions": exclusions,
        "categorical_columns": CATEGORICAL,
        "semantic_warning": "Missing egg counts are predominantly transfer-only Cycle_Type=1 records; they must not be labeled cancellations. Source retrieval links for frozen transfers are unavailable.",
    }
    for split in ["train", "test"]:
        parent = (
            parents.loc[parents.split.eq(split)]
            .drop(columns="split")
            .reset_index(drop=True)
        )
        child = (
            children.loc[children.split.eq(split)]
            .drop(columns="split")
            .reset_index(drop=True)
        )
        sequence = child.merge(
            parent, on="Patient_ID", how="left", validate="many_to_one"
        )
        parent.to_csv(PREPARED / f"parents_{split}.csv", index=False)
        child.to_csv(PREPARED / f"cycles_{split}.csv", index=False)
        sequence.to_csv(PREPARED / f"sequences_{split}.csv", index=False)
        audit[split] = {
            "patients": int(len(parent)),
            "cycles": int(len(child)),
            "repeat_patients": int((child.groupby("Patient_ID").size() > 1).sum()),
            "retrieval_observed": int(child.Retrieval_Observed.sum()),
            "transfer_only_cycle_type_1": int(child.Cycle_Type.eq(1).sum()),
            "stage_observed": int(child.Stage_Observed.sum()),
            "stage_discordant": int(child.Stage_Discordant.sum()),
        }
    audit["patient_overlap"] = int(
        len(
            set(parents.loc[parents.split.eq("train"), "Patient_ID"])
            & set(parents.loc[parents.split.eq("test"), "Patient_ID"])
        )
    )
    (PREPARED / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


PREPARED = PROJECT / "data_full_pathway"
PARTITIONS = PROJECT / "prediction_restart"
RESERVED = PARTITIONS / "sealed_test"
SALT = "EVC_IVF_PREDICTION_RESTART_20260905_V1"
EXPECTED_PATIENTS = 4710
EXPECTED_CYCLES = 10795
TEST_PATIENTS = 942
DEVELOPMENT_FOLDS = 4


def digest(value: str, suffix: str = "") -> str:
    return hashlib.sha256(f"{SALT}|{suffix}|{value}".encode()).hexdigest()


def load_table(stem: str) -> pd.DataFrame:
    train = pd.read_csv(PREPARED / f"{stem}_train.csv")
    test = pd.read_csv(PREPARED / f"{stem}_test.csv")
    for frame in (train, test):
        frame["Patient_ID"] = frame.Patient_ID.astype(str)
    return pd.concat([train, test], ignore_index=True, sort=False)


def split_development() -> None:
    if PARTITIONS.exists() and (PARTITIONS / "split_manifest.json").exists():
        raise RuntimeError("Frozen split already exists; refusing to overwrite it")
    PARTITIONS.mkdir(parents=True, exist_ok=True)
    RESERVED.mkdir(parents=True, exist_ok=True)
    parents_train = pd.read_csv(PREPARED / "parents_train.csv")
    parents_test = pd.read_csv(PREPARED / "parents_test.csv")
    parents_train["Patient_ID"] = parents_train.Patient_ID.astype(str)
    parents_test["Patient_ID"] = parents_test.Patient_ID.astype(str)
    overlap = set(parents_train.Patient_ID) & set(parents_test.Patient_ID)
    if overlap:
        raise RuntimeError(f"Source patient overlap: {len(overlap)}")
    parents = load_table("parents")
    cycles = load_table("cycles")
    cycles["Cycle_ID"] = cycles.Cycle_ID.astype(str)
    if (
        parents.Patient_ID.nunique() != EXPECTED_PATIENTS
        or len(parents) != EXPECTED_PATIENTS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} unique patients; got {len(parents)}/{parents.Patient_ID.nunique()}"
        )
    if len(cycles) != EXPECTED_CYCLES:
        raise RuntimeError(f"Expected {EXPECTED_CYCLES} cycles; got {len(cycles)}")
    if cycles.Cycle_ID.duplicated().any():
        raise RuntimeError("Cycle_ID is not unique after combining sources")
    if set(cycles.Patient_ID) - set(parents.Patient_ID):
        raise RuntimeError("A cycle references a missing patient")
    ordered = sorted(parents.Patient_ID.tolist(), key=lambda x: digest(x, "TEST"))
    test_ids = set(ordered[:TEST_PATIENTS])
    development_ids = ordered[TEST_PATIENTS:]
    if len(development_ids) != EXPECTED_PATIENTS - TEST_PATIENTS:
        raise RuntimeError("Unexpected development size")
    fold_order = sorted(development_ids, key=lambda x: digest(x, "FOLD"))
    folds = pd.DataFrame(
        {
            "Patient_ID": fold_order,
            "Fold": [index % DEVELOPMENT_FOLDS for index in range(len(fold_order))],
        }
    )
    if folds.groupby("Fold").size().tolist() != [942, 942, 942, 942]:
        raise RuntimeError("Development folds are not exactly balanced")
    development_parents = parents.loc[parents.Patient_ID.isin(development_ids)].copy()
    development_cycles = cycles.loc[cycles.Patient_ID.isin(development_ids)].copy()
    test_parents = parents.loc[parents.Patient_ID.isin(test_ids)].copy()
    test_cycles = cycles.loc[cycles.Patient_ID.isin(test_ids)].copy()
    if set(development_parents.Patient_ID) & set(test_parents.Patient_ID):
        raise RuntimeError("Patient leakage between development and test")
    if set(development_cycles.Patient_ID) & set(test_cycles.Patient_ID):
        raise RuntimeError("Cycle-patient leakage between development and test")
    development_parents.to_csv(PARTITIONS / "development_parents.csv", index=False)
    development_cycles.to_csv(PARTITIONS / "development_cycles.csv", index=False)
    folds.to_csv(PARTITIONS / "development_folds.csv", index=False)
    test_parents.to_csv(RESERVED / "test_parents.csv", index=False)
    test_cycles.to_csv(RESERVED / "test_cycles.csv", index=False)
    split_rows = pd.concat(
        [
            pd.DataFrame(
                {"Patient_ID": sorted(development_ids), "Partition": "development"}
            ),
            pd.DataFrame({"Patient_ID": sorted(test_ids), "Partition": "sealed_test"}),
        ],
        ignore_index=True,
    )
    split_rows["Patient_Hash"] = split_rows.Patient_ID.map(lambda x: digest(x, "AUDIT"))
    split_rows.drop(columns="Patient_ID").to_csv(
        PARTITIONS / "deidentified_split_audit.csv", index=False
    )
    manifest = {
        "status": "split_frozen_test_unopened",
        "salt_sha256": hashlib.sha256(SALT.encode()).hexdigest(),
        "source_patients": int(len(parents)),
        "source_cycles": int(len(cycles)),
        "development_patients": int(len(development_parents)),
        "development_cycles": int(len(development_cycles)),
        "sealed_test_patients": int(len(test_parents)),
        "sealed_test_cycles": int(len(test_cycles)),
        "development_fold_sizes": {
            str(key): int(value) for key, value in folds.groupby("Fold").size().items()
        },
        "patient_overlap": 0,
        "cycle_patient_overlap": 0,
        "test_performance_accessed": False,
        "source_files": [
            "data_full_pathway/parents_train.csv",
            "data_full_pathway/cycles_train.csv",
            "data_full_pathway/parents_test.csv",
            "data_full_pathway/cycles_test.csv",
        ],
    }
    (PARTITIONS / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


PARENT_CATEGORICAL = [
    "Producer_Race",
    "GC",
    "DX_Dim_Ovar_Rsrv",
    "DX_Endometriosis",
    "DX_MaleFactor",
    "DX_Ovul_dysf",
    "DX_Tubal_disease",
    "DX_Unknown",
    "DX_Uterine_factor",
    "DX_Other",
]


def augment(cycles: pd.DataFrame, parents: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in parents if column != "Patient_ID" and column not in cycles
    ]
    return cycles.merge(
        parents[["Patient_ID", *missing]],
        on="Patient_ID",
        how="left",
        validate="many_to_one",
    )


def update_types(metadata: Metadata, table: str, columns: list[str]) -> None:
    available = metadata.to_dict()["tables"][table]["columns"]
    for column in columns:
        if column in available:
            metadata.update_column(column, table_name=table, sdtype="categorical")


def relational_metadata(parents: pd.DataFrame, cycles: pd.DataFrame) -> Metadata:
    metadata = Metadata.detect_from_dataframes(
        {"parents": parents, "cycles": cycles},
        infer_sdtypes=True,
        infer_keys="primary_and_foreign",
    )
    for table, primary in [("parents", "Patient_ID"), ("cycles", "Cycle_ID")]:
        metadata.update_column(primary, table_name=table, sdtype="id")
        metadata.set_primary_key(primary, table_name=table)
    metadata.update_column("Patient_ID", table_name="cycles", sdtype="id")
    update_types(metadata, "parents", PARENT_CATEGORICAL)
    update_types(metadata, "cycles", CATEGORICAL)
    expected = {
        "parent_table_name": "parents",
        "child_table_name": "cycles",
        "parent_primary_key": "Patient_ID",
        "child_foreign_key": "Patient_ID",
    }
    relationships = metadata.to_dict().get("relationships", [])
    if not any(
        (
            all((item.get(key) == value for key, value in expected.items()))
            for item in relationships
        )
    ):
        metadata.add_relationship(**expected)
    metadata.validate()
    metadata.validate_data({"parents": parents, "cycles": cycles})
    return metadata


def clone_patients(
    parents: pd.DataFrame, cycles: pd.DataFrame, source_ids: np.ndarray, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.DataFrame(
        {
            "source_id": source_ids,
            "new_id": [f"{prefix}P{index:06d}" for index in range(len(source_ids))],
        }
    )
    parent_source = parents.rename(columns={"Patient_ID": "source_id"})
    cycle_source = cycles.rename(columns={"Patient_ID": "source_id"}).drop(
        columns="Cycle_ID"
    )
    new_parents = (
        mapping.merge(parent_source, on="source_id", how="left", validate="many_to_one")
        .drop(columns="source_id")
        .rename(columns={"new_id": "Patient_ID"})
    )
    new_cycles = (
        mapping.merge(cycle_source, on="source_id", how="left", validate="many_to_many")
        .drop(columns="source_id")
        .rename(columns={"new_id": "Patient_ID"})
    )
    order = new_cycles.groupby("Patient_ID").cumcount() + 1
    new_cycles["Cycle_ID"] = [
        f"{prefix}C{index:08d}" for index in range(len(new_cycles))
    ]
    new_cycles["Cycle_Order"] = order.to_numpy()
    return (new_parents[list(parents.columns)], new_cycles[list(cycles.columns)])


EPOCHS = 300


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


FIT_SEED = 202609060
IDS = {"Patient_ID", "Cycle_ID"}


def validate(p, c, n=None, forbidden_ids=()):
    assert p.Patient_ID.is_unique and c.Cycle_ID.is_unique
    assert set(c.Patient_ID.astype(str)) <= set(p.Patient_ID.astype(str))
    assert set(p.Patient_ID.astype(str)).isdisjoint(set(map(str, forbidden_ids)))
    assert n is None or len(p) == n
    assert (
        c.groupby("Patient_ID").size().reindex(p.Patient_ID, fill_value=0).gt(0).all()
    )
    assert not any((x.startswith("_") for x in list(p) + list(c)))
    return {
        "patients": len(p),
        "cycles": len(c),
        "keys_unique": True,
        "ownership_valid": True,
    }


def _metadata(frame):
    m = Metadata.detect_from_dataframe(
        frame, table_name="flat", infer_sdtypes=True, infer_keys=None
    )
    for col in CATEGORICAL:
        if col in frame:
            m.update_column(col, table_name="flat", sdtype="categorical")
    m.validate()
    return m


class CohortAdapter:
    def __init__(self, parents, cycles):
        self.parents = parents.copy()
        self.cycles = cycles.copy()
        validate(self.parents, self.cycles)
        self.models = {}
        self.training = augment(self.cycles, self.parents)
        self.flat = self.training.drop(columns=list(IDS), errors="ignore")
        self.counts = (
            self.cycles.groupby("Patient_ID")
            .size()
            .reindex(self.parents.Patient_ID)
            .to_numpy(int)
        )

    def _model(self, method):
        if method in self.models:
            return self.models[method]

        def build():
            seed_everything(FIT_SEED)
            if method == "generic_hma":
                m = HMASynthesizer(
                    relational_metadata(self.parents, self.cycles), verbose=False
                )
                m.fit({"parents": self.parents, "cycles": self.cycles})
            else:
                cls = {
                    "gaussian_copula": GaussianCopulaSynthesizer,
                    "tvae": TVAESynthesizer,
                }[method]
                kw = (
                    {}
                    if method == "gaussian_copula"
                    else dict(epochs=EPOCHS, enable_gpu=True, verbose=False)
                )
                m = cls(_metadata(self.flat), **kw)
                m.fit(self.flat)
            return m

        self.models[method] = build()
        return self.models[method]

    def _hma_sample(self, n_patients, seed):
        m = self._model("generic_hma")
        m._numpy_seed = int(seed)
        for submodel in m._table_synthesizers.values():
            submodel._set_random_state(seed)
        raw = m.sample(scale=max(n_patients / len(self.parents) * 1.12, 0.1))
        p, c = (raw["parents"], raw["cycles"])
        p = p.loc[p.Patient_ID.isin(c.Patient_ID)].copy()
        if not len(p):
            raise RuntimeError("HMA returned no patients with cycles")
        ids = np.random.default_rng(seed).choice(
            p.Patient_ID.to_numpy(), n_patients, replace=len(p) < n_patients
        )
        return clone_patients(p, c, ids, f"S{seed}_")

    def _flat_relational(self, flat, counts, seed):
        flat = flat.reset_index(drop=True)
        pid = np.repeat([f"S{seed}_P{i:06d}" for i in range(len(counts))], counts)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        pc = [x for x in self.parents if x != "Patient_ID"]
        cc = [x for x in self.cycles if x not in IDS]
        p = flat.iloc[starts][pc].copy().reset_index(drop=True)
        p.insert(0, "Patient_ID", pid[starts])
        c = flat[cc].copy()
        c.insert(0, "Cycle_ID", [f"S{seed}_C{i:08d}" for i in range(len(c))])
        c.insert(0, "Patient_ID", pid)
        c["Cycle_Order"] = c.groupby("Patient_ID").cumcount() + 1
        return (p[list(self.parents)], c[list(self.cycles)])

    def sample(self, method, n_patients, seed):
        rng = np.random.default_rng(seed)
        seed_everything(seed)
        if method in ["gaussian_copula", "tvae"]:
            counts = rng.choice(self.counts, n_patients, replace=True)
            size = int(counts.sum())
            model = self._model(method)
            model._set_random_state(seed)
            flat = model.sample(num_rows=size)
            p, c = self._flat_relational(flat, counts, seed)
        elif method == "generic_hma":
            p, c = self._hma_sample(n_patients, seed)
        else:
            raise ValueError(method)
        check = validate(p, c, n_patients, self.parents.Patient_ID)
        return (p, c, check)
