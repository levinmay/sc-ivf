"""BWH prediction tasks."""

from __future__ import annotations
import numpy as np
import pandas as pd

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
UTILITY_BASE = [
    "Patient_Age",
    "Producer_Age",
    "Producer_BMI",
    "AMH",
    "Day3_FSH",
    "Day3_E2",
    "Cycle_Type",
    "CET_cycle_type",
    "FSH",
    "HMG",
    "Fresh_Stim_Protocol",
    *PARENT_CATEGORICAL[2:],
]
TRANSFER_UTILITY_FEATURES = [
    *UTILITY_BASE,
    "Trigger_E2",
    "Num>=18",
    "Num17",
    "Num16",
    "Num>=12<16",
    "Sum_Eggs",
    "Sum_MII",
    "Sum_Fert",
]
BABY_UTILITY_FEATURES = [
    *UTILITY_BASE,
    "Retrieval_Observed",
    "Sum_Eggs",
    "Sum_MII",
    "Sum_Fert",
    "Sum_Transferred",
    "Day_of_transfer",
]
OBSERVATION_UTILITY_FEATURES = [*BABY_UTILITY_FEATURES, "Transfer"]
BASE_FEATURES = [
    "Patient_Age",
    "Producer_Age",
    "Producer_BMI",
    "AMH",
    "Day3_FSH",
    "Day3_E2",
    "FSH",
    "HMG",
    "Fresh_Stim_Protocol",
    "Ovulatory_trigger",
    "DX_Dim_Ovar_Rsrv",
    "DX_Endometriosis",
    "DX_MaleFactor",
    "DX_Ovul_dysf",
    "DX_Tubal_disease",
    "DX_Unknown",
]
MII_FEATURES = [
    *BASE_FEATURES,
    "Trigger_E2",
    "Baseline_Endo",
    "Num>=18",
    "Num17",
    "Num16",
    "Num>=12<16",
    "Sum_Eggs",
]


def targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["transfer_target"] = np.where(
        frame.Retrieval_Observed.eq(1), frame.Transfer.eq(1).astype(float), np.nan
    )
    frame["baby_target"] = np.where(
        frame.Transfer.eq(1) & frame["#_Babies_(>24wks)"].notna(),
        frame["#_Babies_(>24wks)"].gt(0).astype(float),
        np.nan,
    )
    frame["observation_target"] = frame.Stage_Observed.eq(1).astype(int)
    frame["high_maturity"] = np.where(
        frame.Retrieval_Observed.eq(1) & frame.Sum_Eggs.notna() & frame.Sum_MII.notna(),
        (frame.Sum_MII / frame.Sum_Eggs.clip(lower=1)).ge(0.75).astype(float),
        np.nan,
    )
    frame["poor_response"] = np.where(
        frame.Retrieval_Observed.eq(1) & frame.Sum_Eggs.notna(),
        frame.Sum_Eggs.le(3).astype(float),
        np.nan,
    )
    frame["high_response"] = np.where(
        frame.Retrieval_Observed.eq(1) & frame.Sum_Eggs.notna(),
        frame.Sum_Eggs.ge(20).astype(float),
        np.nan,
    )
    return frame


def task_frame(frame: pd.DataFrame, task: str):
    frame = targets(frame)
    if task == "transfer":
        return (
            frame.loc[frame.transfer_target.notna()].copy(),
            TRANSFER_UTILITY_FEATURES,
            "transfer_target",
            "binary",
        )
    if task == "live_birth":
        return (
            frame.loc[frame.baby_target.notna()].copy(),
            BABY_UTILITY_FEATURES,
            "baby_target",
            "binary",
        )
    if task == "observation":
        return (
            frame.loc[frame.observation_target.notna()].copy(),
            OBSERVATION_UTILITY_FEATURES,
            "observation_target",
            "binary",
        )
    retrieval = frame.loc[frame.Retrieval_Observed.eq(1)].copy()
    if task == "high_maturity":
        return (
            retrieval.loc[retrieval.high_maturity.notna()].copy(),
            MII_FEATURES,
            "high_maturity",
            "binary",
        )
    if task == "poor_response":
        return (
            retrieval.loc[retrieval.poor_response.notna()].copy(),
            BASE_FEATURES,
            "poor_response",
            "binary",
        )
    if task == "high_response":
        return (
            retrieval.loc[retrieval.high_response.notna()].copy(),
            BASE_FEATURES,
            "high_response",
            "binary",
        )
    raise KeyError(task)
