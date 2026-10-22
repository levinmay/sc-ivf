import numpy as np
import pandas as pd

UPSTREAM_NUM = [
    "age",
    "inf_years",
    "afc",
    "bmi",
    "previous_et_number",
    "stimulation_days",
    "daily_fsh",
    "daily_hmg",
    "total_fsh",
]
UPSTREAM_CAT = ["primary_infertility", "diagnosis", "stimulation_type", "trigger_type"]
COUNT_COLS = ["eggs", "mii", "fertilized", "blastocysts", "frozen_blastocysts"]
MASK_COLS = [
    "first_fet_transferred_observed",
    "first_fet_endo_observed",
    "first_fet_pregnancy_observed",
    "first_fet_live_birth_observed",
    "first_fet_eligible",
]
OUTCOME_COLS = ["first_fet_pregnancy", "first_fet_live_birth"]
NUM = (
    UPSTREAM_NUM
    + COUNT_COLS
    + ["first_fet_transferred", "first_fet_endo_mm"]
    + OUTCOME_COLS
    + MASK_COLS
)
CAT = UPSTREAM_CAT + ["first_fet_embryo_score"]
COLS = NUM + CAT
BASE_FEATURES = UPSTREAM_NUM + UPSTREAM_CAT
TASKS = [
    "poor_response",
    "high_response",
    "egg_count",
    "high_maturation",
    "mii_count",
    "fertilized_count",
    "blastocyst_count",
    "frozen_blastocyst_count",
    "first_fet_live_birth",
    "first_fet_recorded_pr",
]
TASK_KIND = {
    t: "binary"
    if t
    in [
        "poor_response",
        "high_response",
        "high_maturation",
        "first_fet_live_birth",
        "first_fet_recorded_pr",
    ]
    else "count"
    for t in TASKS
}
FEATURES = {
    "poor_response": BASE_FEATURES.copy(),
    "high_response": BASE_FEATURES.copy(),
    "egg_count": BASE_FEATURES.copy(),
    "high_maturation": BASE_FEATURES + ["eggs"],
    "mii_count": BASE_FEATURES + ["eggs"],
    "fertilized_count": BASE_FEATURES + ["eggs", "mii"],
    "blastocyst_count": BASE_FEATURES + ["eggs", "mii", "fertilized"],
    "frozen_blastocyst_count": BASE_FEATURES
    + ["eggs", "mii", "fertilized", "blastocysts"],
}
FET_FEATURES = (
    BASE_FEATURES
    + COUNT_COLS
    + [
        "first_fet_transferred",
        "first_fet_endo_mm",
        "first_fet_endo_observed",
        "first_fet_embryo_score",
    ]
)
FEATURES.update(
    {t: FET_FEATURES.copy() for t in ["first_fet_live_birth", "first_fet_recorded_pr"]}
)


def _valid_count(values):
    x = pd.to_numeric(values, errors="coerce")
    return x.notna() & np.isfinite(x) & x.ge(0) & x.eq(np.rint(x))


def task_frame(frame, task):
    if task not in TASKS:
        raise KeyError(task)
    missing = set(FEATURES[task]) - set(frame)
    if missing:
        raise ValueError("Task feature columns missing: " + str(sorted(missing)))
    if task in ["poor_response", "high_response", "egg_count"]:
        eligible = _valid_count(frame.eggs) & frame.eggs.gt(0)
        y = frame.eggs
        if task == "poor_response":
            y = y.le(3).astype(float)
        if task == "high_response":
            y = y.ge(20).astype(float)
    elif task in ["high_maturation", "mii_count"]:
        eligible = (
            _valid_count(frame.eggs)
            & frame.eggs.gt(0)
            & _valid_count(frame.mii)
            & frame.mii.le(frame.eggs)
        )
        y = (
            frame.mii
            if task == "mii_count"
            else (frame.mii / frame.eggs).ge(0.75).astype(float)
        )
    elif task in ["fertilized_count", "blastocyst_count", "frozen_blastocyst_count"]:
        name = {
            "fertilized_count": "fertilized",
            "blastocyst_count": "blastocysts",
            "frozen_blastocyst_count": "frozen_blastocysts",
        }[task]
        y = frame[name]
        eligible = _valid_count(y)
    else:
        name = (
            "first_fet_live_birth"
            if task == "first_fet_live_birth"
            else "first_fet_pregnancy"
        )
        y = frame[name]
        eligible = (
            frame.first_fet_transferred_observed.eq(1)
            & frame.first_fet_transferred.isin([1, 2])
            & frame[name + "_observed"].eq(1)
            & y.isin([0, 1])
        )
    return (frame.loc[eligible].copy(), y.loc[eligible].to_numpy(float))
