"""Source-schema declarations only; no cohort-specific estimators or tuning."""

from .engine import Schema, Stage

ANT_NUM = (
    "age",
    "inf_years",
    "afc",
    "bmi",
    "previous_et_number",
    "stimulation_days",
    "daily_fsh",
    "daily_hmg",
    "total_fsh",
)
ANT_CAT = ("primary_infertility", "diagnosis", "stimulation_type", "trigger_type")
ANT_BASE = ANT_NUM + ANT_CAT
ANT_COUNTS = ("eggs", "mii", "fertilized", "blastocysts", "frozen_blastocysts")
ANT_MASKS = (
    ("first_fet_transferred", "first_fet_transferred_observed"),
    ("first_fet_endo_mm", "first_fet_endo_observed"),
    ("first_fet_pregnancy", "first_fet_pregnancy_observed"),
    ("first_fet_live_birth", "first_fet_live_birth_observed"),
)


def antalya():
    stages = [
        Stage("eggs", "positive_count", ANT_BASE),
        Stage(
            "mii", "bounded_count", ANT_BASE + ("eggs",), total="eggs", log_total=True
        ),
    ]
    features = ANT_BASE + ("eggs", "mii")
    for target, total, excess in [
        ("fertilized", "mii", True),
        ("blastocysts", "fertilized", False),
        ("frozen_blastocysts", "blastocysts", True),
    ]:
        stages.append(
            Stage(
                target,
                "bounded_count",
                features,
                total=total,
                allow_excess=excess,
                log_total=True,
            )
        )
        features += (target,)
    stages.append(Stage("first_fet_eligible", "categorical", features))
    features += ("first_fet_eligible",)
    gate = (("first_fet_eligible", (1,)),)
    for target in [
        "first_fet_transferred",
        "first_fet_embryo_score",
        "first_fet_endo_mm",
        "first_fet_pregnancy",
        "first_fet_live_birth",
    ]:
        stages.append(
            Stage(
                target,
                "continuous" if target == "first_fet_endo_mm" else "categorical",
                features,
                gates=gate,
            )
        )
        features += (target,)
    return Schema(
        "Antalya",
        "Patient_Code",
        "Cycle_ID",
        ANT_CAT + ("first_fet_embryo_score",),
        tuple(stages),
        ANT_MASKS,
        (("first_fet_eligible", "positive", ("first_fet_transferred",)),),
        mask_groups=(
            (
                tuple(v for v, _ in ANT_MASKS),
                ANT_BASE + ANT_COUNTS + ("first_fet_eligible",),
                gate,
            ),
        ),
    )


BWH_PARENT_CAT = (
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
)
BWH_CAT = BWH_PARENT_CAT + (
    "Cycle_Type",
    "CET_cycle_type",
    "Fresh_Stim_Protocol",
    "Ovulatory_trigger",
)
BWH_BASE = (
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
    *BWH_PARENT_CAT[2:8],
)
BWH_UPSTREAM = tuple(
    dict.fromkeys(
        BWH_BASE
        + (
            "Cycle_Order",
            "Cycle_Type",
            "CET_cycle_type",
            "Trigger_E2",
            "Day_Ovulatory_Trigger",
            "Baseline_Endo",
            "Num>=18",
            "Num17",
            "Num16",
            "Num>=12<16",
        )
    )
)


def bwh():
    retrieval = (("Retrieval_Observed", (1,)),)
    stages = [
        Stage("Retrieval_Observed", "categorical", BWH_UPSTREAM),
        Stage("Sum_Eggs", "positive_count", BWH_UPSTREAM, gates=retrieval),
        Stage(
            "Sum_MII",
            "bounded_count",
            BWH_UPSTREAM + ("Sum_Eggs",),
            total="Sum_Eggs",
            gates=retrieval,
            log_total=True,
        ),
        Stage(
            "Sum_Fert",
            "bounded_count",
            BWH_UPSTREAM + ("Sum_Eggs", "Sum_MII"),
            total="Sum_MII",
            gates=retrieval,
            allow_excess=True,
            log_total=True,
        ),
    ]
    features = BWH_UPSTREAM + ("Retrieval_Observed", "Sum_Eggs", "Sum_MII", "Sum_Fert")
    stages.append(Stage("Transfer", "categorical", features))
    features += ("Transfer",)
    gate = (("Transfer", (1,)),)
    stages.append(
        Stage("Sum_Transferred", "categorical", features, gates=gate, inactive=0.0)
    )
    features += ("Sum_Transferred",)
    stages.append(Stage("Day_of_transfer", "categorical", features, gates=gate))
    features += ("Day_of_transfer",)
    for target, total in [
        ("Sacs_Max", "Sum_Transferred"),
        ("FH>12wk", "Sacs_Max"),
        ("#_Babies_(>24wks)", "FH>12wk"),
    ]:
        stages.append(
            Stage(
                target,
                "bounded_count",
                features,
                total=total,
                gates=gate,
                allow_excess=True,
                log_total=True,
                inactive=0.0,
            )
        )
        features += (target,)
    return Schema(
        "BWH",
        "Patient_ID",
        "Cycle_ID",
        BWH_CAT,
        tuple(stages),
        derived=(
            ("Fert_Exceeds_MII", "greater", ("Sum_Fert", "Sum_MII")),
            (
                "Stage_Observed",
                "all_observed",
                ("Sum_Transferred", "Sacs_Max", "FH>12wk", "#_Babies_(>24wks)"),
            ),
            (
                "Stage_Discordant",
                "discordant",
                ("Sum_Transferred", "Sacs_Max", "FH>12wk", "#_Babies_(>24wks)"),
            ),
        ),
        mask_groups=(
            (
                ("Sum_Transferred", "Sacs_Max", "FH>12wk", "#_Babies_(>24wks)"),
                BWH_UPSTREAM
                + ("Retrieval_Observed", "Sum_Eggs", "Sum_MII", "Sum_Fert", "Transfer"),
                (),
            ),
        ),
    )


SCHEMAS = {"antalya": antalya, "bwh": bwh}
