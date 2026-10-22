"""Build the camera-ready robustness tables from saved aggregate results."""

from pathlib import Path
import csv
import argparse

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(
    description="Build the paper's robustness tables from aggregate CSVs."
)
parser.add_argument("--results", type=Path, default=ROOT / "paper_results/tables")
parser.add_argument(
    "--output", type=Path, default=ROOT / "output/tables/robustness_checks.tex"
)
args = parser.parse_args()


def read(name):
    with (args.results / name).open() as handle:
        return list(csv.DictReader(handle))


pairs = read("paired_comparisons.csv")
release = {(r["cohort"], r["task"]): r for r in read("release_variability.csv")}
TASKS = {
    "antalya": [
        ("poor_response", "Poor response"),
        ("high_response", "High response"),
        ("high_maturation", "High maturation"),
        ("first_fet_live_birth", "First-FET live birth"),
        ("egg_count", "Egg count"),
        ("mii_count", "MII count"),
        ("fertilized_count", "Fertilized count"),
        ("blastocyst_count", "Blastocyst count"),
        ("frozen_blastocyst_count", "Frozen blastocysts"),
    ],
    "bwh": [
        ("poor_response", "Poor response"),
        ("high_response", "High response"),
        ("high_maturity", "High maturation"),
        ("transfer", "Transfer"),
        ("observation", "Outcome observation"),
        ("live_birth", "Live birth"),
        ("egg_count", "Egg count"),
        ("mii_count", "MII count"),
    ],
}


def pair(cohort, task, left, right):
    rows = [
        r
        for r in pairs
        if r["cohort"] == cohort
        and r["task"] == task
        and r["left"] == left
        and r["right"] == right
        and r["left_family"] in ("hgb", "direct_stage")
    ]
    assert len(rows) <= 1
    return rows[0] if rows else None


def fmt(value):
    return f"{float(value):.3f}"


def interval(row, reverse=False):
    if row is None:
        return "--"
    gain, low, high = (float(row[k]) for k in ("gain", "low", "high"))
    if reverse:
        gain, low, high = -gain, -high, -low
    return f"${gain:.3f}$ [${low:.3f}$, ${high:.3f}$]"


lines = [
    r"\section{Robustness checks}",
    r"\label{app:robustness}",
    r"Tables~\ref{tab:robustness-scores} and~\ref{tab:robustness-paired} report the additional analyses on the same development folds. The $1\times$ condition is a deterministic whole-patient subset of the original $20\times$ release. All synthetic-trained HGB predictors retain the original 180 iterations. For real-data training, the iteration count is selected from 60, 120, 180, and 300 separately for each task and outer fold by mean inner-validation AUROC or MAE. Other settings remain fixed.",
    r"Direct predictions use the observed real validation inputs, with stage-model inputs checked against the downstream task inputs. We estimate conditional means for counts and event probabilities for binary tasks from 4,000 draws per record, including the fitted support and excess-count rules. This comparison does not evaluate conditional medians, which minimize expected absolute error. Direct first-FET live-birth prediction is omitted because its generator inputs differ from the downstream task inputs.",
    r"For repeated sampling, we regenerate the clinical stages five times per fold and retrain the fixed HGB predictors. The fitted generator and original HMA background records remain fixed. Standard deviations are calculated across five four-fold mean scores. They do not include generator refitting or background-record resampling.",
    r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}",
    r"\caption{Robustness analyses: four-fold means. SC denotes SC-IVF. Synthetic $1\times$ and $20\times$ use fixed HGB predictors. Real tuned selects the HGB iteration count. Direct uses the fitted stage models. Repeat SD denotes the standard deviation across five additional outcome samples.}",
    r"\label{tab:robustness-scores}",
    r"\begin{tabular}{llrrrrr}\toprule",
    r"Task & Metric & SC $1\times$ & SC $20\times$ & Real tuned & Direct & Repeat SD \\ \midrule",
]
interval_lines = []
for cohort, tasks in TASKS.items():
    lines.append(
        r"\multicolumn{7}{l}{\textit{"
        + ("Antalya" if cohort == "antalya" else "BWH")
        + r"}} \\"
    )
    interval_lines.append(
        r"\multicolumn{4}{l}{\textit{"
        + ("Antalya" if cohort == "antalya" else "BWH")
        + r"}} \\"
    )
    for task, label in tasks:
        size = pair(cohort, task, "sc_ivf_1x_fixed", "sc_ivf_20x_fixed")
        tuned = pair(cohort, task, "sc_ivf_20x_fixed", "real_tuned_iter")
        direct = pair(cohort, task, "direct_stage", "sc_ivf_20x_fixed")
        assert size and tuned
        assert abs(float(size["right_score"]) - float(tuned["left_score"])) < 1e-12
        values = [
            label,
            "AUROC" if size["kind"] == "binary" else "MAE",
            fmt(size["left_score"]),
            fmt(size["right_score"]),
            fmt(tuned["right_score"]),
            fmt(direct["left_score"]) if direct else "--",
            fmt(release[(cohort, task)]["std"]),
        ]
        lines.append(" & ".join(values) + r" \\")
        interval_lines.append(
            " & ".join(
                [label, interval(size, True), interval(tuned), interval(direct, True)]
            )
            + r" \\"
        )
    if cohort == "antalya":
        lines.append(r"\midrule")
        interval_lines.append(r"\midrule")
lines.extend(
    [
        r"\bottomrule\end{tabular}",
        r"\end{table}",
        r"\FloatBarrier",
        r"\begin{table}[htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{SC-IVF $20\times$ gains over each comparator, with paired 95\% CIs. Positive values favor SC-IVF (AUROC increases or MAE reductions).}",
        r"\label{tab:robustness-paired}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Task & Versus SC $1\times$ & Versus Real tuned & Versus Direct \\ \midrule",
        *interval_lines,
        r"\bottomrule\end{tabular}",
        r"\end{table}",
        r"\FloatBarrier",
    ]
)
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n\n".join(lines[:6]) + "\n" + "\n".join(lines[6:]) + "\n")
print(output)
