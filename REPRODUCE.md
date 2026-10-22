# Reproduce the experiments

Run commands from the repository root after installing `requirements.txt`.
The primary pipeline uses four patient-grouped development folds and 20x synthetic
releases. Reserved test partitions are excluded from fitting and evaluation.
Preparation reconstructs the source cohorts and their fixed partitions.

## Primary results

```bash
python paper_results/scripts/run_pipeline.py --output output --workers 4
```

Use `--stage prepare`, `baselines`, `evaluate`, or `analyze` to run one stage.
Completed jobs are reused. Failed or partial fits stop for inspection.
Results are written to `output/evaluation/aggregate`.

For a single GPU, use `--workers 1` to avoid simultaneous TVAE fits.

## Additional checks

These commands reuse the primary pipeline's fitted models and releases.

```bash
python paper_results/scripts/run_robustness.py core --evaluation output/evaluation --output output/robustness
python paper_results/scripts/run_robustness.py release --evaluation output/evaluation --output output/robustness
python paper_results/scripts/run_robustness.py diagnostics --evaluation output/evaluation --output output/robustness
python paper_results/scripts/analyze_robustness.py --evaluation output/evaluation --results output/robustness
python paper_results/scripts/build_robustness_tables.py --results output/robustness/summary
```

`core` evaluates a 1x patient subset, tunes the real-data predictor's iteration
count, and tests direct stage predictions for compatible Antalya tasks. Direct
count predictions use sampled means, not medians.

`release` repeats outcome sampling five times with the fitted generator and HMA
background records fixed. `diagnostics` computes membership-attack intervals.
Each phase writes a `DONE.json` receipt when complete. Use a new output directory for a separate run.

To rebuild the robustness tables from the included aggregate results without
training, run `python paper_results/scripts/build_robustness_tables.py`.

## Reproducibility details

`requirements.txt` pins the paper's environment. Seeds and estimator settings
are fixed in the scripts. `paper_results/tables/bootstrap_states.json`
preserves the random-number states for the published paired confidence intervals.
Raw downloads and experiment inputs are checked by SHA-256. Keep a run in its original location because its input manifest records
absolute paths.

BWH HMA sampling retains the published batch order, thread counts, and missingness
random-state sequence in `paper_results/scripts/bwh_hma_batches.py`. Changing these
can change the sampled records. The two cohort modules in `paper_results/scripts`
contain source preprocessing and baseline routines.

## Saved results

The CSVs in `paper_results/tables/` contain the aggregate results used in the camera-ready manuscript.
No patient-level predictions or records are included.

- `means.csv`, `absolute.csv`: primary and linear-predictor scores and intervals.
- `paired.csv`, `generic_comparisons.csv`: comparisons with real training and synthetic baselines.
- `linear_secondary_losses.csv`: secondary losses for linear predictors.
- `paired_comparisons.csv`: 1x release, tuned real-data, and direct-stage comparisons.
- `release_variability.csv`: variation across five additional outcome samples.
- `task_counts.csv`: eligible validation records and binary outcomes.
- `fidelity_summary.csv`: count-distribution Wasserstein distances.
- `membership.csv`: SC-IVF membership-attack scores and intervals.

Positive gains in `paired.csv` and `generic_comparisons.csv` favor SC-IVF.
In `paired_comparisons.csv`, they favor the `left` condition. AUROC gains are score increases. MAE gains are error reductions.
The `shared_ivf_v1_1` method identifier denotes SC-IVF.
