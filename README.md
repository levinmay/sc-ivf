# SC-IVF

SC-IVF generates linked patient and treatment-cycle records by modeling successive
stages of in vitro fertilization (IVF). The same implementation supports the
Antalya and Brigham and Women's Hospital cohorts.

Code for [Stage-Conditional Synthetic IVF Records for Predictive Modeling](https://openreview.net/forum?id=eyaaR7i3OF),
GenAI for Health at NeurIPS 2026.

## Install

Use Python 3.11.14 on Linux. The paper's TVAE baseline requires a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Run

```bash
python paper_results/scripts/run_pipeline.py --output output --workers 4
```

This downloads the public source data and runs the four-fold experiments.
See [reproduction instructions](REPRODUCE.md) for individual stages and additional checks.

The generator API is `shared_ivf.IVFGenerator`. Its `fit(patients, cycles)` method
learns from linked tables, and `sample(n_patients, seed)` returns synthetic tables.
Cohort schemas are in `shared_ivf/schemas.py`.

Aggregate paper results are in [paper_results](paper_results/tables).
Patient records and fitted models are not included. Synthetic output has no
formal differential-privacy guarantee.

Code: [MIT](LICENSE). Source datasets:
[BWH](https://doi.org/10.6084/m9.figshare.15153525.v1) and
[Antalya](https://doi.org/10.17632/xy2t2cxm8g.1), both CC BY 4.0.
