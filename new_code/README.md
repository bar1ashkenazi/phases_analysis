# Intake Phase Analysis

All analysis logic lives in `functions.py` (single source of truth). Parameters and
paths live in `settings.py`. Everything else only calls `functions.py` and renders
the results.

Terminology: an epoch is the EEG window around one stimulation event (one BOSS
decision, t=0); a memory-task trial contains several. Epochs are analyzed
independently. An Optuna "trial" (`OPT_N_TRIALS`) is one candidate causal parameter
set, not an EEG epoch.

Files:

- `functions.py`: loading (`get_data`), phase estimation (`noncausal_phase`,
  `causal_phase`, `estimate_all_subjects`), causal parameter optimization,
  classification/scoring (`classify_phase`, `score_vs_boss`), deviations and
  circular statistics (`analyze_phase_results`), and static figures.
- `settings.py`: subjects, data path, preprocessing, band, tolerances, causal
  parameters and optimization ranges.
- `run_intake_phase_html_report.py`: computes estimates once, caches them as JSON,
  and writes a self-contained interactive HTML report plus a publication figure.
- `phase_pipeline.ipynb`: the pipeline for one subject, section by section (load,
  one epoch step by step, optional optimization, BOSS success, figures). Each section
  sets its own parameters; it does not read `settings.py`.
- `inspect_intake_metadata.py`: prints intake metadata columns and values, useful
  for confirming which column contains the BOSS positive/negative labels.

Ground-truth filter order: with `CAUSAL_PARAMS_MODE = "optimize"` each subject's
optimized causal `filter_order` is also used for the non-causal estimate; with
`"manual"` (or `CAUSAL_ESTIMATION = False`) `FILTER_ORDER` is used.

## Interactive HTML report

Run from the project root:

```bash
./.venv/bin/python new_code/run_intake_phase_html_report.py
```

Outputs are written to `new_code/intake_phase_report/`:

- `intake_phase_results.json`: cached estimates and selected causal params.
- `intake_phase_report.html`: phase circle, epoch signals, deviation histograms
  (phase − BOSS target, causal − non-causal; per subject or pooled) and BOSS
  success vs tolerance.
- `success_vs_tolerance.pdf` / `.png`: publication version of the success figure
  (one dot per subject, mean ± SD, chance = 2T/360).

To rebuild the HTML and figures without touching the EEG data (all derived numbers
are recomputed from the cached phases):

```bash
./.venv/bin/python new_code/run_intake_phase_html_report.py --from-cache
```

Synthetic demo (no data needed): add `--demo --output-dir /tmp/intake_phase_report_demo`.

## Notebook

In VS Code: open `new_code/phase_pipeline.ipynb`, choose the `.venv` kernel (Select
Kernel, top right), set `DATA_ROOT` and `SUBJECT` in section 1, then Run All. The data
share must be mounted (VPN + Finder > Go > Connect to Server). Set `DEMO = True` to run
on synthetic epochs without data.

## Checks

```bash
./.venv/bin/python scripts/check_new_code.py              # compile, imports, synthetic tests, demo report
./.venv/bin/python scripts/check_new_code.py --notebook   # also execute the notebook in DEMO mode
./.venv/bin/python scripts/check_new_code.py --with-data --subject sub_103 --n-epochs 2   # NAS mounted
```
