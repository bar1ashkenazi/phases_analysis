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
- `phase_pipeline_walkthrough.ipynb`: step-by-step notebook (load one epoch as a
  NumPy array, each processing step with plots and parameter explanations, final
  results). Update `DATA_ROOT` in its settings cell to your data location.
- `phase_pipeline_minimal.ipynb`: the same pipeline for one subject in a few short
  cells, with all parameters in its first cells (independent of `settings.py`).
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
- `intake_phase_report.html`: phase circle, trial signals, deviation histograms
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

```bash
./.venv/bin/jupyter lab new_code/phase_pipeline_walkthrough.ipynb
```

In VS Code: open the notebook, choose the `.venv` kernel (Select Kernel, top right), Run All.
Set `DEMO = True` in the first code cell to run on synthetic epochs.

## Checks

```bash
./.venv/bin/python scripts/check_new_code.py              # compile, imports, synthetic tests, demo report
./.venv/bin/python scripts/check_new_code.py --notebook   # also execute both notebooks in DEMO mode
./.venv/bin/python scripts/check_new_code.py --with-data --subject sub_103 --n-trials 2   # NAS mounted
```
