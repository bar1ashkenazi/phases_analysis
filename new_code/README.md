# Intake Phase Viewer

Reusable code lives in `functions.py`.
Analysis parameters and paths live in the runner scripts, not in `functions.py`.

Manual runners:

- `run_intake_phase_viewer.py` loads one subject or a list of subjects, estimates the
  non-causal phase at `t=0`, optionally estimates causal AR phase, and shows a
  circular clickable plot colored by BOSS classification.
- `inspect_intake_metadata.py` prints intake metadata columns and values, useful for
  confirming which column contains the BOSS positive/negative labels.
- `run_intake_phase_html_report.py` runs the calculation once, caches the results
  as JSON, and writes a self-contained interactive HTML report. Re-run it with
  `--from-cache` to rebuild only the HTML after visualization edits.

Run from the project root:

```bash
./.venv/bin/python new_code/run_intake_phase_viewer.py
```

Set `CAUSAL_ESTIMATION = False` for the old non-causal-only plot. Set
`CAUSAL_PARAMS_MODE = "manual"` to use the runner's AR parameters, or `"optimize"`
to fit one causal AR parameter set per subject and print the selected values.

Interactive HTML report:

```bash
./.venv/bin/python new_code/run_intake_phase_html_report.py
```

Outputs are written to `new_code/intake_phase_report/`:

- `intake_phase_results.json`: cached estimates and selected causal params.
- `intake_phase_report.html`: the interactive report.

To avoid rerunning optimization while iterating on the HTML, use:

```bash
./.venv/bin/python new_code/run_intake_phase_html_report.py --from-cache
```
