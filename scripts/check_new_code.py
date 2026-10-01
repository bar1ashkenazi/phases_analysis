"""Headless checks for the intake phase code.

By default this compiles and imports the analysis code and runs synthetic-data tests
(no EEG files needed). Use --with-data on a machine that has the NAS mounted to run a
small real-data smoke test, and --notebook to execute the walkthrough notebook in
DEMO mode (needs nbconvert + ipykernel).
"""

from __future__ import annotations

import argparse
import compileall
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_CODE = PROJECT_ROOT / "new_code"
NOTEBOOKS = [NEW_CODE / "phase_pipeline_walkthrough.ipynb", NEW_CODE / "phase_pipeline_minimal.ipynb"]
if str(NEW_CODE) not in sys.path:
    sys.path.insert(0, str(NEW_CODE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))


def compile_sources() -> None:
    paths = [
        PROJECT_ROOT / "phastimate.py",
        PROJECT_ROOT / "pipeline.py",
        PROJECT_ROOT / "causal_vs_boss.py",
        PROJECT_ROOT / "run.py",
        PROJECT_ROOT / "visualize_trials.py",
        NEW_CODE,
    ]
    for path in paths:
        if not compileall.compile_file(str(path), quiet=1) if path.is_file() else not compileall.compile_dir(str(path), quiet=1):
            raise RuntimeError(f"Compilation failed for {path}")


def import_new_code() -> None:
    import functions  # noqa: F401
    import settings  # noqa: F401
    import run_intake_phase_html_report  # noqa: F401
    import inspect_intake_metadata  # noqa: F401


def run_synthetic_tests() -> None:
    """The public array API and the pipeline agree, and report numbers match direct counts."""
    import numpy as np

    import functions as f
    import settings as s

    data = f.make_synthetic_epochs(n_epochs=20, seed=1)
    estimates = f.estimate_all_subjects(
        [data],
        band=s.BAND,
        filter_order=s.FILTER_ORDER,
        cutoff_ms=s.CUTOFF_MS,
        n_trials=False,
        causal_estimation=True,
        causal_params_mode="manual",
        manual_causal_params=s.MANUAL_CAUSAL_PARAMS,
        optimization_config=s.CAUSAL_OPTIMIZATION_CONFIG,
    )[data.subject]

    for est, x in zip(estimates, data.X):
        nc = f.noncausal_phase(x, data.times_ms, data.fs, s.BAND, s.FILTER_ORDER, s.CUTOFF_MS)
        c = f.causal_phase(x, data.times_ms, data.fs, s.BAND, s.CUTOFF_MS, s.MANUAL_CAUSAL_PARAMS)
        assert nc["phase_deg"] == est.phase_deg, "noncausal_phase differs from the pipeline"
        assert c is not None and c.phase_deg == est.causal.phase_deg, "causal_phase differs from the pipeline"

    arrays = f.estimates_to_arrays(estimates)
    analysis = f.analyze_phase_results({data.subject: arrays}, tolerances_deg=s.TOLERANCES_DEG)
    for tol in s.TOLERANCES_DEG:
        labeled = [e for e in estimates if e.label in (f.DEFAULT_LABELS.positive, f.DEFAULT_LABELS.negative)]
        correct = sum(f.classify_phase(e.phase_deg, tol) == e.label for e in labeled)
        entry = analysis["success"]["by_tolerance"][str(tol)]
        assert entry["correct"][0] == correct and entry["n_labeled"][0] == len(labeled)
        assert np.isclose(entry["chance_pct"], 100 * 2 * tol / 360)

    hist = analysis["subjects"][data.subject]["deviations"]["boss_target"]
    assert sum(sum(c) for c in hist["counts_by_class"].values()) == hist["stats"]["n"]
    stats = f.circular_stats([10.0, 10.0, 10.0], tolerances_deg=[15])
    assert np.isclose(stats["mean_deg"], 10.0) and np.isclose(stats["R"], 1.0) and stats["pct_within"]["15"] == 100.0
    assert f.resolve_noncausal_filter_order(None, 350) == 350
    assert f.resolve_noncausal_filter_order({"filter_order": 265}, 350) == 265


def run_report_demo() -> None:
    """Demo report and a cache round-trip, written to a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        script = NEW_CODE / "run_intake_phase_html_report.py"
        for flag in ("--demo", "--from-cache"):
            subprocess.run([sys.executable, str(script), flag, "--output-dir", tmp], check=True, capture_output=True)
        for name in ("intake_phase_report.html", "success_vs_tolerance.pdf", "success_vs_tolerance.png"):
            if not (Path(tmp) / name).exists():
                raise RuntimeError(f"Report demo did not write {name}")


def run_notebook_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PHASE_NOTEBOOK_DEMO": "1"}
        for notebook in NOTEBOOKS:
            subprocess.run(
                [
                    sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=600", "--output-dir", tmp, str(notebook),
                ],
                check=True,
                env=env,
                cwd=NEW_CODE,
            )


def run_data_smoke(subject: str, n_trials: int, causal_params_mode: str) -> None:
    import settings as s
    from functions import estimate_all_subjects, get_data

    if not s.DATA_ROOT.exists():
        print(f"Skipping data smoke: data root not found: {s.DATA_ROOT}")
        return

    estimates = estimate_all_subjects(
        [get_data(subject, s.LOAD_CONFIG)],
        band=s.BAND,
        filter_order=s.FILTER_ORDER,
        cutoff_ms=s.CUTOFF_MS,
        n_trials=n_trials,
        causal_estimation=s.CAUSAL_ESTIMATION,
        causal_params_mode=causal_params_mode,
        manual_causal_params=s.MANUAL_CAUSAL_PARAMS,
        optimization_config=s.CAUSAL_OPTIMIZATION_CONFIG,
    )
    trial_estimates = estimates[subject]
    if len(trial_estimates) != n_trials:
        raise RuntimeError(f"Expected {n_trials} estimates for {subject}, got {len(trial_estimates)}")
    if s.CAUSAL_ESTIMATION and not all(est.causal is not None for est in trial_estimates):
        raise RuntimeError("Causal estimation is enabled but at least one trial has no causal phase.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-data", action="store_true", help="Run a small real-data smoke test if data is mounted.")
    parser.add_argument("--notebook", action="store_true", help="Execute the notebooks in DEMO mode.")
    parser.add_argument("--subject", default="sub_103")
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument(
        "--causal-params-mode",
        choices=("manual", "optimize"),
        default="manual",
        help="Mode to use for the optional data smoke test. Defaults to manual so small smoke tests stay fast.",
    )
    args = parser.parse_args()

    compile_sources()
    import_new_code()
    run_synthetic_tests()
    run_report_demo()
    if args.notebook:
        run_notebook_demo()
    if args.with_data:
        run_data_smoke(args.subject, args.n_trials, args.causal_params_mode)
    print("checks ok")


if __name__ == "__main__":
    main()
