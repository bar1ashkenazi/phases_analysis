"""Headless checks for the intake phase viewer code.

This is intentionally light enough for Codex cloud. By default it only compiles
and imports the analysis code. Use --with-data on a machine that has the NAS
mounted to run a small real-data smoke test.
"""

from __future__ import annotations

import argparse
import compileall
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_CODE = PROJECT_ROOT / "new_code"
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
    import run_intake_phase_viewer  # noqa: F401
    import run_intake_phase_html_report  # noqa: F401
    import inspect_intake_metadata  # noqa: F401


def run_data_smoke(subject: str, n_trials: int, causal_params_mode: str) -> None:
    import run_intake_phase_viewer as r
    from functions import estimate_all_subjects, load_intake_subjects, plot_intake_phase_circle

    if not r.DATA_ROOT.exists():
        print(f"Skipping data smoke: data root not found: {r.DATA_ROOT}")
        return

    subject_data = load_intake_subjects(
        [subject],
        data_root=r.DATA_ROOT,
        intake_filename=r.INTAKE_FILENAME,
        channel=r.CHANNEL,
        condition_column=r.CONDITION_COLUMN,
        lowpass_before_downsample_hz=r.LOWPASS_BEFORE_DOWNSAMPLE_HZ,
        downsample=r.DOWNSAMPLE,
        downsample_fs=r.DOWNSAMPLE_FS,
        show_metadata_summary=False,
        metadata_max_values=r.METADATA_MAX_VALUES,
        hjorth_channel=r.HJORTH_CHANNEL,
        hjorth_weights=r.HJORTH_WEIGHTS,
        hjorth_scale_reference=r.HJORTH_SCALE_REFERENCE,
        positive_labels=r.POSITIVE_LABELS,
        negative_labels=r.NEGATIVE_LABELS,
        positive_name=r.POSITIVE_NAME,
        negative_name=r.NEGATIVE_NAME,
        unknown_name=r.UNKNOWN_NAME,
        condition_auto_keywords=r.CONDITION_AUTO_KEYWORDS,
    )
    estimates = estimate_all_subjects(
        subject_data,
        band=r.BAND,
        filter_order=r.FILTER_ORDER,
        cutoff_ms=r.CUTOFF_MS,
        n_trials=n_trials,
        phase_class_tolerance_deg=r.PHASE_CLASS_TOLERANCE_DEG,
        positive_name=r.POSITIVE_NAME,
        negative_name=r.NEGATIVE_NAME,
        unclassified_name=r.NONCAUSAL_UNCLASSIFIED_NAME,
        causal_estimation=r.CAUSAL_ESTIMATION,
        causal_params_mode=causal_params_mode,
        manual_causal_params=r.MANUAL_CAUSAL_PARAMS,
        optimization_config=r.CAUSAL_OPTIMIZATION_CONFIG,
    )
    trial_estimates = estimates[subject]
    if len(trial_estimates) != n_trials:
        raise RuntimeError(f"Expected {n_trials} estimates for {subject}, got {len(trial_estimates)}")
    if r.CAUSAL_ESTIMATION and not all(est.causal_phase_deg is not None for est in trial_estimates):
        raise RuntimeError("Causal estimation is enabled but at least one trial has no causal phase.")
    plot_intake_phase_circle(
        estimates,
        causal_estimation=r.CAUSAL_ESTIMATION,
        cutoff_ms=r.CUTOFF_MS,
        band=r.BAND,
        phase_class_tolerance_deg=r.PHASE_CLASS_TOLERANCE_DEG,
        class_order=r.CLASS_ORDER,
        class_colors=r.CLASS_COLORS,
        positive_name=r.POSITIVE_NAME,
        negative_name=r.NEGATIVE_NAME,
        unclassified_name=r.NONCAUSAL_UNCLASSIFIED_NAME,
        show=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-data", action="store_true", help="Run a small real-data smoke test if data is mounted.")
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
    if args.with_data:
        run_data_smoke(args.subject, args.n_trials, args.causal_params_mode)
    print("checks ok")


if __name__ == "__main__":
    main()
