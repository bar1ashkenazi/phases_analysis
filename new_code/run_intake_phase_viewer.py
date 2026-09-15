"""Manual runner for intake phase vs. BOSS classification.

Edit the CAPS settings below, then run from the project root:

    ./.venv/bin/python new_code/run_intake_phase_viewer.py
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
(XDG_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

from functions import (
    estimate_all_subjects,
    load_intake_subjects,
    plot_intake_phase_circle,
    print_phase_summary,
)


# ---- MANUAL SETTINGS -----------------------------------------------------
# Subjects with available intake_stim epoch files: 11
# Good intake subjects:
# ["sub_102", "sub_103", "sub_104", "sub_105", "sub_107", "sub_108",
#  "sub_109", "sub_110", "sub_111", "sub_112", "sub_113"]
# Missing intake_stim files: sub_101, sub_106
SUBJECTS = ["sub_102"] #["sub_102", "sub_103", "sub_104", "sub_105", "sub_107", "sub_108",  "sub_109", "sub_110", "sub_111", "sub_112", "sub_113"]

# Intake epochs are loaded from:
# DATA_ROOT / subject / "EEG" / "processed" / f"{subject}_intake_stim-epo.fif"
DATA_ROOT = Path("/Volumes/CENSORLAB$/Or/Cortical Communication/CorticalCommunication/Data")
INTAKE_FILENAME = "{subject}_intake_stim-epo.fif"

CHANNEL = "Fz_hjorth"
HJORTH_CHANNEL = "Fz_hjorth"
HJORTH_WEIGHTS = {"Fz": 1, "AF3": -0.25, "AF4": -0.25, "FC1": -0.25, "FC2": -0.25}
HJORTH_SCALE_REFERENCE = "Fz"

CONDITION_COLUMN = "Condition"
CONDITION_AUTO_KEYWORDS = ("condition", "boss", "classification", "class")
POSITIVE_NAME = "positive"
NEGATIVE_NAME = "negative"
UNKNOWN_NAME = "unknown"
NONCAUSAL_UNCLASSIFIED_NAME = "unclassified"
POSITIVE_LABELS = {"positive", "pos", "p", "1", "true", "peak"}
NEGATIVE_LABELS = {"negative", "neg", "n", "0", "false", "trough"}
CLASS_ORDER = (POSITIVE_NAME, NEGATIVE_NAME, UNKNOWN_NAME)
CLASS_COLORS = {
    POSITIVE_NAME: "tab:green",
    NEGATIVE_NAME: "tab:red",
    UNKNOWN_NAME: "tab:gray",
}

BAND = (4.0, 8.0)
FILTER_ORDER = 350
CUTOFF_MS = 0.0
PHASE_CLASS_TOLERANCE_DEG = 45.0

DOWNSAMPLE = True
DOWNSAMPLE_FS = 1000.0
LOWPASS_BEFORE_DOWNSAMPLE_HZ = 100.0
N_TRIALS = False

# Set False for the old non-causal-only view.
CAUSAL_ESTIMATION = True
CAUSAL_PARAMS_MODE = "optimize"  # "manual" or "optimize"

# Manual causal AR params, matching the previous intake/phastimate analysis.
WINDOW_MS = 510.0
EDGE = 65
AR_ORDER = 78
HILBERT_WINDOW = 128
OFFSET = 0
MANUAL_CAUSAL_PARAMS = {
    "window_ms": WINDOW_MS,
    "filter_order": FILTER_ORDER,
    "edge": EDGE,
    "ar_order": AR_ORDER,
    "hilbert_window": HILBERT_WINDOW,
    "offset": OFFSET,
}

# Optimization ranges copied from pipeline.py's theta-adjusted Optuna setup.
OPT_N_TRIALS = 200
OPT_TRAIN_FRACTION = 0.8
OPT_RANDOM_SEED = 0
OPT_MIN_TRIALS = 20
OPT_WINDOW_MS_RANGE = (300, 950)
OPT_WINDOW_MS_STEP = 10
OPT_FILTER_ORDER_RANGE = (150, 450)
OPT_FILTER_ORDER_STEP = 5
OPT_EDGE_RANGE = (20, 200)
OPT_EDGE_STEP = 5
OPT_AR_ORDER_RANGE = (5, 80)
OPT_INFEASIBLE_PENALTY = 10.0
CAUSAL_OPTIMIZATION_CONFIG = {
    "n_opt_trials": OPT_N_TRIALS,
    "train_fraction": OPT_TRAIN_FRACTION,
    "random_seed": OPT_RANDOM_SEED,
    "min_usable_trials": OPT_MIN_TRIALS,
    "window_ms_range": OPT_WINDOW_MS_RANGE,
    "window_ms_step": OPT_WINDOW_MS_STEP,
    "filter_order_range": OPT_FILTER_ORDER_RANGE,
    "filter_order_step": OPT_FILTER_ORDER_STEP,
    "edge_range": OPT_EDGE_RANGE,
    "edge_step": OPT_EDGE_STEP,
    "ar_order_range": OPT_AR_ORDER_RANGE,
    "hilbert_window": HILBERT_WINDOW,
    "offset": OFFSET,
    "infeasible_penalty": OPT_INFEASIBLE_PENALTY,
}

SHOW_METADATA_SUMMARY = True
METADATA_MAX_VALUES = 12
SAVE_FIGURE = False
OUT_FIGURE = Path("new_code/intake_phase_vs_boss.png")


def main() -> None:
    subject_data = load_intake_subjects(
        SUBJECTS,
        data_root=DATA_ROOT,
        intake_filename=INTAKE_FILENAME,
        channel=CHANNEL,
        condition_column=CONDITION_COLUMN,
        lowpass_before_downsample_hz=LOWPASS_BEFORE_DOWNSAMPLE_HZ,
        downsample=DOWNSAMPLE,
        downsample_fs=DOWNSAMPLE_FS,
        show_metadata_summary=SHOW_METADATA_SUMMARY,
        metadata_max_values=METADATA_MAX_VALUES,
        hjorth_channel=HJORTH_CHANNEL,
        hjorth_weights=HJORTH_WEIGHTS,
        hjorth_scale_reference=HJORTH_SCALE_REFERENCE,
        positive_labels=POSITIVE_LABELS,
        negative_labels=NEGATIVE_LABELS,
        positive_name=POSITIVE_NAME,
        negative_name=NEGATIVE_NAME,
        unknown_name=UNKNOWN_NAME,
        condition_auto_keywords=CONDITION_AUTO_KEYWORDS,
    )
    estimates = estimate_all_subjects(
        subject_data,
        band=BAND,
        filter_order=FILTER_ORDER,
        cutoff_ms=CUTOFF_MS,
        n_trials=N_TRIALS,
        phase_class_tolerance_deg=PHASE_CLASS_TOLERANCE_DEG,
        positive_name=POSITIVE_NAME,
        negative_name=NEGATIVE_NAME,
        unclassified_name=NONCAUSAL_UNCLASSIFIED_NAME,
        causal_estimation=CAUSAL_ESTIMATION,
        causal_params_mode=CAUSAL_PARAMS_MODE,
        manual_causal_params=MANUAL_CAUSAL_PARAMS,
        optimization_config=CAUSAL_OPTIMIZATION_CONFIG,
    )
    print_phase_summary(
        estimates,
        class_order=CLASS_ORDER,
        positive_name=POSITIVE_NAME,
        negative_name=NEGATIVE_NAME,
        unclassified_name=NONCAUSAL_UNCLASSIFIED_NAME,
    )
    plot_intake_phase_circle(
        estimates,
        causal_estimation=CAUSAL_ESTIMATION,
        cutoff_ms=CUTOFF_MS,
        band=BAND,
        phase_class_tolerance_deg=PHASE_CLASS_TOLERANCE_DEG,
        class_order=CLASS_ORDER,
        class_colors=CLASS_COLORS,
        positive_name=POSITIVE_NAME,
        negative_name=NEGATIVE_NAME,
        unclassified_name=NONCAUSAL_UNCLASSIFIED_NAME,
        show=True,
        save_path=OUT_FIGURE if SAVE_FIGURE else None,
    )


if __name__ == "__main__":
    main()
