"""Analysis parameters for the intake phase vs. BOSS comparison.

Edit the CAPS settings below. They are used by ``run_intake_phase_html_report.py``,
``inspect_intake_metadata.py`` and ``scripts/check_new_code.py``. The notebook
(``phase_pipeline.ipynb``) sets its own parameters in each section.
"""

from pathlib import Path

from functions import DEFAULT_LABELS, LoadConfig


# ---- Subjects and data ---------------------------------------------------
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
LABELS = DEFAULT_LABELS  # class names + raw metadata values for positive/negative

DOWNSAMPLE = True
DOWNSAMPLE_FS = 1000.0
LOWPASS_BEFORE_DOWNSAMPLE_HZ = 100.0
N_EPOCHS = False  # False = all epochs; an int caps epochs per subject (fast smoke runs)

SHOW_METADATA_SUMMARY = True
METADATA_MAX_VALUES = 12

LOAD_CONFIG = LoadConfig(
    data_root=DATA_ROOT,
    intake_filename=INTAKE_FILENAME,
    channel=CHANNEL,
    condition_column=CONDITION_COLUMN,
    condition_auto_keywords=CONDITION_AUTO_KEYWORDS,
    lowpass_before_downsample_hz=LOWPASS_BEFORE_DOWNSAMPLE_HZ,
    downsample=DOWNSAMPLE,
    downsample_fs=DOWNSAMPLE_FS,
    hjorth_channel=HJORTH_CHANNEL,
    hjorth_weights=HJORTH_WEIGHTS,
    hjorth_scale_reference=HJORTH_SCALE_REFERENCE,
    labels=LABELS,
    show_metadata_summary=SHOW_METADATA_SUMMARY,
    metadata_max_values=METADATA_MAX_VALUES,
)

# ---- Phase estimation ----------------------------------------------------
BAND = (4.0, 8.0)
CUTOFF_MS = 0.0

# Classification windows (+/- deg around 0 = positive and 180 = negative).
TOLERANCES_DEG = (45, 30, 15)
DEFAULT_TOLERANCE_DEG = 30

# Non-causal (ground-truth) filter order rule:
#   CAUSAL_PARAMS_MODE = "optimize" -> each subject's optimized causal filter_order is used
#                                      for the non-causal estimate too (one shared filter).
#   CAUSAL_PARAMS_MODE = "manual" or CAUSAL_ESTIMATION = False -> FILTER_ORDER below.
# See functions.resolve_noncausal_filter_order.
FILTER_ORDER = 350

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
# OPT_N_TRIALS counts Optuna trials (candidate parameter sets), not EEG epochs.
OPT_N_TRIALS = 200
OPT_TRAIN_FRACTION = 0.8
OPT_RANDOM_SEED = 0
OPT_MIN_EPOCHS = 20
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
    "min_usable_epochs": OPT_MIN_EPOCHS,
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
