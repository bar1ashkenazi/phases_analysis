"""Inspect intake epoch metadata before choosing the BOSS label column.

Edit SUBJECTS, then run from the project root:

    ./.venv/bin/python new_code/inspect_intake_metadata.py
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

from functions import available_intake_subjects, intake_epoch_path, load_intake_subject


SUBJECTS = ["sub_103"]
SHOW_AVAILABLE_SUBJECTS = True

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
POSITIVE_LABELS = {"positive", "pos", "p", "1", "true", "peak"}
NEGATIVE_LABELS = {"negative", "neg", "n", "0", "false", "trough"}

DOWNSAMPLE = True
DOWNSAMPLE_FS = 1000.0
LOWPASS_BEFORE_DOWNSAMPLE_HZ = 100.0
METADATA_MAX_VALUES = 12


def main() -> None:
    if SHOW_AVAILABLE_SUBJECTS:
        subjects = available_intake_subjects(DATA_ROOT, INTAKE_FILENAME)
        if subjects:
            print("Available intake subjects:", ", ".join(subjects))
        else:
            print(f"No intake subjects found under {DATA_ROOT}")

    for subject in SUBJECTS:
        print(f"\nInspecting {subject}: {intake_epoch_path(subject, DATA_ROOT, INTAKE_FILENAME)}")
        load_intake_subject(
            subject,
            data_root=DATA_ROOT,
            intake_filename=INTAKE_FILENAME,
            channel=CHANNEL,
            condition_column=CONDITION_COLUMN,
            lowpass_before_downsample_hz=LOWPASS_BEFORE_DOWNSAMPLE_HZ,
            downsample=DOWNSAMPLE,
            downsample_fs=DOWNSAMPLE_FS,
            show_metadata_summary=True,
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


if __name__ == "__main__":
    main()
