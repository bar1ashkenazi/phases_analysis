"""Inspect intake epoch metadata before choosing the BOSS label column.

Edit SUBJECTS (load settings come from settings.py), then run from the project root:

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

from dataclasses import replace

import settings
from functions import available_intake_subjects, get_data, intake_epoch_path


SUBJECTS = ["sub_103"]
SHOW_AVAILABLE_SUBJECTS = True


def main() -> None:
    config = replace(settings.LOAD_CONFIG, show_metadata_summary=True)
    if SHOW_AVAILABLE_SUBJECTS:
        subjects = available_intake_subjects(config.data_root, config.intake_filename)
        if subjects:
            print("Available intake subjects:", ", ".join(subjects))
        else:
            print(f"No intake subjects found under {config.data_root}")

    for subject in SUBJECTS:
        print(f"\nInspecting {subject}: {intake_epoch_path(subject, config.data_root, config.intake_filename)}")
        get_data(subject, config)


if __name__ == "__main__":
    main()
