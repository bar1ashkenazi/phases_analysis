# Codex Working Notes

## Project
- This repository contains EEG phase-estimation analysis code.
- The active clean implementation is under `new_code/`.
- Reusable logic belongs in `new_code/functions.py`.
- Analysis parameters, subject choices, paths, and run modes belong in `new_code/settings.py`.
- Runners, the HTML report and `new_code/phase_pipeline_walkthrough.ipynb` only call `functions.py`; no analysis math elsewhere (including the report JavaScript).

## Data
- Real intake data is not stored in git.
- The local data path currently used by the runner is:
  `/Volumes/CENSORLAB$/Or/Cortical Communication/CorticalCommunication/Data`
- Cloud environments usually will not have this NAS mounted. Do not assume data tests can run in Codex cloud unless the user explicitly says the data is available.

## Verification
- After cloning, initialize the MATLAB reference submodule if needed:
  `git submodule update --init --recursive`
- Basic no-data check:
  `./.venv/bin/python scripts/check_new_code.py`
- Optional data smoke check, when the NAS is mounted:
  `./.venv/bin/python scripts/check_new_code.py --with-data --subject sub_103 --n-epochs 2`

## Constraints
- Do not commit `.venv`, caches, generated figures, or raw EEG files.
- Preserve the user's local data paths unless explicitly asked to change them.
- Keep changes focused on intake-session analysis unless the user asks to expand scope.
