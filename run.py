"""Estimate the pre-stimulus phase (and band SNR) at each TMS pulse from preprocessed epoch files.

Edit the CAPS parameters below, then run:  python run.py
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from phastimate import design_bandpass, phastimate, estimate_snr

# ---- DATA -------------------------------------------------------------------
# Served from the lab NAS over SMB; mount via Finder > Go > Connect to Server >
# smb://censorlabfs.tau.ac.il/CENSORLAB$ (VPN must be active) before running.
# Bar/phase_estimation_analysis on the share only has sub_103 -- the full cohort below
# lives under Or's Cortical Communication project instead.
# DATA_ROOT = Path("/Volumes/CENSORLAB$/Bar/phase_estimation_analysis")
DATA_ROOT = Path("/Volumes/CENSORLAB$/Or/Cortical Communication/CorticalCommunication/Data")

# One subject, or a list to run several at once (each subject's rows are tagged with a
# "subject" column in the output CSV).
SUBJECT = ["sub_103"]
# known subjects under DATA_ROOT above:
# sub_101, sub_102, sub_103, sub_104, sub_105, sub_106, sub_107, sub_108, sub_109,
# sub_110, sub_111, sub_112, sub_113
SESSION = "exp"                # "exp" -> {SUBJECT}_exp-epo.fif, "intake" -> {SUBJECT}_intake_stim-epo.fif
CHANNEL = "C3"                 # channel over the stimulated area

# ---- OSCILLATION / FILTER (phastimate.m) ------------------------------------
BAND = (8.0, 13.0)            # target band [Hz]; sensorimotor mu / alpha
FILTER_ORDER = 190            # FIR order (demo default at 1 kHz)
AR_ORDER = 30                 # Yule-Walker AR order
EDGE = 64                     # filter-edge samples removed
HILBERT_WINDOW = 128          # samples for the Hilbert transform
OFFSET = 0                    # phase-index offset correction

# ---- ANALYSIS WINDOW --------------------------------------------------------
WINDOW_MS = 750.0             # length of the window ending near each pulse
PRE_PULSE_MS = 5.0            # window ends this many ms BEFORE the pulse (avoid TMS artifact)

# ---- SNR (estimate_SNR.m; 1/f-corrected) ------------------------------------
# The epoch files only carry 1000 ms of pre-pulse data (tmin=-1.0s), so the SNR
# window is capped well below the 4s used in the original continuous-recording
# design. Set COMPUTE_SNR = False if that's too short to be meaningful for your band.
COMPUTE_SNR = True
SNR_WINDOW_MS = 900.0         # window ending at PRE_PULSE_MS used for the spectrum

# Background-only bins used to fit the 1/f trend (estimate_SNR.m default). Safe for BAND
# (8-13 Hz) here since it doesn't overlap this range -- re-check before reusing for a
# lower target band (e.g. theta), where 0.5-7 Hz would run into the band itself.
SNR_FIT_RANGES = [(0.5, 7.0), (35.0, 65.0)]

# Welch segment length/overlap for estimate_snr()'s spectrum. The original estimate_SNR.m
# default (nperseg=2s, no overlap) needs several seconds of data to average over; our
# 900ms SNR_WINDOW_MS can't fit even one such segment, so it silently fell back to a
# single raw (high-variance) periodogram. A shorter, overlapping segment gets several
# averaged periodograms out of the same 900ms instead, at the cost of frequency resolution.
SNR_NPERSEG_MS = 300.0         # Welch segment length
SNR_OVERLAP = 0.5              # fraction of SNR_NPERSEG_MS overlapping between segments

# ---- OUTPUT -----------------------------------------------------------------
OUT_CSV = "pulse_phases.csv"

_EPO_FILENAME = {
    "exp": "{subject}_exp-epo.fif",
    "intake": "{subject}_intake_stim-epo.fif",
}


def _process_subject(subject):
    epo_path = (
        DATA_ROOT / subject / "EEG" / "processed"
        / _EPO_FILENAME[SESSION].format(subject=subject)
    )
    epochs = mne.read_epochs(epo_path, preload=True)
    fs = epochs.info["sfreq"]
    data = epochs.get_data(picks=CHANNEL)[:, 0, :]  # (n_epochs, n_times)

    pulse_sample = int(round(-epochs.tmin * fs))    # sample index of the TMS pulse (t=0) within each epoch
    b = design_bandpass(FILTER_ORDER, BAND[0], BAND[1], fs)
    win = round(WINDOW_MS / 1000 * fs)
    pre = round(PRE_PULSE_MS / 1000 * fs)
    snr_win = round(SNR_WINDOW_MS / 1000 * fs)
    snr_nperseg = round(SNR_NPERSEG_MS / 1000 * fs)
    snr_noverlap = round(snr_nperseg * SNR_OVERLAP)

    rows = []
    for i, x in enumerate(data):
        end = pulse_sample - pre                       # last sample used (pre-pulse)
        if end - win < 0:
            continue                                   # not enough data around the pulse
        phase, amp = phastimate(x[end - win:end], b, EDGE, AR_ORDER, HILBERT_WINDOW, OFFSET)
        row = {"subject": subject, "epoch": i, "phase_rad": phase, "phase_deg": np.degrees(phase), "amplitude": amp}
        if COMPUTE_SNR and end - snr_win >= 0:
            row["snr_peak_hz"], row["snr_db"] = estimate_snr(
                x[end - snr_win:end], fs, BAND, SNR_FIT_RANGES, snr_nperseg, snr_noverlap
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if epochs.metadata is not None:
        df = df.join(epochs.metadata.reset_index(drop=True), on="epoch")
    return df


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"{DATA_ROOT} is not reachable. Check that the VPN is connected and the "
            "CENSORLAB$ share is mounted (Finder > Go > Connect to Server > "
            "smb://censorlabfs.tau.ac.il/CENSORLAB$)."
        )

    subjects = [SUBJECT] if isinstance(SUBJECT, str) else list(SUBJECT)
    df = pd.concat([_process_subject(subject) for subject in subjects], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"{len(df)} pulses across {len(subjects)} subject(s) -> {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
