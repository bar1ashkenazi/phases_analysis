"""Step through TMS-pulse epochs and look at the phase estimation on the C3 trace.

Each trial opens its own window; close it to advance to the next one.
Edit the toggles below, then run:  python visualize_trials.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import filtfilt

from phastimate import design_bandpass, phastimate
from run import (
    DATA_ROOT, SUBJECT, SESSION, CHANNEL, BAND, FILTER_ORDER, AR_ORDER,
    EDGE, HILBERT_WINDOW, OFFSET, WINDOW_MS, PRE_PULSE_MS, _EPO_FILENAME,
)

# ---- TOGGLES ------------------------------------------------------------
SHOW = True                    # open an interactive window per trial (close it to advance)
SAVE = False                   # also save each trial's plot as a PNG
SHOW_ESTIMATE = False          # overlay the filtered oscillation + the fitted sine wave

OUT_DIR = Path("trial_plots")  # only used when SAVE = True


def main():
    # This tool steps through trials one window at a time, so it only makes sense for one
    # subject; if SUBJECT is a list (as in run.py/pipeline.py), the first entry is used.
    subject = SUBJECT if isinstance(SUBJECT, str) else SUBJECT[0]
    epo_path = (
        DATA_ROOT / subject / "EEG" / "processed"
        / _EPO_FILENAME[SESSION].format(subject=subject)
    )
    epochs = mne.read_epochs(epo_path, preload=True)
    fs = epochs.info["sfreq"]
    data_uv = epochs.get_data(picks=CHANNEL, units="uV")[:, 0, :]  # (n_epochs, n_times)
    times_ms = epochs.times * 1000

    pulse_sample = int(round(-epochs.tmin * fs))   # sample index of the TMS pulse (t=0)
    pre = round(PRE_PULSE_MS / 1000 * fs)
    win = round(WINDOW_MS / 1000 * fs)
    end = pulse_sample - pre                       # last sample fed into phastimate

    # Where phastimate actually reads the phase out, relative to the pulse (see run.py
    # discussion): reported sample = end + OFFSET, i.e. -PRE_PULSE_MS + OFFSET/fs.
    t_estimate_ms = -PRE_PULSE_MS + (OFFSET / fs) * 1000

    b = design_bandpass(FILTER_ORDER, BAND[0], BAND[1], fs)
    f0 = sum(BAND) / 2  # sine overlay frequency: center of the target band

    if SAVE:
        OUT_DIR.mkdir(exist_ok=True)

    for i, x in enumerate(data_uv):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(times_ms, x, color="black", lw=0.8, label=f"raw {CHANNEL}")

        if SHOW_ESTIMATE and end - win >= 0:
            window = x[end - win:end]
            window_t = times_ms[end - win:end]
            phase_rad, amplitude = phastimate(window, b, EDGE, AR_ORDER, HILBERT_WINDOW, OFFSET)

            # the filtered "core" phastimate actually used (filtfilt edges trimmed off)
            filtered = filtfilt(b, 1.0, window - window.mean())[EDGE:-EDGE]
            core_t = window_t[EDGE:-EDGE]
            ax.plot(core_t, filtered, color="tab:blue", lw=1.2,
                     label=f"{BAND[0]:g}-{BAND[1]:g} Hz filtered")

            # simple sine reconstruction from the estimated phase, ending at t_estimate
            sine_t = window_t
            sine = amplitude * np.cos(2 * np.pi * f0 * (sine_t - t_estimate_ms) / 1000 + phase_rad)
            ax.plot(sine_t, sine, color="tab:red", lw=1.2, ls="--",
                     label=f"fitted sine ({np.degrees(phase_rad):.0f}°)")

        ax.axvline(0, color="gray", ls=":", label="TMS pulse (t=0)")
        ax.axvline(t_estimate_ms, color="tab:red", ls="--", alpha=0.6,
                   label=f"phase read out ({t_estimate_ms:.1f} ms)")

        cond = ""
        if epochs.metadata is not None and "Condition" in epochs.metadata.columns:
            cond = f", condition={epochs.metadata.iloc[i]['Condition']}"
        ax.set_title(f"Trial {i + 1}/{len(data_uv)}{cond}")
        ax.set_xlabel("Time relative to TMS pulse (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()

        if SAVE:
            fig.savefig(OUT_DIR / f"trial_{i + 1:03d}.png", dpi=150)
        if SHOW:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
