"""Validate the causal (AR-forecast) phase estimate against the online phase-triggering
device ("boss device") on real TMS sessions (Zrenner 2018/2020).

Unlike pipeline.py's intake epochs (no TMS, full pre/post data -> non-causal filtfilt
ground truth available), exp-session epochs are pulse-triggered BY the device's own
real-time phase decision: epochs.metadata only carries a categorical "Condition"
("positive" / "negative" / "random" -- the peak the device targeted, or no target),
not a continuous phase value. So there is no offline ground-truth phase to compare
against here -- instead this checks whether phastimate()'s own causal AR estimate,
computed independently from the same pre-pulse data, at CUTOFF_MS agrees with what the
device decided online. "random" epochs have no target and are excluded from the
phase-error/accuracy stats (shown separately for reference).

Edit the CAPS parameters below, then run:  python causal_vs_boss.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from phastimate import design_bandpass, phastimate

# ---- WHAT TO RUN --------------------------------------------------------
SHOW = True                    # open interactive plots
SAVE = False                   # save plots to disk
DOWNSAMPLE = 1                  # if True, resample to DOWNSAMPLE_FS once on load (pipeline.py
                                 # convention) -- exp-epo.fif is natively 5000 Hz, but
                                 # FILTER_ORDER/EDGE/AR_ORDER/HILBERT_WINDOW below (from
                                 # run.py) are sample counts tuned for ~1 kHz; running them
                                 # unresampled against 5000 Hz data makes EDGE/HILBERT_WINDOW/
                                 # AR_ORDER cover a small fraction of the time span they were
                                 # designed for (e.g. EDGE=64 samples = 12.8 ms, nowhere near
                                 # enough to absorb a 190-tap filter's ~38 ms transient at
                                 # 5000 Hz) and silently wrecks the phase estimate.
DOWNSAMPLE_FS = 1000.0          # Hz

# ---- DATA -----------------------------------------------------------------
# Same NAS layout as pipeline.py / run.py. "exp" = real TMS session, phase-triggered
# online by the device.
DATA_ROOT = Path("/Volumes/CENSORLAB$/Or/Cortical Communication/CorticalCommunication/Data")

# One subject, or a list to run several at once. Each subject plotted in its own color.
SUBJECT = ["sub_101", "sub_103", "sub_104"]
# known subjects under DATA_ROOT above with a {subject}_exp-epo.fif file:
# sub_101, sub_103, sub_104, sub_109, sub_110, sub_111, sub_112, sub_113
# (sub_102, sub_105, sub_106, sub_107, sub_108 have no exp-epo.fif -- load() will fail)
CHANNEL = "Fz_hjorth"           # built from FZ_HJORTH_WEIGHTS below, same as utils/data_processing.py
FZ_HJORTH_WEIGHTS = {"Fz": 1, "AF3": -0.25, "AF4": -0.25, "FC1": -0.25, "FC2": -0.25}

# ---- OSCILLATION / FILTER (phastimate.m) -----------------------------------
# Matches pipeline.py's theta-band params (Fz_hjorth, same filter/AR settings) so the
# causal estimate here is directly comparable to the intake validation there.
BAND = (4.0, 8.0)              # theta
FILTER_ORDER = 350
AR_ORDER = 78
EDGE = 65
HILBERT_WINDOW = 128
OFFSET = 0
WINDOW_MS = 510.0               # length of the window feeding the AR predictor

# ---- causal-vs-device: AR-forecasted phase at the pulse, vs. device's targeted class ----
CUTOFF_MS = 0.0                  # split point: AR core before, AR-predicted after (t=0 = pulse)
N_EPOCHS = False                 # limit to the first N epochs; False = use all

# Real EEG in the ~50ms right before the pulse is contaminated (coil-related artifact --
# confirmed empirically: raw-signal correlation across epochs in that window is ~0.98-1.0,
# vs ~0 hundreds of ms earlier). phastimate()'s own EDGE trimming already keeps the last
# EDGE samples out of the AR-fit core (they're filter-edge artifact, not real data used for
# fitting), but that's separate from this recording artifact -- ARTIFACT_MARGIN_MS pushes
# the core's real data back by this much *more*, on top of EDGE, so nothing from the
# contaminated window ever reaches the AR fit; the extra gap is bridged by forecasting
# further forward (via `offset`/`iterations`) rather than by trimming more filter edge.
ARTIFACT_MARGIN_MS = 50.0

# device's declared target, mapped to a phase (deg) for angular comparison
CONDITION_TARGET_DEG = {"positive": 0.0, "negative": 180.0}


def to_0_360(phase_rad):
    """0-360 deg, 0 = positive peak (top), 180 = negative peak (bottom)."""
    return np.degrees(phase_rad) % 360


def _phase_diff_deg(a_deg, b_deg):
    """Signed angular difference a-b (deg), wrapped to (-180, 180]."""
    return (a_deg - b_deg + 180) % 360 - 180


def add_fz_hjorth(epochs):
    """Fz Hjorth spatial filter (Laplacian), same as pipeline.py's add_fz_hjorth()."""
    weights = np.array([FZ_HJORTH_WEIGHTS.get(ch, 0) for ch in epochs.ch_names])
    data = epochs.get_data()
    hjorth = np.einsum("ect,c->et", data, weights)

    ref_idx = epochs.ch_names.index("Fz")
    scale = np.std(data[:, ref_idx, :]) / np.std(hjorth)
    hjorth = (hjorth * scale)[:, np.newaxis, :]

    info = mne.create_info(["Fz_hjorth"], epochs.info["sfreq"], ["eeg"])
    new_epo = mne.EpochsArray(hjorth, info, tmin=epochs.tmin, metadata=epochs.metadata)
    epochs.add_channels([new_epo])
    return epochs


def load(subject):
    epo_path = DATA_ROOT / subject / "EEG" / "processed" / f"{subject}_exp-epo.fif"
    epochs = mne.read_epochs(epo_path, preload=True)
    # exp epochs carry a genuine TMS discharge right after t=0 (post-pulse amplitude is
    # ~4000x pre-pulse here) -- resample()'s anti-aliasing filter runs over the whole
    # epoch and its ringing smears that huge spike backward through the pre-pulse window
    # (empirically: cross-epoch correlation in the "clean" pre-pulse data jumps from ~0 to
    # ~0.98-1.0 after a naive resample). We only ever use pre-pulse data (CUTOFF_MS=0 is
    # the latest point used), so cropping the post-pulse artifact out *before* resampling
    # keeps the filter from ever seeing it. Matches the source preprocessing scripts'
    # explicit "epochs.resample(sfreq=1000) DONT DO THAT WITH THE TMS ARTIFACT" warning --
    # cropping first is what makes resampling safe here.
    epochs.crop(tmax=0.0, include_tmax=False)
    if DOWNSAMPLE:
        epochs.resample(DOWNSAMPLE_FS)
    epochs = add_fz_hjorth(epochs)
    fs = epochs.info["sfreq"]
    data = epochs.get_data(picks=CHANNEL)[:, 0, :]
    times_ms = epochs.times * 1000
    conditions = epochs.metadata["Condition"].to_numpy()
    return epochs, fs, data, times_ms, conditions


def _causal_estimate(x, times_ms, fs, b, cutoff, win):
    """Causal AR-forecasted phase at CUTOFF_MS. None if not enough pre-cutoff data.

    ARTIFACT_MARGIN_MS shifts the window fed to phastimate() back by that many samples
    (`cutoff_used`), so the AR-fit core never touches raw data from the contaminated
    window right before the pulse; `offset` is bumped by the same amount so the estimated
    point still lands at the true CUTOFF_MS, and `iterations` is extended to match so the
    Hilbert-window centering (phastimate()'s default) is preserved.
    """
    margin = round(ARTIFACT_MARGIN_MS / 1000 * fs)
    cutoff_used = cutoff - margin
    if cutoff_used - win < 0:
        return None
    offset = OFFSET + margin
    iterations = EDGE + margin + int(np.ceil(HILBERT_WINDOW / 2))
    phase, amplitude, core, pred_future = phastimate(
        x[cutoff_used - win:cutoff_used], b, EDGE, AR_ORDER, HILBERT_WINDOW, offset,
        iterations=iterations, return_trace=True,
    )
    core_times_ms = times_ms[cutoff_used - win:cutoff_used][EDGE:-EDGE]
    future_t = core_times_ms[-1] + (np.arange(1, len(pred_future) + 1)) / fs * 1000
    return dict(
        phase=phase, amplitude=amplitude,
        core=core, pred_future=pred_future,
        core_times_ms=core_times_ms, future_t=future_t,
    )


def _predicted_class(phase_deg):
    """Nearest of the device's two target classes (positive=0deg, negative=180deg)."""
    return "positive" if abs(_phase_diff_deg(phase_deg, 0.0)) < 90 else "negative"


def _plot_epoch(i, n, x, times_ms, est, condition):
    """Per-epoch signal overlay: raw / causal (AR-forecast), device's declared condition."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times_ms, x, color="black", lw=1, label=f"raw ({CHANNEL})")
    ax.plot(est["core_times_ms"], est["core"], color="tab:red", lw=1.2, label="AR core (causal, filtered)")
    ax.plot(est["future_t"], est["pred_future"], color="tab:red", lw=1.2, ls="--", label="AR-predicted (causal)")
    ax.axvspan(est["core_times_ms"][-1], CUTOFF_MS, color="gray", alpha=0.2,
               label="edge-trimmed + artifact margin (AR forecast bridges to t=0)")
    ax.axvline(CUTOFF_MS, color="gray", ls=":")
    causal_phase_deg = to_0_360(est["phase"])
    predicted = _predicted_class(causal_phase_deg)
    ax.plot([], [], " ", label=f"causal phase @ t=0: {causal_phase_deg:.1f}°")
    ax.plot([], [], " ", label=f"device condition: {condition}")
    ax.plot([], [], " ", label=f"causal-implied class: {predicted} ({'match' if predicted == condition else 'mismatch'})")
    ax.set_title(f"Epoch {i + 1}/{n}  |  device={condition}, causal phase={causal_phase_deg:.1f}°")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if SAVE:
        Path("epoch_plots").mkdir(exist_ok=True)
        fig.savefig(f"epoch_plots/boss_epoch_{i + 1:03d}.png", dpi=150)
    return fig


def _causal_vs_boss_one(subject, fs, data, times_ms, conditions):
    """Compute the causal AR phase at CUTOFF_MS per epoch and compare to the device's
    declared condition. Returns (phase_deg, conditions, epoch_points) aligned arrays, where
    `epoch_points` holds per-point data needed for click-through (epoch_index, x, est, condition, n)."""
    b = design_bandpass(FILTER_ORDER, BAND[0], BAND[1], fs)
    n = N_EPOCHS or len(data)
    cutoff = int(np.argmin(np.abs(times_ms - CUTOFF_MS)))
    win = round(WINDOW_MS / 1000 * fs)

    epoch_points = []
    phase_deg, conds = [], []
    for i, (x, condition) in enumerate(zip(data[:n], conditions[:n])):
        est = _causal_estimate(x, times_ms, fs, b, cutoff, win)
        if est is None or np.isnan(est["phase"]):
            continue
        phase_deg.append(to_0_360(est["phase"]))
        conds.append(condition)
        epoch_points.append((i, x, est, condition, n))

    phase_deg = np.array(phase_deg)
    conds = np.array(conds)

    targeted = np.isin(conds, list(CONDITION_TARGET_DEG))
    if targeted.any():
        target_deg = np.array([CONDITION_TARGET_DEG[c] for c in conds[targeted]])
        errors_deg = np.array([_phase_diff_deg(p, t) for p, t in zip(phase_deg[targeted], target_deg)])
        predicted = np.array([_predicted_class(p) for p in phase_deg[targeted]])
        matches = predicted == conds[targeted]
        print(
            f"[{subject}] causal vs device @ t=0 (n={targeted.sum()}/{len(conds)} targeted epochs, "
            f"{(~targeted).sum()} random excluded): "
            f"|phase error| {np.mean(np.abs(errors_deg)):.1f}° ± {np.std(np.abs(errors_deg)):.1f}°, "
            f"class agreement {100 * matches.mean():.1f}%"
        )
        for c in CONDITION_TARGET_DEG:
            sel = conds[targeted] == c
            if sel.any():
                print(
                    f"    {c:>8s} (n={sel.sum()}): |phase error| {np.mean(np.abs(errors_deg[sel])):.1f}° "
                    f"± {np.std(np.abs(errors_deg[sel])):.1f}°, agreement {100 * matches[sel].mean():.1f}%"
                )
    else:
        print(f"[{subject}] no positive/negative-targeted epochs with a usable causal estimate")

    return phase_deg, conds, epoch_points


_CONDITION_COLOR = {"positive": "tab:red", "negative": "tab:blue", "random": "tab:gray"}


def causal_vs_boss(subject_data):
    """subject_data: list of (subject, fs, data, times_ms, conditions). Plots causal
    phase (deg) per epoch, colored by the device's declared condition, one panel per
    subject side by side in a single figure."""
    fig, axes = plt.subplots(1, len(subject_data), figsize=(5 * len(subject_data), 5),
                              squeeze=False, subplot_kw=dict(projection="polar"))
    axes = axes[0]

    scatters_by_ax = []
    epoch_points_by_ax = []
    times_ms_by_ax = []

    for ax, (subject, fs, data, times_ms, conditions) in zip(axes, subject_data):
        phase_deg, conds, epoch_points = _causal_vs_boss_one(subject, fs, data, times_ms, conditions)
        scatters = []
        for condition, color in _CONDITION_COLOR.items():
            sel = conds == condition
            if not sel.any():
                continue
            r = np.arange(sel.sum())  # spread points radially so overlapping angles are visible
            scatter = ax.scatter(np.radians(phase_deg[sel]), r, s=20, alpha=0.7, color=color,
                                  label=condition, picker=True, pickradius=6)
            scatters.append((scatter, [t for t, keep in zip(epoch_points, sel) if keep]))
        for target_deg in CONDITION_TARGET_DEG.values():
            ax.axvline(np.radians(target_deg), color="black", ls=":", lw=1, alpha=0.5)
        ax.set_title(subject)
        ax.set_yticklabels([])
        ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.1))
        scatters_by_ax.append(scatters)
        epoch_points_by_ax.append(epoch_points)
        times_ms_by_ax.append(times_ms)

    fig.suptitle("Causal (AR-forecast) phase @ t=0 vs. device-declared condition -- "
                  "dotted lines = device targets (0°/180°); click a point for its signal plot")
    fig.tight_layout()

    def on_pick(event):
        for ax_idx, scatters in enumerate(scatters_by_ax):
            for scatter, sub_epoch_points in scatters:
                if event.artist is scatter and len(event.ind):
                    i, x, est, condition, n = sub_epoch_points[event.ind[0]]
                    _plot_epoch(i, n, x, times_ms_by_ax[ax_idx], est, condition).show()
                    return

    fig.canvas.mpl_connect("pick_event", on_pick)

    if SAVE:
        fig.savefig("causal_vs_boss.png", dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def main():
    subjects = [SUBJECT] if isinstance(SUBJECT, str) else list(SUBJECT)
    subject_data = [(subject, *load(subject)[1:]) for subject in subjects]
    causal_vs_boss(subject_data)


if __name__ == "__main__":
    main()
