"""Validate phase estimation on intake (non-TMS) epochs (Zrenner 2018/2020).

CALCULATION options:
  "deviation_snr_signal" -- causal (AR-forecast) vs non-causal (filtfilt) phase estimate
        at the cutoff, over N_TRIALS epochs -> scatter of |phase error| (deg) vs band SNR
        (dB), one point per epoch. Click a point to open that trial's signal plot (raw /
        causal AR-forecast / non-causal ground-truth overlay -- formerly "step 1") plus its
        SNR-window spectrum (log-log, 1/f fit line + R², peak frequency, SNR).
  "phase_metadata" -- one phase estimate per epoch at t=0 -> new column on epochs.metadata.
  "optimize_params" -- Optuna search (subject-level, single param set for all its epochs)
        over WINDOW_MS/FILTER_ORDER/EDGE/AR_ORDER, minimizing circular variance of the
        causal-vs-non-causal phase error (same objective as phastimate_optimize.m's GA),
        with a held-out split to check the winning params generalize. See OPT_* below.

Edit the CAPS parameters below, then run:  python pipeline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import optuna
from scipy.signal import filtfilt, hilbert

from phastimate import design_bandpass, estimate_snr, phastimate

# ---- WHAT TO RUN --------------------------------------------------------
CALCULATION = "deviation_snr_signal"  # "deviation_snr_signal" | "phase_metadata" | "optimize_params" -- see docstring
SHOW = True                    # open interactive plots
SAVE = False                   # save plots / metadata to disk
DOWNSAMPLE = 1             # if True, resample to DOWNSAMPLE_FS once on load;
                                # every step below then runs on that resampled data
DOWNSAMPLE_FS = 1000.0         # Hz

# ---- DATA -----------------------------------------------------------------
# Same NAS layout as run.py. "intake" = memory task, no TMS.
# Bar/phase_estimation_analysis on the share only has sub_103 -- the full cohort below
# lives under Or's Cortical Communication project instead.
# DATA_ROOT = Path("/Volumes/CENSORLAB$/Bar/phase_estimation_analysis")
DATA_ROOT = Path("/Volumes/CENSORLAB$/Or/Cortical Communication/CorticalCommunication/Data")

# One subject, or a list to run several at once. With multiple subjects, deviation_snr_signal
# plots each subject in its own color with a legend; phase_metadata / optimize_params just
# run once per subject in turn.
SUBJECT = ["sub_102", "sub_103", "sub_104"]
# known subjects under DATA_ROOT above, all with sub_*_intake_stim-epo.fif EXCEPT
# sub_101 and sub_106 (no intake_stim file for those two -- pipeline.py's load() will
# fail on them):
# sub_101(!), sub_102, sub_103, sub_104, sub_105, sub_106(!), sub_107, sub_108,
# sub_109, sub_110, sub_111, sub_112, sub_113
CHANNEL = "Fz_hjorth"           # built from FZ_HJORTH_WEIGHTS below, same as utils/data_processing.py
FZ_HJORTH_WEIGHTS = {"Fz": 1, "AF3": -0.25, "AF4": -0.25, "FC1": -0.25, "FC2": -0.25}

# ---- OSCILLATION / FILTER (phastimate.m) -----------------------------------
BAND = (4.0, 8.0)              # theta -- confirm with lab convention
FILTER_ORDER = 350             # Zrenner2018 (phastimate/main_script.m:411)
AR_ORDER = 78
EDGE =  65 #FILTER_ORDER//2
HILBERT_WINDOW = 128
OFFSET = 0
WINDOW_MS = 510.0              # length of the window feeding the AR predictor

# ---- deviation_snr_signal: causal-vs-non-causal phase estimate at the cutoff ----
CUTOFF_MS = 0.0                 # split point: AR core before, AR-predicted after
N_TRIALS = False                # limit to the first N trials; False = use all

# what to plot |phase error| against (x-axis of the scatter):
#   "snr"         band SNR (dB), windowed 1/f-corrected spectral estimate (original metric)
#   "amp_true"    non-causal (filtfilt) instantaneous amplitude at the cutoff -- offline-only
#                 ground truth, but the cleanest predictor of phase reliability
#   "amp_causal"  causal (AR-forecast) instantaneous amplitude at the cutoff -- the same
#                 online-usable quantity phastimate() already returns alongside the phase
X_AXIS = "amp_true"

# Window length from the original Zrenner2018 parameter set (phastimate/main_script.m,
# "500ms window" real-time causal fit). The forecast horizon (`iterations`) is left at
# phastimate()'s own default (edge + ceil(hilbert_window/2)) rather than fixed here --
# that default centers the point being estimated in the Hilbert window (~equal buffer of
# forecasted samples on both sides), which matters because the FFT-based Hilbert
# transform has boundary artifacts right at the edges of whatever window it's given.
# A shorter, fixed forecast (just enough to reach t=0) leaves the estimate right next to
# that trailing boundary instead.
PRE_CUTOFF_WINDOW_MS = 500.0    # AR core: real data used before the cutoff

# ---- deviation_snr_signal: SNR (estimate_SNR.m; 1/f-corrected), per epoch -----
# Epochs only carry 1000 ms of pre-cutoff data (tmin=-1.0s), same constraint as
# run.py -- window capped well below the 4s used in the original continuous-recording
# design.
SNR_WINDOW_MS = 900.0           # window ending at CUTOFF_MS used for the spectrum

# Background-only bins used to fit the 1/f trend that gets subtracted before peak-finding.
# The original estimate_SNR.m default (0.5-7, 35-65 Hz) overlaps our BAND (theta, 4-8 Hz)
# on 4-7 Hz, biasing the fit toward the peak it's supposed to measure. Kept clear of BAND.
SNR_FIT_RANGES = [(0.5, 4.0), (35.0, 65.0)]

# Multitaper spectral smoothing bandwidth for estimate_snr()'s spectrum. The original
# estimate_SNR.m used Welch (nperseg=2s, no overlap), which needs several seconds of data
# to average over; our 900ms SNR_WINDOW_MS can't fit even one such segment, so it silently
# fell back to a single raw (high-variance, "bumpy") periodogram -- and chopping it into
# shorter Welch segments to get averaging just traded that noise for coarser frequency
# resolution instead. Multitaper averages over orthogonal tapers on the full window
# instead of over shorter time segments, so it gets the same noise averaging without
# sacrificing resolution.
# Full bandwidth (not half-bandwidth); NW = bandwidth * SNR_WINDOW_MS/1000 / 2. At 900ms,
# 4.5 Hz gives NW ~= 2.0 -> 3 tapers, the minimum for MNE's adaptive combining to actually
# engage (fewer silently degrades back toward a ~1-taper periodogram). If SNR_WINDOW_MS
# changes, re-check this still clears NW = 2 (bandwidth >= 4 / (SNR_WINDOW_MS/1000)).
SNR_MULTITAPER_BANDWIDTH_HZ = 4.5

# ---- phase_metadata -----------------------------------------------------------
METADATA_COLUMN = "phase_deg"

# ---- optimize_params: Optuna search over phastimate() geometry (this subject only) -----
# Fits ONE param set for all of this subject's epochs (not per-epoch -- per-epoch would
# need the answer before estimating it, and wouldn't generalize). BAND/HILBERT_WINDOW/
# OFFSET above stay fixed and shared; only the 4 params below are searched, with their own
# ranges here -- kept separate from deviation_snr_signal's fixed FILTER_ORDER/EDGE/AR_ORDER/
# PRE_CUTOFF_WINDOW_MS above so tweaking one calculation's params can't silently affect the
# other's.
#
# Ranges are theta-adjusted, not copied from phastimate_optimize.m's alpha-tuned defaults
# (filter_order 100-250, window 400-750ms, edge 30-120, ar_order 5-60): theta (4-8 Hz) is a
# narrower, lower band than the original 8-13 Hz alpha, so needs a longer/sharper filter,
# and OPT_EDGE_RANGE is searched independently rather than tied to FILTER_ORDER//2 like the
# fixed EDGE above. OPT_WINDOW_MS_RANGE tops out under 1000ms since epochs only carry that
# much pre-cutoff data.
OPT_N_TRIALS = 200              # number of Optuna trials (param combos tried)
OPT_TRAIN_FRACTION = 0.8        # fraction of epochs used to fit; rest held out to check generalization
OPT_RANDOM_SEED = 0
OPT_MIN_TRIALS = 20             # penalize a param combo if fewer than this many epochs are usable

OPT_WINDOW_MS_RANGE = (300, 950)
OPT_WINDOW_MS_STEP = 10
OPT_FILTER_ORDER_RANGE = (150, 450)
OPT_FILTER_ORDER_STEP = 5
OPT_EDGE_RANGE = (20, 200)
OPT_EDGE_STEP = 5
OPT_AR_ORDER_RANGE = (5, 80)

OPT_INFEASIBLE_PENALTY = 10.0   # circular variance is in [0, 1]; well above any real value


def to_0_360(phase_rad):
    """0-360 deg, 0 = positive peak (top), 180 = negative peak (bottom)."""
    return np.degrees(phase_rad) % 360


def _phase_diff_deg(a_rad, b_rad):
    """Signed angular difference a-b, wrapped to (-180, 180]."""
    return (np.degrees(a_rad - b_rad) + 180) % 360 - 180


def add_fz_hjorth(epochs):
    """Fz Hjorth spatial filter (Laplacian), same as utils/data_processing.py."""
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
    epo_path = DATA_ROOT / subject / "EEG" / "processed" / f"{subject}_intake_stim-epo.fif"
    epochs = mne.read_epochs(epo_path, preload=True)
    if DOWNSAMPLE:
        epochs.resample(DOWNSAMPLE_FS)
    epochs = add_fz_hjorth(epochs)
    fs = epochs.info["sfreq"]
    data = epochs.get_data(picks=CHANNEL)[:, 0, :]
    times_ms = epochs.times * 1000
    return epochs, fs, data, times_ms


def _cutoff_estimate(x, times_ms, fs, b, cutoff, win):
    """Causal AR-forecasted phase vs. non-causal (filtfilt) ground truth, both at CUTOFF_MS.

    Uses phastimate()'s default forecast horizon (edge + ceil(hilbert_window/2)), which
    centers the estimated point in the Hilbert window instead of just reaching t=0.

    Returns None if there isn't enough data around the cutoff, else a dict with the
    causal/non-causal phase and amplitude, their error/ratio, and the raw traces
    needed to plot them (core/pred_future/core_times_ms/future_t).
    """
    if cutoff - win < 0:
        return None

    phase, amplitude, core, pred_future = phastimate(
        x[cutoff - win:cutoff], b, EDGE, AR_ORDER, HILBERT_WINDOW, OFFSET,
        return_trace=True,
    )
    core_times_ms = times_ms[cutoff - win:cutoff][EDGE:-EDGE]
    future_t = core_times_ms[-1] + (np.arange(1, len(pred_future) + 1)) / fs * 1000

    # non-causal (offline, zero-phase) ground truth over the whole trial, evaluated
    # at CUTOFF_MS so it's directly comparable to the causal phastimate() estimate.
    real_filtered = filtfilt(b, 1.0, x - x.mean())
    real_analytic = hilbert(real_filtered)
    real_phase, real_amplitude = np.angle(real_analytic[cutoff]), np.abs(real_analytic[cutoff])

    return dict(
        phase=phase, amplitude=amplitude,
        real_phase=real_phase, real_amplitude=real_amplitude,
        phase_error_deg=_phase_diff_deg(phase, real_phase),
        amp_ratio=amplitude / real_amplitude,
        real_filtered=real_filtered, core=core, pred_future=pred_future,
        core_times_ms=core_times_ms, future_t=future_t,
    )


def _plot_trial(i, n, x, times_ms, est):
    """Per-trial signal overlay: raw / causal (AR-forecast) / non-causal (filtfilt) ground truth."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times_ms, x, color="black", lw=1, label=f"raw ({CHANNEL})")
    ax.plot(times_ms, est["real_filtered"], color="tab:blue", lw=1, alpha=0.7,
            label="non-causal (filtfilt, ground truth)")
    ax.plot(est["core_times_ms"], est["core"], color="tab:red", lw=1.2, label="AR core (causal, filtered)")
    ax.plot(est["future_t"], est["pred_future"], color="tab:red", lw=1.2, ls="--", label="AR-predicted (causal)")
    # filtfilt boundary transient: the last EDGE samples of the raw pre-cutoff window are
    # dropped from `core` (see phastimate()) since they're contaminated by the edge of the
    # window itself; the AR forecast bridges exactly this gap to reach the CUTOFF_MS estimate.
    ax.axvspan(est["core_times_ms"][-1], CUTOFF_MS, color="gray", alpha=0.2,
               label="edge-trimmed (AR forecast bridges to t=0)")
    ax.axvline(CUTOFF_MS, color="gray", ls=":")
    causal_phase_deg = to_0_360(est["phase"])
    noncausal_phase_deg = to_0_360(est["real_phase"])
    ax.plot([], [], " ", label=f"causal phase @ t=0: {causal_phase_deg:.1f}°")
    ax.plot([], [], " ", label=f"non-causal phase @ t=0: {noncausal_phase_deg:.1f}°")
    ax.plot([], [], " ", label=f"Δphase (causal − non-causal): {est['phase_error_deg']:+.1f}°")
    ax.set_title(
        f"Trial {i + 1}/{n}  |  causal vs non-causal @ cutoff: "
        f"Δphase={est['phase_error_deg']:+.1f}°, amp ratio={est['amp_ratio']:.2f}"
    )
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if SAVE:
        Path("trial_plots").mkdir(exist_ok=True)
        fig.savefig(f"trial_plots/trial_{i + 1:03d}.png", dpi=150)
    return fig


def _plot_spectrum(i, n, peak_freq, snr_db, fit_info):
    """Log-log (semilogx, dB) spectrum for one trial's SNR window: the 1/f fit line,
    which points fed it, and the detected peak -- for sanity-checking estimate_snr()."""
    f, log_p = fit_info["f"], fit_info["log_p"]
    slope, intercept, r2, fit_mask = fit_info["slope"], fit_info["intercept"], fit_info["r2"], fit_info["fit_mask"]
    fit_line = slope * np.log10(f) + intercept

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(f, log_p, color="black", lw=1, label="power spectrum")
    ax.semilogx(f[fit_mask], log_p[fit_mask], "o", color="tab:gray", ms=4, label="1/f fit points")
    ax.semilogx(f, fit_line, color="tab:orange", lw=1.5, label=f"1/f fit (R²={r2:.3f})")
    ax.axvspan(BAND[0], BAND[1], color="tab:blue", alpha=0.08, label=f"target band ({BAND[0]}-{BAND[1]} Hz)")
    if not np.isnan(peak_freq):
        ax.axvline(peak_freq, color="tab:red", ls="--", lw=1.2,
                   label=f"band max @ {peak_freq:.2f} Hz (SNR={snr_db:.1f} dB)")
    else:
        ax.plot([], [], " ", label="band outside evaluated 2-45 Hz range")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    ax.set_title(f"Trial {i + 1}/{n} spectrum (SNR window)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if SAVE:
        Path("trial_plots").mkdir(exist_ok=True)
        fig.savefig(f"trial_plots/trial_{i + 1:03d}_spectrum.png", dpi=150)
    return fig


_X_AXIS_LABELS = {
    "snr": "band SNR (dB)",
    "amp_true": "non-causal (ground-truth) amplitude at cutoff",
    "amp_causal": "causal (AR-forecast) amplitude at cutoff",
}


def _deviation_snr_signal_one(subject, fs, data, times_ms):
    """Compute causal-vs-non-causal phase error + band SNR + both amplitudes per trial.

    Returns (x_values, abs_errors_deg, trials) where x_values is whichever quantity
    X_AXIS selects, and `trials` holds the per-point data needed for click-through
    (trial_index, x, est, peak_freq, snr_db, fit_info, n)."""
    b = design_bandpass(FILTER_ORDER, BAND[0], BAND[1], fs)
    n = N_TRIALS or len(data)
    cutoff = int(np.argmin(np.abs(times_ms - CUTOFF_MS)))
    win = round(PRE_CUTOFF_WINDOW_MS / 1000 * fs)
    snr_win = round(SNR_WINDOW_MS / 1000 * fs)

    trials = []
    phase_errors_deg, amp_ratios, abs_errors_deg = [], [], []
    snr_db, amp_true, amp_causal = [], [], []
    for i, x in enumerate(data[:n]):
        est = _cutoff_estimate(x, times_ms, fs, b, cutoff, win)
        if est is None or cutoff - snr_win < 0:
            continue
        peak_freq, peak_snr_db, fit_info = estimate_snr(
            x[cutoff - snr_win:cutoff], fs, BAND, SNR_FIT_RANGES,
            SNR_MULTITAPER_BANDWIDTH_HZ, return_fit=True,
        )
        if np.isnan(peak_snr_db):
            continue
        phase_errors_deg.append(est["phase_error_deg"])
        amp_ratios.append(est["amp_ratio"])
        abs_errors_deg.append(abs(est["phase_error_deg"]))
        snr_db.append(peak_snr_db)
        amp_true.append(est["real_amplitude"])
        amp_causal.append(est["amplitude"])
        trials.append((i, x, est, peak_freq, peak_snr_db, fit_info, n))

    if phase_errors_deg:
        print(
            f"[{subject}] causal vs non-causal @ cutoff (n={len(phase_errors_deg)}/{n} trials): "
            f"phase error {np.mean(phase_errors_deg):+.1f}° ± {np.std(phase_errors_deg):.1f}°, "
            f"amp ratio {np.mean(amp_ratios):.2f} ± {np.std(amp_ratios):.2f}"
        )

    x_values = {"snr": snr_db, "amp_true": amp_true, "amp_causal": amp_causal}[X_AXIS]
    return x_values, abs_errors_deg, trials


def deviation_snr_signal(subject_data):
    """subject_data: list of (subject, fs, data, times_ms). One subject -> plain scatter;
    multiple subjects -> one point cloud per subject, each in its own color, with a legend."""
    fig, ax = plt.subplots(figsize=(6, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    scatters = []       # per-subject scatter artists, for click-through lookup
    trials_by_subject = []
    times_ms_by_subject = []
    total_points = 0
    for idx, (subject, fs, data, times_ms) in enumerate(subject_data):
        x_values, abs_errors_deg, trials = _deviation_snr_signal_one(subject, fs, data, times_ms)
        color = color_cycle[idx % len(color_cycle)]
        scatter = ax.scatter(x_values, abs_errors_deg, s=25, alpha=0.7, color=color,
                              label=subject, picker=True, pickradius=6)
        scatters.append(scatter)
        trials_by_subject.append(trials)
        times_ms_by_subject.append(times_ms)
        total_points += len(abs_errors_deg)

    ax.set_xlabel(_X_AXIS_LABELS[X_AXIS])
    ax.set_ylabel("|Δphase| (causal − non-causal, deg)")
    title = f"Phase error vs. {_X_AXIS_LABELS[X_AXIS]} (n={total_points} trials) -- click a point for its signal + spectrum plots"
    if len(subject_data) > 1:
        ax.legend(loc="best", fontsize=8, title="subject")
    ax.set_title(title)
    fig.tight_layout()

    def on_pick(event):
        if event.artist not in scatters or not len(event.ind):
            return
        subj_idx = scatters.index(event.artist)
        trials = trials_by_subject[subj_idx]
        times_ms = times_ms_by_subject[subj_idx]
        i, x, est, peak_freq, peak_snr_db, fit_info, n = trials[event.ind[0]]
        _plot_trial(i, n, x, times_ms, est).show()
        _plot_spectrum(i, n, peak_freq, peak_snr_db, fit_info).show()

    fig.canvas.mpl_connect("pick_event", on_pick)

    if SAVE:
        fig.savefig("deviation_snr_signal.png", dpi=150)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def _predicted_phase(x, t_idx, win, b):
    if t_idx - win < 0:
        return None
    phase, _ = phastimate(x[t_idx - win:t_idx], b, EDGE, AR_ORDER, HILBERT_WINDOW, OFFSET)
    return phase


def phase_metadata(subject, epochs, fs, data, times_ms):
    b = design_bandpass(FILTER_ORDER, BAND[0], BAND[1], fs)
    win = round(WINDOW_MS / 1000 * fs)
    t_idx = int(np.argmin(np.abs(times_ms - 0.0)))

    phases_deg = []
    for x in data:
        pred_phase = _predicted_phase(x, t_idx, win, b)
        phases_deg.append(to_0_360(pred_phase) if pred_phase is not None else np.nan)

    epochs.metadata[METADATA_COLUMN] = phases_deg
    print(f"[{subject}]")
    print(epochs.metadata[[METADATA_COLUMN]].describe())

    if SAVE:
        out_epo_path = DATA_ROOT / subject / "EEG" / "processed" / f"{subject}_intake_stim_phase-epo.fif"
        epochs.save(out_epo_path, overwrite=True)
        print(f"saved -> {out_epo_path}")


def _circular_variance(errors_rad):
    """1 - |resultant vector length|; 0 = all errors identical, 1 = uniformly scattered.
    Same objective as phastimate_optimize.m's ang_var_of_diff."""
    return 1 - np.abs(np.mean(np.exp(1j * np.asarray(errors_rad))))


def _phase_error_rad(x, b, edge, ar_order, window_samples, cutoff):
    """Causal phastimate() phase minus non-causal (filtfilt) ground-truth phase, both at
    `cutoff`. Unwrapped -- fed straight into exp(1j*.) by the caller, which handles the
    wraparound, so no explicit (-180, 180] clamp is needed here."""
    phase, _ = phastimate(x[cutoff - window_samples:cutoff], b, edge, ar_order, HILBERT_WINDOW, OFFSET)
    if np.isnan(phase):
        return None
    real_filtered = filtfilt(b, 1.0, x - x.mean())
    real_phase = np.angle(hilbert(real_filtered)[cutoff])
    return phase - real_phase


def _evaluate_params(data, indices, cutoff, fs, window_ms, filter_order, edge, ar_order):
    """Circular variance of the causal-vs-non-causal phase error over `indices`' epochs,
    for one candidate (window_ms, filter_order, edge, ar_order). None if infeasible (not
    enough samples for the filter/AR fit given the window) or too few epochs came out
    usable (phastimate() can return nan, e.g. if AR fit is unstable for this combo)."""
    window_samples = round(window_ms / 1000 * fs)
    core_len = window_samples - 2 * edge
    if window_samples <= filter_order or window_samples > cutoff or core_len <= ar_order + 10:
        return None

    b = design_bandpass(filter_order, BAND[0], BAND[1], fs)
    errors = []
    for i in indices:
        try:
            e = _phase_error_rad(data[i], b, edge, ar_order, window_samples, cutoff)
        except Exception:
            continue
        if e is not None:
            errors.append(e)
    if len(errors) < OPT_MIN_TRIALS:
        return None
    return _circular_variance(errors), len(errors)


def _make_objective(data, train_idx, cutoff, fs):
    def objective(trial):
        window_ms = trial.suggest_int("window_ms", *OPT_WINDOW_MS_RANGE, step=OPT_WINDOW_MS_STEP)
        filter_order = trial.suggest_int("filter_order", *OPT_FILTER_ORDER_RANGE, step=OPT_FILTER_ORDER_STEP)
        edge = trial.suggest_int("edge", *OPT_EDGE_RANGE, step=OPT_EDGE_STEP)
        ar_order = trial.suggest_int("ar_order", *OPT_AR_ORDER_RANGE)

        result = _evaluate_params(data, train_idx, cutoff, fs, window_ms, filter_order, edge, ar_order)
        if result is None:
            return OPT_INFEASIBLE_PENALTY
        circ_var, _ = result
        return circ_var

    return objective


def optimize_params(subject, fs, data, times_ms):
    """Optuna (TPE) search for one subject-level phastimate() param set: minimizes circular
    variance of the causal-vs-non-causal phase error (phastimate_optimize.m's objective),
    fit on a train split and checked on a held-out split so the winner isn't just fit to
    this subject's noise."""
    cutoff = int(np.argmin(np.abs(times_ms - CUTOFF_MS)))
    n = len(data)
    rng = np.random.default_rng(OPT_RANDOM_SEED)
    shuffled = rng.permutation(n)
    n_train = round(n * OPT_TRAIN_FRACTION)
    train_idx, test_idx = shuffled[:n_train], shuffled[n_train:]

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPT_RANDOM_SEED)
    )
    study.optimize(_make_objective(data, train_idx, cutoff, fs), n_trials=OPT_N_TRIALS)

    best = study.best_params
    train_result = _evaluate_params(data, train_idx, cutoff, fs, **best)
    test_result = _evaluate_params(data, test_idx, cutoff, fs, **best)

    print(f"\n[{subject}] best params ({n_train}/{n} epochs for training): {best}")
    if train_result:
        cv, n_used = train_result
        print(f"  train:     circular variance {cv:.4f}  (R={1 - cv:.3f}, n={n_used}/{len(train_idx)})")
    if test_result:
        cv, n_used = test_result
        print(f"  held-out:  circular variance {cv:.4f}  (R={1 - cv:.3f}, n={n_used}/{len(test_idx)})")
    else:
        print("  held-out:  infeasible / too few usable epochs with the winning params")

    try:
        importances = optuna.importance.get_param_importances(study)
        print("param importances:", {k: round(v, 3) for k, v in importances.items()})
    except Exception as exc:
        print(f"param importances unavailable: {exc}")

    if SHOW or SAVE:
        import optuna.visualization.matplotlib as opt_vis
        fig1 = opt_vis.plot_optimization_history(study).figure
        fig2 = opt_vis.plot_param_importances(study).figure
        fig1.tight_layout()
        fig2.tight_layout()
        if SAVE:
            fig1.savefig(f"optimize_params_history_{subject}.png", dpi=150)
            fig2.savefig(f"optimize_params_importance_{subject}.png", dpi=150)
        if SHOW:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)


def main():
    subjects = [SUBJECT] if isinstance(SUBJECT, str) else list(SUBJECT)

    if CALCULATION == "deviation_snr_signal":
        # one combined plot, colored by subject -- needs everyone's data loaded first
        subject_data = [(subject, *load(subject)[1:]) for subject in subjects]
        deviation_snr_signal(subject_data)
    elif CALCULATION == "phase_metadata":
        for subject in subjects:
            epochs, fs, data, times_ms = load(subject)
            phase_metadata(subject, epochs, fs, data, times_ms)
    elif CALCULATION == "optimize_params":
        for subject in subjects:
            epochs, fs, data, times_ms = load(subject)
            optimize_params(subject, fs, data, times_ms)
    else:
        raise ValueError(f"unknown CALCULATION: {CALCULATION!r}")


if __name__ == "__main__":
    main()
