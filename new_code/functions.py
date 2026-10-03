"""Reusable helpers for intake-session phase/BOSS comparison.

This module is the single source of analysis logic. Runners, the HTML report and the
notebook (phase_pipeline.ipynb) only call these functions and render what they return.

Terminology: an epoch is the EEG window cut around one stimulation event (one BOSS
decision, t=0). A memory-task trial contains several such events; epochs are analyzed
independently. An Optuna "trial" (``n_opt_trials``) is something else: one evaluated
causal parameter set.

Phase convention: 0-360 deg, 0 = positive peak, 180 = negative trough.

Contents
--------
1. Constants          BOSS class names, targets, colors
2. Loading            LoadConfig, get_data, make_synthetic_epochs
3. Phase estimation   noncausal_phase, causal_phase, estimate_subject_epochs, estimate_all_subjects
4. Optimization       optimize_causal_params
5. Scoring and stats  classify_phase, score_vs_boss, circular_stats, analyze_phase_results
6. Figures            plot_epoch_estimates, plot_phase_circle, plot_circular_histogram,
                      plot_success_vs_tolerance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import filtfilt, hilbert

from phastimate import design_bandpass, phastimate


# ---- 1. Constants --------------------------------------------------------

POSITIVE, NEGATIVE, UNKNOWN = "positive", "negative", "unknown"
UNCLASSIFIED = "unclassified"   # phase outside both +/-tolerance windows
BOSS_CLASSES = (POSITIVE, NEGATIVE, UNKNOWN)
BOSS_TARGET_DEG = {POSITIVE: 0.0, NEGATIVE: 180.0}

# Raw metadata values accepted for each BOSS class (compared lower-case).
POSITIVE_VALUES = {"positive", "pos", "p", "1", "true", "peak"}
NEGATIVE_VALUES = {"negative", "neg", "n", "0", "false", "trough"}

# Paul Tol "vibrant" (colorblind-safe). Pink/blue mean BOSS class only; method traces
# use neutral charcoal (non-causal) and orange (causal) so they never read as a class.
CLASS_COLORS = {POSITIVE: "#EE3377", NEGATIVE: "#0077BB", UNKNOWN: "#858b93"}
METHOD_COLORS = {"raw": "#9a9e98", "noncausal": "#33363b", "causal": "#EE7733"}

METHODS = ("noncausal", "causal")
# Circular histograms computed for the report: "phase" kinds are 0-360 deg,
# "deviation" kinds are signed -180..180 deg.
HISTOGRAM_AXES = {"noncausal_phase": "phase", "boss_target": "deviation", "causal_error": "deviation"}


# ---- 2. Loading ----------------------------------------------------------

@dataclass(frozen=True)
class LoadConfig:
    """Where intake epochs live and how they are preprocessed before phase estimation.

    Only ``data_root`` is required; the defaults are the intake analysis settings.

    data_root                   folder containing ``<subject>/EEG/processed/<intake_filename>``
    intake_filename             epoch file name pattern, formatted with ``subject``
    channel                     channel to analyze; if it equals ``hjorth_channel`` it is built
                                from ``hjorth_weights`` and rescaled to the std of ``hjorth_scale_reference``
    condition_column            metadata column with the BOSS label; if missing, columns whose
                                name contains one of ``condition_auto_keywords`` are tried
    lowpass_before_downsample_hz  anti-alias low-pass (Hz) before resampling, or None
    downsample, downsample_fs   resample epochs to ``downsample_fs`` Hz
    """

    data_root: Path
    intake_filename: str = "{subject}_intake_stim-epo.fif"
    channel: str = "Fz_hjorth"
    condition_column: str = "Condition"
    condition_auto_keywords: tuple[str, ...] = ("condition", "boss", "classification", "class")
    lowpass_before_downsample_hz: float | None = 100.0
    downsample: bool = True
    downsample_fs: float = 1000.0
    hjorth_channel: str = "Fz_hjorth"
    hjorth_weights: dict[str, float] = field(
        default_factory=lambda: {"Fz": 1, "AF3": -0.25, "AF4": -0.25, "FC1": -0.25, "FC2": -0.25}
    )
    hjorth_scale_reference: str = "Fz"
    show_metadata_summary: bool = False
    metadata_max_values: int = 12


@dataclass
class EpochArrays:
    """One subject's epochs as plain arrays.

    ``X`` (n_epochs, n_times) in volts, ``times_ms`` relative to the stimulus (t=0),
    ``fs`` in Hz, ``labels`` one BOSS class name per epoch.
    """

    subject: str
    X: np.ndarray
    times_ms: np.ndarray
    fs: float
    labels: np.ndarray
    channel: str


def intake_epoch_path(subject: str, data_root: Path, intake_filename: str) -> Path:
    return data_root / subject / "EEG" / "processed" / intake_filename.format(subject=subject)


def available_intake_subjects(data_root: Path, intake_filename: str) -> list[str]:
    if not data_root.exists():
        return []
    return [
        path.name for path in sorted(data_root.glob("sub_*"))
        if intake_epoch_path(path.name, data_root, intake_filename).exists()
    ]


def add_weighted_hjorth_channel(
    epochs: mne.Epochs,
    output_channel: str,
    weights_by_channel: dict[str, float],
    scale_reference_channel: str,
) -> mne.Epochs:
    """Add a weighted Hjorth/Laplacian-style channel, rescaled to the reference channel's std."""
    if output_channel in epochs.ch_names:
        return epochs

    missing = [ch for ch in weights_by_channel if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Cannot build {output_channel}; missing channel(s): {missing}")

    weights = np.array([weights_by_channel.get(ch, 0.0) for ch in epochs.ch_names])
    data = epochs.get_data()
    hjorth = np.einsum("ect,c->et", data, weights)
    reference = data[:, epochs.ch_names.index(scale_reference_channel), :]
    hjorth = (hjorth * np.std(reference) / np.std(hjorth))[:, np.newaxis, :]

    info = mne.create_info([output_channel], epochs.info["sfreq"], ["eeg"])
    epochs.add_channels([mne.EpochsArray(hjorth, info, tmin=epochs.tmin, metadata=epochs.metadata)],
                        force_update_info=True)
    return epochs


def normalize_boss_label(value) -> str:
    """Map a raw metadata value (bool, 0/1, or text such as "pos"/"trough") to a BOSS class name."""
    if value is None:
        return UNKNOWN
    if isinstance(value, (bool, np.bool_)):
        return POSITIVE if value else NEGATIVE
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return UNKNOWN
        if np.isclose(value, 1):
            return POSITIVE
        if np.isclose(value, 0):
            return NEGATIVE

    text = str(value).strip().lower().removesuffix(".0")
    if text in POSITIVE_VALUES:
        return POSITIVE
    if text in NEGATIVE_VALUES:
        return NEGATIVE
    return UNKNOWN


def summarize_metadata(epochs: mne.Epochs, max_values: int) -> None:
    """Print metadata columns and a few values of each."""
    if epochs.metadata is None:
        print("metadata: none")
        return
    print("metadata columns:")
    for column in epochs.metadata.columns:
        values = epochs.metadata[column].dropna().unique()[:max_values]
        print(f"  - {column}: {', '.join(repr(v) for v in values)}")


def _find_condition_column(epochs: mne.Epochs, config: LoadConfig) -> str:
    """The configured BOSS label column, or the first keyword-matching column with BOSS labels."""
    metadata = epochs.metadata
    if metadata is None:
        raise ValueError("The epochs file has no metadata, so BOSS labels cannot be read.")
    if config.condition_column in metadata.columns:
        return config.condition_column

    for column in metadata.columns:
        if not any(key in column.lower() for key in config.condition_auto_keywords):
            continue
        if metadata[column].map(normalize_boss_label).isin([POSITIVE, NEGATIVE]).any():
            print(f"Using metadata column {column!r} for BOSS labels.")
            return column

    raise ValueError(
        f"Could not find BOSS label column {config.condition_column!r}. "
        f"Available columns: {list(metadata.columns)}"
    )


def get_data(subject: str, config: LoadConfig) -> EpochArrays:
    """Load one subject's intake epochs into arrays.

    Steps: read ``.fif`` -> optional low-pass -> optional resample -> optional Hjorth
    channel -> pick ``config.channel`` -> map BOSS metadata to class names.
    """
    epo_path = intake_epoch_path(subject, config.data_root, config.intake_filename)
    if not epo_path.exists():
        raise FileNotFoundError(f"Intake epochs not found for {subject}: {epo_path}")

    epochs = mne.read_epochs(epo_path, preload=True)
    if config.lowpass_before_downsample_hz is not None:
        epochs.filter(l_freq=None, h_freq=config.lowpass_before_downsample_hz, picks="eeg", verbose=False)
    if config.downsample:
        epochs.resample(config.downsample_fs)
    if config.channel == config.hjorth_channel:
        epochs = add_weighted_hjorth_channel(
            epochs, config.hjorth_channel, config.hjorth_weights, config.hjorth_scale_reference
        )

    if config.show_metadata_summary:
        print(f"\n[{subject}] {epo_path}")
        summarize_metadata(epochs, config.metadata_max_values)

    condition_column = _find_condition_column(epochs, config)
    return EpochArrays(
        subject=subject,
        X=epochs.get_data(picks=config.channel)[:, 0, :],
        times_ms=epochs.times * 1000.0,
        fs=float(epochs.info["sfreq"]),
        labels=epochs.metadata[condition_column].map(normalize_boss_label).to_numpy(),
        channel=config.channel,
    )


def make_synthetic_epochs(
    subject: str = "demo_sub_001",
    n_epochs: int = 40,
    fs: float = 1000.0,
    tmin_ms: float = -1000.0,
    tmax_ms: float = 500.0,
    freq_hz: float = 6.0,
    phase_jitter_deg: float = 35.0,
    noise_rel: float = 0.35,
    seed: int = 7,
) -> EpochArrays:
    """Synthetic theta epochs for testing without data.

    Each epoch is a ``freq_hz`` cosine whose phase at t=0 is near the BOSS target
    (0 deg positive, 180 deg negative, every 5th epoch unknown/random) plus a 14 Hz
    component and white noise. Amplitudes are in volts (~8 uV).
    """
    rng = np.random.default_rng(seed)
    times_ms = np.arange(tmin_ms, tmax_ms + 0.5 * 1000.0 / fs, 1000.0 / fs)
    cycle = (POSITIVE, NEGATIVE, POSITIVE, NEGATIVE, UNKNOWN)
    labels = np.array([cycle[i % len(cycle)] for i in range(n_epochs)], dtype=object)
    X = np.empty((n_epochs, len(times_ms)))
    for i, label in enumerate(labels):
        target = BOSS_TARGET_DEG.get(label)
        phase0 = rng.uniform(0, 360) if target is None else target + rng.normal(0, phase_jitter_deg)
        # cos(2*pi*f*t + phi) has Hilbert phase phi at t=0 (0 deg = peak).
        theta = np.cos(2 * np.pi * freq_hz * times_ms / 1000.0 + np.radians(phase0))
        beta = 0.25 * np.sin(2 * np.pi * 14.0 * times_ms / 1000.0 + rng.uniform(0, 2 * np.pi))
        X[i] = (theta + beta + rng.normal(0, noise_rel, size=times_ms.shape)) * 8e-6
    return EpochArrays(subject=subject, X=X, times_ms=times_ms, fs=float(fs), labels=labels, channel="synthetic")


# ---- 3. Phase estimation -------------------------------------------------

@dataclass
class CausalEstimate:
    phase_rad: float
    phase_deg: float
    amplitude: float
    core: np.ndarray             # filtered pre-stimulus window after trimming the edges
    pred_future: np.ndarray      # AR forecast continuing the core past t=0
    core_times_ms: np.ndarray
    future_times_ms: np.ndarray
    params: dict[str, float | int]
    phase_error_deg: float | None = None   # causal - non-causal, set by estimate_subject_epochs


@dataclass
class EpochEstimate:
    subject: str
    epoch_index: int
    label: str                   # BOSS class
    phase_rad: float             # non-causal phase at t=0
    phase_deg: float
    amplitude: float
    raw: np.ndarray
    filtered: np.ndarray         # non-causal filtered epoch
    times_ms: np.ndarray
    channel: str
    noncausal_filter_order: int
    causal: CausalEstimate | None = None


def to_0_360(phase_rad: float | np.ndarray) -> float | np.ndarray:
    """Radians -> 0-360 deg (0 = positive peak, 180 = negative trough)."""
    return np.degrees(phase_rad) % 360.0


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Signed circular difference a - b in degrees, wrapped to [-180, 180)."""
    return float((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def cutoff_index(times_ms: np.ndarray, cutoff_ms: float) -> int:
    return int(np.argmin(np.abs(times_ms - cutoff_ms)))


def bandpass_fir(fs: float, band: tuple[float, float], filter_order: int) -> np.ndarray:
    """Windowed-sinc FIR band-pass taps (``filter_order + 1`` taps).

    Longer filters give sharper band edges but need more signal; at 1000 Hz a 4-8 Hz
    band typically uses orders 150-450.
    """
    return design_bandpass(filter_order, band[0], band[1], fs)


def _noncausal_phase_with_filter(x: np.ndarray, times_ms: np.ndarray, taps: np.ndarray, cutoff_ms: float) -> dict:
    filtered = filtfilt(taps, 1.0, x - x.mean())
    value = hilbert(filtered)[cutoff_index(times_ms, cutoff_ms)]
    phase_rad = float(np.angle(value))
    return {
        "phase_rad": phase_rad,
        "phase_deg": float(to_0_360(phase_rad)),
        "amplitude": float(np.abs(value)),
        "filtered": filtered,
    }


def noncausal_phase(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
) -> dict:
    """Offline ("ground truth") phase at ``cutoff_ms``.

    Zero-phase ``filtfilt`` with an FIR band-pass over the whole epoch, then Hilbert;
    phase and amplitude are read at the sample nearest ``cutoff_ms``. Uses samples after
    the stimulus, so it cannot run online.

    Returns a dict with ``phase_rad`` (-pi..pi), ``phase_deg`` (0..360), ``amplitude``
    and ``filtered`` (the whole filtered epoch).
    """
    return _noncausal_phase_with_filter(x, times_ms, bandpass_fir(fs, band, filter_order), cutoff_ms)


def validate_causal_params(params: dict[str, float | int]) -> dict[str, float | int]:
    """Check all causal parameters are present and cast them to their types."""
    types = {"window_ms": float, "filter_order": int, "edge": int, "ar_order": int, "hilbert_window": int, "offset": int}
    missing = [key for key in types if key not in params]
    if missing:
        raise ValueError(f"Missing causal parameter(s): {missing}")
    return {key: cast(params[key]) for key, cast in types.items()}


def _window_samples(params: dict, fs: float) -> int:
    return round(float(params["window_ms"]) / 1000.0 * fs)


def _causal_params_fit(params: dict, cutoff: int, fs: float) -> bool:
    """The window fits before the cutoff, is longer than the filter, and leaves enough core for the AR fit."""
    window = _window_samples(params, fs)
    core_len = window - 2 * int(params["edge"])
    return cutoff - window >= 0 and window > int(params["filter_order"]) and core_len > int(params["ar_order"]) + 10


def _causal_phase_with_filter(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    taps: np.ndarray,
    cutoff_ms: float,
    params: dict[str, float | int],
) -> CausalEstimate | None:
    cutoff = cutoff_index(times_ms, cutoff_ms)
    if not _causal_params_fit(params, cutoff, fs):
        return None

    window = _window_samples(params, fs)
    edge = int(params["edge"])
    try:
        phase_rad, amplitude, core, pred_future = phastimate(
            x[cutoff - window:cutoff], taps, edge, int(params["ar_order"]),
            int(params["hilbert_window"]), int(params["offset"]), return_trace=True,
        )
    except Exception:
        return None
    if np.isnan(phase_rad):
        return None

    window_times = times_ms[cutoff - window:cutoff]
    core_times_ms = window_times[edge:-edge] if edge else window_times
    if len(core_times_ms) == 0:
        return None
    return CausalEstimate(
        phase_rad=float(phase_rad),
        phase_deg=float(to_0_360(phase_rad)),
        amplitude=float(amplitude),
        core=core,
        pred_future=pred_future,
        core_times_ms=core_times_ms,
        future_times_ms=core_times_ms[-1] + np.arange(1, len(pred_future) + 1) / fs * 1000.0,
        params=dict(params),
    )


def causal_phase(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    params: dict[str, float | int],
) -> CausalEstimate | None:
    """Online-style phase at ``cutoff_ms`` using only samples before it (phastimate).

    Takes the ``window_ms`` before the cutoff, band-passes it, trims ``edge`` samples at
    both ends, fits a Yule-Walker AR(``ar_order``) model, forecasts forward, and reads the
    phase from a Hilbert transform over the last ``hilbert_window`` samples.
    Returns None when the parameters don't fit in the epoch or the estimate fails.
    """
    params = validate_causal_params(params)
    return _causal_phase_with_filter(x, times_ms, fs, bandpass_fir(fs, band, params["filter_order"]), cutoff_ms, params)


def resolve_noncausal_filter_order(causal_params: dict | None, default_filter_order: int) -> int:
    """Filter order for the non-causal (ground-truth) estimate.

    With causal params (manual or optimized) the non-causal estimate uses the same
    ``filter_order``, so both methods see an identical band-pass; the optimizer scores
    parameter sets the same way. Without causal estimation the default is used.
    """
    return int(default_filter_order if causal_params is None else causal_params["filter_order"])


def _limit_epochs(X: np.ndarray, n_epochs: int | bool) -> np.ndarray:
    """First ``n_epochs`` rows, or all of them when ``n_epochs`` is False/0."""
    return X[:int(n_epochs)] if n_epochs else X


def estimate_subject_epochs(
    data: EpochArrays,
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
    n_epochs: int | bool,
    causal_params: dict[str, float | int] | None,
) -> list[EpochEstimate]:
    """Non-causal (and optional causal) phase for every epoch of one subject.

    ``filter_order`` is the non-causal filter order; see ``resolve_noncausal_filter_order``.
    """
    noncausal_taps = bandpass_fir(data.fs, band, filter_order)
    if causal_params is not None:
        causal_params = validate_causal_params(causal_params)
        causal_taps = bandpass_fir(data.fs, band, causal_params["filter_order"])

    estimates = []
    for i, (x, label) in enumerate(zip(_limit_epochs(data.X, n_epochs), data.labels)):
        est = _noncausal_phase_with_filter(x, data.times_ms, noncausal_taps, cutoff_ms)
        causal = None
        if causal_params is not None:
            causal = _causal_phase_with_filter(x, data.times_ms, data.fs, causal_taps, cutoff_ms, causal_params)
            if causal is not None:
                causal.phase_error_deg = angle_diff_deg(causal.phase_deg, est["phase_deg"])
        estimates.append(EpochEstimate(
            subject=data.subject,
            epoch_index=i,
            label=label,
            phase_rad=est["phase_rad"],
            phase_deg=est["phase_deg"],
            amplitude=est["amplitude"],
            raw=x,
            filtered=est["filtered"],
            times_ms=data.times_ms,
            channel=data.channel,
            noncausal_filter_order=int(filter_order),
            causal=causal,
        ))
    return estimates


def estimate_all_subjects(
    subject_data: Iterable[EpochArrays],
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
    n_epochs: int | bool,
    causal_estimation: bool,
    causal_params_mode: str,
    manual_causal_params: dict[str, float | int],
    optimization_config: dict,
) -> dict[str, list[EpochEstimate]]:
    """Estimate phases for each subject.

    ``causal_params_mode``: ``"manual"`` uses ``manual_causal_params`` for everyone;
    ``"optimize"`` fits one causal parameter set per subject (``optimize_causal_params``).
    """
    if causal_params_mode not in ("manual", "optimize"):
        raise ValueError("causal_params_mode must be 'manual' or 'optimize'.")

    estimates_by_subject = {}
    for data in subject_data:
        causal_params = None
        if causal_estimation and causal_params_mode == "manual":
            causal_params = validate_causal_params(manual_causal_params)
        elif causal_estimation:
            causal_params = optimize_causal_params(
                data.X, data.times_ms, data.fs, band, cutoff_ms, optimization_config,
                n_epochs=n_epochs, name=data.subject,
            )
        estimates_by_subject[data.subject] = estimate_subject_epochs(
            data,
            band=band,
            filter_order=resolve_noncausal_filter_order(causal_params, filter_order),
            cutoff_ms=cutoff_ms,
            n_epochs=n_epochs,
            causal_params=causal_params,
        )
    return estimates_by_subject


# ---- 4. Causal parameter optimization -----------------------------------

def _evaluate_causal_params(
    X: np.ndarray,
    indices: Sequence[int],
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    params: dict[str, float | int],
    min_usable_epochs: int,
) -> dict | None:
    """Circular variance of (causal - non-causal) phase over the given epochs, both using
    the candidate band-pass. None if the parameters don't fit or too few epochs are usable."""
    params = validate_causal_params(params)
    if not _causal_params_fit(params, cutoff_index(times_ms, cutoff_ms), fs):
        return None

    taps = bandpass_fir(fs, band, params["filter_order"])
    errors_rad = []
    for i in indices:
        try:
            causal = _causal_phase_with_filter(X[i], times_ms, fs, taps, cutoff_ms, params)
            if causal is None:
                continue
            noncausal = _noncausal_phase_with_filter(X[i], times_ms, taps, cutoff_ms)
        except Exception:
            continue
        errors_rad.append(causal.phase_rad - noncausal["phase_rad"])

    if len(errors_rad) < min_usable_epochs:
        return None
    resultant_length = float(np.abs(np.mean(np.exp(1j * np.asarray(errors_rad)))))
    return {
        "circular_variance": 1.0 - resultant_length,
        "resultant_length": resultant_length,
        "n_used": len(errors_rad),
        "n_total": len(indices),
    }


def _format_fit(result: dict) -> str:
    return (f"circular variance {result['circular_variance']:.4f}, R={result['resultant_length']:.3f}, "
            f"n={result['n_used']}/{result['n_total']}")


def optimize_causal_params(
    X: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    optimization_config: dict,
    n_epochs: int | bool = False,
    name: str = "data",
) -> dict[str, float | int]:
    """Choose one causal AR parameter set against the non-causal phase (Optuna TPE).

    Epochs are split into train/held-out (``train_fraction``). Each Optuna trial (one
    candidate parameter set, ``n_opt_trials`` in total) is scored by the circular
    variance of (causal - non-causal) phase over training epochs; lower is better.
    Both estimates use the candidate ``filter_order``. Held-out performance is printed.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optimization mode requires optuna. Install the project requirements first.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = optimization_config
    X = _limit_epochs(X, n_epochs)
    if len(X) == 0:
        raise ValueError(f"No epochs available for {name}.")

    shuffled = np.random.default_rng(int(cfg["random_seed"])).permutation(len(X))
    n_train = min(max(round(len(X) * float(cfg["train_fraction"])), 1), max(len(X) - 1, 1))
    train_idx, test_idx = shuffled[:n_train], shuffled[n_train:]

    def objective(trial):
        params = {
            "window_ms": trial.suggest_int("window_ms", *cfg["window_ms_range"], step=cfg["window_ms_step"]),
            "filter_order": trial.suggest_int("filter_order", *cfg["filter_order_range"], step=cfg["filter_order_step"]),
            "edge": trial.suggest_int("edge", *cfg["edge_range"], step=cfg["edge_step"]),
            "ar_order": trial.suggest_int("ar_order", *cfg["ar_order_range"]),
            "hilbert_window": cfg["hilbert_window"],
            "offset": cfg["offset"],
        }
        result = _evaluate_causal_params(X, train_idx, times_ms, fs, band, cutoff_ms, params, cfg["min_usable_epochs"])
        return cfg["infeasible_penalty"] if result is None else result["circular_variance"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(cfg["random_seed"])))
    study.optimize(objective, n_trials=int(cfg["n_opt_trials"]))

    best_params = validate_causal_params(
        {**study.best_params, "hilbert_window": cfg["hilbert_window"], "offset": cfg["offset"]}
    )
    train_result = _evaluate_causal_params(X, train_idx, times_ms, fs, band, cutoff_ms, best_params, cfg["min_usable_epochs"])
    if train_result is None:
        raise RuntimeError(f"No feasible causal AR parameters found for {name}.")
    test_result = None
    if len(test_idx):
        test_result = _evaluate_causal_params(X, test_idx, times_ms, fs, band, cutoff_ms, best_params, 1)

    print(f"\n[{name}] optimized causal AR params: {best_params}")
    print(f"  train: {_format_fit(train_result)}")
    print(f"  held-out: {'unavailable / too few usable epochs' if test_result is None else _format_fit(test_result)}")
    return best_params


# ---- 5. Classification, scoring and circular statistics ----------------

def classify_phase(phase_deg: float | None, tolerance_deg: float) -> str | None:
    """POSITIVE within +/-tolerance of 0 deg, NEGATIVE within +/-tolerance of 180 deg
    (the nearer one if both), otherwise UNCLASSIFIED. A missing phase (None/NaN) gives None."""
    if phase_deg is None or np.isnan(phase_deg):
        return None
    to_positive = abs(angle_diff_deg(phase_deg, 0.0))
    to_negative = abs(angle_diff_deg(phase_deg, 180.0))
    if min(to_positive, to_negative) > tolerance_deg:
        return UNCLASSIFIED
    return POSITIVE if to_positive <= to_negative else NEGATIVE


def comparison_status(boss_label: str, phase_class: str | None) -> str:
    """``correct`` / ``wrong`` / ``unclassified`` (or missing phase) / ``unknown`` (no BOSS label)."""
    if boss_label not in BOSS_TARGET_DEG:
        return "unknown"
    if phase_class in (None, UNCLASSIFIED):
        return "unclassified"
    return "correct" if phase_class == boss_label else "wrong"


def chance_success_pct(tolerance_deg: float) -> float:
    """Success expected if BOSS fired at a random phase: the +/-T window covers 2T of 360 deg."""
    return 100.0 * 2.0 * tolerance_deg / 360.0


def score_vs_boss(boss_labels: Sequence[str], phases_deg: Sequence[float | None], tolerance_deg: float) -> dict:
    """BOSS success at one tolerance.

    Only BOSS-labeled (positive/negative) epochs count. ``success_pct`` =
    correct / n_labeled * 100, so unclassified epochs count as failures.
    ``by_class`` gives n and correct separately for BOSS positive and negative.
    """
    counts = {"correct": 0, "wrong": 0, "unclassified": 0}
    by_class = {name: {"n": 0, "correct": 0} for name in BOSS_TARGET_DEG}
    for boss, phase in zip(boss_labels, phases_deg):
        status = comparison_status(boss, classify_phase(phase, tolerance_deg))
        if status == "unknown":
            continue
        counts[status] += 1
        by_class[boss]["n"] += 1
        by_class[boss]["correct"] += status == "correct"
    n_labeled = sum(counts.values())
    return {
        "n_labeled": n_labeled,
        **counts,
        "success_pct": 100.0 * counts["correct"] / n_labeled if n_labeled else None,
        "by_class": by_class,
    }


def _as_float_array(values: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def boss_target_deviations(boss_labels: Sequence[str], phases_deg: Sequence[float | None]) -> np.ndarray:
    """Signed deviation of each phase from its BOSS target (NaN for unknown label or missing phase)."""
    return np.array([
        angle_diff_deg(phase, BOSS_TARGET_DEG[boss])
        if boss in BOSS_TARGET_DEG and phase is not None and not np.isnan(phase) else np.nan
        for boss, phase in zip(boss_labels, phases_deg)
    ])


def circular_stats(angles_deg: Sequence[float], tolerances_deg: Sequence[float] = (), axis: str = "deviation") -> dict:
    """Circular summary of angles (NaNs ignored).

    ``mean_deg``: circular mean (signed for ``axis="deviation"``, 0-360 for ``axis="phase"``);
    ``R``: mean resultant length (1 = all equal); ``sd_deg``: circular SD = sqrt(-2 ln R);
    ``pct_within``: % with |angle| <= T (deviations only).
    """
    values = _as_float_array(angles_deg)
    values = values[~np.isnan(values)]
    within_keys = tolerances_deg if axis == "deviation" else ()
    if len(values) == 0:
        return {"n": 0, "mean_deg": None, "sd_deg": None, "R": None, "pct_within": {str(t): None for t in within_keys}}
    vector = np.mean(np.exp(1j * np.radians(values)))
    R = float(np.abs(vector))
    mean_deg = angle_diff_deg(float(np.degrees(np.angle(vector))), 0.0)
    return {
        "n": len(values),
        "mean_deg": mean_deg if axis == "deviation" else mean_deg % 360.0,
        "sd_deg": float(np.degrees(np.sqrt(-2.0 * np.log(R)))) if R > 0 else None,
        "R": R,
        "pct_within": {str(t): float(100.0 * np.mean(np.abs(values) <= t)) for t in within_keys},
    }


def circular_histogram(
    angles_deg: Sequence[float | None],
    boss_labels: Sequence[str],
    tolerances_deg: Sequence[float] = (),
    axis: str = "deviation",
    bin_width_deg: float = 10.0,
) -> dict:
    """Counts per angle bin split by BOSS class, plus circular stats overall and per class.

    ``axis="deviation"`` bins -180..180 deg, ``axis="phase"`` bins 0..360 deg. Per-class
    stats matter because opposite biases of positive and negative epochs would cancel
    in the pooled mean.
    """
    start = -180.0 if axis == "deviation" else 0.0
    edges = np.arange(start, start + 360.0 + bin_width_deg / 2, bin_width_deg)
    values = _as_float_array(angles_deg)
    boss = np.asarray(boss_labels, dtype=object)
    return {
        "axis": axis,
        "bin_edges_deg": edges.tolist(),
        "counts_by_class": {
            name: np.histogram(values[(boss == name) & ~np.isnan(values)], bins=edges)[0].astype(int).tolist()
            for name in BOSS_CLASSES
        },
        "stats": circular_stats(values, tolerances_deg, axis),
        "stats_by_class": {name: circular_stats(values[boss == name], tolerances_deg, axis) for name in BOSS_CLASSES},
    }


def boss_success_by_tolerance(subjects: dict[str, tuple[Sequence[str], Sequence[float]]], tolerances_deg: Sequence[float]) -> dict:
    """Per-subject non-causal BOSS success for each tolerance, plus mean and sample SD across subjects.

    ``subjects`` maps subject id -> (BOSS labels, non-causal phases in deg).
    """
    result = {"subjects": list(subjects), "tolerances_deg": list(tolerances_deg), "by_tolerance": {}}
    for tol in tolerances_deg:
        scores = [score_vs_boss(boss, phases, tol) for boss, phases in subjects.values()]
        pct = [s["success_pct"] for s in scores]
        valid = [p for p in pct if p is not None]
        result["by_tolerance"][str(tol)] = {
            "success_pct": pct,
            "correct": [s["correct"] for s in scores],
            "n_labeled": [s["n_labeled"] for s in scores],
            "mean_pct": float(np.mean(valid)) if valid else None,
            "sd_pct": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
            "chance_pct": chance_success_pct(tol),
        }
    return result


def _histogram_angles(data: dict[str, Sequence]) -> dict[str, np.ndarray]:
    """The angles behind each entry of HISTOGRAM_AXES, for one subject's per-epoch arrays."""
    return {
        "noncausal_phase": _as_float_array(data["noncausal_deg"]),
        "boss_target": boss_target_deviations(data["boss"], data["noncausal_deg"]),
        "causal_error": _as_float_array(data["causal_error_deg"]),
    }


def analyze_phase_results(subjects: dict[str, dict[str, Sequence]], tolerances_deg: Sequence[float]) -> dict:
    """Everything the report shows, precomputed for every tolerance.

    ``subjects`` maps subject id -> dict with per-epoch sequences ``boss`` (labels),
    ``noncausal_deg``, ``causal_deg`` and ``causal_error_deg`` (None where missing),
    e.g. from ``estimates_to_arrays``.

    Returns:
      ``subjects[id]["classes" / "status" / "counts"][method][str(tol)]``  per-epoch class,
          per-epoch comparison status, and ``score_vs_boss`` counts;
      ``subjects[id]["histograms"][kind]`` and ``pooled[kind]``  ``circular_histogram`` for each
          kind in HISTOGRAM_AXES (non-causal phase; non-causal - BOSS target; causal - non-causal);
      ``success``  ``boss_success_by_tolerance``.
    """
    out = {"tolerances_deg": list(tolerances_deg), "subjects": {}, "pooled": {}}
    pooled_angles = {kind: [] for kind in HISTOGRAM_AXES}
    pooled_boss = []

    for subject_id, data in subjects.items():
        boss = list(data["boss"])
        phases = {"noncausal": list(data["noncausal_deg"]), "causal": list(data["causal_deg"])}
        subject_out = {"classes": {}, "status": {}, "counts": {}, "histograms": {}}
        for method in METHODS:
            classes = {str(t): [classify_phase(p, t) for p in phases[method]] for t in tolerances_deg}
            subject_out["classes"][method] = classes
            subject_out["status"][method] = {
                key: [comparison_status(b, c) for b, c in zip(boss, values)] for key, values in classes.items()
            }
            subject_out["counts"][method] = {str(t): score_vs_boss(boss, phases[method], t) for t in tolerances_deg}

        for kind, angles in _histogram_angles(data).items():
            subject_out["histograms"][kind] = circular_histogram(angles, boss, tolerances_deg, HISTOGRAM_AXES[kind])
            pooled_angles[kind].extend(angles.tolist())
        pooled_boss.extend(boss)
        out["subjects"][subject_id] = subject_out

    for kind, angles in pooled_angles.items():
        out["pooled"][kind] = circular_histogram(angles, pooled_boss, tolerances_deg, HISTOGRAM_AXES[kind])

    out["success"] = boss_success_by_tolerance(
        {sid: (data["boss"], data["noncausal_deg"]) for sid, data in subjects.items()}, tolerances_deg
    )
    return out


def estimates_to_arrays(estimates: Sequence[EpochEstimate]) -> dict[str, list]:
    """Per-epoch sequences in the shape ``analyze_phase_results`` expects."""
    return {
        "boss": [est.label for est in estimates],
        "noncausal_deg": [est.phase_deg for est in estimates],
        "causal_deg": [None if est.causal is None else est.causal.phase_deg for est in estimates],
        "causal_error_deg": [None if est.causal is None else est.causal.phase_error_deg for est in estimates],
    }


# ---- 6. Figures (notebook + publication export) ------------------------

def _figure_axes(ax, figsize, polar: bool = False):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"} if polar else None)


def _setup_polar(ax) -> None:
    """0 deg at the top, angles increasing clockwise."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def _shade_windows(ax, centers_deg: Sequence[float], tolerance_deg: float, radius: float, label: str | None = None) -> None:
    """Grey +/-tolerance sectors around each center on a polar axis."""
    for i, center in enumerate(centers_deg):
        ax.bar(np.radians(center), radius, width=np.radians(2 * tolerance_deg), color="0.9", zorder=0,
               label=label if i == 0 else None)


def _time_window_mask(times_ms: np.ndarray, window_ms: tuple[float, float] | None) -> np.ndarray:
    if window_ms is None:
        return np.ones_like(times_ms, dtype=bool)
    return (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])


def plot_epoch_estimates(
    x: np.ndarray,
    times_ms: np.ndarray,
    noncausal: dict,
    causal: CausalEstimate | None,
    band: tuple[float, float],
    cutoff_ms: float,
    title: str = "",
    window_ms: tuple[float, float] = (-600.0, 300.0),
):
    """One epoch, step by step, in three stacked panels sharing the time axis (signal in uV).

    1. raw signal; 2. raw + non-causal filtfilt trace with its phase at the cutoff;
    3. non-causal reference + causal filtered core and AR forecast with the causal phase.
    ``noncausal`` / ``causal`` are the outputs of ``noncausal_phase`` / ``causal_phase``.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    mask = _time_window_mask(times_ms, window_ms)
    t, raw, filtered = times_ms[mask], x[mask] * 1e6, noncausal["filtered"][mask] * 1e6

    axes[0].plot(t, raw, color=METHOD_COLORS["raw"], lw=0.8)
    axes[0].set_title("1. Raw signal", fontsize=10, loc="left")

    axes[1].plot(t, raw, color=METHOD_COLORS["raw"], lw=0.6, alpha=0.6, label="raw")
    axes[1].plot(t, filtered, color=METHOD_COLORS["noncausal"], lw=1.6, label=f"non-causal filtfilt {band[0]:g}-{band[1]:g} Hz")
    axes[1].set_title(f"2. Non-causal (uses the whole epoch): phase at t=0 = {noncausal['phase_deg']:.1f} deg",
                      fontsize=10, loc="left")

    axes[2].plot(t, filtered, color=METHOD_COLORS["noncausal"], lw=1.4, label="non-causal (reference)")
    if causal is None:
        axes[2].set_title("3. Causal: unavailable for these parameters", fontsize=10, loc="left")
    else:
        core = _time_window_mask(causal.core_times_ms, window_ms)
        future = _time_window_mask(causal.future_times_ms, window_ms)
        axes[2].plot(causal.core_times_ms[core], causal.core[core] * 1e6, color=METHOD_COLORS["causal"], lw=1.6,
                     label="causal: filtered pre-stimulus core")
        axes[2].plot(causal.future_times_ms[future], causal.pred_future[future] * 1e6, color=METHOD_COLORS["causal"],
                     lw=1.6, ls="--", label="causal: AR forecast")
        diff = angle_diff_deg(causal.phase_deg, noncausal["phase_deg"])
        axes[2].set_title(f"3. Causal (only data before t=0): phase = {causal.phase_deg:.1f} deg "
                          f"(causal - non-causal = {diff:+.1f} deg)", fontsize=10, loc="left")

    for ax in axes:
        ax.axvline(cutoff_ms, color="gray", ls=":", lw=1.2)
        ax.set_ylabel("uV")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1:]:
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[-1].set_xlabel("Time relative to stimulus (ms)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_phase_circle(
    phases_deg: Sequence[float | None],
    boss_labels: Sequence[str],
    tolerance_deg: float,
    title: str = "",
    ax=None,
):
    """One dot per epoch at its phase, colored by BOSS label; shaded +/-T windows around 0 and 180 deg."""
    fig, ax = _figure_axes(ax, figsize=(5.5, 5.5), polar=True)
    _setup_polar(ax)
    _shade_windows(ax, (0.0, 180.0), tolerance_deg, 2.1)
    for center in (0.0, 180.0):
        ax.axvline(np.radians(center), color="black", ls=":", lw=1, alpha=0.55)
    for class_name in BOSS_CLASSES:
        selected = [p for p, b in zip(phases_deg, boss_labels) if b == class_name and p is not None]
        if selected:
            ax.scatter(np.radians(selected), np.linspace(1, 2, len(selected)), s=26, color=CLASS_COLORS[class_name],
                       alpha=0.85, label=f"BOSS {class_name} (n={len(selected)})")
    ax.set_yticklabels([])
    ax.set_ylim(0, 2.1)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    fig.tight_layout()
    return fig


def plot_circular_histogram(histogram: dict, tolerance_deg: float, title: str = "", ax=None):
    """Circular (rose) histogram from ``circular_histogram`` / ``analyze_phase_results``.

    Wedges = epochs per bin, stacked by BOSS class; 0 deg at the top, clockwise.
    Deviation histograms shade +/-T around 0; phase histograms shade +/-T around 0
    (positive target) and 180 deg (negative target). One vector per BOSS class (line
    ending in a dot) is that class's mean resultant vector: direction = circular mean,
    length = R relative to the outer ring. The title gives the pooled statistics
    (per class for phase histograms).
    """
    fig, ax = _figure_axes(ax, figsize=(5.2, 5.4), polar=True)
    _setup_polar(ax)
    is_phase = histogram["axis"] == "phase"
    edges = np.asarray(histogram["bin_edges_deg"])
    totals = sum(np.asarray(c) for c in histogram["counts_by_class"].values())
    r_max = max(float(np.max(totals)), 1.0) * 1.08

    _shade_windows(ax, (0.0, 180.0) if is_phase else (0.0,), tolerance_deg, r_max, label=f"+/-{tolerance_deg:g} deg")
    bottom = np.zeros(len(edges) - 1)
    for class_name in BOSS_CLASSES:
        counts = np.asarray(histogram["counts_by_class"][class_name])
        if counts.sum():
            ax.bar(np.radians(edges[:-1]), counts, width=np.radians(edges[1] - edges[0]), bottom=bottom, align="edge",
                   color=CLASS_COLORS[class_name], edgecolor="white", lw=0.5, zorder=2, label=f"BOSS {class_name}")
            bottom += counts

    for class_name, class_stats in histogram["stats_by_class"].items():
        if class_stats["n"]:
            theta, radius, color = np.radians(class_stats["mean_deg"]), class_stats["R"] * r_max, CLASS_COLORS[class_name]
            ax.plot([theta, theta], [0, radius], color="white", lw=6, solid_capstyle="round", zorder=4)
            ax.plot([theta, theta], [0, radius], color=color, lw=3, solid_capstyle="round", zorder=5,
                    marker="o", markevery=[1], markersize=8, markeredgecolor="white", markeredgewidth=1.2)
            ax.plot([], [], color=color, lw=3, label=f"mean vector {class_name}")

    stats = histogram["stats"]
    if is_phase:
        # A pooled mean over positive and negative epochs is meaningless here; show each class.
        text = "\n".join(
            f"{name}: n={st['n']}  mean={st['mean_deg']:.1f} deg  R={st['R']:.2f}"
            for name, st in histogram["stats_by_class"].items() if name in BOSS_TARGET_DEG and st["n"]
        )
    elif stats["n"]:
        text = f"n={stats['n']}  mean={stats['mean_deg']:+.1f} deg  R={stats['R']:.2f}\ncirc SD={stats['sd_deg']:.1f} deg"
        within = stats["pct_within"].get(str(tolerance_deg))
        if within is not None:
            text += f"  within +/-{tolerance_deg:g}: {within:.0f}%"
    else:
        text = ""
    ax.set_title("\n".join(part for part in (title, text) if part), fontsize=9, pad=14)

    ticks = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([str(t) for t in ticks] if is_phase else ["0", "+45", "+90", "+135", "±180", "-135", "-90", "-45"])
    ax.set_ylim(0, r_max)
    ax.set_rlabel_position(112.5)
    ax.tick_params(axis="y", labelsize=7, colors="0.4")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.92, 1.12), frameon=False)
    fig.tight_layout()
    return fig


def plot_success_vs_tolerance(success: dict, highlight_subject: str | None = None, ax=None, jitter: float = 0.09):
    """% BOSS success vs tolerance: one dot per subject, paired lines, mean +/- SD, chance line.

    ``success`` is the output of ``boss_success_by_tolerance`` (tolerances plotted in the
    given order). ``highlight_subject`` draws that subject in black and dims the rest.
    Jitter is deterministic so the figure is reproducible.
    """
    fig, ax = _figure_axes(ax, figsize=(4.6, 4.4))
    tolerances, subjects = success["tolerances_deg"], success["subjects"]
    offsets = np.linspace(-jitter, jitter, len(subjects)) if len(subjects) > 1 else np.zeros(1)
    xs = np.arange(len(tolerances))
    by_tol = [success["by_tolerance"][str(t)] for t in tolerances]
    dim = highlight_subject in subjects

    for s, subject in enumerate(subjects):
        ys = [entry["success_pct"][s] for entry in by_tol]
        if subject == highlight_subject:
            ax.plot(xs + offsets[s], ys, color="black", lw=1.8, zorder=4)
            ax.scatter(xs + offsets[s], ys, s=46, color="black", zorder=5, edgecolors="white", linewidths=0.8, label=subject)
        else:
            ax.plot(xs + offsets[s], ys, color="0.85" if dim else "0.75", lw=0.7, zorder=1)
            ax.scatter(xs + offsets[s], ys, s=22, color="0.7" if dim else METHOD_COLORS["noncausal"],
                       alpha=0.75, zorder=2, edgecolors="white", linewidths=0.5)
    for x, entry in zip(xs, by_tol):
        if entry["mean_pct"] is not None:
            ax.hlines(entry["mean_pct"], x - 0.28, x + 0.28, color="black", lw=2.2, zorder=3)
        if entry["sd_pct"] is not None:
            ax.errorbar(x + 0.33, entry["mean_pct"], yerr=entry["sd_pct"], color="black", capsize=4, lw=1.3, zorder=3)
        ax.hlines(entry["chance_pct"], x - 0.4, x + 0.4, color=METHOD_COLORS["causal"], ls="--", lw=1.3, zorder=0)
    ax.plot([], [], color=METHOD_COLORS["causal"], ls="--", label="chance (2T/360)")
    ax.plot([], [], color="black", lw=2.2, label="mean +/- SD")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"+/-{t:g}" for t in tolerances])
    ax.set_xlim(-0.6, len(tolerances) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Tolerance (deg)")
    ax.set_ylabel("BOSS success (% of labeled epochs)")
    ax.set_title(f"BOSS phase accuracy (non-causal, n={len(subjects)} subjects)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    return fig
