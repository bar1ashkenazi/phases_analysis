"""Reusable helpers for intake-session phase/BOSS comparison.

This module is the single source of analysis logic. Runners, the HTML report and the
notebook (phase_pipeline.ipynb) only call these functions and render what they return.

Terminology: an epoch is the EEG window cut around one stimulation event (one BOSS
decision, t=0). A memory-task trial contains several such events; epochs are analyzed
independently. An Optuna
"trial" (``n_opt_trials``) is something else: one evaluated causal parameter set.

The non-causal phase estimate intentionally follows the working logic in
``pipeline.py``: zero-phase FIR filtering across the full epoch, Hilbert transform,
then phase/amplitude read out at the sample nearest t=0.
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


# ---- Labels and colors ---------------------------------------------------

@dataclass(frozen=True)
class LabelScheme:
    """BOSS label vocabulary: output class names and the raw metadata values mapped to them."""

    positive: str = "positive"
    negative: str = "negative"
    unknown: str = "unknown"
    unclassified: str = "unclassified"
    positive_labels: frozenset[str] = frozenset({"positive", "pos", "p", "1", "true", "peak"})
    negative_labels: frozenset[str] = frozenset({"negative", "neg", "n", "0", "false", "trough"})

    @property
    def class_order(self) -> tuple[str, str, str]:
        return (self.positive, self.negative, self.unknown)

    def target_deg(self, label: str) -> float | None:
        """Intended BOSS phase: 0 deg (positive peak) or 180 deg (negative trough)."""
        if label == self.positive:
            return 0.0
        if label == self.negative:
            return 180.0
        return None


DEFAULT_LABELS = LabelScheme()

# Paul Tol "vibrant" (colorblind-safe). Pink/blue mean BOSS class only; method traces
# use neutral charcoal (non-causal) and orange (causal) so they never read as a class.
CLASS_COLORS = {
    DEFAULT_LABELS.positive: "#EE3377",
    DEFAULT_LABELS.negative: "#0077BB",
    DEFAULT_LABELS.unknown: "#858b93",
}
METHOD_COLORS = {"raw": "#9a9e98", "noncausal": "#33363b", "causal": "#EE7733"}

METHODS = ("noncausal", "causal")
DEVIATION_KINDS = ("causal_error", "boss_target")


# ---- Loading -------------------------------------------------------------

@dataclass(frozen=True)
class LoadConfig:
    """Where intake epochs live and how they are preprocessed before phase estimation.

    Parameters
    ----------
    data_root : Path
        Folder containing ``<subject>/EEG/processed/<intake_filename>``.
    intake_filename : str
        Epoch file name pattern, formatted with ``subject``.
    channel : str
        Channel to analyze. If equal to ``hjorth_channel`` it is built from ``hjorth_weights``.
    condition_column : str
        Metadata column with the BOSS label. If missing, columns matching
        ``condition_auto_keywords`` are tried.
    lowpass_before_downsample_hz : float or None
        Anti-alias low-pass applied before resampling (Hz).
    downsample, downsample_fs : bool, float
        Resample epochs to ``downsample_fs`` Hz.
    hjorth_channel, hjorth_weights, hjorth_scale_reference
        Weighted Hjorth/Laplacian channel definition; it is rescaled to the std of
        ``hjorth_scale_reference``.

    Only ``data_root`` is required; the defaults are the intake analysis settings
    (Fz Hjorth channel, 100 Hz low-pass, resampling to 1000 Hz).
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
    labels: LabelScheme = DEFAULT_LABELS
    show_metadata_summary: bool = False
    metadata_max_values: int = 12


@dataclass
class EpochArrays:
    """One subject's epochs as plain arrays.

    ``X`` has shape (n_epochs, n_times) in volts, ``times_ms`` is relative to the
    stimulus (t=0), ``labels`` holds one BOSS class name per epoch.
    """

    subject: str
    X: np.ndarray
    times_ms: np.ndarray
    fs: float
    labels: np.ndarray
    channel: str
    condition_column: str


def as_subject_list(subjects: str | Sequence[str]) -> list[str]:
    if isinstance(subjects, str):
        return [subjects]
    return list(subjects)


def intake_epoch_path(subject: str, data_root: Path, intake_filename: str) -> Path:
    return data_root / subject / "EEG" / "processed" / intake_filename.format(subject=subject)


def available_intake_subjects(data_root: Path, intake_filename: str) -> list[str]:
    if not data_root.exists():
        return []
    subjects = []
    for path in sorted(data_root.glob("sub_*")):
        if intake_epoch_path(path.name, data_root, intake_filename).exists():
            subjects.append(path.name)
    return subjects


def add_weighted_hjorth_channel(
    epochs: mne.Epochs,
    output_channel: str,
    weights_by_channel: dict[str, float],
    scale_reference_channel: str,
) -> mne.Epochs:
    """Add a weighted Hjorth/Laplacian-style channel."""
    if output_channel in epochs.ch_names:
        return epochs

    missing = [ch for ch in weights_by_channel if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Cannot build {output_channel}; missing channel(s): {missing}")

    weights = np.array([weights_by_channel.get(ch, 0.0) for ch in epochs.ch_names])
    data = epochs.get_data()
    hjorth = np.einsum("ect,c->et", data, weights)

    ref_idx = epochs.ch_names.index(scale_reference_channel)
    scale = np.std(data[:, ref_idx, :]) / np.std(hjorth)
    hjorth = (hjorth * scale)[:, np.newaxis, :]

    info = mne.create_info([output_channel], epochs.info["sfreq"], ["eeg"])
    new_epo = mne.EpochsArray(hjorth, info, tmin=epochs.tmin, metadata=epochs.metadata)
    epochs.add_channels([new_epo], force_update_info=True)
    return epochs


def normalize_boss_label(value, labels: LabelScheme = DEFAULT_LABELS) -> str:
    """Map a raw metadata value to positive/negative/unknown."""
    if value is None:
        return labels.unknown

    if isinstance(value, (bool, np.bool_)):
        return labels.positive if value else labels.negative

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return labels.unknown
        if np.isclose(value, 1):
            return labels.positive
        if np.isclose(value, 0):
            return labels.negative

    normalized = str(value).strip().lower()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    if normalized in labels.positive_labels:
        return labels.positive
    if normalized in labels.negative_labels:
        return labels.negative
    return labels.unknown


def _map_labels(column, labels: LabelScheme):
    return column.map(lambda value: normalize_boss_label(value, labels))


def summarize_metadata(epochs: mne.Epochs, max_values: int) -> None:
    """Print metadata columns and a compact value summary."""
    if epochs.metadata is None:
        print("metadata: none")
        return

    print("metadata columns:")
    for column in epochs.metadata.columns:
        values = epochs.metadata[column].dropna().unique()[:max_values]
        values_str = ", ".join(repr(v) for v in values)
        print(f"  - {column}: {values_str}")


def _resolve_condition_column(epochs: mne.Epochs, config: LoadConfig) -> str:
    if epochs.metadata is None:
        raise ValueError("The epochs file has no metadata, so BOSS labels cannot be read.")
    if config.condition_column in epochs.metadata.columns:
        return config.condition_column

    labels = config.labels
    candidates = [
        column for column in epochs.metadata.columns
        if any(key in column.lower() for key in config.condition_auto_keywords)
    ]
    for column in candidates:
        if _map_labels(epochs.metadata[column], labels).isin([labels.positive, labels.negative]).any():
            print(f"Using metadata column {column!r} for BOSS labels.")
            return column

    raise ValueError(
        f"Could not find BOSS label column {config.condition_column!r}. "
        f"Available columns: {list(epochs.metadata.columns)}"
    )


def get_data(subject: str, config: LoadConfig) -> EpochArrays:
    """Load one subject's intake epochs into arrays.

    Steps: read ``.fif`` -> optional low-pass -> optional resample -> optional Hjorth
    channel -> pick ``config.channel`` -> map BOSS metadata to class names.

    Returns
    -------
    EpochArrays
        ``X`` (n_epochs, n_times) in volts, ``times_ms``, ``fs`` (Hz), ``labels``.
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
            epochs,
            output_channel=config.hjorth_channel,
            weights_by_channel=config.hjorth_weights,
            scale_reference_channel=config.hjorth_scale_reference,
        )

    if config.show_metadata_summary:
        print(f"\n[{subject}] {epo_path}")
        summarize_metadata(epochs, config.metadata_max_values)

    condition_column = _resolve_condition_column(epochs, config)
    return EpochArrays(
        subject=subject,
        X=epochs.get_data(picks=config.channel)[:, 0, :],
        times_ms=epochs.times * 1000.0,
        fs=float(epochs.info["sfreq"]),
        labels=_map_labels(epochs.metadata[condition_column], config.labels).to_numpy(),
        channel=config.channel,
        condition_column=condition_column,
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
    labels: LabelScheme = DEFAULT_LABELS,
) -> EpochArrays:
    """Synthetic theta epochs for testing without data.

    Each epoch is a ``freq_hz`` sine whose phase at t=0 is near the BOSS target
    (0 deg positive, 180 deg negative, every 5th epoch unknown/random) plus a
    14 Hz component and white noise. Amplitudes are in volts (~8 uV).
    """
    rng = np.random.default_rng(seed)
    times_ms = np.arange(tmin_ms, tmax_ms + 0.5 * 1000.0 / fs, 1000.0 / fs)
    cycle = (labels.positive, labels.negative, labels.positive, labels.negative, labels.unknown)
    epoch_labels = np.array([cycle[i % len(cycle)] for i in range(n_epochs)], dtype=object)
    X = np.empty((n_epochs, len(times_ms)))
    for i, label in enumerate(epoch_labels):
        target = labels.target_deg(label)
        phase0 = rng.uniform(0, 360) if target is None else target + rng.normal(0, phase_jitter_deg)
        # cos(2*pi*f*t + phi) has Hilbert phase phi at t=0 (0 deg = peak, matching to_0_360).
        theta = np.cos(2 * np.pi * freq_hz * times_ms / 1000.0 + np.radians(phase0))
        beta = 0.25 * np.sin(2 * np.pi * 14.0 * times_ms / 1000.0 + rng.uniform(0, 2 * np.pi))
        X[i] = (theta + beta + rng.normal(0, noise_rel, size=times_ms.shape)) * 8e-6
    return EpochArrays(
        subject=subject,
        X=X,
        times_ms=times_ms,
        fs=float(fs),
        labels=epoch_labels,
        channel="synthetic",
        condition_column="synthetic",
    )


# ---- Phase estimation ----------------------------------------------------

@dataclass
class CausalEstimate:
    phase_rad: float
    phase_deg: float
    amplitude: float
    core: np.ndarray
    pred_future: np.ndarray
    core_times_ms: np.ndarray
    future_times_ms: np.ndarray
    params: dict[str, float | int]
    phase_error_deg: float | None = None


@dataclass
class EpochEstimate:
    subject: str
    epoch_index: int
    label: str
    phase_rad: float
    phase_deg: float
    amplitude: float
    raw: np.ndarray
    filtered: np.ndarray
    times_ms: np.ndarray
    channel: str
    noncausal_filter_order: int
    causal: CausalEstimate | None = None


def to_0_360(phase_rad: float | np.ndarray) -> float | np.ndarray:
    """0-360 deg, where 0 is a positive peak and 180 is a negative peak."""
    return np.degrees(phase_rad) % 360.0


def cutoff_index(times_ms: np.ndarray, cutoff_ms: float) -> int:
    return int(np.argmin(np.abs(times_ms - cutoff_ms)))


def bandpass_fir(fs: float, band: tuple[float, float], filter_order: int) -> np.ndarray:
    """Windowed-sinc FIR band-pass taps (``filter_order + 1`` taps).

    Longer filters give sharper band edges but need more signal; at 1000 Hz a 4-8 Hz
    band typically uses orders 150-450.
    """
    return design_bandpass(filter_order, band[0], band[1], fs)


def _noncausal_phase_with_filter(
    x: np.ndarray,
    times_ms: np.ndarray,
    bandpass_filter: np.ndarray,
    cutoff_ms: float,
) -> dict:
    idx = cutoff_index(times_ms, cutoff_ms)
    filtered = filtfilt(bandpass_filter, 1.0, x - x.mean())
    analytic = hilbert(filtered)
    phase_rad = float(np.angle(analytic[idx]))
    amplitude = float(np.abs(analytic[idx]))
    return {
        "phase_rad": phase_rad,
        "phase_deg": float(to_0_360(phase_rad)),
        "amplitude": amplitude,
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
    phase and amplitude are read at the sample nearest ``cutoff_ms``. Uses future
    samples, so it cannot run online.

    Returns
    -------
    dict
        ``phase_rad`` (-pi..pi), ``phase_deg`` (0..360, 0 = peak, 180 = trough),
        ``amplitude`` (signal units), ``filtered`` (full filtered epoch).
    """
    return _noncausal_phase_with_filter(x, times_ms, bandpass_fir(fs, band, filter_order), cutoff_ms)


def _causal_core_times(times_ms: np.ndarray, cutoff: int, window_samples: int, edge: int) -> np.ndarray:
    window_times = times_ms[cutoff - window_samples:cutoff]
    return window_times[edge:-edge] if edge else window_times


def _causal_params_are_feasible(
    cutoff: int,
    fs: float,
    window_ms: float,
    filter_order: int,
    edge: int,
    ar_order: int,
) -> bool:
    window_samples = round(window_ms / 1000.0 * fs)
    core_len = window_samples - 2 * edge
    return cutoff - window_samples >= 0 and window_samples > filter_order and core_len > ar_order + 10


def _causal_phase_with_filter(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    bandpass_filter: np.ndarray,
    cutoff_ms: float,
    params: dict[str, float | int],
) -> CausalEstimate | None:
    cutoff = cutoff_index(times_ms, cutoff_ms)
    window_ms = float(params["window_ms"])
    filter_order = int(params["filter_order"])
    edge = int(params["edge"])
    ar_order = int(params["ar_order"])
    hilbert_window = int(params["hilbert_window"])
    offset = int(params["offset"])
    window_samples = round(window_ms / 1000.0 * fs)

    if not _causal_params_are_feasible(cutoff, fs, window_ms, filter_order, edge, ar_order):
        return None

    try:
        phase_rad, amplitude, core, pred_future = phastimate(
            x[cutoff - window_samples:cutoff],
            bandpass_filter,
            edge,
            ar_order,
            hilbert_window,
            offset,
            return_trace=True,
        )
    except Exception:
        return None

    if np.isnan(phase_rad):
        return None

    core_times_ms = _causal_core_times(times_ms, cutoff, window_samples, edge)
    if len(core_times_ms) == 0:
        return None

    future_times_ms = core_times_ms[-1] + (np.arange(1, len(pred_future) + 1) / fs * 1000.0)
    return CausalEstimate(
        phase_rad=float(phase_rad),
        phase_deg=float(to_0_360(phase_rad)),
        amplitude=float(amplitude),
        core=core,
        pred_future=pred_future,
        core_times_ms=core_times_ms,
        future_times_ms=future_times_ms,
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

    Takes the ``window_ms`` before the cutoff, band-passes it, trims ``edge`` samples
    at both ends, fits a Yule-Walker AR(``ar_order``) model, forecasts forward, and reads
    the phase from a Hilbert transform over the last ``hilbert_window`` samples.
    Returns None when the parameters don't fit in the epoch or the estimate fails.
    """
    params = validate_causal_params(params)
    b = bandpass_fir(fs, band, int(params["filter_order"]))
    return _causal_phase_with_filter(x, times_ms, fs, b, cutoff_ms, params)


def validate_causal_params(params: dict[str, float | int]) -> dict[str, float | int]:
    required = ("window_ms", "filter_order", "edge", "ar_order", "hilbert_window", "offset")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing causal parameter(s): {missing}")
    return {
        "window_ms": float(params["window_ms"]),
        "filter_order": int(params["filter_order"]),
        "edge": int(params["edge"]),
        "ar_order": int(params["ar_order"]),
        "hilbert_window": int(params["hilbert_window"]),
        "offset": int(params["offset"]),
    }


def resolve_noncausal_filter_order(
    causal_params: dict[str, float | int] | None,
    default_filter_order: int,
) -> int:
    """Filter order for the non-causal (ground-truth) estimate.

    With causal params (manual or optimized) the non-causal estimate uses the same
    ``filter_order`` as the causal one, so both methods see an identical band-pass;
    the optimizer scores parameter sets the same way. Without causal estimation the
    default is used. In manual mode the causal ``filter_order`` normally equals the
    default, so this only differs per subject in optimize mode.
    """
    if causal_params is None:
        return int(default_filter_order)
    return int(causal_params["filter_order"])


def _limit_epochs(X: np.ndarray, n_epochs: int | bool) -> np.ndarray:
    return X if not n_epochs else X[:min(int(n_epochs), len(X))]


def signed_angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """Signed circular difference a-b in degrees, wrapped to [-180, 180)."""
    return float((a_deg - b_deg + 180.0) % 360.0 - 180.0)


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
    X = _limit_epochs(data.X, n_epochs)
    noncausal_filter = bandpass_fir(data.fs, band, filter_order)
    causal_filter = None
    if causal_params is not None:
        causal_params = validate_causal_params(causal_params)
        causal_filter = bandpass_fir(data.fs, band, int(causal_params["filter_order"]))

    estimates = []
    for i, (x, label) in enumerate(zip(X, data.labels)):
        est = _noncausal_phase_with_filter(x, data.times_ms, noncausal_filter, cutoff_ms)
        causal = None
        if causal_filter is not None:
            causal = _causal_phase_with_filter(x, data.times_ms, data.fs, causal_filter, cutoff_ms, causal_params)
            if causal is not None:
                causal.phase_error_deg = signed_angular_difference_deg(causal.phase_deg, est["phase_deg"])

        estimates.append(
            EpochEstimate(
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
            )
        )
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
    estimates_by_subject = {}
    for data in subject_data:
        causal_params = None
        if causal_estimation:
            if causal_params_mode == "manual":
                causal_params = validate_causal_params(manual_causal_params)
            elif causal_params_mode == "optimize":
                causal_params = optimize_causal_params(
                    data.X,
                    data.times_ms,
                    data.fs,
                    band=band,
                    cutoff_ms=cutoff_ms,
                    optimization_config=optimization_config,
                    n_epochs=n_epochs,
                    name=data.subject,
                )
            else:
                raise ValueError("causal_params_mode must be 'manual' or 'optimize'.")

        estimates_by_subject[data.subject] = estimate_subject_epochs(
            data,
            band=band,
            filter_order=resolve_noncausal_filter_order(causal_params, filter_order),
            cutoff_ms=cutoff_ms,
            n_epochs=n_epochs,
            causal_params=causal_params,
        )
    return estimates_by_subject


# ---- Causal parameter optimization --------------------------------------

def _circular_variance(errors_rad: Sequence[float]) -> float:
    return float(1.0 - np.abs(np.mean(np.exp(1j * np.asarray(errors_rad)))))


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
    params = validate_causal_params(params)
    cutoff = cutoff_index(times_ms, cutoff_ms)
    if not _causal_params_are_feasible(
        cutoff,
        fs,
        float(params["window_ms"]),
        int(params["filter_order"]),
        int(params["edge"]),
        int(params["ar_order"]),
    ):
        return None

    b = bandpass_fir(fs, band, int(params["filter_order"]))
    errors_rad = []
    for i in indices:
        try:
            causal = _causal_phase_with_filter(X[i], times_ms, fs, b, cutoff_ms, params)
            if causal is None:
                continue
            noncausal = _noncausal_phase_with_filter(X[i], times_ms, b, cutoff_ms)
        except Exception:
            continue
        errors_rad.append(causal.phase_rad - noncausal["phase_rad"])

    if len(errors_rad) < min_usable_epochs:
        return None

    circ_var = _circular_variance(errors_rad)
    return {
        "circular_variance": circ_var,
        "resultant_length": 1.0 - circ_var,
        "n_used": len(errors_rad),
        "n_total": len(indices),
    }


def _make_causal_optimization_objective(
    X: np.ndarray,
    train_idx: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    optimization_config: dict,
):
    def objective(trial):
        params = {
            "window_ms": trial.suggest_int(
                "window_ms",
                *optimization_config["window_ms_range"],
                step=optimization_config["window_ms_step"],
            ),
            "filter_order": trial.suggest_int(
                "filter_order",
                *optimization_config["filter_order_range"],
                step=optimization_config["filter_order_step"],
            ),
            "edge": trial.suggest_int(
                "edge",
                *optimization_config["edge_range"],
                step=optimization_config["edge_step"],
            ),
            "ar_order": trial.suggest_int("ar_order", *optimization_config["ar_order_range"]),
            "hilbert_window": optimization_config["hilbert_window"],
            "offset": optimization_config["offset"],
        }
        result = _evaluate_causal_params(
            X,
            train_idx,
            times_ms,
            fs,
            band,
            cutoff_ms,
            params,
            min_usable_epochs=optimization_config["min_usable_epochs"],
        )
        if result is None:
            return optimization_config["infeasible_penalty"]
        return result["circular_variance"]

    return objective


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

    X = _limit_epochs(X, n_epochs)
    if len(X) == 0:
        raise ValueError(f"No epochs available for {name}.")

    rng = np.random.default_rng(int(optimization_config["random_seed"]))
    shuffled = rng.permutation(len(X))
    if len(X) == 1:
        train_idx = shuffled
        test_idx = np.array([], dtype=int)
    else:
        n_train = round(len(X) * float(optimization_config["train_fraction"]))
        n_train = min(max(n_train, 1), len(X) - 1)
        train_idx = shuffled[:n_train]
        test_idx = shuffled[n_train:]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(optimization_config["random_seed"])),
    )
    study.optimize(
        _make_causal_optimization_objective(X, train_idx, times_ms, fs, band, cutoff_ms, optimization_config),
        n_trials=int(optimization_config["n_opt_trials"]),
    )

    best_params = validate_causal_params(
        {
            **study.best_params,
            "hilbert_window": optimization_config["hilbert_window"],
            "offset": optimization_config["offset"],
        }
    )
    train_result = _evaluate_causal_params(
        X,
        train_idx,
        times_ms,
        fs,
        band,
        cutoff_ms,
        best_params,
        min_usable_epochs=optimization_config["min_usable_epochs"],
    )
    if train_result is None:
        raise RuntimeError(f"No feasible causal AR parameters found for {name}.")

    test_result = None
    if len(test_idx):
        test_result = _evaluate_causal_params(
            X, test_idx, times_ms, fs, band, cutoff_ms, best_params, min_usable_epochs=1
        )

    print(f"\n[{name}] optimized causal AR params: {best_params}")
    print(
        "  train: "
        f"circular variance {train_result['circular_variance']:.4f}, "
        f"R={train_result['resultant_length']:.3f}, "
        f"n={train_result['n_used']}/{train_result['n_total']}"
    )
    if test_result is None:
        print("  held-out: unavailable / too few usable epochs")
    else:
        print(
            "  held-out: "
            f"circular variance {test_result['circular_variance']:.4f}, "
            f"R={test_result['resultant_length']:.3f}, "
            f"n={test_result['n_used']}/{test_result['n_total']}"
        )

    return best_params


# ---- Classification, scoring and deviations -----------------------------

def angular_distance_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute circular distance between two angles in degrees."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def classify_phase(
    phase_deg: float | None,
    tolerance_deg: float,
    labels: LabelScheme = DEFAULT_LABELS,
) -> str | None:
    """Positive if within +/-tolerance of 0 deg, negative if within +/-tolerance of 180 deg.

    Anything else is ``labels.unclassified``; a missing phase (None/NaN) returns None.
    """
    if phase_deg is None or (isinstance(phase_deg, float) and np.isnan(phase_deg)):
        return None
    positive_distance = angular_distance_deg(phase_deg, 0.0)
    negative_distance = angular_distance_deg(phase_deg, 180.0)
    is_positive = positive_distance <= tolerance_deg
    is_negative = negative_distance <= tolerance_deg

    if is_positive and is_negative:
        return labels.positive if positive_distance <= negative_distance else labels.negative
    if is_positive:
        return labels.positive
    if is_negative:
        return labels.negative
    return labels.unclassified


def comparison_status(boss_label: str, phase_class: str | None, labels: LabelScheme = DEFAULT_LABELS) -> str:
    """``correct`` / ``wrong`` / ``unclassified`` (or missing phase) / ``unknown`` (no BOSS label)."""
    if boss_label not in (labels.positive, labels.negative):
        return "unknown"
    if phase_class is None or phase_class == labels.unclassified:
        return "unclassified"
    return "correct" if phase_class == boss_label else "wrong"


def chance_success_pct(tolerance_deg: float) -> float:
    """Chance of a uniformly random phase landing in the correct +/-T window: 2T/360."""
    return 100.0 * 2.0 * tolerance_deg / 360.0


def score_vs_boss(
    boss_labels: Sequence[str],
    phases_deg: Sequence[float | None],
    tolerance_deg: float,
    labels: LabelScheme = DEFAULT_LABELS,
) -> dict[str, float | int | None]:
    """BOSS success at one tolerance.

    Only BOSS-labeled (positive/negative) epochs count. ``success_pct`` =
    correct / n_labeled * 100, so unclassified epochs count as failures.
    ``by_class`` gives n and correct separately for BOSS positive and negative.
    """
    counts = {"correct": 0, "wrong": 0, "unclassified": 0}
    by_class = {name: {"n": 0, "correct": 0} for name in (labels.positive, labels.negative)}
    for boss, phase in zip(boss_labels, phases_deg):
        status = comparison_status(boss, classify_phase(phase, tolerance_deg, labels), labels)
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


def phase_deviation_deg(phase_deg: float, target_deg: float) -> float:
    """Signed deviation of a phase from a target, wrapped to [-180, 180)."""
    return signed_angular_difference_deg(phase_deg, target_deg)


def boss_target_deviations(
    boss_labels: Sequence[str],
    phases_deg: Sequence[float | None],
    labels: LabelScheme = DEFAULT_LABELS,
) -> np.ndarray:
    """Signed deviation of each phase from its BOSS target (NaN for unknown label or phase)."""
    out = np.full(len(boss_labels), np.nan)
    for i, (boss, phase) in enumerate(zip(boss_labels, phases_deg)):
        target = labels.target_deg(boss)
        if target is not None and phase is not None and not np.isnan(phase):
            out[i] = phase_deviation_deg(phase, target)
    return out


def circular_stats(deviations_deg: Sequence[float], tolerances_deg: Sequence[float] = ()) -> dict:
    """Circular summary of signed deviations (NaNs ignored).

    ``mean_deg``: circular mean (bias); ``R``: mean resultant length (1 = all equal);
    ``sd_deg``: circular SD = sqrt(-2 ln R); ``pct_within``: % with |deviation| <= T.
    """
    values = np.asarray(deviations_deg, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean_deg": None, "sd_deg": None, "R": None, "pct_within": {str(t): None for t in tolerances_deg}}
    vector = np.mean(np.exp(1j * np.radians(values)))
    R = float(np.abs(vector))
    return {
        "n": n,
        "mean_deg": signed_angular_difference_deg(float(np.degrees(np.angle(vector))), 0.0),
        "sd_deg": float(np.degrees(np.sqrt(-2.0 * np.log(R)))) if R > 0 else None,
        "R": R,
        "pct_within": {str(t): float(100.0 * np.mean(np.abs(values) <= t)) for t in tolerances_deg},
    }


def deviation_histogram(
    deviations_deg: Sequence[float],
    boss_labels: Sequence[str],
    bin_width_deg: float = 10.0,
    labels: LabelScheme = DEFAULT_LABELS,
) -> dict:
    """Counts per bin from -180 to 180, split by BOSS class (NaNs dropped)."""
    edges = np.arange(-180.0, 180.0 + bin_width_deg / 2, bin_width_deg)
    values = np.asarray(deviations_deg, dtype=float)
    boss = np.asarray(boss_labels, dtype=object)
    keep = ~np.isnan(values)
    counts = {}
    for class_name in labels.class_order:
        selected = values[keep & (boss == class_name)]
        counts[class_name] = np.histogram(selected, bins=edges)[0].astype(int).tolist()
    return {"bin_edges_deg": edges.tolist(), "counts_by_class": counts}


def deviation_summary(
    deviations_deg: Sequence[float],
    boss_labels: Sequence[str],
    tolerances_deg: Sequence[float],
    bin_width_deg: float = 10.0,
    labels: LabelScheme = DEFAULT_LABELS,
) -> dict:
    """Histogram + circular stats for all epochs (``stats``) and per BOSS class (``stats_by_class``).

    Per-class stats matter because opposite biases for positive and negative epochs
    would cancel in the pooled mean.
    """
    values = np.asarray(deviations_deg, dtype=float)
    boss = np.asarray(boss_labels, dtype=object)
    return {
        **deviation_histogram(values, boss, bin_width_deg, labels),
        "stats": circular_stats(values, tolerances_deg),
        "stats_by_class": {
            name: circular_stats(values[boss == name], tolerances_deg) for name in labels.class_order
        },
    }


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def boss_success_by_tolerance(
    subjects: dict[str, tuple[Sequence[str], Sequence[float]]],
    tolerances_deg: Sequence[float],
    labels: LabelScheme = DEFAULT_LABELS,
) -> dict:
    """Per-subject non-causal BOSS success for each tolerance, plus mean and sample SD across subjects.

    ``subjects`` maps subject id -> (BOSS labels, non-causal phases in deg).
    """
    result = {"subjects": list(subjects), "tolerances_deg": list(tolerances_deg), "by_tolerance": {}}
    for tol in tolerances_deg:
        scores = [score_vs_boss(boss, phases, tol, labels) for boss, phases in subjects.values()]
        pct = [s["success_pct"] for s in scores]
        valid = [p for p in pct if p is not None]
        result["by_tolerance"][str(tol)] = {
            "success_pct": pct,
            "correct": [s["correct"] for s in scores],
            "n_labeled": [s["n_labeled"] for s in scores],
            "mean_pct": float(np.mean(valid)) if valid else None,
            "sd_pct": _sample_sd(valid),
            "chance_pct": chance_success_pct(tol),
        }
    return result


def analyze_phase_results(
    subjects: dict[str, dict[str, Sequence]],
    tolerances_deg: Sequence[float],
    bin_width_deg: float = 10.0,
    labels: LabelScheme = DEFAULT_LABELS,
) -> dict:
    """Everything the report shows, precomputed for every tolerance.

    ``subjects`` maps subject id -> dict with per-epoch sequences ``boss`` (labels),
    ``noncausal_deg``, ``causal_deg`` and ``causal_error_deg`` (None where missing).

    Returns per subject: per-epoch class/status for each method and tolerance, counts,
    deviation histograms + circular stats (``causal_error`` = causal - non-causal,
    ``boss_target`` = non-causal - BOSS target); the same histograms pooled over
    subjects; and ``success`` from ``boss_success_by_tolerance``.
    """
    out = {"tolerances_deg": list(tolerances_deg), "subjects": {}, "pooled": {}}
    pooled = {kind: ([], []) for kind in DEVIATION_KINDS}

    for subject_id, data in subjects.items():
        boss = list(data["boss"])
        phases = {"noncausal": list(data["noncausal_deg"]), "causal": list(data["causal_deg"])}
        subject_out = {"classes": {}, "status": {}, "counts": {}, "deviations": {}}
        for method in METHODS:
            subject_out["classes"][method] = {}
            subject_out["status"][method] = {}
            subject_out["counts"][method] = {}
            for tol in tolerances_deg:
                classes = [classify_phase(p, tol, labels) for p in phases[method]]
                subject_out["classes"][method][str(tol)] = classes
                subject_out["status"][method][str(tol)] = [
                    comparison_status(b, c, labels) for b, c in zip(boss, classes)
                ]
                subject_out["counts"][method][str(tol)] = score_vs_boss(boss, phases[method], tol, labels)

        deviations = {
            "causal_error": np.array([np.nan if v is None else v for v in data["causal_error_deg"]], dtype=float),
            "boss_target": boss_target_deviations(boss, phases["noncausal"], labels),
        }
        for kind, values in deviations.items():
            subject_out["deviations"][kind] = deviation_summary(values, boss, tolerances_deg, bin_width_deg, labels)
            pooled[kind][0].extend(values.tolist())
            pooled[kind][1].extend(boss)
        out["subjects"][subject_id] = subject_out

    for kind, (values, boss) in pooled.items():
        out["pooled"][kind] = deviation_summary(values, boss, tolerances_deg, bin_width_deg, labels)

    out["success"] = boss_success_by_tolerance(
        {sid: (data["boss"], data["noncausal_deg"]) for sid, data in subjects.items()},
        tolerances_deg,
        labels,
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


# ---- Static figures (notebook + publication export) ---------------------

def plot_epoch_signal(
    estimate: EpochEstimate,
    cutoff_ms: float,
    band: tuple[float, float],
    window_ms: tuple[float, float] | None = None,
    ax=None,
):
    """Raw signal, non-causal filtered trace and (if available) causal AR core + forecast."""
    times_ms = estimate.times_ms
    plot_mask = _time_window_mask(times_ms, window_ms)
    fig, ax = _figure_axes(ax, figsize=(10, 4.5))
    ax.plot(times_ms[plot_mask], estimate.raw[plot_mask], color=METHOD_COLORS["raw"], lw=0.85,
            label=f"raw ({estimate.channel})")
    ax.plot(
        times_ms[plot_mask],
        estimate.filtered[plot_mask],
        color=METHOD_COLORS["noncausal"],
        lw=1.35,
        label=f"non-causal filtered ({band[0]:.0f}-{band[1]:.0f} Hz)",
    )

    causal = estimate.causal
    if causal is not None:
        core_mask = _time_window_mask(causal.core_times_ms, window_ms)
        ax.plot(causal.core_times_ms[core_mask], causal.core[core_mask], color=METHOD_COLORS["causal"],
                lw=1.2, label="causal AR core")
        future_mask = _time_window_mask(causal.future_times_ms, window_ms)
        ax.plot(causal.future_times_ms[future_mask], causal.pred_future[future_mask], color=METHOD_COLORS["causal"],
                lw=1.2, ls="--", label="causal AR predicted")

    ax.axvline(cutoff_ms, color="gray", ls=":", lw=1.2)
    ax.plot([], [], " ", label=f"BOSS: {estimate.label}")
    ax.plot([], [], " ", label=f"non-causal: {estimate.phase_deg:.1f} deg")
    if causal is not None:
        ax.plot([], [], " ", label=f"causal: {causal.phase_deg:.1f} deg")
        ax.plot([], [], " ", label=f"causal - non-causal: {causal.phase_error_deg:+.1f} deg")
    ax.set_title(f"{estimate.subject} epoch {estimate.epoch_index + 1}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


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
    t = times_ms[mask]
    raw = x[mask] * 1e6
    filtered = noncausal["filtered"][mask] * 1e6

    axes[0].plot(t, raw, color=METHOD_COLORS["raw"], lw=0.8)
    axes[0].set_title("1. Raw signal", fontsize=10, loc="left")

    axes[1].plot(t, raw, color=METHOD_COLORS["raw"], lw=0.6, alpha=0.6, label="raw")
    axes[1].plot(t, filtered, color=METHOD_COLORS["noncausal"], lw=1.6,
                 label=f"non-causal filtfilt {band[0]:g}-{band[1]:g} Hz")
    axes[1].set_title(f"2. Non-causal (uses the whole epoch): phase at t=0 = {noncausal['phase_deg']:.1f} deg",
                      fontsize=10, loc="left")

    axes[2].plot(t, filtered, color=METHOD_COLORS["noncausal"], lw=1.4, label="non-causal (reference)")
    if causal is not None:
        core_mask = _time_window_mask(causal.core_times_ms, window_ms)
        future_mask = _time_window_mask(causal.future_times_ms, window_ms)
        axes[2].plot(causal.core_times_ms[core_mask], causal.core[core_mask] * 1e6,
                     color=METHOD_COLORS["causal"], lw=1.6, label="causal: filtered pre-stimulus core")
        axes[2].plot(causal.future_times_ms[future_mask], causal.pred_future[future_mask] * 1e6,
                     color=METHOD_COLORS["causal"], lw=1.6, ls="--", label="causal: AR forecast")
        diff = signed_angular_difference_deg(causal.phase_deg, noncausal["phase_deg"])
        axes[2].set_title(f"3. Causal (only data before t=0): phase = {causal.phase_deg:.1f} deg "
                          f"(causal - non-causal = {diff:+.1f} deg)", fontsize=10, loc="left")
    else:
        axes[2].set_title("3. Causal: unavailable for these parameters", fontsize=10, loc="left")

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
    labels: LabelScheme = DEFAULT_LABELS,
    ax=None,
):
    """Static polar plot: one dot per epoch at its phase, colored by BOSS label; shaded +/-T windows."""
    fig, ax = _figure_axes(ax, figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title(title)
    for center in (0.0, 180.0):
        start, end = (center - tolerance_deg) % 360.0, (center + tolerance_deg) % 360.0
        spans = [(start, 360.0), (0.0, end)] if start > end else [(start, end)]
        for a, b in spans:
            ax.axvspan(np.radians(a), np.radians(b), color="0.85", alpha=0.6, zorder=0)
        ax.axvline(np.radians(center), color="black", ls=":", lw=1, alpha=0.55)
    for class_name in labels.class_order:
        selected = [p for p, b in zip(phases_deg, boss_labels) if b == class_name and p is not None]
        if selected:
            radius = np.linspace(1, 2, len(selected))
            ax.scatter(np.radians(selected), radius, s=26, color=CLASS_COLORS.get(class_name, "0.5"),
                       alpha=0.85, label=f"BOSS {class_name} (n={len(selected)})")
    ax.set_ylim(0, 2.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    fig.tight_layout()
    return fig


def plot_deviation_histogram(
    histogram: dict,
    tolerance_deg: float,
    title: str = "",
    labels: LabelScheme = DEFAULT_LABELS,
    ax=None,
):
    """Circular (rose) histogram from ``deviation_histogram`` / ``analyze_phase_results``.

    0 deg (no deviation) is at the top and positive deviations run clockwise. Wedges are
    stacked by BOSS class, wedge length = epoch count. The shaded sector is +/-tolerance.
    One vector (line ending in a dot) per BOSS class is that class's mean resultant
    vector: direction = circular mean (bias), length = R (1 = all deviations identical) relative to the outer ring.
    The title gives the pooled statistics.
    """
    fig, ax = _figure_axes(ax, figsize=(5.2, 5.4), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    edges = np.asarray(histogram["bin_edges_deg"])
    width = np.radians(edges[1] - edges[0])
    totals = sum(np.asarray(c) for c in histogram["counts_by_class"].values())
    r_max = max(float(np.max(totals)), 1.0) * 1.08

    ax.bar(0.0, r_max, width=np.radians(2 * tolerance_deg), color="0.9", zorder=0,
           label=f"+/-{tolerance_deg:g} deg")
    bottom = np.zeros(len(edges) - 1)
    for class_name in labels.class_order:
        counts = np.asarray(histogram["counts_by_class"].get(class_name, []))
        if counts.sum() == 0:
            continue
        ax.bar(np.radians(edges[:-1]), counts, width=width, bottom=bottom, align="edge",
               color=CLASS_COLORS.get(class_name, "0.5"), edgecolor="white", lw=0.5, zorder=2,
               label=f"BOSS {class_name}")
        bottom += counts

    for class_name, class_stats in histogram.get("stats_by_class", {}).items():
        if not class_stats["n"]:
            continue
        color = CLASS_COLORS.get(class_name, "0.5")
        theta = np.radians(class_stats["mean_deg"])
        radius = class_stats["R"] * r_max
        ax.plot([theta, theta], [0, radius], color="white", lw=6, solid_capstyle="round", zorder=4)
        ax.plot([theta, theta], [0, radius], color=color, lw=3, solid_capstyle="round", zorder=5,
                marker="o", markevery=[1], markersize=8, markeredgecolor="white", markeredgewidth=1.2)
        ax.plot([], [], color=color, lw=3, label=f"mean vector {class_name}")

    stats = histogram.get("stats")
    if stats and stats["n"]:
        within = stats["pct_within"].get(str(tolerance_deg))
        text = f"n={stats['n']}  mean={stats['mean_deg']:+.1f} deg  R={stats['R']:.2f}\ncirc SD={stats['sd_deg']:.1f} deg"
        if within is not None:
            text += f"  within +/-{tolerance_deg:g}: {within:.0f}%"
        ax.set_title(f"{title}\n{text}" if title else text, fontsize=9, pad=14)
    ax.set_ylim(0, r_max)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0", "+45", "+90", "+135", "\u00b1180", "-135", "-90", "-45"])
    ax.set_rlabel_position(112.5)
    ax.tick_params(axis="y", labelsize=7, colors="0.4")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.92, 1.12), frameon=False)
    fig.tight_layout()
    return fig


def plot_success_vs_tolerance(
    success: dict,
    highlight_subject: str | None = None,
    ax=None,
    jitter: float = 0.09,
):
    """% BOSS success vs tolerance: one dot per subject, paired lines, mean +/- SD, chance line.

    ``success`` is the output of ``boss_success_by_tolerance`` (tolerances plotted in
    the given order). ``highlight_subject`` draws that subject in black and dims the
    rest. Jitter is deterministic so the figure is reproducible.
    """
    fig, ax = _figure_axes(ax, figsize=(4.6, 4.4))
    tolerances = success["tolerances_deg"]
    subjects = success["subjects"]
    n_subjects = len(subjects)
    offsets = np.linspace(-jitter, jitter, n_subjects) if n_subjects > 1 else np.zeros(1)
    xs = np.arange(len(tolerances))
    by_tol = [success["by_tolerance"][str(t)] for t in tolerances]
    dim = highlight_subject in subjects

    for s, subject in enumerate(subjects):
        ys = [entry["success_pct"][s] for entry in by_tol]
        if subject == highlight_subject:
            ax.plot(xs + offsets[s], ys, color="black", lw=1.8, zorder=4)
            ax.scatter(xs + offsets[s], ys, s=46, color="black", zorder=5, edgecolors="white", linewidths=0.8,
                       label=subject)
            continue
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
    ax.set_title(f"BOSS phase accuracy (non-causal, n={n_subjects} subjects)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def _figure_axes(ax, figsize, subplot_kw=None):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize, subplot_kw=subplot_kw)


def _time_window_mask(times_ms: np.ndarray, window_ms: tuple[float, float] | None) -> np.ndarray:
    if window_ms is None:
        return np.ones_like(times_ms, dtype=bool)
    return (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])
