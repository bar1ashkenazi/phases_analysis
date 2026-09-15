"""Reusable helpers for intake-session phase/BOSS comparison.

The non-causal phase estimate intentionally follows the working logic in
``pipeline.py``: zero-phase FIR filtering across the full epoch, Hilbert transform,
then phase/amplitude read out at the sample nearest t=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import filtfilt, hilbert

from phastimate import design_bandpass, phastimate


@dataclass
class SubjectData:
    subject: str
    epochs: mne.Epochs
    fs: float
    data: np.ndarray
    times_ms: np.ndarray
    conditions: np.ndarray
    condition_column: str
    channel: str


@dataclass
class TrialEstimate:
    subject: str
    epoch_index: int
    condition: str
    noncausal_class: str
    phase_rad: float
    phase_deg: float
    amplitude: float
    raw: np.ndarray
    filtered: np.ndarray
    times_ms: np.ndarray
    channel: str
    causal_class: str | None = None
    causal_phase_rad: float | None = None
    causal_phase_deg: float | None = None
    causal_amplitude: float | None = None
    causal_core: np.ndarray | None = None
    causal_pred_future: np.ndarray | None = None
    causal_core_times_ms: np.ndarray | None = None
    causal_future_times_ms: np.ndarray | None = None
    causal_phase_error_deg: float | None = None
    causal_params: dict[str, float | int] | None = None


def as_subject_list(subjects: str | Sequence[str]) -> list[str]:
    if isinstance(subjects, str):
        return [subjects]
    return list(subjects)


def intake_epoch_path(subject: str, data_root: Path, intake_filename: str) -> Path:
    return data_root / subject / "EEG" / "processed" / intake_filename.format(subject=subject)


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


def normalize_boss_label(
    value,
    positive_labels: set[str],
    negative_labels: set[str],
    positive_name: str,
    negative_name: str,
    unknown_name: str,
) -> str:
    """Map raw metadata values to positive/negative/unknown."""
    if value is None:
        return unknown_name

    if isinstance(value, (bool, np.bool_)):
        return positive_name if value else negative_name

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return unknown_name
        if np.isclose(value, 1):
            return positive_name
        if np.isclose(value, 0):
            return negative_name

    normalized = str(value).strip().lower()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    if normalized in positive_labels:
        return positive_name
    if normalized in negative_labels:
        return negative_name
    return unknown_name


def available_intake_subjects(data_root: Path, intake_filename: str) -> list[str]:
    if not data_root.exists():
        return []
    subjects = []
    for path in sorted(data_root.glob("sub_*")):
        if intake_epoch_path(path.name, data_root, intake_filename).exists():
            subjects.append(path.name)
    return subjects


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


def _resolve_condition_column(
    epochs: mne.Epochs,
    requested: str,
    positive_labels: set[str],
    negative_labels: set[str],
    positive_name: str,
    negative_name: str,
    unknown_name: str,
    auto_keywords: Sequence[str],
) -> str:
    if epochs.metadata is None:
        raise ValueError("The epochs file has no metadata, so BOSS labels cannot be read.")
    if requested in epochs.metadata.columns:
        return requested

    candidates = [
        column for column in epochs.metadata.columns
        if any(key in column.lower() for key in auto_keywords)
    ]
    for column in candidates:
        mapped = epochs.metadata[column].map(
            lambda value: normalize_boss_label(
                value,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                positive_name=positive_name,
                negative_name=negative_name,
                unknown_name=unknown_name,
            )
        )
        if mapped.isin([positive_name, negative_name]).any():
            print(f"Using metadata column {column!r} for BOSS labels.")
            return column

    raise ValueError(
        f"Could not find BOSS label column {requested!r}. "
        f"Available columns: {list(epochs.metadata.columns)}"
    )


def load_intake_subject(
    subject: str,
    data_root: Path,
    intake_filename: str,
    channel: str,
    condition_column: str,
    lowpass_before_downsample_hz: float | None,
    downsample: bool,
    downsample_fs: float,
    show_metadata_summary: bool,
    metadata_max_values: int,
    hjorth_channel: str,
    hjorth_weights: dict[str, float],
    hjorth_scale_reference: str,
    positive_labels: set[str],
    negative_labels: set[str],
    positive_name: str,
    negative_name: str,
    unknown_name: str,
    condition_auto_keywords: Sequence[str],
) -> SubjectData:
    epo_path = intake_epoch_path(subject, data_root, intake_filename)
    if not epo_path.exists():
        raise FileNotFoundError(f"Intake epochs not found for {subject}: {epo_path}")

    epochs = mne.read_epochs(epo_path, preload=True)
    if lowpass_before_downsample_hz is not None:
        epochs.filter(l_freq=None, h_freq=lowpass_before_downsample_hz, picks="eeg", verbose=False)
    if downsample:
        epochs.resample(downsample_fs)
    if channel == hjorth_channel:
        epochs = add_weighted_hjorth_channel(
            epochs,
            output_channel=hjorth_channel,
            weights_by_channel=hjorth_weights,
            scale_reference_channel=hjorth_scale_reference,
        )

    if show_metadata_summary:
        print(f"\n[{subject}] {epo_path}")
        summarize_metadata(epochs, metadata_max_values)

    resolved_condition_column = _resolve_condition_column(
        epochs,
        requested=condition_column,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        positive_name=positive_name,
        negative_name=negative_name,
        unknown_name=unknown_name,
        auto_keywords=condition_auto_keywords,
    )
    conditions = epochs.metadata[resolved_condition_column].map(
        lambda value: normalize_boss_label(
            value,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            positive_name=positive_name,
            negative_name=negative_name,
            unknown_name=unknown_name,
        )
    ).to_numpy()

    fs = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=channel)[:, 0, :]
    times_ms = epochs.times * 1000.0

    return SubjectData(
        subject=subject,
        epochs=epochs,
        fs=fs,
        data=data,
        times_ms=times_ms,
        conditions=conditions,
        condition_column=resolved_condition_column,
        channel=channel,
    )


def load_intake_subjects(
    subjects: str | Sequence[str],
    data_root: Path,
    intake_filename: str,
    channel: str,
    condition_column: str,
    lowpass_before_downsample_hz: float | None,
    downsample: bool,
    downsample_fs: float,
    show_metadata_summary: bool,
    metadata_max_values: int,
    hjorth_channel: str,
    hjorth_weights: dict[str, float],
    hjorth_scale_reference: str,
    positive_labels: set[str],
    negative_labels: set[str],
    positive_name: str,
    negative_name: str,
    unknown_name: str,
    condition_auto_keywords: Sequence[str],
) -> list[SubjectData]:
    return [
        load_intake_subject(
            subject,
            data_root=data_root,
            intake_filename=intake_filename,
            channel=channel,
            condition_column=condition_column,
            lowpass_before_downsample_hz=lowpass_before_downsample_hz,
            downsample=downsample,
            downsample_fs=downsample_fs,
            show_metadata_summary=show_metadata_summary,
            metadata_max_values=metadata_max_values,
            hjorth_channel=hjorth_channel,
            hjorth_weights=hjorth_weights,
            hjorth_scale_reference=hjorth_scale_reference,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            positive_name=positive_name,
            negative_name=negative_name,
            unknown_name=unknown_name,
            condition_auto_keywords=condition_auto_keywords,
        )
        for subject in as_subject_list(subjects)
    ]


def to_0_360(phase_rad: float | np.ndarray) -> float | np.ndarray:
    """0-360 deg, where 0 is a positive peak and 180 is a negative peak."""
    return np.degrees(phase_rad) % 360.0


def cutoff_index(times_ms: np.ndarray, cutoff_ms: float) -> int:
    return int(np.argmin(np.abs(times_ms - cutoff_ms)))


def _noncausal_phase_estimate_with_filter(
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


def noncausal_phase_estimate(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
) -> dict:
    """Estimate offline/non-causal phase at cutoff using filtfilt + Hilbert."""
    b = design_bandpass(filter_order, band[0], band[1], fs)
    return _noncausal_phase_estimate_with_filter(x, times_ms, b, cutoff_ms)


def _causal_core_times(
    times_ms: np.ndarray,
    cutoff: int,
    window_samples: int,
    edge: int,
) -> np.ndarray:
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


def _causal_phase_estimate_with_filter(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    bandpass_filter: np.ndarray,
    cutoff_ms: float,
    params: dict[str, float | int],
) -> dict | None:
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
    return {
        "phase_rad": float(phase_rad),
        "phase_deg": float(to_0_360(phase_rad)),
        "amplitude": float(amplitude),
        "core": core,
        "pred_future": pred_future,
        "core_times_ms": core_times_ms,
        "future_times_ms": future_times_ms,
        "params": dict(params),
    }


def causal_phase_estimate(
    x: np.ndarray,
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    params: dict[str, float | int],
) -> dict | None:
    """Estimate online/causal phase at cutoff using phastimate's AR forecast."""
    b = design_bandpass(int(params["filter_order"]), band[0], band[1], fs)
    return _causal_phase_estimate_with_filter(x, times_ms, fs, b, cutoff_ms, params)


def signed_angular_difference_deg(a_deg: float, b_deg: float) -> float:
    """Signed circular difference a-b in degrees, wrapped to (-180, 180]."""
    return float((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _circular_variance(errors_rad: Sequence[float]) -> float:
    return float(1.0 - np.abs(np.mean(np.exp(1j * np.asarray(errors_rad)))))


def _validate_causal_params(params: dict[str, float | int]) -> dict[str, float | int]:
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


def _evaluate_causal_params(
    data: np.ndarray,
    indices: Sequence[int],
    times_ms: np.ndarray,
    fs: float,
    band: tuple[float, float],
    cutoff_ms: float,
    params: dict[str, float | int],
    min_usable_trials: int,
) -> dict | None:
    params = _validate_causal_params(params)
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

    b = design_bandpass(int(params["filter_order"]), band[0], band[1], fs)
    errors_rad = []
    for i in indices:
        try:
            causal = _causal_phase_estimate_with_filter(data[i], times_ms, fs, b, cutoff_ms, params)
            if causal is None:
                continue
            noncausal = _noncausal_phase_estimate_with_filter(data[i], times_ms, b, cutoff_ms)
        except Exception:
            continue
        errors_rad.append(causal["phase_rad"] - noncausal["phase_rad"])

    if len(errors_rad) < min_usable_trials:
        return None

    circ_var = _circular_variance(errors_rad)
    return {
        "circular_variance": circ_var,
        "resultant_length": 1.0 - circ_var,
        "n_used": len(errors_rad),
        "n_total": len(indices),
    }


def _make_causal_optimization_objective(
    data: np.ndarray,
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
            data,
            train_idx,
            times_ms,
            fs,
            band,
            cutoff_ms,
            params,
            min_usable_trials=optimization_config["min_usable_trials"],
        )
        if result is None:
            return optimization_config["infeasible_penalty"]
        return result["circular_variance"]

    return objective


def optimize_causal_params_for_subject(
    subject_data: SubjectData,
    band: tuple[float, float],
    cutoff_ms: float,
    n_trials: int | bool,
    optimization_config: dict,
) -> dict[str, float | int]:
    """Choose one causal AR parameter set for a subject against non-causal phase."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optimization mode requires optuna. Install the project requirements first.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(subject_data.data) if not n_trials else min(int(n_trials), len(subject_data.data))
    data = subject_data.data[:n]
    if len(data) == 0:
        raise ValueError(f"No trials available for {subject_data.subject}.")

    rng = np.random.default_rng(int(optimization_config["random_seed"]))
    shuffled = rng.permutation(len(data))
    if len(data) == 1:
        train_idx = shuffled
        test_idx = np.array([], dtype=int)
    else:
        n_train = round(len(data) * float(optimization_config["train_fraction"]))
        n_train = min(max(n_train, 1), len(data) - 1)
        train_idx = shuffled[:n_train]
        test_idx = shuffled[n_train:]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(optimization_config["random_seed"])),
    )
    study.optimize(
        _make_causal_optimization_objective(
            data,
            train_idx,
            subject_data.times_ms,
            subject_data.fs,
            band,
            cutoff_ms,
            optimization_config,
        ),
        n_trials=int(optimization_config["n_opt_trials"]),
    )

    best_params = _validate_causal_params(
        {
            **study.best_params,
            "hilbert_window": optimization_config["hilbert_window"],
            "offset": optimization_config["offset"],
        }
    )
    train_result = _evaluate_causal_params(
        data,
        train_idx,
        subject_data.times_ms,
        subject_data.fs,
        band,
        cutoff_ms,
        best_params,
        min_usable_trials=optimization_config["min_usable_trials"],
    )
    if train_result is None:
        raise RuntimeError(f"No feasible causal AR parameters found for {subject_data.subject}.")

    test_result = None
    if len(test_idx):
        test_result = _evaluate_causal_params(
            data,
            test_idx,
            subject_data.times_ms,
            subject_data.fs,
            band,
            cutoff_ms,
            best_params,
            min_usable_trials=1,
        )

    print(f"\n[{subject_data.subject}] optimized causal AR params: {best_params}")
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


def estimate_subject_trials(
    subject_data: SubjectData,
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
    n_trials: int | bool,
    phase_class_tolerance_deg: float,
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
    causal_params: dict[str, float | int] | None,
) -> list[TrialEstimate]:
    n = len(subject_data.data) if not n_trials else min(int(n_trials), len(subject_data.data))
    estimates = []
    noncausal_filter = design_bandpass(filter_order, band[0], band[1], subject_data.fs)
    causal_filter = None
    if causal_params is not None:
        causal_params = _validate_causal_params(causal_params)
        causal_filter = design_bandpass(int(causal_params["filter_order"]), band[0], band[1], subject_data.fs)

    for i, (x, condition) in enumerate(zip(subject_data.data[:n], subject_data.conditions[:n])):
        est = _noncausal_phase_estimate_with_filter(
            x,
            subject_data.times_ms,
            noncausal_filter,
            cutoff_ms,
        )
        noncausal_class = phase_class(
            est["phase_deg"],
            tolerance_deg=phase_class_tolerance_deg,
            positive_name=positive_name,
            negative_name=negative_name,
            unclassified_name=unclassified_name,
        )
        causal_est = None
        causal_class = None
        causal_phase_error_deg = None
        if causal_params is not None and causal_filter is not None:
            causal_est = _causal_phase_estimate_with_filter(
                x,
                subject_data.times_ms,
                subject_data.fs,
                causal_filter,
                cutoff_ms,
                causal_params,
            )
            if causal_est is not None:
                causal_class = phase_class(
                    causal_est["phase_deg"],
                    tolerance_deg=phase_class_tolerance_deg,
                    positive_name=positive_name,
                    negative_name=negative_name,
                    unclassified_name=unclassified_name,
                )
                causal_phase_error_deg = signed_angular_difference_deg(
                    causal_est["phase_deg"],
                    est["phase_deg"],
                )

        estimates.append(
            TrialEstimate(
                subject=subject_data.subject,
                epoch_index=i,
                condition=condition,
                noncausal_class=noncausal_class,
                phase_rad=est["phase_rad"],
                phase_deg=est["phase_deg"],
                amplitude=est["amplitude"],
                raw=x,
                filtered=est["filtered"],
                times_ms=subject_data.times_ms,
                channel=subject_data.channel,
                causal_class=causal_class,
                causal_phase_rad=None if causal_est is None else causal_est["phase_rad"],
                causal_phase_deg=None if causal_est is None else causal_est["phase_deg"],
                causal_amplitude=None if causal_est is None else causal_est["amplitude"],
                causal_core=None if causal_est is None else causal_est["core"],
                causal_pred_future=None if causal_est is None else causal_est["pred_future"],
                causal_core_times_ms=None if causal_est is None else causal_est["core_times_ms"],
                causal_future_times_ms=None if causal_est is None else causal_est["future_times_ms"],
                causal_phase_error_deg=causal_phase_error_deg,
                causal_params=None if causal_params is None else dict(causal_params),
            )
        )
    return estimates


def estimate_all_subjects(
    subject_data: Iterable[SubjectData],
    band: tuple[float, float],
    filter_order: int,
    cutoff_ms: float,
    n_trials: int | bool,
    phase_class_tolerance_deg: float,
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
    causal_estimation: bool,
    causal_params_mode: str,
    manual_causal_params: dict[str, float | int],
    optimization_config: dict,
) -> dict[str, list[TrialEstimate]]:
    estimates_by_subject = {}
    for data in subject_data:
        causal_params = None
        subject_filter_order = filter_order
        if causal_estimation:
            if causal_params_mode == "manual":
                causal_params = _validate_causal_params(manual_causal_params)
            elif causal_params_mode == "optimize":
                causal_params = optimize_causal_params_for_subject(
                    data,
                    band=band,
                    cutoff_ms=cutoff_ms,
                    n_trials=n_trials,
                    optimization_config=optimization_config,
                )
            else:
                raise ValueError("causal_params_mode must be 'manual' or 'optimize'.")
            subject_filter_order = int(causal_params["filter_order"])

        estimates_by_subject[data.subject] = estimate_subject_trials(
            data,
            band=band,
            filter_order=subject_filter_order,
            cutoff_ms=cutoff_ms,
            n_trials=n_trials,
            phase_class_tolerance_deg=phase_class_tolerance_deg,
            positive_name=positive_name,
            negative_name=negative_name,
            unclassified_name=unclassified_name,
            causal_params=causal_params,
        )
    return estimates_by_subject


def print_phase_summary(
    estimates_by_subject: dict[str, list[TrialEstimate]],
    class_order: Sequence[str],
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> None:
    for subject, estimates in estimates_by_subject.items():
        has_causal = any(est.causal_phase_deg is not None for est in estimates)
        title = "phase estimates" if has_causal else "non-causal phase estimates"
        print(f"\n[{subject}] {title}: n={len(estimates)}")
        for condition in class_order:
            selected = [est.phase_deg for est in estimates if est.condition == condition]
            if selected:
                print(
                    f"  {condition:>8s}: n={len(selected)}, "
                    f"non-causal mean={circular_mean_deg(selected):.1f} deg"
                )
                causal_selected = [
                    est.causal_phase_deg for est in estimates
                    if est.condition == condition and est.causal_phase_deg is not None
                ]
                if causal_selected:
                    print(f"            causal mean={circular_mean_deg(causal_selected):.1f} deg")
        for phase_class_name in (positive_name, negative_name, unclassified_name):
            selected = [est for est in estimates if est.noncausal_class == phase_class_name]
            if selected:
                print(f"  non-causal {phase_class_name:>12s}: n={len(selected)}")
            if has_causal:
                causal_selected = [est for est in estimates if est.causal_class == phase_class_name]
                if causal_selected:
                    print(f"      causal {phase_class_name:>12s}: n={len(causal_selected)}")
        counts = boss_vs_noncausal_counts(estimates, positive_name, negative_name, unclassified_name)
        if counts["compared"]:
            print(
                f"  BOSS wrong vs non-causal: {counts['wrong']}/{counts['compared']} "
                f"({100 * counts['wrong'] / counts['compared']:.1f}%)"
            )
        if has_causal:
            causal_counts = boss_vs_causal_counts(estimates, positive_name, negative_name, unclassified_name)
            if causal_counts["compared"]:
                print(
                    f"  BOSS wrong vs causal:     {causal_counts['wrong']}/{causal_counts['compared']} "
                    f"({100 * causal_counts['wrong'] / causal_counts['compared']:.1f}%)"
                )


def circular_mean_deg(values_deg: Sequence[float]) -> float:
    values_rad = np.radians(values_deg)
    return float(np.degrees(np.angle(np.mean(np.exp(1j * values_rad)))) % 360.0)


def angular_distance_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute circular distance between two angles in degrees."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def phase_class(
    phase_deg: float,
    tolerance_deg: float,
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> str:
    """Classify phase only inside windows around 0 and 180 degrees."""
    positive_distance = angular_distance_deg(phase_deg, 0.0)
    negative_distance = angular_distance_deg(phase_deg, 180.0)
    is_positive = positive_distance <= tolerance_deg
    is_negative = negative_distance <= tolerance_deg

    if is_positive and is_negative:
        return positive_name if positive_distance <= negative_distance else negative_name
    if is_positive:
        return positive_name
    if is_negative:
        return negative_name
    return unclassified_name


def boss_vs_noncausal_counts(
    estimates: Sequence[TrialEstimate],
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> dict[str, int]:
    return boss_vs_phase_method_counts(
        estimates,
        method="noncausal",
        positive_name=positive_name,
        negative_name=negative_name,
        unclassified_name=unclassified_name,
    )


def boss_vs_causal_counts(
    estimates: Sequence[TrialEstimate],
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> dict[str, int]:
    return boss_vs_phase_method_counts(
        estimates,
        method="causal",
        positive_name=positive_name,
        negative_name=negative_name,
        unclassified_name=unclassified_name,
    )


def boss_vs_phase_method_counts(
    estimates: Sequence[TrialEstimate],
    method: str,
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> dict[str, int]:
    comparable_classes = {positive_name, negative_name}
    compared = [
        est for est in estimates
        if est.condition in comparable_classes and _phase_method_class(est, method) in comparable_classes
    ]
    wrong = [est for est in compared if est.condition != _phase_method_class(est, method)]
    unclassified = [est for est in estimates if _phase_method_class(est, method) == unclassified_name]
    return {
        "compared": len(compared),
        "wrong": len(wrong),
        "correct": len(compared) - len(wrong),
        "unclassified": len(unclassified),
    }


def boss_class_counts(
    estimates: Sequence[TrialEstimate],
    class_name: str,
    method: str,
) -> dict[str, int]:
    boss_trials = [est for est in estimates if est.condition == class_name]
    correct = [est for est in boss_trials if _phase_method_class(est, method) == class_name]
    return {"n_boss": len(boss_trials), "n_correct": len(correct)}


def _phase_method_class(estimate: TrialEstimate, method: str) -> str | None:
    if method == "noncausal":
        return estimate.noncausal_class
    if method == "causal":
        return estimate.causal_class
    raise ValueError("method must be 'noncausal' or 'causal'.")


def _phase_method_deg(estimate: TrialEstimate, method: str) -> float | None:
    if method == "noncausal":
        return estimate.phase_deg
    if method == "causal":
        return estimate.causal_phase_deg
    raise ValueError("method must be 'noncausal' or 'causal'.")


def comparison_status(
    estimate: TrialEstimate,
    method: str,
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
) -> str:
    comparable_classes = {positive_name, negative_name}
    phase_class_name = _phase_method_class(estimate, method)
    if estimate.condition not in comparable_classes:
        return "unknown_boss"
    if phase_class_name is None:
        return "unclassified"
    if phase_class_name == unclassified_name:
        return "unclassified"
    if phase_class_name == estimate.condition:
        return "correct"
    return "wrong"


def plot_intake_phase_circle(
    estimates_by_subject: dict[str, list[TrialEstimate]],
    causal_estimation: bool,
    cutoff_ms: float,
    band: tuple[float, float],
    phase_class_tolerance_deg: float,
    class_order: Sequence[str],
    class_colors: dict[str, str],
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
    show: bool,
    save_path: str | Path | None = None,
):
    """Circular phase plot colored by BOSS label. Click points for signal plots."""
    figures = []
    n_subjects = len(estimates_by_subject)
    for subject, estimates in estimates_by_subject.items():
        fig = _plot_one_subject_phase_circle(
            subject=subject,
            estimates=estimates,
            causal_estimation=causal_estimation,
            cutoff_ms=cutoff_ms,
            band=band,
            phase_class_tolerance_deg=phase_class_tolerance_deg,
            class_order=class_order,
            class_colors=class_colors,
            positive_name=positive_name,
            negative_name=negative_name,
            unclassified_name=unclassified_name,
        )
        if save_path is not None:
            fig.savefig(_save_path_for_subject(save_path, subject, n_subjects), dpi=150)
        figures.append(fig)

    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return figures[0] if len(figures) == 1 else figures


def _plot_one_subject_phase_circle(
    subject: str,
    estimates: list[TrialEstimate],
    causal_estimation: bool,
    cutoff_ms: float,
    band: tuple[float, float],
    phase_class_tolerance_deg: float,
    class_order: Sequence[str],
    class_colors: dict[str, str],
    positive_name: str,
    negative_name: str,
    unclassified_name: str,
):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), subplot_kw={"projection": "polar"})
    scatter_lookup = []

    _configure_phase_axis(ax, subject)
    _shade_phase_windows(ax, phase_class_tolerance_deg)

    radius_by_estimate = {}
    for condition in class_order:
        condition_estimates = [est for est in estimates if est.condition == condition]
        for radius, estimate in enumerate(condition_estimates, start=1):
            radius_by_estimate[id(estimate)] = radius

    if causal_estimation:
        for estimate in estimates:
            if estimate.causal_phase_deg is None:
                continue
            radius = radius_by_estimate[id(estimate)]
            theta = _short_arc_pair(np.radians(estimate.phase_deg), np.radians(estimate.causal_phase_deg))
            ax.plot(theta, [radius, radius], color="0.55", lw=0.75, alpha=0.35, zorder=1)

    method_markers = [("noncausal", "o")]
    if causal_estimation:
        method_markers.append(("causal", "^"))

    for method, marker in method_markers:
        for condition in class_order:
            condition_estimates = [est for est in estimates if est.condition == condition]
            for status in ("correct", "wrong", "unclassified", "unknown_boss"):
                selected = [
                    est for est in condition_estimates
                    if _phase_method_deg(est, method) is not None
                    and comparison_status(est, method, positive_name, negative_name, unclassified_name) == status
                ]
                if not selected:
                    continue
                theta = np.radians([_phase_method_deg(est, method) for est in selected])
                radius = [radius_by_estimate[id(est)] for est in selected]
                scatter_kwargs = _scatter_style(
                    class_colors[condition],
                    status,
                    marker,
                    paired_methods=causal_estimation,
                )
                scatter = ax.scatter(
                    theta,
                    radius,
                    picker=True,
                    pickradius=6,
                    label="_nolegend_",
                    **scatter_kwargs,
                )
                scatter_lookup.append((scatter, selected))

    _add_boss_count_legend(
        ax,
        estimates,
        class_colors,
        positive_name,
        negative_name,
        causal_estimation=causal_estimation,
    )
    _add_method_marker_legend(ax, causal_estimation)
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.38, 1.12))
    fig.tight_layout()

    def on_pick(event):
        for scatter, selected in scatter_lookup:
            if event.artist is scatter and len(event.ind):
                plot_trial_signal(selected[event.ind[0]], cutoff_ms=cutoff_ms, band=band).show()
                return

    fig.canvas.mpl_connect("pick_event", on_pick)
    return fig


def _short_arc_pair(theta_a: float, theta_b: float) -> list[float]:
    if abs(theta_b - theta_a) <= np.pi:
        return [theta_a, theta_b]
    if theta_a < theta_b:
        theta_a += 2.0 * np.pi
    else:
        theta_b += 2.0 * np.pi
    return [theta_a, theta_b]


def _save_path_for_subject(save_path: str | Path, subject: str, n_subjects: int) -> Path:
    path = Path(save_path)
    if n_subjects == 1:
        return path
    return path.with_name(f"{path.stem}_{subject}{path.suffix}")


def _configure_phase_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_rlabel_position(90)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0 deg", "90 deg", "180 deg", "270 deg"])
    for target_deg in (0, 180):
        ax.axvline(np.radians(target_deg), color="black", ls=":", lw=1, alpha=0.55)


def _shade_phase_windows(ax, tolerance_deg: float) -> None:
    for center_deg in (0.0, 180.0):
        _shade_circular_window(ax, center_deg, tolerance_deg)


def _shade_circular_window(ax, center_deg: float, tolerance_deg: float) -> None:
    start_deg = (center_deg - tolerance_deg) % 360.0
    end_deg = (center_deg + tolerance_deg) % 360.0
    spans = [(start_deg, 360.0), (0.0, end_deg)] if start_deg > end_deg else [(start_deg, end_deg)]
    for start, end in spans:
        ax.axvspan(np.radians(start), np.radians(end), color="0.82", alpha=0.35, zorder=0)


def _scatter_style(color: str, status: str, marker: str, paired_methods: bool) -> dict:
    if status == "wrong":
        if not paired_methods:
            return {"s": 52, "marker": "x", "color": color, "alpha": 0.9, "linewidths": 1.5, "zorder": 4}
        return {
            "s": 48,
            "marker": marker,
            "facecolors": "none",
            "edgecolors": color,
            "alpha": 0.95,
            "linewidths": 1.5,
            "zorder": 4,
        }
    if status == "unclassified":
        return {
            "s": 34,
            "marker": marker,
            "facecolors": "none",
            "edgecolors": color,
            "alpha": 0.45,
            "linewidths": 1.0,
            "zorder": 3,
        }
    return {"s": 34, "marker": marker, "color": color, "alpha": 0.78, "zorder": 3}


def _add_boss_count_legend(
    ax,
    estimates: Sequence[TrialEstimate],
    class_colors: dict[str, str],
    positive_name: str,
    negative_name: str,
    causal_estimation: bool,
) -> None:
    for class_name in (positive_name, negative_name):
        noncausal_counts = boss_class_counts(estimates, class_name, method="noncausal")
        label = (
            f"BOSS {class_name}: n={noncausal_counts['n_boss']}, "
            f"non-causal correct={noncausal_counts['n_correct']}"
        )
        if causal_estimation:
            causal_counts = boss_class_counts(estimates, class_name, method="causal")
            label = f"{label}, causal correct={causal_counts['n_correct']}"
        ax.scatter(
            [],
            [],
            s=32,
            marker="o",
            color=class_colors[class_name],
            label=label,
        )


def _add_method_marker_legend(ax, causal_estimation: bool) -> None:
    ax.scatter([], [], s=32, marker="o", color="0.25", label="non-causal phase")
    if causal_estimation:
        ax.scatter([], [], s=36, marker="^", color="0.25", label="causal AR phase")


def plot_trial_signal(
    estimate: TrialEstimate,
    cutoff_ms: float,
    band: tuple[float, float],
    window_ms: tuple[float, float] | None = None,
):
    """Plot raw signal plus non-causal and optional causal estimates for a clicked point."""
    times_ms = estimate.times_ms
    raw = estimate.raw
    filtered = estimate.filtered

    if window_ms is None:
        plot_mask = np.ones_like(times_ms, dtype=bool)
    else:
        plot_mask = (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times_ms[plot_mask], raw[plot_mask], color="black", lw=0.85, label=f"raw ({estimate.channel})")
    ax.plot(
        times_ms[plot_mask],
        filtered[plot_mask],
        color="tab:blue",
        lw=1.35,
        alpha=0.85,
        label=f"non-causal filtered ({band[0]:.0f}-{band[1]:.0f} Hz)",
    )

    if estimate.causal_core is not None and estimate.causal_core_times_ms is not None:
        core_mask = _time_window_mask(estimate.causal_core_times_ms, window_ms)
        ax.plot(
            estimate.causal_core_times_ms[core_mask],
            estimate.causal_core[core_mask],
            color="tab:red",
            lw=1.2,
            alpha=0.9,
            label="causal AR core",
        )
    if estimate.causal_pred_future is not None and estimate.causal_future_times_ms is not None:
        future_mask = _time_window_mask(estimate.causal_future_times_ms, window_ms)
        ax.plot(
            estimate.causal_future_times_ms[future_mask],
            estimate.causal_pred_future[future_mask],
            color="tab:red",
            lw=1.2,
            ls="--",
            alpha=0.9,
            label="causal AR predicted",
        )

    ax.axvline(cutoff_ms, color="gray", ls=":", lw=1.2)
    ax.plot([], [], " ", label=f"BOSS: {estimate.condition}")
    ax.plot([], [], " ", label=f"non-causal: {estimate.noncausal_class}, {estimate.phase_deg:.1f} deg")
    if estimate.causal_phase_deg is not None:
        ax.plot([], [], " ", label=f"causal: {estimate.causal_class}, {estimate.causal_phase_deg:.1f} deg")
    if estimate.causal_phase_error_deg is not None:
        ax.plot([], [], " ", label=f"causal - non-causal: {estimate.causal_phase_error_deg:+.1f} deg")
    ax.set_title(f"{estimate.subject} trial {estimate.epoch_index + 1}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _time_window_mask(times_ms: np.ndarray, window_ms: tuple[float, float] | None) -> np.ndarray:
    if window_ms is None:
        return np.ones_like(times_ms, dtype=bool)
    return (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])
