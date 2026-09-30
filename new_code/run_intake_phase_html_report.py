"""Export an interactive intake phase/BOSS HTML report.

The expensive EEG loading, non-causal phase estimation, and optional causal
optimization run once and are saved to JSON. The HTML report embeds that cached
payload, so visual/layout edits can be made without touching the EEG data again.

Run from the project root:

    ./.venv/bin/python new_code/run_intake_phase_html_report.py

Useful local workflows:

    # Rebuild only the HTML from an existing cache.
    ./.venv/bin/python new_code/run_intake_phase_html_report.py --from-cache

    # Force a new calculation/cache.
    ./.venv/bin/python new_code/run_intake_phase_html_report.py --force

    # Fast no-data smoke/demo report.
    ./.venv/bin/python new_code/run_intake_phase_html_report.py --demo --output-dir /tmp/intake_phase_report_demo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_CODE = PROJECT_ROOT / "new_code"
if str(NEW_CODE) not in sys.path:
    sys.path.insert(0, str(NEW_CODE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
(XDG_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import run_intake_phase_viewer as settings
from functions import TrialEstimate, estimate_all_subjects, load_intake_subjects


REPORT_DIR = Path("new_code/intake_phase_report")
CACHE_FILENAME = "intake_phase_results.json"
HTML_FILENAME = "intake_phase_report.html"
TOLERANCE_OPTIONS_DEG = [15, 30, 45]
DEFAULT_TOLERANCE_DEG = 30
MAX_SIGNAL_POINTS = 4000
FLOAT_DECIMALS = 6
SIGNAL_SCALE = 1_000_000.0
SIGNAL_DECIMALS = 4
SIGNAL_UNIT = "uV"

CLASS_HEX_COLORS = {
    settings.POSITIVE_NAME: "#2f8f5b",
    settings.NEGATIVE_NAME: "#c7564c",
    settings.UNKNOWN_NAME: "#858b93",
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_path = output_dir / CACHE_FILENAME
    html_path = output_dir / HTML_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_written = False
    if args.demo:
        payload = make_demo_payload()
        save_json(cache_path, payload)
        cache_written = True
    elif args.from_cache or (cache_path.exists() and not args.force):
        payload = load_json(cache_path)
        print(f"Loaded cached results: {cache_path}")
    else:
        payload = compute_payload(args)
        save_json(cache_path, payload)
        cache_written = True

    write_html_report(html_path, payload)
    cache_label = "Wrote cache" if cache_written else "Used cache"
    print(f"{cache_label}: {cache_path}")
    print(f"Wrote HTML:  {html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR,
        help=f"Folder for {CACHE_FILENAME} and {HTML_FILENAME}.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip EEG loading/optimization and rebuild the HTML from the existing JSON cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the JSON cache even if one already exists.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write a synthetic no-data report for testing the HTML.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject list. Defaults to SUBJECTS in run_intake_phase_viewer.py.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optional trial cap for fast local smoke runs. Defaults to N_TRIALS in the runner.",
    )
    parser.add_argument(
        "--n-opt-trials",
        type=int,
        default=None,
        help="Optional Optuna trial cap. Defaults to OPT_N_TRIALS in the runner.",
    )
    return parser.parse_args()


def compute_payload(args: argparse.Namespace) -> dict[str, Any]:
    subjects = args.subjects if args.subjects is not None else settings.SUBJECTS
    n_trials = args.n_trials if args.n_trials is not None else settings.N_TRIALS
    optimization_config = dict(settings.CAUSAL_OPTIMIZATION_CONFIG)
    if args.n_opt_trials is not None:
        optimization_config["n_opt_trials"] = args.n_opt_trials

    print(f"Loading subjects: {subjects}")
    subject_data = load_intake_subjects(
        subjects,
        data_root=settings.DATA_ROOT,
        intake_filename=settings.INTAKE_FILENAME,
        channel=settings.CHANNEL,
        condition_column=settings.CONDITION_COLUMN,
        lowpass_before_downsample_hz=settings.LOWPASS_BEFORE_DOWNSAMPLE_HZ,
        downsample=settings.DOWNSAMPLE,
        downsample_fs=settings.DOWNSAMPLE_FS,
        show_metadata_summary=settings.SHOW_METADATA_SUMMARY,
        metadata_max_values=settings.METADATA_MAX_VALUES,
        hjorth_channel=settings.HJORTH_CHANNEL,
        hjorth_weights=settings.HJORTH_WEIGHTS,
        hjorth_scale_reference=settings.HJORTH_SCALE_REFERENCE,
        positive_labels=settings.POSITIVE_LABELS,
        negative_labels=settings.NEGATIVE_LABELS,
        positive_name=settings.POSITIVE_NAME,
        negative_name=settings.NEGATIVE_NAME,
        unknown_name=settings.UNKNOWN_NAME,
        condition_auto_keywords=settings.CONDITION_AUTO_KEYWORDS,
    )

    print("Estimating phase and causal parameters...")
    estimates_by_subject = estimate_all_subjects(
        subject_data,
        band=settings.BAND,
        filter_order=settings.FILTER_ORDER,
        cutoff_ms=settings.CUTOFF_MS,
        n_trials=n_trials,
        phase_class_tolerance_deg=settings.PHASE_CLASS_TOLERANCE_DEG,
        positive_name=settings.POSITIVE_NAME,
        negative_name=settings.NEGATIVE_NAME,
        unclassified_name=settings.NONCAUSAL_UNCLASSIFIED_NAME,
        causal_estimation=settings.CAUSAL_ESTIMATION,
        causal_params_mode=settings.CAUSAL_PARAMS_MODE,
        manual_causal_params=settings.MANUAL_CAUSAL_PARAMS,
        optimization_config=optimization_config,
    )

    payload = estimates_to_payload(
        estimates_by_subject,
        n_trials=n_trials,
        optimization_config=optimization_config,
    )
    return payload


def estimates_to_payload(
    estimates_by_subject: dict[str, list[TrialEstimate]],
    n_trials: int | bool,
    optimization_config: dict[str, Any],
) -> dict[str, Any]:
    subject_payloads = []
    total_trials = 0
    causal_available = False

    for subject, estimates in estimates_by_subject.items():
        trials = [trial_to_payload(est) for est in estimates]
        total_trials += len(trials)
        subject_causal_available = any(trial["causal_phase_deg"] is not None for trial in trials)
        causal_available = causal_available or subject_causal_available
        subject_payloads.append(
            {
                "id": subject,
                "n_trials": len(trials),
                "channel": estimates[0].channel if estimates else settings.CHANNEL,
                "causal_available": subject_causal_available,
                "causal_params": first_causal_params(estimates),
                "trials": trials,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "subjects": list(estimates_by_subject.keys()),
            "data_root": str(settings.DATA_ROOT),
            "intake_filename": settings.INTAKE_FILENAME,
            "channel": settings.CHANNEL,
            "band_hz": list(settings.BAND),
            "filter_order": settings.FILTER_ORDER,
            "cutoff_ms": settings.CUTOFF_MS,
            "n_trials": n_trials,
            "causal_estimation": settings.CAUSAL_ESTIMATION,
            "causal_params_mode": settings.CAUSAL_PARAMS_MODE,
            "optimization_config": json_safe(optimization_config),
            "downsample": settings.DOWNSAMPLE,
            "downsample_fs": settings.DOWNSAMPLE_FS,
            "lowpass_before_downsample_hz": settings.LOWPASS_BEFORE_DOWNSAMPLE_HZ,
            "condition_column": settings.CONDITION_COLUMN,
            "raw_trace_label": f"{settings.CHANNEL} raw",
            "signal_unit": SIGNAL_UNIT,
            "signal_scale": SIGNAL_SCALE,
        },
        "class_names": {
            "positive": settings.POSITIVE_NAME,
            "negative": settings.NEGATIVE_NAME,
            "unknown": settings.UNKNOWN_NAME,
            "unclassified": settings.NONCAUSAL_UNCLASSIFIED_NAME,
        },
        "class_order": list(settings.CLASS_ORDER),
        "class_colors": CLASS_HEX_COLORS,
        "tolerance_options_deg": TOLERANCE_OPTIONS_DEG,
        "default_tolerance_deg": DEFAULT_TOLERANCE_DEG,
        "max_signal_points": MAX_SIGNAL_POINTS,
        "causal_available": causal_available,
        "total_trials": total_trials,
        "subjects": subject_payloads,
    }


def trial_to_payload(est: TrialEstimate) -> dict[str, Any]:
    signal_idx = decimation_indices(len(est.times_ms), MAX_SIGNAL_POINTS)
    payload = {
        "trial_id": f"{est.subject}:{est.epoch_index}",
        "subject": est.subject,
        "epoch_index": est.epoch_index,
        "boss_class": est.condition,
        "phase_deg": round_float(est.phase_deg),
        "phase_rad": round_float(est.phase_rad),
        "amplitude": round_float(est.amplitude),
        "causal_phase_deg": round_float(est.causal_phase_deg),
        "causal_phase_rad": round_float(est.causal_phase_rad),
        "causal_amplitude": round_float(est.causal_amplitude),
        "causal_phase_error_deg": round_float(est.causal_phase_error_deg),
        "signal": {
            "t": round_array(est.times_ms[signal_idx]),
            "raw": signal_array(est.raw[signal_idx]),
            "filtered": signal_array(est.filtered[signal_idx]),
        },
        "causal_core": trace_payload(est.causal_core_times_ms, est.causal_core),
        "causal_future": trace_payload(est.causal_future_times_ms, est.causal_pred_future),
    }
    return payload


def first_causal_params(estimates: Sequence[TrialEstimate]) -> dict[str, Any] | None:
    for est in estimates:
        if est.causal_params is not None:
            return json_safe(est.causal_params)
    return None


def trace_payload(times: np.ndarray | None, values: np.ndarray | None) -> dict[str, list[float | None]] | None:
    if times is None or values is None:
        return None
    idx = decimation_indices(len(times), MAX_SIGNAL_POINTS)
    return {"t": round_array(times[idx]), "y": signal_array(values[idx])}


def decimation_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def round_float(value: Any, decimals: int = FLOAT_DECIMALS) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    return round(as_float, decimals)


def round_array(values: np.ndarray, decimals: int = FLOAT_DECIMALS) -> list[float | None]:
    return [round_float(value, decimals) for value in np.asarray(values)]


def signal_array(values: np.ndarray) -> list[float | None]:
    return round_array(np.asarray(values) * SIGNAL_SCALE, SIGNAL_DECIMALS)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round_float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def make_demo_payload() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    times = np.linspace(-700, 450, 1151)
    trials = []
    conditions = [
        settings.POSITIVE_NAME,
        settings.NEGATIVE_NAME,
        settings.POSITIVE_NAME,
        settings.NEGATIVE_NAME,
        settings.UNKNOWN_NAME,
    ]
    for i in range(28):
        condition = conditions[i % len(conditions)]
        target = 0.0 if condition == settings.POSITIVE_NAME else 180.0
        if condition == settings.UNKNOWN_NAME:
            target = rng.uniform(0, 360)
        phase_deg = (target + rng.normal(0, 34)) % 360
        causal_phase_deg = (phase_deg + rng.normal(0, 18)) % 360
        phase_rad = math.radians(phase_deg)
        raw = (
            0.9 * np.sin(2 * np.pi * 6 * times / 1000 + phase_rad)
            + 0.25 * np.sin(2 * np.pi * 14 * times / 1000)
            + rng.normal(0, 0.12, size=times.shape)
        ) * 8e-6
        filtered = 0.9 * np.sin(2 * np.pi * 6 * times / 1000 + phase_rad) * 8e-6
        core_mask = (times >= -450) & (times <= -40)
        future_mask = (times > -40) & (times <= 130)
        trial = {
            "trial_id": f"demo_sub_001:{i}",
            "subject": "demo_sub_001",
            "epoch_index": i,
            "boss_class": condition,
            "phase_deg": round_float(phase_deg),
            "phase_rad": round_float(math.radians(phase_deg)),
            "amplitude": round_float(abs(rng.normal(1.0, 0.15))),
            "causal_phase_deg": round_float(causal_phase_deg),
            "causal_phase_rad": round_float(math.radians(causal_phase_deg)),
            "causal_amplitude": round_float(abs(rng.normal(0.92, 0.15))),
            "causal_phase_error_deg": round_float(((causal_phase_deg - phase_deg + 180) % 360) - 180),
            "signal": {
                "t": round_array(times),
                "raw": signal_array(raw),
                "filtered": signal_array(filtered),
            },
            "causal_core": {
                "t": round_array(times[core_mask]),
                "y": signal_array(filtered[core_mask] + rng.normal(0, 0.025e-6, size=core_mask.sum())),
            },
            "causal_future": {
                "t": round_array(times[future_mask]),
                "y": signal_array(filtered[future_mask] + rng.normal(0, 0.06e-6, size=future_mask.sum())),
            },
        }
        trials.append(trial)

    optimization_config = dict(settings.CAUSAL_OPTIMIZATION_CONFIG)
    subject = {
        "id": "demo_sub_001",
        "n_trials": len(trials),
        "channel": settings.CHANNEL,
        "causal_available": True,
        "causal_params": {
            "window_ms": 510.0,
            "filter_order": 220,
            "edge": 65,
            "ar_order": 38,
            "hilbert_window": settings.HILBERT_WINDOW,
            "offset": settings.OFFSET,
        },
        "trials": trials,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "subjects": ["demo_sub_001"],
            "data_root": "synthetic demo",
            "intake_filename": settings.INTAKE_FILENAME,
            "channel": settings.CHANNEL,
            "band_hz": list(settings.BAND),
            "filter_order": settings.FILTER_ORDER,
            "cutoff_ms": settings.CUTOFF_MS,
            "n_trials": len(trials),
            "causal_estimation": True,
            "causal_params_mode": "demo",
            "optimization_config": json_safe(optimization_config),
            "downsample": settings.DOWNSAMPLE,
            "downsample_fs": settings.DOWNSAMPLE_FS,
            "lowpass_before_downsample_hz": settings.LOWPASS_BEFORE_DOWNSAMPLE_HZ,
            "condition_column": settings.CONDITION_COLUMN,
            "raw_trace_label": f"{settings.CHANNEL} raw",
            "signal_unit": SIGNAL_UNIT,
            "signal_scale": SIGNAL_SCALE,
        },
        "class_names": {
            "positive": settings.POSITIVE_NAME,
            "negative": settings.NEGATIVE_NAME,
            "unknown": settings.UNKNOWN_NAME,
            "unclassified": settings.NONCAUSAL_UNCLASSIFIED_NAME,
        },
        "class_order": list(settings.CLASS_ORDER),
        "class_colors": CLASS_HEX_COLORS,
        "tolerance_options_deg": TOLERANCE_OPTIONS_DEG,
        "default_tolerance_deg": DEFAULT_TOLERANCE_DEG,
        "max_signal_points": MAX_SIGNAL_POINTS,
        "causal_available": True,
        "total_trials": len(trials),
        "subjects": [subject],
    }


def write_html_report(path: Path, payload: dict[str, Any]) -> None:
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__REPORT_DATA__", data_json)
    path.write_text(html, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Intake Phase Report</title>
  <style>
    :root {
      --bg: #f7f8f5;
      --ink: #20211f;
      --muted: #6a6f68;
      --line: #d9ddd4;
      --soft: #eef1eb;
      --panel: #ffffff;
      --accent: #245f73;
      --accent-ink: #ffffff;
      --positive: #2f8f5b;
      --negative: #c7564c;
      --unknown: #858b93;
      --warn: #9c6b19;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
    }

    h1, h2, h3, p { margin: 0; }

    h1 {
      font-size: clamp(30px, 4vw, 54px);
      line-height: 1;
      font-weight: 780;
    }

    h2 {
      font-size: 18px;
      font-weight: 720;
    }

    h3 {
      font-size: 13px;
      font-weight: 760;
      text-transform: uppercase;
      color: var(--muted);
    }

    .eyebrow {
      margin-bottom: 8px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
    }

    .run-meta {
      max-width: 460px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      text-align: right;
    }

    main {
      padding: 18px clamp(18px, 4vw, 48px) 36px;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      padding: 14px 0 18px;
    }

    .control {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    select, button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }

    select {
      min-width: 150px;
      padding: 0 34px 0 10px;
    }

    button {
      padding: 0 12px;
      cursor: pointer;
      transition: background 160ms ease, color 160ms ease, border-color 160ms ease, transform 160ms ease;
    }

    button:hover:not(:disabled) {
      transform: translateY(-1px);
      border-color: #aab3a8;
    }

    button:disabled {
      cursor: not-allowed;
      color: #a1a69f;
      background: #f2f3f0;
    }

    .segmented {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }

    .segmented button {
      border: 0;
      border-radius: 0;
      border-right: 1px solid var(--line);
      background: transparent;
    }

    .segmented button:last-child {
      border-right: 0;
    }

    .segmented button.active,
    button[aria-pressed="true"] {
      background: var(--accent);
      color: var(--accent-ink);
      border-color: var(--accent);
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 370px);
      gap: 28px;
      align-items: start;
      border-top: 1px solid var(--line);
      padding-top: 20px;
    }

    .figure-head,
    .signal-head {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: baseline;
      margin-bottom: 10px;
    }

    .subtle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .figure-wrap {
      min-height: 610px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }

    svg {
      display: block;
      width: 100%;
      height: auto;
    }

    .summary {
      border-left: 1px solid var(--line);
      padding-left: 20px;
    }

    .summary-grid {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }

    .summary-row {
      display: grid;
      grid-template-columns: 8px minmax(0, 1fr);
      gap: 10px;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }

    .color-bar {
      width: 8px;
      min-height: 100%;
      border-radius: 6px;
    }

    .summary-row strong {
      display: block;
      margin-bottom: 4px;
      font-size: 15px;
    }

    .summary-row p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .method-totals {
      display: grid;
      gap: 8px;
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .signal-section,
    .params-section {
      border-top: 1px solid var(--line);
      margin-top: 28px;
      padding-top: 22px;
    }

    .signal-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }

    .params-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 16px 24px;
      margin-top: 14px;
    }

    .params-block {
      border-top: 3px solid var(--accent);
      padding-top: 10px;
    }

    .params-block table,
    .settings-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    td {
      padding: 5px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }

    td:first-child {
      color: var(--muted);
      padding-right: 16px;
      white-space: nowrap;
    }

    .point {
      cursor: pointer;
      transition: opacity 140ms ease, transform 140ms ease;
    }

    .point:hover {
      opacity: 1;
    }

    .selected-ring {
      pointer-events: none;
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.4;
    }

    .axis-label {
      fill: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .legend-text {
      fill: var(--muted);
      font-size: 12px;
    }

    .empty {
      fill: var(--muted);
      font-size: 16px;
      text-anchor: middle;
    }

    @media (max-width: 980px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .run-meta {
        text-align: left;
      }

      .workspace {
        grid-template-columns: 1fr;
      }

      .summary {
        border-left: 0;
        padding-left: 0;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <p class="eyebrow">EEG phase / BOSS comparison</p>
      <h1>Intake phase report</h1>
    </div>
    <p class="run-meta" id="runMeta"></p>
  </header>

  <main>
    <section class="toolbar" aria-label="Report controls">
      <label class="control">Subject <select id="subjectSelect"></select></label>
      <div class="control">
        +/- deg
        <span class="segmented" id="toleranceButtons"></span>
      </div>
      <div class="control">
        Phase
        <span class="segmented" id="methodButtons">
          <button type="button" data-phase-mode="noncausal">Non-causal</button>
          <button type="button" data-phase-mode="causal">Causal</button>
          <button type="button" data-phase-mode="both">Both</button>
        </span>
      </div>
    </section>

    <section class="workspace">
      <div>
        <div class="figure-head">
          <h2 id="phaseTitle">Phase circle</h2>
          <p class="subtle" id="phaseSubtitle"></p>
        </div>
        <div class="figure-wrap">
          <svg id="phaseSvg" viewBox="0 0 820 620" role="img" aria-label="Phase circle"></svg>
        </div>
      </div>

      <aside class="summary" aria-label="BOSS summary">
        <h2>BOSS classification</h2>
        <div class="summary-grid" id="summaryGrid"></div>
        <div class="method-totals" id="methodTotals"></div>
      </aside>
    </section>

    <section class="signal-section">
      <div class="signal-head">
        <h2 id="signalTitle">Trial signal</h2>
        <p class="subtle" id="signalMeta"></p>
      </div>
      <div class="signal-wrap">
        <svg id="signalSvg" viewBox="0 0 1080 380" role="img" aria-label="Selected trial signal"></svg>
      </div>
    </section>

    <section class="params-section">
      <h2>Selected optimization params</h2>
      <div class="params-grid" id="paramsGrid"></div>
    </section>
  </main>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const NS = "http://www.w3.org/2000/svg";
    const POS = report.class_names.positive;
    const NEG = report.class_names.negative;
    const UNKNOWN = report.class_names.unknown;
    const UNCLASSIFIED = report.class_names.unclassified;
    const classOrder = report.class_order || [POS, NEG, UNKNOWN];
    const colors = report.class_colors || {};
    const state = {
      subjectId: report.subjects[0]?.id || "",
      tolerance: report.default_tolerance_deg || 30,
      phaseMode: "noncausal",
      selectedTrialId: null
    };

    const subjectSelect = document.getElementById("subjectSelect");
    const toleranceButtons = document.getElementById("toleranceButtons");
    const methodButtons = document.getElementById("methodButtons");
    const phaseSvg = document.getElementById("phaseSvg");
    const signalSvg = document.getElementById("signalSvg");

    function el(name, attrs = {}, text = null) {
      const node = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attrs)) {
        if (value !== null && value !== undefined) node.setAttribute(key, value);
      }
      if (text !== null) node.textContent = text;
      return node;
    }

    function htmlEscape(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#039;"
      }[ch]));
    }

    function currentSubject() {
      return report.subjects.find(subject => subject.id === state.subjectId) || report.subjects[0];
    }

    function currentTrials() {
      return currentSubject()?.trials || [];
    }

    function trialById(trialId) {
      return currentTrials().find(trial => trial.trial_id === trialId);
    }

    function ensureSelectedTrial() {
      const trials = currentTrials();
      if (!trials.length) {
        state.selectedTrialId = null;
        return null;
      }
      const current = trialById(state.selectedTrialId);
      if (current) return current;
      state.selectedTrialId = trials[0].trial_id;
      return trials[0];
    }

    function phaseModeMethods() {
      if (state.phaseMode === "both") return ["noncausal", "causal"];
      return [state.phaseMode];
    }

    function methodLabel(method) {
      return method === "causal" ? "Causal" : "Non-causal";
    }

    function hasCausal(trial) {
      return trial?.causal_phase_deg !== null && trial?.causal_phase_deg !== undefined;
    }

    function angularDistance(a, b) {
      return Math.abs(((a - b + 180) % 360 + 360) % 360 - 180);
    }

    function phaseClass(phaseDeg, tolerance) {
      if (phaseDeg === null || phaseDeg === undefined) return null;
      const positiveDistance = angularDistance(phaseDeg, 0);
      const negativeDistance = angularDistance(phaseDeg, 180);
      const isPositive = positiveDistance <= tolerance;
      const isNegative = negativeDistance <= tolerance;
      if (isPositive && isNegative) return positiveDistance <= negativeDistance ? POS : NEG;
      if (isPositive) return POS;
      if (isNegative) return NEG;
      return UNCLASSIFIED;
    }

    function methodClass(trial, method) {
      if (method === "noncausal") return phaseClass(trial.phase_deg, state.tolerance);
      if (method === "causal") return phaseClass(trial.causal_phase_deg, state.tolerance);
      return null;
    }

    function methodPhase(trial, method) {
      return method === "causal" ? trial.causal_phase_deg : trial.phase_deg;
    }

    function comparisonStatus(trial, method) {
      const cls = methodClass(trial, method);
      if (![POS, NEG].includes(trial.boss_class)) return "unknown";
      if (!cls || cls === UNCLASSIFIED) return "unclassified";
      return cls === trial.boss_class ? "correct" : "wrong";
    }

    function polarPoint(cx, cy, radius, phaseDeg) {
      const theta = phaseDeg * Math.PI / 180;
      return [cx + radius * Math.sin(theta), cy - radius * Math.cos(theta)];
    }

    function linePath(xs, ys, xScale, yScale) {
      let path = "";
      let drawing = false;
      for (let i = 0; i < xs.length; i += 1) {
        const x = xs[i];
        const y = ys[i];
        if (x === null || y === null || Number.isNaN(x) || Number.isNaN(y)) {
          drawing = false;
          continue;
        }
        path += `${drawing ? "L" : "M"}${xScale(x).toFixed(2)},${yScale(y).toFixed(2)}`;
        drawing = true;
      }
      return path;
    }

    function sectorPath(cx, cy, r0, r1, startDeg, endDeg) {
      const startOuter = polarPoint(cx, cy, r1, startDeg);
      const endOuter = polarPoint(cx, cy, r1, endDeg);
      const startInner = polarPoint(cx, cy, r0, endDeg);
      const endInner = polarPoint(cx, cy, r0, startDeg);
      const delta = Math.abs(endDeg - startDeg);
      const largeArc = delta > 180 ? 1 : 0;
      return [
        `M${startOuter[0].toFixed(2)},${startOuter[1].toFixed(2)}`,
        `A${r1},${r1} 0 ${largeArc} 1 ${endOuter[0].toFixed(2)},${endOuter[1].toFixed(2)}`,
        `L${startInner[0].toFixed(2)},${startInner[1].toFixed(2)}`,
        `A${r0},${r0} 0 ${largeArc} 0 ${endInner[0].toFixed(2)},${endInner[1].toFixed(2)}`,
        "Z"
      ].join(" ");
    }

    function circularConnectorPath(cx, cy, radius, startDeg, endDeg) {
      const delta = ((endDeg - startDeg + 540) % 360) - 180;
      if (Math.abs(delta) < 0.01) return null;
      const adjustedEnd = startDeg + delta;
      const [x1, y1] = polarPoint(cx, cy, radius, startDeg);
      const [x2, y2] = polarPoint(cx, cy, radius, adjustedEnd);
      const sweep = delta >= 0 ? 1 : 0;
      return `M${x1.toFixed(2)},${y1.toFixed(2)} A${radius.toFixed(2)},${radius.toFixed(2)} 0 0 ${sweep} ${x2.toFixed(2)},${y2.toFixed(2)}`;
    }

    function drawToleranceWindow(svg, cx, cy, r0, r1, centerDeg, tolerance) {
      let start = centerDeg - tolerance;
      let end = centerDeg + tolerance;
      const spans = [];
      if (start < 0) {
        spans.push([360 + start, 360]);
        spans.push([0, end]);
      } else if (end > 360) {
        spans.push([start, 360]);
        spans.push([0, end - 360]);
      } else {
        spans.push([start, end]);
      }
      spans.forEach(([a, b]) => {
        svg.appendChild(el("path", {
          d: sectorPath(cx, cy, r0, r1, a, b),
          fill: "#dfe5dc",
          opacity: "0.78"
        }));
      });
    }

    function pointAttrs(trial, method) {
      const status = comparisonStatus(trial, method);
      const base = colors[trial.boss_class] || colors[UNKNOWN] || "#858b93";
      if (status === "wrong") {
        return { fill: "#ffffff", stroke: base, "stroke-width": "2.2", opacity: "0.98" };
      }
      if (status === "unclassified") {
        return { fill: "#ffffff", stroke: base, "stroke-width": "1.4", opacity: "0.48" };
      }
      if (status === "unknown") {
        return { fill: base, stroke: "#ffffff", "stroke-width": "1", opacity: "0.5" };
      }
      return { fill: base, stroke: "#ffffff", "stroke-width": "1", opacity: "0.88" };
    }

    function addPoint(svg, trial, method, x, y) {
      const attrs = {
        ...pointAttrs(trial, method),
        class: "point",
        "data-trial-id": trial.trial_id,
        tabindex: "0"
      };
      let node;
      if (method === "causal") {
        const size = 8.5;
        node = el("path", {
          ...attrs,
          d: `M${x.toFixed(2)},${(y - size).toFixed(2)} L${(x - size).toFixed(2)},${(y + size).toFixed(2)} L${(x + size).toFixed(2)},${(y + size).toFixed(2)} Z`
        });
      } else {
        node = el("circle", { ...attrs, cx: x.toFixed(2), cy: y.toFixed(2), r: "6.5" });
      }
      const title = el("title");
      const label = method === "causal" ? "causal" : "non-causal";
      title.textContent = `${trial.subject} trial ${trial.epoch_index + 1} | BOSS ${trial.boss_class} | ${label} ${methodClass(trial, method)} | ${methodPhase(trial, method)?.toFixed(1)} deg`;
      node.appendChild(title);
      svg.appendChild(node);
    }

    function renderPhase() {
      const subject = currentSubject();
      const trials = currentTrials();
      const selected = ensureSelectedTrial();
      const visibleMethods = phaseModeMethods();
      phaseSvg.replaceChildren();
      document.getElementById("phaseTitle").textContent = `${subject?.id || "Subject"} phase circle`;
      const modeText = visibleMethods.map(methodLabel).join(" + ");
      document.getElementById("phaseSubtitle").textContent = `${trials.length} trials, ${modeText}, +/-${state.tolerance} deg window`;
      if (!trials.length) {
        phaseSvg.appendChild(el("text", { x: 410, y: 310, class: "empty" }, "No trials in this cache"));
        return;
      }

      const cx = 360;
      const cy = 300;
      const minR = 42;
      const maxR = 248;
      const groups = new Map(classOrder.map(name => [name, []]));
      trials.forEach(trial => {
        if (!groups.has(trial.boss_class)) groups.set(trial.boss_class, []);
        groups.get(trial.boss_class).push(trial);
      });
      const rank = new Map();
      groups.forEach(group => group.forEach((trial, index) => rank.set(trial.trial_id, { index, n: group.length })));
      const radiusFor = trial => {
        const item = rank.get(trial.trial_id) || { index: 0, n: 1 };
        if (item.n <= 1) return (minR + maxR) / 2;
        return minR + (item.index / (item.n - 1)) * (maxR - minR);
      };

      drawToleranceWindow(phaseSvg, cx, cy, 0, maxR + 20, 0, state.tolerance);
      drawToleranceWindow(phaseSvg, cx, cy, 0, maxR + 20, 180, state.tolerance);

      [64, 112, 160, 208, 256].forEach(radius => {
        phaseSvg.appendChild(el("circle", { cx, cy, r: radius, fill: "none", stroke: "#d9ddd4", "stroke-width": "1" }));
      });
      [0, 90, 180, 270].forEach(deg => {
        const [x2, y2] = polarPoint(cx, cy, maxR + 20, deg);
        phaseSvg.appendChild(el("line", { x1: cx, y1: cy, x2, y2, stroke: deg % 180 === 0 ? "#3a3c38" : "#c9cec6", "stroke-dasharray": deg % 180 === 0 ? "3 4" : "0", "stroke-width": "1.2" }));
        const [lx, ly] = polarPoint(cx, cy, maxR + 42, deg);
        phaseSvg.appendChild(el("text", { x: lx, y: ly + 4, class: "axis-label", "text-anchor": "middle" }, `${deg} deg`));
      });

      if (visibleMethods.length === 2) {
        trials.forEach(trial => {
          if (!hasCausal(trial)) return;
          const radius = radiusFor(trial);
          const d = circularConnectorPath(cx, cy, radius, trial.phase_deg, trial.causal_phase_deg);
          if (d) {
            phaseSvg.appendChild(el("path", {
              d,
              fill: "none",
              stroke: "#7c8279",
              "stroke-width": "1",
              opacity: "0.34"
            }));
          }
        });
      }

      trials.forEach(trial => {
        const radius = radiusFor(trial);
        if (visibleMethods.includes("noncausal")) {
          const [x, y] = polarPoint(cx, cy, radius, trial.phase_deg);
          addPoint(phaseSvg, trial, "noncausal", x, y);
        }
        if (visibleMethods.includes("causal") && hasCausal(trial)) {
          const [x, y] = polarPoint(cx, cy, radius, trial.causal_phase_deg);
          addPoint(phaseSvg, trial, "causal", x, y);
        }
      });

      if (selected) {
        const radius = radiusFor(selected);
        drawSelectedPair(phaseSvg, selected, cx, cy, radius);
      }

      drawPhaseLegend(phaseSvg, visibleMethods);
    }

    function drawSelectedPair(svg, trial, cx, cy, radius) {
      const [nonX, nonY] = polarPoint(cx, cy, radius, trial.phase_deg);
      if (hasCausal(trial)) {
        const [causalX, causalY] = polarPoint(cx, cy, radius, trial.causal_phase_deg);
        const d = circularConnectorPath(cx, cy, radius, trial.phase_deg, trial.causal_phase_deg);
        if (d) {
          svg.appendChild(el("path", {
            d,
            fill: "none",
            stroke: "#245f73",
            "stroke-width": "2",
            opacity: "0.72"
          }));
        }
        addPoint(svg, trial, "causal", causalX, causalY);
        svg.appendChild(el("circle", { cx: causalX, cy: causalY, r: "13", class: "selected-ring" }));
      }
      addPoint(svg, trial, "noncausal", nonX, nonY);
      svg.appendChild(el("circle", { cx: nonX, cy: nonY, r: "13", class: "selected-ring" }));
    }

    function drawPhaseLegend(svg, visibleMethods) {
      const x = 650;
      let y = 80;
      classOrder.forEach(name => {
        svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: colors[name] || "#858b93" }));
        svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, `BOSS ${name}`));
        y += 23;
      });
      y += 14;
      svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: "#ffffff", stroke: "#555", "stroke-width": "1.5" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, visibleMethods.includes("noncausal") ? "non-causal" : "selected non-causal"));
      y += 24;
      svg.appendChild(el("path", { d: `M${x},${y - 8} L${x - 8},${y + 8} L${x + 8},${y + 8} Z`, fill: "#ffffff", stroke: "#555", "stroke-width": "1.5" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 5, class: "legend-text" }, visibleMethods.includes("causal") ? "causal" : "selected causal"));
      y += 29;
      svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: "#ffffff", stroke: "#9c6b19", "stroke-width": "2.2" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, "BOSS mismatch"));
    }

    function countSummary(method) {
      const trials = currentTrials();
      const comparable = trials.filter(trial => [POS, NEG].includes(trial.boss_class));
      let correct = 0;
      let wrong = 0;
      let unclassified = 0;
      comparable.forEach(trial => {
        const status = comparisonStatus(trial, method);
        if (status === "correct") correct += 1;
        if (status === "wrong") wrong += 1;
        if (status === "unclassified") unclassified += 1;
      });
      return { correct, wrong, unclassified, compared: comparable.length };
    }

    function renderSummary() {
      const grid = document.getElementById("summaryGrid");
      const totals = document.getElementById("methodTotals");
      const trials = currentTrials();
      const visibleMethods = phaseModeMethods();
      const rows = [POS, NEG].map(name => {
        const bossTrials = trials.filter(trial => trial.boss_class === name);
        const noncausalSame = bossTrials.filter(trial => methodClass(trial, "noncausal") === name).length;
        const causalSame = bossTrials.filter(trial => methodClass(trial, "causal") === name).length;
        const pieces = [];
        if (visibleMethods.includes("noncausal")) pieces.push(`Non-causal labels ${noncausalSame} as ${name}.`);
        if (visibleMethods.includes("causal")) pieces.push(`Causal labels ${causalSame} as ${name}.`);
        return `
          <div class="summary-row">
            <span class="color-bar" style="background:${htmlEscape(colors[name] || "#858b93")}"></span>
            <div>
              <strong>BOSS ${htmlEscape(name)}: ${bossTrials.length}</strong>
              <p>${pieces.map(htmlEscape).join(" ")}</p>
            </div>
          </div>
        `;
      }).join("");
      grid.innerHTML = rows;

      const items = visibleMethods.map(method => {
        const counts = countSummary(method);
        return `<strong>${methodLabel(method)}</strong>: ${counts.correct}/${counts.compared} BOSS matches, ${counts.wrong} mismatches, ${counts.unclassified} unclassified at +/-${state.tolerance} deg.`;
      });
      totals.innerHTML = items.map(item => `<p>${item}</p>`).join("");
    }

    function renderSignal() {
      const trials = currentTrials();
      const trial = ensureSelectedTrial() || trials[0];
      signalSvg.replaceChildren();
      if (!trial) {
        signalSvg.appendChild(el("text", { x: 540, y: 190, class: "empty" }, "No selected trial"));
        return;
      }
      state.selectedTrialId = trial.trial_id;

      document.getElementById("signalTitle").textContent = `${trial.subject} trial ${trial.epoch_index + 1}`;
      const nonCls = methodClass(trial, "noncausal");
      let meta = `BOSS ${trial.boss_class}; non-causal ${nonCls}, ${trial.phase_deg?.toFixed(1)} deg`;
      if (hasCausal(trial)) {
        meta += `; causal ${methodClass(trial, "causal")}, ${trial.causal_phase_deg.toFixed(1)} deg`;
      }
      document.getElementById("signalMeta").textContent = meta;

      const margin = { left: 64, right: 28, top: 24, bottom: 44 };
      const width = 1080;
      const height = 380;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const t = trial.signal.t;
      const channelLabel = report.settings.raw_trace_label || `${report.settings.channel || "channel"} raw`;
      const series = [
        { name: channelLabel, t, y: trial.signal.raw, color: "#20211f", width: 1.05, opacity: 0.78 },
        { name: "non-causal filtered", t, y: trial.signal.filtered, color: "#245f73", width: 1.8, opacity: 0.95 }
      ];
      if (trial.causal_core) {
        series.push({ name: "causal AR core", t: trial.causal_core.t, y: trial.causal_core.y, color: "#c7564c", width: 1.55, opacity: 0.92 });
      }
      if (trial.causal_future) {
        series.push({ name: "causal AR prediction", t: trial.causal_future.t, y: trial.causal_future.y, color: "#c7564c", width: 1.65, opacity: 0.86, dash: "5 5" });
      }
      const xMin = Math.min(...series.flatMap(s => s.t));
      const xMax = Math.max(...series.flatMap(s => s.t));
      const yValues = series.flatMap(s => s.y).filter(v => v !== null && Number.isFinite(v));
      let yMin = Math.min(...yValues);
      let yMax = Math.max(...yValues);
      if (yMin === yMax) {
        yMin -= 1;
        yMax += 1;
      }
      const pad = (yMax - yMin) * 0.12;
      yMin -= pad;
      yMax += pad;

      const xScale = value => margin.left + ((value - xMin) / (xMax - xMin)) * innerW;
      const yScale = value => margin.top + (1 - ((value - yMin) / (yMax - yMin))) * innerH;

      signalSvg.appendChild(el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      if (trial.causal_future?.t?.length) {
        const start = Math.min(...trial.causal_future.t);
        const end = Math.max(...trial.causal_future.t);
        const x1 = xScale(start);
        const x2 = xScale(end);
        signalSvg.appendChild(el("rect", {
          x: x1,
          y: margin.top,
          width: Math.max(1, x2 - x1),
          height: innerH,
          fill: "#c7564c",
          opacity: "0.09"
        }));
        signalSvg.appendChild(el("text", {
          x: x1 + 8,
          y: margin.top + 16,
          class: "legend-text"
        }, "AR prediction"));
      }
      for (let i = 0; i <= 4; i += 1) {
        const y = margin.top + (innerH / 4) * i;
        signalSvg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: "#e2e6df", "stroke-width": "1" }));
      }
      const verticalGridStart = Math.ceil(xMin / 200) * 200;
      for (let value = verticalGridStart; value <= xMax + 0.001; value += 200) {
        const x = xScale(value);
        signalSvg.appendChild(el("line", {
          x1: x,
          x2: x,
          y1: margin.top,
          y2: height - margin.bottom,
          stroke: "#edf0ea",
          "stroke-width": "1"
        }));
      }
      const zeroX = xScale(report.settings.cutoff_ms || 0);
      signalSvg.appendChild(el("line", { x1: zeroX, x2: zeroX, y1: margin.top, y2: height - margin.bottom, stroke: "#8b9189", "stroke-dasharray": "4 5", "stroke-width": "1.3" }));

      signalSvg.appendChild(el("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#aeb6ab" }));
      signalSvg.appendChild(el("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#aeb6ab" }));
      [xMin, 0, xMax].forEach(value => {
        if (value < xMin || value > xMax) return;
        const x = xScale(value);
        signalSvg.appendChild(el("text", { x, y: height - 16, class: "axis-label", "text-anchor": "middle" }, `${Math.round(value)} ms`));
      });
      const unit = report.settings.signal_unit || "a.u.";
      signalSvg.appendChild(el("text", { x: 18, y: 28, class: "axis-label" }, `Amplitude (${unit})`));

      series.forEach(s => {
        signalSvg.appendChild(el("path", {
          d: linePath(s.t, s.y, xScale, yScale),
          fill: "none",
          stroke: s.color,
          "stroke-width": s.width,
          opacity: s.opacity,
          "stroke-dasharray": s.dash || null
        }));
      });

      let lx = margin.left + 10;
      let ly = margin.top + 18;
      series.forEach(s => {
        signalSvg.appendChild(el("line", { x1: lx, y1: ly - 4, x2: lx + 24, y2: ly - 4, stroke: s.color, "stroke-width": "2", "stroke-dasharray": s.dash || null }));
        signalSvg.appendChild(el("text", { x: lx + 31, y: ly, class: "legend-text" }, s.name));
        lx += s.name.length * 7 + 64;
        if (lx > width - 240) {
          lx = margin.left + 10;
          ly += 22;
        }
      });
    }

    function renderParams() {
      const paramsGrid = document.getElementById("paramsGrid");
      const subjectBlocks = report.subjects.map(subject => {
        const params = subject.causal_params;
        const rows = params
          ? Object.entries(params).map(([key, value]) => `<tr><td>${htmlEscape(key)}</td><td>${htmlEscape(value)}</td></tr>`).join("")
          : `<tr><td>causal params</td><td>not available</td></tr>`;
        return `
          <div class="params-block">
            <h3>${htmlEscape(subject.id)}</h3>
            <table>${rows}</table>
          </div>
        `;
      }).join("");
      const settingsRows = [
        ["band_hz", (report.settings.band_hz || []).join("-")],
        ["cutoff_ms", report.settings.cutoff_ms],
        ["channel", report.settings.channel],
        ["causal_params_mode", report.settings.causal_params_mode],
        ["n_opt_trials", report.settings.optimization_config?.n_opt_trials],
        ["train_fraction", report.settings.optimization_config?.train_fraction],
        ["random_seed", report.settings.optimization_config?.random_seed]
      ].map(([key, value]) => `<tr><td>${htmlEscape(key)}</td><td>${htmlEscape(value)}</td></tr>`).join("");
      paramsGrid.innerHTML = `
        ${subjectBlocks}
        <div class="params-block">
          <h3>Run settings</h3>
          <table class="settings-table">${settingsRows}</table>
        </div>
      `;
    }

    function setupControls() {
      subjectSelect.innerHTML = report.subjects.map(subject => `<option value="${htmlEscape(subject.id)}">${htmlEscape(subject.id)}</option>`).join("");
      subjectSelect.value = state.subjectId;
      subjectSelect.addEventListener("change", event => {
        state.subjectId = event.target.value;
        state.selectedTrialId = null;
        renderAll();
      });

      toleranceButtons.innerHTML = (report.tolerance_options_deg || [15, 30, 45]).map(value => (
        `<button type="button" data-tolerance="${value}">+/-${value}</button>`
      )).join("");
      toleranceButtons.addEventListener("click", event => {
        const button = event.target.closest("button[data-tolerance]");
        if (!button) return;
        state.tolerance = Number(button.dataset.tolerance);
        renderAll();
      });

      methodButtons.querySelectorAll("[data-phase-mode]").forEach(button => {
        const mode = button.dataset.phaseMode;
        if (!report.causal_available && mode !== "noncausal") button.disabled = true;
      });
      methodButtons.addEventListener("click", event => {
        const button = event.target.closest("button[data-phase-mode]");
        if (!button || button.disabled) return;
        state.phaseMode = button.dataset.phaseMode;
        renderAll();
      });

      phaseSvg.addEventListener("click", event => {
        const target = event.target.closest("[data-trial-id]");
        if (!target) return;
        state.selectedTrialId = target.getAttribute("data-trial-id");
        renderAll();
      });
      phaseSvg.addEventListener("keydown", event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const target = event.target.closest("[data-trial-id]");
        if (!target) return;
        event.preventDefault();
        state.selectedTrialId = target.getAttribute("data-trial-id");
        renderAll();
      });
    }

    function updateControlState() {
      [...toleranceButtons.querySelectorAll("button")].forEach(button => {
        button.classList.toggle("active", Number(button.dataset.tolerance) === state.tolerance);
      });
      [...methodButtons.querySelectorAll("button[data-phase-mode]")].forEach(button => {
        button.classList.toggle("active", button.dataset.phaseMode === state.phaseMode);
      });
    }

    function renderRunMeta() {
      const generated = report.generated_at ? `Generated ${report.generated_at}` : "Generated from cached results";
      const subjectCount = report.subjects.length;
      const trialCount = report.total_trials || report.subjects.reduce((sum, subject) => sum + subject.n_trials, 0);
      const band = report.settings.band_hz || [];
      document.getElementById("runMeta").textContent = `${generated}. ${subjectCount} subject(s), ${trialCount} trial(s), ${report.settings.channel} signal, ${band.join("-")} Hz, cutoff ${report.settings.cutoff_ms} ms.`;
    }

    function renderAll() {
      updateControlState();
      renderPhase();
      renderSummary();
      renderSignal();
      renderParams();
    }

    setupControls();
    renderRunMeta();
    renderAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
