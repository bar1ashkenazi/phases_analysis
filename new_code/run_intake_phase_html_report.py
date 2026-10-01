"""Export an interactive intake phase/BOSS HTML report.

The expensive EEG loading, non-causal phase estimation, and optional causal
optimization run once and are saved to JSON. Everything derived from the phases
(classes per tolerance, BOSS success, deviation histograms) is computed by
``functions.analyze_phase_results`` each time the HTML is built, so analysis and
visual edits never require touching the EEG data again.

Run from the project root:

    ./.venv/bin/python new_code/run_intake_phase_html_report.py

Useful local workflows:

    # Rebuild only the HTML (and figures) from an existing cache.
    ./.venv/bin/python new_code/run_intake_phase_html_report.py --from-cache

    # Force a new calculation/cache.
    ./.venv/bin/python new_code/run_intake_phase_html_report.py --force

    # Fast no-data report from synthetic epochs run through the real pipeline.
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

import matplotlib.pyplot as plt

import settings
from functions import (
    CLASS_COLORS,
    METHOD_COLORS,
    TrialEstimate,
    analyze_phase_results,
    estimate_all_subjects,
    get_data,
    make_synthetic_epochs,
    plot_success_vs_tolerance,
    resolve_noncausal_filter_order,
)


REPORT_DIR = Path("new_code/intake_phase_report")
CACHE_FILENAME = "intake_phase_results.json"
HTML_FILENAME = "intake_phase_report.html"
SUCCESS_FIGURE_STEM = "success_vs_tolerance"
HISTOGRAM_BIN_WIDTH_DEG = 10.0
MAX_SIGNAL_POINTS = 4000
FLOAT_DECIMALS = 6
SIGNAL_SCALE = 1_000_000.0
SIGNAL_DECIMALS = 4
SIGNAL_UNIT = "uV"
DEMO_SUBJECTS = 6
DEMO_EPOCHS = 30


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

    report = prepare_report(payload)
    write_html_report(html_path, report)
    figure_paths = write_success_figure(output_dir, report)
    cache_label = "Wrote cache" if cache_written else "Used cache"
    print(f"{cache_label}: {cache_path}")
    print(f"Wrote HTML:  {html_path}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR,
        help=f"Folder for {CACHE_FILENAME}, {HTML_FILENAME} and the exported figures.",
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
        help="Optional subject list. Defaults to SUBJECTS in settings.py.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optional epoch cap per subject for fast local smoke runs. Defaults to N_TRIALS in settings.py.",
    )
    parser.add_argument(
        "--n-opt-trials",
        type=int,
        default=None,
        help="Optional Optuna trial (parameter-set) cap. Defaults to OPT_N_TRIALS in settings.py.",
    )
    return parser.parse_args()


def compute_payload(args: argparse.Namespace) -> dict[str, Any]:
    subjects = args.subjects if args.subjects is not None else settings.SUBJECTS
    n_trials = args.n_trials if args.n_trials is not None else settings.N_TRIALS
    optimization_config = dict(settings.CAUSAL_OPTIMIZATION_CONFIG)
    if args.n_opt_trials is not None:
        optimization_config["n_opt_trials"] = args.n_opt_trials

    print(f"Loading subjects: {subjects}")
    subject_data = [get_data(subject, settings.LOAD_CONFIG) for subject in subjects]

    print("Estimating phase and causal parameters...")
    estimates_by_subject = estimate_all_subjects(
        subject_data,
        band=settings.BAND,
        filter_order=settings.FILTER_ORDER,
        cutoff_ms=settings.CUTOFF_MS,
        n_trials=n_trials,
        causal_estimation=settings.CAUSAL_ESTIMATION,
        causal_params_mode=settings.CAUSAL_PARAMS_MODE,
        manual_causal_params=settings.MANUAL_CAUSAL_PARAMS,
        optimization_config=optimization_config,
    )
    return estimates_to_payload(
        estimates_by_subject,
        n_trials=n_trials,
        optimization_config=optimization_config,
        causal_params_mode=settings.CAUSAL_PARAMS_MODE,
        data_root=str(settings.DATA_ROOT),
    )


def estimates_to_payload(
    estimates_by_subject: dict[str, list[TrialEstimate]],
    n_trials: int | bool,
    optimization_config: dict[str, Any],
    causal_params_mode: str,
    data_root: str,
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
                "noncausal_filter_order": estimates[0].noncausal_filter_order if estimates else None,
                "trials": trials,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "subjects": list(estimates_by_subject.keys()),
            "data_root": data_root,
            "intake_filename": settings.INTAKE_FILENAME,
            "channel": settings.CHANNEL,
            "band_hz": list(settings.BAND),
            "filter_order": settings.FILTER_ORDER,
            "cutoff_ms": settings.CUTOFF_MS,
            "n_trials": n_trials,
            "causal_estimation": settings.CAUSAL_ESTIMATION,
            "causal_params_mode": causal_params_mode,
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
            "positive": settings.LABELS.positive,
            "negative": settings.LABELS.negative,
            "unknown": settings.LABELS.unknown,
            "unclassified": settings.LABELS.unclassified,
        },
        "class_order": list(settings.LABELS.class_order),
        "max_signal_points": MAX_SIGNAL_POINTS,
        "causal_available": causal_available,
        "total_trials": total_trials,
        "subjects": subject_payloads,
    }


def trial_to_payload(est: TrialEstimate) -> dict[str, Any]:
    signal_idx = decimation_indices(len(est.times_ms), MAX_SIGNAL_POINTS)
    causal = est.causal
    return {
        "trial_id": f"{est.subject}:{est.epoch_index}",
        "subject": est.subject,
        "epoch_index": est.epoch_index,
        "boss_class": est.label,
        "phase_deg": round_float(est.phase_deg),
        "phase_rad": round_float(est.phase_rad),
        "amplitude": round_float(est.amplitude),
        "causal_phase_deg": None if causal is None else round_float(causal.phase_deg),
        "causal_phase_rad": None if causal is None else round_float(causal.phase_rad),
        "causal_amplitude": None if causal is None else round_float(causal.amplitude),
        "causal_phase_error_deg": None if causal is None else round_float(causal.phase_error_deg),
        "signal": {
            "t": round_array(est.times_ms[signal_idx]),
            "raw": signal_array(est.raw[signal_idx]),
            "filtered": signal_array(est.filtered[signal_idx]),
        },
        "causal_core": None if causal is None else trace_payload(causal.core_times_ms, causal.core),
        "causal_future": None if causal is None else trace_payload(causal.future_times_ms, causal.pred_future),
    }


def first_causal_params(estimates: Sequence[TrialEstimate]) -> dict[str, Any] | None:
    for est in estimates:
        if est.causal is not None:
            return json_safe(est.causal.params)
    return None


def prepare_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the current analysis, colors and tolerances to a (possibly old) cached payload."""
    report = dict(payload)
    run_settings = report["settings"]
    subjects = []
    for subject in report["subjects"]:
        subject = dict(subject)
        if subject.get("noncausal_filter_order") is None:
            causal_params = subject.get("causal_params") if run_settings.get("causal_estimation") else None
            subject["noncausal_filter_order"] = resolve_noncausal_filter_order(causal_params, run_settings["filter_order"])
        subjects.append(subject)
    report["subjects"] = subjects

    report["class_colors"] = CLASS_COLORS
    report["method_colors"] = METHOD_COLORS
    report["tolerance_options_deg"] = sorted(settings.TOLERANCES_DEG)
    report["default_tolerance_deg"] = settings.DEFAULT_TOLERANCE_DEG
    report["analysis"] = json_safe(analyze_phase_results(
        {
            subject["id"]: {
                "boss": [trial["boss_class"] for trial in subject["trials"]],
                "noncausal_deg": [trial["phase_deg"] for trial in subject["trials"]],
                "causal_deg": [trial["causal_phase_deg"] for trial in subject["trials"]],
                "causal_error_deg": [trial["causal_phase_error_deg"] for trial in subject["trials"]],
            }
            for subject in subjects
        },
        tolerances_deg=list(settings.TOLERANCES_DEG),
        bin_width_deg=HISTOGRAM_BIN_WIDTH_DEG,
        labels=settings.LABELS,
    ))
    return report


def write_success_figure(output_dir: Path, report: dict[str, Any]) -> list[Path]:
    fig = plot_success_vs_tolerance(report["analysis"]["success"])
    paths = [output_dir / f"{SUCCESS_FIGURE_STEM}.pdf", output_dir / f"{SUCCESS_FIGURE_STEM}.png"]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round_float(value)
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
    """Synthetic epochs for several fake subjects, run through the real estimation pipeline."""
    subject_data = [
        make_synthetic_epochs(
            subject=f"demo_sub_{i + 1:03d}",
            n_epochs=DEMO_EPOCHS,
            phase_jitter_deg=20.0 + 8.0 * i,
            seed=7 + i,
            labels=settings.LABELS,
        )
        for i in range(DEMO_SUBJECTS)
    ]
    estimates_by_subject = estimate_all_subjects(
        subject_data,
        band=settings.BAND,
        filter_order=settings.FILTER_ORDER,
        cutoff_ms=settings.CUTOFF_MS,
        n_trials=False,
        causal_estimation=True,
        causal_params_mode="manual",
        manual_causal_params=settings.MANUAL_CAUSAL_PARAMS,
        optimization_config=settings.CAUSAL_OPTIMIZATION_CONFIG,
    )
    payload = estimates_to_payload(
        estimates_by_subject,
        n_trials=False,
        optimization_config=settings.CAUSAL_OPTIMIZATION_CONFIG,
        causal_params_mode="manual",
        data_root="synthetic demo",
    )
    payload["settings"]["causal_estimation"] = True
    payload["settings"]["raw_trace_label"] = "synthetic raw"
    return payload


def write_html_report(path: Path, report: dict[str, Any]) -> None:
    data_json = json.dumps(report, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
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
      --accent: #33363b;
      --accent-ink: #ffffff;
      --positive: #EE3377;
      --negative: #0077BB;
      --unknown: #858b93;
      --causal: #EE7733;
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
    .deviation-section,
    .success-section,
    .params-section {
      border-top: 1px solid var(--line);
      margin-top: 28px;
      padding-top: 22px;
    }

    .section-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 12px;
    }

    .stats-line {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .success-layout {
      display: grid;
      grid-template-columns: minmax(0, 640px) minmax(260px, 1fr);
      gap: 28px;
      align-items: start;
    }

    .table-scroll {
      overflow-x: auto;
    }

    .success-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }

    .success-table th {
      padding: 5px 8px 5px 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-weight: 650;
      text-align: left;
      white-space: nowrap;
    }

    .success-table td {
      padding: 5px 8px 5px 0;
      white-space: nowrap;
    }

    .success-table tr.selected td {
      font-weight: 720;
    }

    #successTable tbody tr[data-subject] {
      cursor: pointer;
    }

    #successTable tbody tr[data-subject]:hover td {
      background: var(--soft);
    }

    #successTable tr.highlighted td {
      background: #20211f;
      color: #ffffff;
      font-weight: 720;
    }

    #successTable.dimmed tbody tr[data-subject]:not(.highlighted) td {
      color: var(--muted);
    }

    .subject-hit {
      cursor: pointer;
    }

    .success-table tr.summary-line td {
      color: var(--ink);
      font-weight: 650;
    }

    .signal-wrap,
    .deviation-wrap,
    .success-wrap {
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

      .success-layout {
        grid-template-columns: 1fr;
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

    <section class="deviation-section">
      <div class="signal-head">
        <h2 id="deviationTitle">Phase deviation</h2>
        <p class="subtle" id="deviationSubtitle"></p>
      </div>
      <div class="section-toolbar">
        <div class="control">
          Deviation
          <span class="segmented" id="deviationKindButtons">
            <button type="button" data-deviation-kind="boss_target">Phase − BOSS target</button>
            <button type="button" data-deviation-kind="causal_error">Causal − non-causal</button>
          </span>
        </div>
        <label class="control">Subject <select id="deviationSubjectSelect"></select></label>
      </div>
      <div class="success-layout">
        <div class="deviation-wrap">
          <svg id="deviationSvg" viewBox="0 0 640 600" role="img" aria-label="Circular deviation histogram"></svg>
        </div>
        <div>
          <table class="success-table" id="deviationStats"></table>
          <p class="stats-line">Wedges: trials per 10° bin, stacked by BOSS class; 0° (no deviation) at the top, positive deviations clockwise. Arrows: mean resultant vector per BOSS class; direction = circular mean (bias), length = R (1 = all deviations identical, reaching the outer circle). Separate arrows keep opposite biases of positive and negative trials from cancelling out. Shaded: ±tolerance.</p>
        </div>
      </div>
    </section>

    <section class="success-section">
      <div class="signal-head">
        <h2>BOSS success vs tolerance</h2>
        <p class="subtle" id="successSubtitle"></p>
      </div>
      <div class="section-toolbar">
        <label class="control">Highlight subject <select id="highlightSelect"></select></label>
        <span class="subtle">or click a dot, line or table row</span>
      </div>
      <div class="success-layout">
        <div class="success-wrap">
          <svg id="successSvg" viewBox="0 0 640 440" role="img" aria-label="BOSS success versus tolerance"></svg>
        </div>
        <div class="table-scroll">
          <table class="success-table" id="successTable"></table>
        </div>
      </div>
    </section>

    <section class="params-section">
      <h2>Selected optimization params</h2>
      <div class="params-grid" id="paramsGrid"></div>
    </section>
  </main>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    // All analysis numbers (classes, statuses, counts, histograms, circular stats, success %)
    // are precomputed in functions.py and embedded in report.analysis; this script only draws them.
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const analysis = report.analysis;
    const NS = "http://www.w3.org/2000/svg";
    const POS = report.class_names.positive;
    const NEG = report.class_names.negative;
    const UNKNOWN = report.class_names.unknown;
    const classOrder = report.class_order || [POS, NEG, UNKNOWN];
    const colors = report.class_colors || {};
    const methodColors = report.method_colors || { raw: "#9a9e98", noncausal: "#33363b", causal: "#EE7733" };
    const ACCENT = "#33363b";
    const state = {
      subjectId: report.subjects[0]?.id || "",
      tolerance: report.default_tolerance_deg || 30,
      phaseMode: "noncausal",
      selectedTrialId: null,
      deviationKind: "boss_target",
      deviationSubject: "",
      highlightSubject: null
    };
    report.subjects.forEach(subject => subject.trials.forEach((trial, index) => { trial._index = index; }));

    const subjectSelect = document.getElementById("subjectSelect");
    const toleranceButtons = document.getElementById("toleranceButtons");
    const methodButtons = document.getElementById("methodButtons");
    const deviationKindButtons = document.getElementById("deviationKindButtons");
    const deviationSubjectSelect = document.getElementById("deviationSubjectSelect");
    const phaseSvg = document.getElementById("phaseSvg");
    const signalSvg = document.getElementById("signalSvg");
    const deviationSvg = document.getElementById("deviationSvg");
    const successSvg = document.getElementById("successSvg");
    const highlightSelect = document.getElementById("highlightSelect");

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

    function fmt(value, digits = 1, signed = false) {
      if (value === null || value === undefined) return "n/a";
      const text = value.toFixed(digits);
      return signed && value >= 0 ? `+${text}` : text;
    }

    function tolKey() {
      return String(state.tolerance);
    }

    function currentSubject() {
      return report.subjects.find(subject => subject.id === state.subjectId) || report.subjects[0];
    }

    function currentAnalysis() {
      return analysis.subjects[currentSubject()?.id] || null;
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

    function methodClass(trial, method) {
      return currentAnalysis()?.classes[method][tolKey()][trial._index] ?? null;
    }

    function methodPhase(trial, method) {
      return method === "causal" ? trial.causal_phase_deg : trial.phase_deg;
    }

    function comparisonStatus(trial, method) {
      return currentAnalysis()?.status[method][tolKey()][trial._index] ?? "unknown";
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
      const start = centerDeg - tolerance;
      const end = centerDeg + tolerance;
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
          fill: "#e4e6e2",
          opacity: "0.85"
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
          const d = circularConnectorPath(cx, cy, radiusFor(trial), trial.phase_deg, trial.causal_phase_deg);
          if (d) {
            phaseSvg.appendChild(el("path", { d, fill: "none", stroke: "#7c8279", "stroke-width": "1", opacity: "0.34" }));
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

      if (selected) drawSelectedPair(phaseSvg, selected, cx, cy, radiusFor(selected));
      drawPhaseLegend(phaseSvg, visibleMethods);
    }

    function drawSelectedPair(svg, trial, cx, cy, radius) {
      const [nonX, nonY] = polarPoint(cx, cy, radius, trial.phase_deg);
      if (hasCausal(trial)) {
        const [causalX, causalY] = polarPoint(cx, cy, radius, trial.causal_phase_deg);
        const d = circularConnectorPath(cx, cy, radius, trial.phase_deg, trial.causal_phase_deg);
        if (d) {
          svg.appendChild(el("path", { d, fill: "none", stroke: ACCENT, "stroke-width": "2", opacity: "0.72" }));
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
      svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: "#555", stroke: "#ffffff", "stroke-width": "1" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, visibleMethods.includes("noncausal") ? "non-causal" : "selected non-causal"));
      y += 24;
      svg.appendChild(el("path", { d: `M${x},${y - 8} L${x - 8},${y + 8} L${x + 8},${y + 8} Z`, fill: "#555", stroke: "#ffffff", "stroke-width": "1" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 5, class: "legend-text" }, visibleMethods.includes("causal") ? "causal" : "selected causal"));
      y += 29;
      svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: "#ffffff", stroke: "#555", "stroke-width": "2.2" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, "BOSS mismatch (hollow)"));
      y += 24;
      svg.appendChild(el("circle", { cx: x, cy: y, r: 6, fill: "#ffffff", stroke: "#555", "stroke-width": "1.4", opacity: "0.48" }));
      svg.appendChild(el("text", { x: x + 16, y: y + 4, class: "legend-text" }, "unclassified (faint)"));
    }

    function renderSummary() {
      const grid = document.getElementById("summaryGrid");
      const totals = document.getElementById("methodTotals");
      const subjectAnalysis = currentAnalysis();
      const visibleMethods = phaseModeMethods();
      if (!subjectAnalysis) {
        grid.innerHTML = "";
        totals.innerHTML = "";
        return;
      }
      const counts = method => subjectAnalysis.counts[method][tolKey()];
      grid.innerHTML = [POS, NEG].map(name => {
        const pieces = visibleMethods.map(method => `${methodLabel(method)} labels ${counts(method).by_class[name].correct} as ${name}.`);
        return `
          <div class="summary-row">
            <span class="color-bar" style="background:${htmlEscape(colors[name] || "#858b93")}"></span>
            <div>
              <strong>BOSS ${htmlEscape(name)}: ${counts("noncausal").by_class[name].n}</strong>
              <p>${pieces.map(htmlEscape).join(" ")}</p>
            </div>
          </div>
        `;
      }).join("");

      totals.innerHTML = visibleMethods.map(method => {
        const c = counts(method);
        return `<p><strong>${methodLabel(method)}</strong>: ${c.correct}/${c.n_labeled} BOSS matches (${fmt(c.success_pct, 1)}%), ${c.wrong} mismatches, ${c.unclassified} unclassified at +/-${state.tolerance} deg.</p>`;
      }).join("");
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
      let meta = `BOSS ${trial.boss_class}; non-causal ${methodClass(trial, "noncausal")}, ${trial.phase_deg?.toFixed(1)} deg`;
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
        { name: channelLabel, t, y: trial.signal.raw, color: methodColors.raw, width: 1.05, opacity: 0.9 },
        { name: "non-causal filtered", t, y: trial.signal.filtered, color: methodColors.noncausal, width: 1.8, opacity: 0.95 }
      ];
      if (trial.causal_core) {
        series.push({ name: "causal AR core", t: trial.causal_core.t, y: trial.causal_core.y, color: methodColors.causal, width: 1.55, opacity: 0.95 });
      }
      if (trial.causal_future) {
        series.push({ name: "causal AR prediction", t: trial.causal_future.t, y: trial.causal_future.y, color: methodColors.causal, width: 1.65, opacity: 0.9, dash: "5 5" });
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
        const x1 = xScale(Math.min(...trial.causal_future.t));
        const x2 = xScale(Math.max(...trial.causal_future.t));
        signalSvg.appendChild(el("rect", { x: x1, y: margin.top, width: Math.max(1, x2 - x1), height: innerH, fill: methodColors.causal, opacity: "0.09" }));
        signalSvg.appendChild(el("text", { x: x1 + 8, y: height - margin.bottom - 8, class: "legend-text" }, "AR prediction"));
      }
      for (let i = 0; i <= 4; i += 1) {
        const y = margin.top + (innerH / 4) * i;
        signalSvg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: "#e2e6df", "stroke-width": "1" }));
      }
      for (let value = Math.ceil(xMin / 200) * 200; value <= xMax + 0.001; value += 200) {
        const x = xScale(value);
        signalSvg.appendChild(el("line", { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, stroke: "#edf0ea", "stroke-width": "1" }));
      }
      const zeroX = xScale(report.settings.cutoff_ms || 0);
      signalSvg.appendChild(el("line", { x1: zeroX, x2: zeroX, y1: margin.top, y2: height - margin.bottom, stroke: "#8b9189", "stroke-dasharray": "4 5", "stroke-width": "1.3" }));

      signalSvg.appendChild(el("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#aeb6ab" }));
      signalSvg.appendChild(el("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "#aeb6ab" }));
      [xMin, 0, xMax].forEach(value => {
        if (value < xMin || value > xMax) return;
        signalSvg.appendChild(el("text", { x: xScale(value), y: height - 16, class: "axis-label", "text-anchor": "middle" }, `${Math.round(value)} ms`));
      });
      signalSvg.appendChild(el("text", { x: 18, y: 28, class: "axis-label" }, `Amplitude (${report.settings.signal_unit || "a.u."})`));

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

    function niceStep(maxValue, targetTicks) {
      const raw = Math.max(maxValue, 1) / targetTicks;
      const power = 10 ** Math.floor(Math.log10(raw));
      const unit = [1, 2, 5, 10].find(m => m * power >= raw) || 10;
      return Math.max(1, unit * power);
    }

    function renderDeviation() {
      const kind = state.deviationKind;
      const pooled = !analysis.subjects[state.deviationSubject];
      const histogram = pooled ? analysis.pooled[kind] : analysis.subjects[state.deviationSubject].deviations[kind];
      deviationSubjectSelect.value = pooled ? "" : state.deviationSubject;
      document.getElementById("deviationTitle").textContent = kind === "causal_error" ? "Causal estimator error (causal − non-causal)" : "Phase deviation from BOSS target (non-causal − target)";
      document.getElementById("deviationSubtitle").textContent = `${pooled ? "All subjects pooled" : state.deviationSubject}, 10° bins, ±${state.tolerance}° shaded`;
      deviationSvg.replaceChildren();
      const statsTable = document.getElementById("deviationStats");

      const stats = histogram?.stats;
      if (!histogram || !stats?.n) {
        deviationSvg.appendChild(el("text", { x: 320, y: 300, class: "empty" }, "No deviations available"));
        statsTable.innerHTML = "";
        return;
      }

      const width = 640;
      const height = 600;
      const cx = width / 2;
      const cy = 312;
      const rMax = 230;
      const edges = histogram.bin_edges_deg;
      const nBins = edges.length - 1;
      const totals = Array.from({ length: nBins }, (_, i) => classOrder.reduce((sum, name) => sum + (histogram.counts_by_class[name]?.[i] || 0), 0));
      const maxTotal = Math.max(...totals, 1);
      const step = niceStep(maxTotal, 4);
      const countMax = maxTotal * 1.05;
      const rScale = count => (count / countMax) * rMax;

      deviationSvg.appendChild(el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      drawToleranceWindow(deviationSvg, cx, cy, 0, rMax, 0, state.tolerance);
      deviationSvg.appendChild(el("circle", { cx, cy, r: rMax, fill: "none", stroke: "#cfd3cc" }));
      for (let value = step; value <= countMax; value += step) {
        deviationSvg.appendChild(el("circle", { cx, cy, r: rScale(value), fill: "none", stroke: "#e2e6df" }));
        const [lx, ly] = polarPoint(cx, cy, rScale(value), 112.5);
        deviationSvg.appendChild(el("text", { x: lx + 4, y: ly, class: "legend-text" }, String(value)));
      }
      [0, 45, 90, 135, 180, 225, 270, 315].forEach(deg => {
        const [x2, y2] = polarPoint(cx, cy, rMax, deg);
        deviationSvg.appendChild(el("line", { x1: cx, y1: cy, x2, y2, stroke: deg === 0 ? "#3a3c38" : "#e2e6df", "stroke-dasharray": deg === 0 ? "3 4" : null }));
        const [lx, ly] = polarPoint(cx, cy, rMax + 24, deg);
        const label = deg === 0 ? "0°" : deg === 180 ? "±180°" : deg < 180 ? `+${deg}°` : `−${360 - deg}°`;
        deviationSvg.appendChild(el("text", { x: lx, y: ly + 4, class: "axis-label", "text-anchor": "middle" }, label));
      });

      for (let i = 0; i < nBins; i += 1) {
        let base = 0;
        classOrder.forEach(name => {
          const count = histogram.counts_by_class[name]?.[i] || 0;
          if (!count) return;
          const wedge = el("path", {
            d: sectorPath(cx, cy, rScale(base), rScale(base + count), edges[i], edges[i + 1]),
            fill: colors[name] || "#858b93", stroke: "#ffffff", "stroke-width": "0.8", opacity: "0.92"
          });
          wedge.appendChild(el("title", {}, `${edges[i]}° to ${edges[i + 1]}° | BOSS ${name}: ${count}`));
          deviationSvg.appendChild(wedge);
          base += count;
        });
      }

      const arrowClasses = classOrder.filter(name => histogram.stats_by_class?.[name]?.n);
      arrowClasses.forEach(name => {
        const classStats = histogram.stats_by_class[name];
        const [ax, ay] = polarPoint(cx, cy, classStats.R * rMax, classStats.mean_deg);
        const theta = classStats.mean_deg * Math.PI / 180;
        const ux = Math.sin(theta);
        const uy = -Math.cos(theta);
        const head = [[ax + ux * 7, ay + uy * 7], [ax - ux * 9 - uy * 8, ay - uy * 9 + ux * 8], [ax - ux * 9 + uy * 8, ay - uy * 9 - ux * 8]];
        const headPath = `M${head.map(p => p.map(v => v.toFixed(2)).join(",")).join(" L")} Z`;
        const color = colors[name] || "#858b93";
        const group = el("g");
        group.appendChild(el("title", {}, `BOSS ${name}: mean ${fmt(classStats.mean_deg, 1, true)}°, R ${fmt(classStats.R, 3)}, n ${classStats.n}`));
        group.appendChild(el("line", { x1: cx, y1: cy, x2: ax, y2: ay, stroke: "#ffffff", "stroke-width": "7", "stroke-linecap": "round" }));
        group.appendChild(el("path", { d: headPath, fill: "#ffffff", stroke: "#ffffff", "stroke-width": "4", "stroke-linejoin": "round" }));
        group.appendChild(el("line", { x1: cx, y1: cy, x2: ax, y2: ay, stroke: color, "stroke-width": "3.5", "stroke-linecap": "round" }));
        group.appendChild(el("path", { d: headPath, fill: color }));
        deviationSvg.appendChild(group);
      });
      deviationSvg.appendChild(el("circle", { cx, cy, r: 3, fill: "#20211f" }));

      let ly = 24;
      classOrder.forEach(name => {
        if (!histogram.counts_by_class[name]?.some(v => v)) return;
        deviationSvg.appendChild(el("rect", { x: 16, y: ly - 10, width: 12, height: 12, fill: colors[name] || "#858b93" }));
        deviationSvg.appendChild(el("text", { x: 34, y: ly, class: "legend-text" }, `BOSS ${name}`));
        ly += 20;
      });
      deviationSvg.appendChild(el("text", { x: 16, y: ly, class: "legend-text" }, "arrows = class mean vectors"));
      deviationSvg.appendChild(el("text", { x: width - 16, y: 24, class: "legend-text", "text-anchor": "end" }, "rings = trials"));

      const columns = [["All", stats], ...arrowClasses.map(name => [name, histogram.stats_by_class[name]])];
      const rows = [
        ["Trials (n)", st => String(st.n)],
        ["Circular mean (bias)", st => `${fmt(st.mean_deg, 1, true)}°`],
        ["Resultant length R", st => fmt(st.R, 3)],
        ["Circular SD", st => `${fmt(st.sd_deg, 1)}°`],
        [`Within ±${state.tolerance}°`, st => `${fmt(st.pct_within[tolKey()], 1)}%`]
      ];
      const head = `<tr><th></th>${columns.map(([name]) => name === "All"
        ? "<th>All</th>"
        : `<th><span style="color:${htmlEscape(colors[name] || "#858b93")}">■</span> ${htmlEscape(name)}</th>`).join("")}</tr>`;
      statsTable.innerHTML = `<thead>${head}</thead><tbody>${rows.map(([label, value]) => `<tr><td>${htmlEscape(label)}</td>${columns.map(([, st]) => `<td>${htmlEscape(value(st))}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }

    function setHighlight(subjectId) {
      state.highlightSubject = state.highlightSubject === subjectId ? null : subjectId;
      renderAll();
    }

    function renderSuccess() {
      const success = analysis.success;
      const tolerances = success.tolerances_deg;
      const subjects = success.subjects;
      const entries = tolerances.map(tol => success.by_tolerance[String(tol)]);
      const highlight = subjects.includes(state.highlightSubject) ? state.highlightSubject : null;
      highlightSelect.value = highlight || "";
      document.getElementById("successSubtitle").textContent =
        `Non-causal phase within ±T of the BOSS target, % of BOSS-labeled trials per subject (unclassified = failure). n = ${subjects.length} subjects; bar = mean, whiskers = ±1 SD; dashed = chance (2T/360).`;
      successSvg.replaceChildren();

      const margin = { left: 64, right: 24, top: 24, bottom: 58 };
      const width = 640;
      const height = 440;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const slot = innerW / tolerances.length;
      const xCenter = i => margin.left + slot * (i + 0.5);
      const yScale = value => margin.top + (1 - value / 100) * innerH;
      const jitter = s => subjects.length > 1 ? (-0.13 + 0.26 * s / (subjects.length - 1)) * slot : 0;
      const subjectPoints = s => entries.map((entry, i) => [xCenter(i) + jitter(s), entry.success_pct[s], i]).filter(([, v]) => v !== null);
      const pathFor = points => points.map(([x, v], k) => `${k ? "L" : "M"}${x.toFixed(2)},${yScale(v).toFixed(2)}`).join("");

      successSvg.appendChild(el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
      for (let value = 0; value <= 100; value += 20) {
        const y = yScale(value);
        successSvg.appendChild(el("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: "#edf0ea" }));
        successSvg.appendChild(el("text", { x: margin.left - 10, y: y + 4, class: "axis-label", "text-anchor": "end" }, `${value}%`));
      }
      successSvg.appendChild(el("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "#aeb6ab" }));

      entries.forEach((entry, i) => {
        const y = yScale(entry.chance_pct);
        successSvg.appendChild(el("line", { x1: xCenter(i) - slot * 0.4, x2: xCenter(i) + slot * 0.4, y1: y, y2: y, stroke: methodColors.causal, "stroke-dasharray": "6 4", "stroke-width": "1.6" }));
        successSvg.appendChild(el("text", { x: xCenter(i), y: height - margin.bottom + 20, class: "axis-label", "text-anchor": "middle" }, `±${tolerances[i]}°`));
      });
      successSvg.appendChild(el("text", { x: margin.left + innerW / 2, y: height - 12, class: "axis-label", "text-anchor": "middle" }, "Tolerance around BOSS target (deg)"));
      successSvg.appendChild(el("text", { x: 14, y: 16, class: "axis-label" }, "BOSS success"));
      const legendX = width - margin.right - 150;
      successSvg.appendChild(el("line", { x1: legendX, x2: legendX + 24, y1: margin.top + 8, y2: margin.top + 8, stroke: methodColors.causal, "stroke-dasharray": "6 4", "stroke-width": "1.6" }));
      successSvg.appendChild(el("text", { x: legendX + 32, y: margin.top + 12, class: "legend-text" }, "chance (2T/360)"));
      successSvg.appendChild(el("line", { x1: legendX, x2: legendX + 24, y1: margin.top + 28, y2: margin.top + 28, stroke: "#20211f", "stroke-width": "3" }));
      successSvg.appendChild(el("text", { x: legendX + 32, y: margin.top + 32, class: "legend-text" }, "mean ± SD"));

      subjects.forEach((subjectId, s) => {
        if (subjectId === highlight) return;
        const points = subjectPoints(s);
        if (points.length > 1) {
          successSvg.appendChild(el("path", { d: pathFor(points), fill: "none", stroke: highlight ? "#dfe2dc" : "#c3c7c0", "stroke-width": "1" }));
        }
      });
      entries.forEach((entry, i) => {
        const x = xCenter(i);
        if (entry.mean_pct === null) return;
        if (entry.sd_pct !== null) {
          const top = yScale(Math.min(100, entry.mean_pct + entry.sd_pct));
          const bottom = yScale(Math.max(0, entry.mean_pct - entry.sd_pct));
          successSvg.appendChild(el("line", { x1: x, x2: x, y1: top, y2: bottom, stroke: "#20211f", "stroke-width": "1.6" }));
          [top, bottom].forEach(yy => successSvg.appendChild(el("line", { x1: x - 8, x2: x + 8, y1: yy, y2: yy, stroke: "#20211f", "stroke-width": "1.6" })));
        }
        const meanY = yScale(entry.mean_pct);
        successSvg.appendChild(el("line", { x1: x - slot * 0.26, x2: x + slot * 0.26, y1: meanY, y2: meanY, stroke: "#20211f", "stroke-width": "3" }));
      });

      const drawSubject = (subjectId, s) => {
        const isHighlighted = subjectId === highlight;
        const points = subjectPoints(s);
        if (isHighlighted && points.length > 1) {
          successSvg.appendChild(el("path", { d: pathFor(points), fill: "none", stroke: "#20211f", "stroke-width": "2.4" }));
        }
        const hit = el("path", { d: pathFor(points), fill: "none", stroke: "transparent", "stroke-width": "10", class: "subject-hit", "data-subject": subjectId });
        hit.appendChild(el("title", {}, subjectId));
        successSvg.appendChild(hit);
        points.forEach(([px, value, i]) => {
          const entry = entries[i];
          const dot = el("circle", {
            cx: px.toFixed(2), cy: yScale(value).toFixed(2), r: isHighlighted ? 6.5 : 4.5,
            fill: isHighlighted ? "#20211f" : highlight ? "#b9bdb6" : methodColors.noncausal,
            opacity: isHighlighted ? "1" : "0.75", stroke: "#ffffff", "stroke-width": isHighlighted ? "1.5" : "0.6",
            class: "subject-hit", "data-subject": subjectId
          });
          dot.appendChild(el("title", {}, `${subjectId} | ±${tolerances[i]}°: ${entry.correct[s]}/${entry.n_labeled[s]} (${fmt(value, 1)}%)`));
          successSvg.appendChild(dot);
        });
        if (isHighlighted && points.length) {
          const [px, value] = points[0];
          successSvg.appendChild(el("text", { x: px - 12, y: yScale(value) + 4, class: "legend-text", "text-anchor": "end", "font-weight": "700", fill: "#20211f" }, subjectId));
        }
      };
      subjects.forEach((subjectId, s) => { if (subjectId !== highlight) drawSubject(subjectId, s); });
      if (highlight) drawSubject(highlight, subjects.indexOf(highlight));

      const header = `<tr><th>Subject</th>${tolerances.map(t => `<th>±${t}°</th>`).join("")}</tr>`;
      const rows = subjects.map((subjectId, s) => `
        <tr data-subject="${htmlEscape(subjectId)}" class="${subjectId === highlight ? "highlighted" : ""}">
          <td>${htmlEscape(subjectId)}</td>
          ${entries.map(entry => `<td>${entry.correct[s]}/${entry.n_labeled[s]} (${fmt(entry.success_pct[s], 0)}%)</td>`).join("")}
        </tr>`).join("");
      const meanRow = `<tr class="summary-line"><td>mean ± SD</td>${entries.map(entry => `<td>${fmt(entry.mean_pct, 1)} ± ${fmt(entry.sd_pct, 1)}%</td>`).join("")}</tr>`;
      const chanceRow = `<tr class="summary-line"><td>chance</td>${entries.map(entry => `<td>${fmt(entry.chance_pct, 1)}%</td>`).join("")}</tr>`;
      const table = document.getElementById("successTable");
      table.classList.toggle("dimmed", Boolean(highlight));
      table.innerHTML = `<thead>${header}</thead><tbody>${rows}${meanRow}${chanceRow}</tbody>`;
    }

    function renderParams() {
      const paramsGrid = document.getElementById("paramsGrid");
      const subjectBlocks = report.subjects.map(subject => {
        const params = subject.causal_params;
        const rows = [`<tr><td>non-causal filter_order (used)</td><td>${htmlEscape(subject.noncausal_filter_order)}</td></tr>`];
        if (params) {
          Object.entries(params).forEach(([key, value]) => rows.push(`<tr><td>${htmlEscape(key)}</td><td>${htmlEscape(value)}</td></tr>`));
        } else {
          rows.push(`<tr><td>causal params</td><td>not available</td></tr>`);
        }
        return `
          <div class="params-block">
            <h3>${htmlEscape(subject.id)}</h3>
            <table>${rows.join("")}</table>
          </div>
        `;
      }).join("");
      const settingsRows = [
        ["band_hz", (report.settings.band_hz || []).join("-")],
        ["cutoff_ms", report.settings.cutoff_ms],
        ["channel", report.settings.channel],
        ["default filter_order (used when not optimizing)", report.settings.filter_order],
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

    function setupSegmented(container, attribute, onSelect) {
      container.addEventListener("click", event => {
        const button = event.target.closest(`button[${attribute}]`);
        if (!button || button.disabled) return;
        onSelect(button.getAttribute(attribute));
        renderAll();
      });
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
      setupSegmented(toleranceButtons, "data-tolerance", value => { state.tolerance = Number(value); });

      methodButtons.querySelectorAll("[data-phase-mode]").forEach(button => {
        if (!report.causal_available && button.dataset.phaseMode !== "noncausal") button.disabled = true;
      });
      setupSegmented(methodButtons, "data-phase-mode", value => { state.phaseMode = value; });

      deviationKindButtons.querySelectorAll("[data-deviation-kind]").forEach(button => {
        if (!report.causal_available && button.dataset.deviationKind === "causal_error") button.disabled = true;
      });
      setupSegmented(deviationKindButtons, "data-deviation-kind", value => { state.deviationKind = value; });
      deviationSubjectSelect.innerHTML = `<option value="">All subjects (pooled)</option>` + report.subjects.map(subject => `<option value="${htmlEscape(subject.id)}">${htmlEscape(subject.id)}</option>`).join("");
      deviationSubjectSelect.addEventListener("change", event => {
        state.deviationSubject = event.target.value;
        renderAll();
      });

      highlightSelect.innerHTML = `<option value="">none</option>` + analysis.success.subjects.map(id => `<option value="${htmlEscape(id)}">${htmlEscape(id)}</option>`).join("");
      highlightSelect.addEventListener("change", event => {
        state.highlightSubject = event.target.value || null;
        renderAll();
      });
      const highlightFromEvent = event => {
        const target = event.target.closest("[data-subject]");
        if (target) setHighlight(target.getAttribute("data-subject"));
      };
      successSvg.addEventListener("click", highlightFromEvent);
      document.getElementById("successTable").addEventListener("click", highlightFromEvent);

      const selectTrial = event => {
        const target = event.target.closest("[data-trial-id]");
        if (!target) return;
        event.preventDefault();
        state.selectedTrialId = target.getAttribute("data-trial-id");
        renderAll();
      };
      phaseSvg.addEventListener("click", selectTrial);
      phaseSvg.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") selectTrial(event);
      });
    }

    function updateControlState() {
      const toggle = (container, attribute, value) => {
        container.querySelectorAll(`button[${attribute}]`).forEach(button => {
          button.classList.toggle("active", button.getAttribute(attribute) === String(value));
        });
      };
      toggle(toleranceButtons, "data-tolerance", state.tolerance);
      toggle(methodButtons, "data-phase-mode", state.phaseMode);
      toggle(deviationKindButtons, "data-deviation-kind", state.deviationKind);
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
      renderDeviation();
      renderSuccess();
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
