#!/usr/bin/env python3
"""Plot diagnostics for recursive heavy training.

This script supports two data sources:
1) `logs` mode: reads `pass_*_logs.csv` (and optional exact metrics files).
2) `models` mode: if logs are missing, it rebuilds pass evaluations from
   `recursive/models/pass_*/block_*.npz` and computes diagnostics directly
   from stitched predictions (optionally with exact solution).
"""

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    PLOTTING_AVAILABLE = False


PASS_LOG_RE = re.compile(r"^pass_(\d+)_logs\.csv$")
EXACT_METRICS_RE = re.compile(r"^exact_metrics_pass(\d+)\.json$")
PASS_DIR_RE = re.compile(r"^pass_(\-?\d+)$")
BLOCK_NPZ_RE = re.compile(r"^block_(\d+)\.npz$")


def _is_finite_number(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _to_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return float(default)
    try:
        return float(text)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    if not _is_finite_number(value):
        return int(default)
    return int(round(float(value)))


def _infer_zero_based(pass_ids: List[int]) -> bool:
    return len(pass_ids) > 0 and min(pass_ids) <= 0


def _pass_index(pass_id: int, zero_based: bool) -> int:
    pid = int(pass_id)
    return pid if zero_based else pid - 1


def _has_pass_logs(recursive_dir: str) -> bool:
    return len(glob.glob(os.path.join(recursive_dir, "pass_*_logs.csv"))) > 0


def _has_models(recursive_dir: str) -> bool:
    models_dir = os.path.join(recursive_dir, "models")
    if not os.path.isdir(models_dir):
        return False
    for name in os.listdir(models_dir):
        if PASS_DIR_RE.match(name) and os.path.isdir(os.path.join(models_dir, name)):
            return True
    return False


def _candidate_recursive_dirs(path: str) -> List[str]:
    path = os.path.abspath(os.path.expanduser(path))
    out: List[str] = []
    if os.path.isdir(path):
        out.append(path)
    nested = os.path.join(path, "recursive")
    if os.path.isdir(nested):
        out.append(nested)
    # Common case: input is a parent folder containing run_xxx subfolders.
    try:
        for name in sorted(os.listdir(path)):
            child = os.path.join(path, name)
            cand = os.path.join(child, "recursive")
            if os.path.isdir(cand):
                out.append(cand)
    except Exception:
        pass

    seen = set()
    dedup: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _resolve_artifact_context(input_dir: str) -> Dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(str(input_dir)))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input directory not found: {path}")

    # If input is exactly a models directory.
    if os.path.basename(path) == "models" and os.path.isdir(path):
        recursive_dir = os.path.dirname(path)
        if os.path.isdir(recursive_dir):
            return {
                "recursive_dir": recursive_dir,
                "models_dir": path,
                "has_logs": _has_pass_logs(recursive_dir),
                "has_models": True,
            }

    best: Optional[Dict[str, Any]] = None
    for recursive_dir in _candidate_recursive_dirs(path):
        ctx = {
            "recursive_dir": recursive_dir,
            "models_dir": os.path.join(recursive_dir, "models"),
            "has_logs": _has_pass_logs(recursive_dir),
            "has_models": _has_models(recursive_dir),
        }
        if ctx["has_logs"]:
            return ctx
        if ctx["has_models"] and best is None:
            best = ctx

    if best is not None:
        return best

    raise FileNotFoundError(
        "Could not find recursive artifacts. Expected either pass_*_logs.csv or "
        f"recursive/models/pass_* folders under: {path}"
    )


def _read_pass_log_file(path: str, fallback_from_name: int) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return None

    normalized: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        for key in (
            "pass",
            "block",
            "t_start",
            "t_end",
            "T_block",
            "eval_mean_loss",
            "eval_std_loss",
            "eval_mean_loss_per_sample",
            "eval_std_loss_per_sample",
            "eval_mean_y0",
            "precision_target",
            "refine_rounds",
        ):
            if key in row:
                row[key] = _to_float(row[key], default=np.nan)
        normalized.append(row)

    normalized = sorted(normalized, key=lambda r: _safe_int(r.get("block", np.nan)))
    pass_col_values = [r.get("pass", np.nan) for r in normalized if _is_finite_number(r.get("pass", np.nan))]
    if len(pass_col_values) > 0:
        pass_id = _safe_int(pass_col_values[0], default=fallback_from_name)
    else:
        pass_id = int(fallback_from_name)
    return int(pass_id), normalized


def _load_pass_logs(recursive_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    by_pass_id: Dict[int, List[Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(recursive_dir, "pass_*_logs.csv"))):
        name = os.path.basename(path)
        match = PASS_LOG_RE.match(name)
        if not match:
            continue

        file_num = int(match.group(1))
        parsed = _read_pass_log_file(path, fallback_from_name=file_num)
        if parsed is None:
            continue
        pass_id, rows = parsed

        existing = by_pass_id.get(pass_id)
        if existing is None or len(rows) >= len(existing):
            by_pass_id[pass_id] = rows

    if len(by_pass_id) == 0:
        raise RuntimeError(f"No valid pass logs found in {recursive_dir}")
    return by_pass_id


def _pick_loss_key(pass_rows_by_index: Dict[int, List[Dict[str, Any]]]) -> str:
    preferred = "eval_mean_loss_per_sample"
    for rows in pass_rows_by_index.values():
        if len(rows) == 0:
            continue
        vals = np.array([_to_float(r.get(preferred, np.nan)) for r in rows], dtype=np.float64)
        if not np.isfinite(vals).any():
            return "eval_mean_loss"
    return preferred


def _load_results_summary(recursive_dir: str) -> Dict[str, Any]:
    path = os.path.join(recursive_dir, "results.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_selected_pass_index(
    results_summary: Dict[str, Any],
    id_to_index: Dict[int, int],
) -> Optional[int]:
    if "selected_pass_index" in results_summary and _is_finite_number(results_summary.get("selected_pass_index")):
        return _safe_int(results_summary.get("selected_pass_index"))

    if "selected_pass_id" in results_summary and _is_finite_number(results_summary.get("selected_pass_id")):
        pid = _safe_int(results_summary.get("selected_pass_id"))
        return id_to_index.get(pid, None)

    return None


def _load_exact_metrics(
    recursive_dir: str,
    id_to_index: Dict[int, int],
    zero_based: bool,
) -> Dict[int, Dict[str, Any]]:
    exact_by_index: Dict[int, Dict[str, Any]] = {}
    pass_ids = set(id_to_index.keys())
    pass_indices = set(id_to_index.values())

    for path in sorted(glob.glob(os.path.join(recursive_dir, "exact_metrics_pass*.json"))):
        name = os.path.basename(path)
        match = EXACT_METRICS_RE.match(name)
        if not match:
            continue
        num = int(match.group(1))

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            continue

        if num in pass_indices and num not in pass_ids:
            idx = num
        elif num in pass_ids and num not in pass_indices:
            idx = id_to_index[num]
        elif num in pass_indices:
            idx = num
        elif num in pass_ids:
            idx = id_to_index[num]
        else:
            idx = num if zero_based else num - 1
        exact_by_index[int(idx)] = summary

    return exact_by_index


def _plot_block_metric(
    pass_rows_by_index: Dict[int, List[Dict[str, Any]]],
    key: str,
    title: str,
    ylabel: str,
    out_path: str,
    ylog: bool,
) -> None:
    if not PLOTTING_AVAILABLE:
        return
    if len(pass_rows_by_index) == 0:
        return

    pass_indices = sorted(pass_rows_by_index.keys())
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(pass_indices), 2)))

    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(pass_indices):
        rows = pass_rows_by_index[idx]
        if len(rows) == 0:
            continue
        blocks = np.array([_safe_int(r.get("block", np.nan), j) for j, r in enumerate(rows)], dtype=np.int32)
        values = np.array([_to_float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        plt.plot(blocks, values, marker="o", linewidth=1.6, color=colors[i], label=f"pass{idx}")

    if ylog:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Block index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pass_summary(
    pass_rows_by_index: Dict[int, List[Dict[str, Any]]],
    loss_key: str,
    selected_index: Optional[int],
    out_path: str,
) -> None:
    if not PLOTTING_AVAILABLE:
        return
    pass_indices = sorted(pass_rows_by_index.keys())
    if len(pass_indices) == 0:
        return

    mean_loss = []
    worst_loss = []
    composite = []
    for idx in pass_indices:
        rows = pass_rows_by_index[idx]
        vals = np.array([_to_float(r.get(loss_key, np.nan)) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean_loss.append(np.nan)
            worst_loss.append(np.nan)
            composite.append(np.nan)
        else:
            m = float(np.mean(vals))
            w = float(np.max(vals))
            mean_loss.append(m)
            worst_loss.append(w)
            composite.append(m + 0.35 * w)

    plt.figure(figsize=(10, 6))
    plt.plot(pass_indices, mean_loss, "o-", linewidth=1.6, label=f"{loss_key} mean")
    plt.plot(pass_indices, worst_loss, "o-", linewidth=1.4, label=f"{loss_key} worst block")
    plt.plot(pass_indices, composite, "o-", linewidth=1.8, label=f"{loss_key} score(mean+0.35*worst)")
    if selected_index is not None:
        plt.axvline(int(selected_index), linestyle="--", linewidth=1.0, color="tab:red", alpha=0.45, label="selected")
    plt.yscale("log")
    plt.title("Heavy training - pass summary score")
    plt.xlabel("Pass index")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_refine_rounds(
    pass_rows_by_index: Dict[int, List[Dict[str, Any]]],
    selected_index: Optional[int],
    out_path: str,
) -> None:
    if not PLOTTING_AVAILABLE:
        return
    pass_indices = sorted(pass_rows_by_index.keys())
    if len(pass_indices) == 0:
        return

    total_rounds = []
    max_rounds = []
    for idx in pass_indices:
        vals = np.array([_to_float(r.get("refine_rounds", np.nan), 0.0) for r in pass_rows_by_index[idx]], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        total_rounds.append(float(np.sum(vals)) if vals.size > 0 else np.nan)
        max_rounds.append(float(np.max(vals)) if vals.size > 0 else np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(pass_indices, total_rounds, "o-", linewidth=1.8, label="Total refine rounds")
    plt.plot(pass_indices, max_rounds, "o-", linewidth=1.5, label="Max refine rounds on one block")
    if selected_index is not None:
        plt.axvline(int(selected_index), linestyle="--", linewidth=1.0, color="tab:red", alpha=0.45, label="selected")
    plt.title("Heavy training - refine rounds across passes")
    plt.xlabel("Pass index")
    plt.ylabel("Rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_exact_summary(
    exact_by_index: Dict[int, Dict[str, Any]],
    selected_index: Optional[int],
    out_path: str,
) -> None:
    if not PLOTTING_AVAILABLE or len(exact_by_index) == 0:
        return

    pass_indices = sorted(exact_by_index.keys())
    keys = [
        ("mean_abs_error_y", "mean_abs_error_y"),
        ("mean_abs_error_z", "mean_abs_error_z"),
        ("abs_error_mean_y0", "abs_error_mean_y0"),
        ("rmse_y", "rmse_y"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, label) in zip(axes.flatten(), keys):
        values = np.array([_to_float(exact_by_index[idx].get(key, np.nan)) for idx in pass_indices], dtype=np.float64)
        ax.plot(pass_indices, values, "o-", linewidth=1.7)
        if np.isfinite(values).any() and np.nanmax(values) > 0.0:
            ax.set_yscale("log")
        if selected_index is not None:
            ax.axvline(int(selected_index), linestyle="--", linewidth=1.0, color="tab:red", alpha=0.45)
        ax.set_title(label)
        ax.set_xlabel("Pass index")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Heavy training - exact metrics across passes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_model_mean_y(stitched_by_index: Dict[int, Dict[str, np.ndarray]], out_path: str) -> None:
    if not PLOTTING_AVAILABLE or len(stitched_by_index) == 0:
        return
    pass_indices = sorted(stitched_by_index.keys())
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, max(len(pass_indices), 2)))

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(pass_indices):
        stitched = stitched_by_index[idx]
        t = stitched["t"][0, :, 0]
        y_mean = np.mean(stitched["Y"][:, :, 0], axis=0)
        plt.plot(t, y_mean, linewidth=1.8, color=colors[i], label=f"pass{idx} mean Y")
    plt.title("Heavy training (models) - mean Y(t) by pass")
    plt.xlabel("Time")
    plt.ylabel("Mean Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_model_y0(stats_rows: List[Dict[str, Any]], selected_index: Optional[int], out_path: str) -> None:
    if not PLOTTING_AVAILABLE or len(stats_rows) == 0:
        return
    rows = sorted(stats_rows, key=lambda r: int(r["pass_index"]))
    idx = [int(r["pass_index"]) for r in rows]
    y0_mean = np.array([_to_float(r.get("mean_y0", np.nan)) for r in rows], dtype=np.float64)
    y0_std = np.array([_to_float(r.get("std_y0", np.nan)) for r in rows], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.errorbar(idx, y0_mean, yerr=y0_std, fmt="o-", linewidth=1.8, capsize=4, label="mean_y0 ± std")
    if selected_index is not None:
        plt.axvline(int(selected_index), linestyle="--", linewidth=1.0, color="tab:red", alpha=0.45, label="selected")
    plt.title("Heavy training (models) - Y0 by pass")
    plt.xlabel("Pass index")
    plt.ylabel("Y0")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_model_drift(stats_rows: List[Dict[str, Any]], out_path: str) -> None:
    if not PLOTTING_AVAILABLE or len(stats_rows) == 0:
        return
    rows = sorted(stats_rows, key=lambda r: int(r["pass_index"]))
    idx = [int(r["pass_index"]) for r in rows]
    drift_y = np.array([_to_float(r.get("drift_y_vs_prev", np.nan)) for r in rows], dtype=np.float64)
    drift_z = np.array([_to_float(r.get("drift_z_vs_prev", np.nan)) for r in rows], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.plot(idx, drift_y, "o-", linewidth=1.8, label="mean |Y - Y_prev|")
    plt.plot(idx, drift_z, "o-", linewidth=1.6, label="mean |Z - Z_prev|")
    finite_vals = np.concatenate([drift_y[np.isfinite(drift_y)], drift_z[np.isfinite(drift_z)]])
    if finite_vals.size > 0 and np.nanmax(finite_vals) > 0.0:
        plt.yscale("log")
    plt.title("Heavy training (models) - pass-to-pass drift")
    plt.xlabel("Pass index")
    plt.ylabel("Drift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_model_y_vs_exact_per_pass(
    stitched_by_index: Dict[int, Dict[str, np.ndarray]],
    exact_y_by_index: Dict[int, np.ndarray],
    out_dir: str,
    sample_paths: int = 6,
) -> None:
    if not PLOTTING_AVAILABLE or len(stitched_by_index) == 0 or len(exact_y_by_index) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    for idx in sorted(stitched_by_index.keys()):
        if idx not in exact_y_by_index:
            continue
        stitched = stitched_by_index[idx]
        Y_exact = exact_y_by_index[idx]
        t = stitched["t"]
        Y_pred = stitched["Y"]
        if Y_pred.shape != Y_exact.shape:
            continue

        n_paths = max(1, min(int(sample_paths), int(Y_pred.shape[0])))
        plt.figure(figsize=(12, 6))
        for i in range(n_paths):
            alpha = 0.95 if i == 0 else 0.25
            width = 1.6 if i == 0 else 0.9
            pred_label = "Y pred (sample paths)" if i == 0 else None
            exact_label = "u exact (sample paths)" if i == 0 else None
            plt.plot(
                t[i, :, 0],
                Y_pred[i, :, 0],
                color="tab:blue",
                alpha=alpha,
                linewidth=width,
                label=pred_label,
            )
            plt.plot(
                t[i, :, 0],
                Y_exact[i, :, 0],
                color="tab:red",
                alpha=alpha,
                linewidth=width,
                linestyle="--",
                label=exact_label,
            )

        y_pred_mean = np.mean(Y_pred[:, :, 0], axis=0)
        y_exact_mean = np.mean(Y_exact[:, :, 0], axis=0)
        plt.plot(
            t[0, :, 0],
            y_pred_mean,
            color="navy",
            linewidth=2.6,
            label="mean Y pred",
        )
        plt.plot(
            t[0, :, 0],
            y_exact_mean,
            color="darkred",
            linewidth=2.6,
            linestyle="--",
            label="mean u exact",
        )

        plt.title(f"Heavy training models - pass{idx}: Y predicted vs u exact")
        plt.xlabel("Time")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heavy_models_y_vs_exact_pass{idx:02d}.png"), dpi=160)
        plt.close()


def _plot_model_y_vs_exact_all_passes_mean(
    stitched_by_index: Dict[int, Dict[str, np.ndarray]],
    exact_y_by_index: Dict[int, np.ndarray],
    out_path: str,
) -> None:
    if not PLOTTING_AVAILABLE or len(stitched_by_index) == 0 or len(exact_y_by_index) == 0:
        return
    pass_indices = [idx for idx in sorted(stitched_by_index.keys()) if idx in exact_y_by_index]
    if len(pass_indices) == 0:
        return

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(pass_indices), 2)))
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(pass_indices):
        stitched = stitched_by_index[idx]
        Y_exact = exact_y_by_index[idx]
        t = stitched["t"][0, :, 0]
        y_pred_mean = np.mean(stitched["Y"][:, :, 0], axis=0)
        y_exact_mean = np.mean(Y_exact[:, :, 0], axis=0)
        plt.plot(
            t,
            y_pred_mean,
            color=colors[i],
            linewidth=2.0,
            label=f"pass{idx} mean Y pred",
        )
        plt.plot(
            t,
            y_exact_mean,
            color=colors[i],
            linewidth=2.0,
            linestyle="--",
            label=f"pass{idx} mean u exact",
        )

    plt.title("Heavy training models - mean Y pred vs mean u exact across passes")
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_summary_csv_logs(
    pass_rows_by_id: Dict[int, List[Dict[str, Any]]],
    id_to_index: Dict[int, int],
    exact_by_index: Dict[int, Dict[str, Any]],
    loss_key: str,
    out_path: str,
) -> None:
    rows_out: List[Dict[str, Any]] = []
    for pass_id in sorted(pass_rows_by_id.keys()):
        idx = id_to_index[pass_id]
        rows = pass_rows_by_id[pass_id]

        losses = np.array([_to_float(r.get(loss_key, np.nan)) for r in rows], dtype=np.float64)
        losses = losses[np.isfinite(losses)]
        y0 = np.array([_to_float(r.get("eval_mean_y0", np.nan)) for r in rows], dtype=np.float64)
        y0 = y0[np.isfinite(y0)]
        ref = np.array([_to_float(r.get("refine_rounds", np.nan), 0.0) for r in rows], dtype=np.float64)
        ref = ref[np.isfinite(ref)]

        row: Dict[str, Any] = {
            "pass_id": int(pass_id),
            "pass_index": int(idx),
            "source_mode": "logs",
            "n_blocks": int(len(rows)),
            "loss_key": loss_key,
            "mean_loss": float(np.mean(losses)) if losses.size > 0 else np.nan,
            "worst_block_loss": float(np.max(losses)) if losses.size > 0 else np.nan,
            "score_mean_plus_0.35_worst": float(np.mean(losses) + 0.35 * np.max(losses))
            if losses.size > 0
            else np.nan,
            "mean_eval_y0": float(np.mean(y0)) if y0.size > 0 else np.nan,
            "std_eval_y0": float(np.std(y0)) if y0.size > 0 else np.nan,
            "total_refine_rounds": float(np.sum(ref)) if ref.size > 0 else np.nan,
            "max_refine_rounds": float(np.max(ref)) if ref.size > 0 else np.nan,
        }

        exact = exact_by_index.get(idx, {})
        if isinstance(exact, dict) and len(exact) > 0:
            row["exact_mean_abs_error_y"] = _to_float(exact.get("mean_abs_error_y", np.nan))
            row["exact_mean_abs_error_z"] = _to_float(exact.get("mean_abs_error_z", np.nan))
            row["exact_abs_error_mean_y0"] = _to_float(exact.get("abs_error_mean_y0", np.nan))
            row["exact_rmse_y"] = _to_float(exact.get("rmse_y", np.nan))
            row["exact_rmse_y0"] = _to_float(exact.get("rmse_y0", np.nan))

        rows_out.append(row)

    if len(rows_out) == 0:
        return

    fieldnames: List[str] = []
    for row in rows_out:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def _write_summary_csv_models(stats_rows: List[Dict[str, Any]], out_path: str) -> None:
    if len(stats_rows) == 0:
        return
    rows = sorted(stats_rows, key=lambda r: int(r["pass_index"]))
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_blob_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _make_deterministic_xi_default(M: int, D: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    if int(D) >= 1:
        Xi[:, 0] = rng.normal(1.0, 1.0, int(M))
    if int(D) >= 2:
        Xi[:, 1] = rng.normal(1.0, 1.0, int(M))
    if int(D) >= 3:
        Xi[:, 2] = rng.normal(0.0, 1.0, int(M))
    if int(D) >= 4:
        Xi[:, 3] = rng.uniform(1.0, 9.0, int(M))
    if int(D) > 4:
        Xi[:, 4:] = rng.normal(0.0, 1.0, size=(int(M), int(D) - 4))
    return Xi.astype(np.float32)


def _build_rollout_inputs(
    blocks: List[Dict[str, float]],
    M: int,
    N_per_block: int,
    D: int,
    seed: int = 1234,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(int(seed))
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        dt = float(block["T_block"]) / float(N_per_block)
        Dt = np.zeros((int(M), int(N_per_block) + 1, 1), dtype=np.float32)
        DW = np.zeros((int(M), int(N_per_block) + 1, int(D)), dtype=np.float32)
        Dt[:, 1:, :] = dt

        if int(M) > 1:
            half_M = int(M) // 2
            DW_half = np.sqrt(dt) * rng.normal(size=(half_M, int(N_per_block), int(D)))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if int(M) % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * rng.normal(size=(int(N_per_block), int(D)))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * rng.normal(size=(int(M), int(N_per_block), int(D)))

        t_abs = float(block["t_start"]) + np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        rollout_inputs.append((t_abs.astype(np.float32), W.astype(np.float32)))
    return rollout_inputs


def _save_eval_bundle(
    path: str,
    Xi_initial: np.ndarray,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    blocks: List[Dict[str, float]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t_stack = np.stack([pair[0] for pair in rollout_inputs], axis=0).astype(np.float32)
    w_stack = np.stack([pair[1] for pair in rollout_inputs], axis=0).astype(np.float32)
    t_start = np.array([float(b["t_start"]) for b in blocks], dtype=np.float32)
    t_end = np.array([float(b["t_end"]) for b in blocks], dtype=np.float32)
    np.savez(
        path,
        Xi_initial=np.asarray(Xi_initial, dtype=np.float32),
        t_bundle=t_stack,
        W_bundle=w_stack,
        block_t_start=t_start,
        block_t_end=t_end,
    )


def _load_eval_bundle(
    path: str,
    n_blocks_expected: int,
    N_per_block_expected: int,
    D_expected: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as data:
        Xi = np.asarray(data["Xi_initial"], dtype=np.float32)
        t_bundle = np.asarray(data["t_bundle"], dtype=np.float32)
        W_bundle = np.asarray(data["W_bundle"], dtype=np.float32)

    if Xi.ndim != 2 or Xi.shape[1] != int(D_expected):
        raise ValueError(f"Invalid Xi_initial shape in eval bundle: {Xi.shape}")
    if t_bundle.shape[0] != int(n_blocks_expected) or W_bundle.shape[0] != int(n_blocks_expected):
        raise ValueError(
            f"Eval bundle blocks mismatch: got t={t_bundle.shape[0]}, W={W_bundle.shape[0]}, expected={int(n_blocks_expected)}"
        )
    if t_bundle.shape[2] != int(N_per_block_expected) + 1:
        raise ValueError("Eval bundle N_per_block mismatch")
    if W_bundle.shape[3] != int(D_expected):
        raise ValueError("Eval bundle D mismatch")
    if t_bundle.shape[1] != Xi.shape[0] or W_bundle.shape[1] != Xi.shape[0]:
        raise ValueError("Eval bundle M mismatch between Xi and rollout tensors")

    rollout_inputs = [(t_bundle[i], W_bundle[i]) for i in range(int(n_blocks_expected))]
    return Xi, rollout_inputs


def _psi_np(x: np.ndarray, d: float, x_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, np.minimum(x / d, (x_max - x) / d)))


def _psi3_np(v: np.ndarray, d: float, v_max: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (v_max - v) / d))


def _psi4_np(v: np.ndarray, d: float, v_min: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(1.0, (v - v_min) / d))


def _f_np(X: np.ndarray, Z: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    S = X[:, 0:1]
    V = X[:, 2:3]
    Z_S = Z[:, 0:1]
    gamma = float(params["gamma"])
    s1 = float(params["s1"])
    d = float(params["d"])
    x_max = float(params["x_max"])
    arg = -np.exp(-S) * Z_S / (gamma * s1)
    return -0.5 * V * _psi_np(arg, d=d, x_max=x_max)


def _mu_np(X: np.ndarray, Z: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    S = X[:, 0:1]
    H = X[:, 1:2]
    V = X[:, 2:3]
    X_state = X[:, 3:4]

    mu1 = float(params["mu1"])
    mu2 = float(params["mu2"])
    c1 = float(params["c1"])
    c2 = float(params["c2"])
    c3 = float(params["c3"])
    c4 = float(params["c4"])
    x_max = float(params["x_max"])
    d = float(params["d"])
    v_min = float(params["v_min"])
    v_max = float(params["v_max"])

    dS = mu1 * (c1 - S)
    dH = mu2 * (c2 - H)
    dV = (
        _f_np(X, Z, params) * _psi_np(X_state, d=d, x_max=x_max)
        + c3 * _psi_np(-X_state, d=d, x_max=x_max) * _psi3_np(V, d=d, v_max=v_max)
        - c4 * _psi_np(X_state - x_max, d=d, x_max=x_max) * _psi4_np(V, d=d, v_min=v_min)
    )
    dX = V
    return np.concatenate([dS, dH, dV, dX], axis=1).astype(np.float32)


def _nn_u_du_np(t: np.ndarray, X: np.ndarray, blob: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape [M,1], got {t.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must have shape [M,D], got {X.shape}")

    D = int(X.shape[1])
    n_layers = int(np.asarray(blob["n_layers"]).reshape(()))
    weights = [np.asarray(blob[f"W_{i}"], dtype=np.float32) for i in range(n_layers)]
    biases = [np.asarray(blob[f"b_{i}"], dtype=np.float32) for i in range(n_layers)]

    use_time = bool(int(np.asarray(blob.get("normalize_time_input", np.array(1))).reshape(())))
    T_total = float(np.asarray(blob.get("T_total", np.array(1.0))).reshape(()))
    x_mean = np.asarray(blob.get("x_norm_mean", np.zeros((1, D), dtype=np.float32)), dtype=np.float32).reshape(1, D)
    x_std = np.asarray(blob.get("x_norm_std", np.ones((1, D), dtype=np.float32)), dtype=np.float32).reshape(1, D)
    x_std = np.maximum(x_std, 1.0e-3).astype(np.float32)

    t_in = (2.0 * (t / T_total) - 1.0).astype(np.float32) if use_time else t
    X_in = ((X - x_mean) / x_std).astype(np.float32)
    H = np.concatenate([t_in, X_in], axis=1).astype(np.float32)

    pre_acts: List[np.ndarray] = []
    for i in range(n_layers - 1):
        A = H @ weights[i] + biases[i]
        pre_acts.append(A)
        H = np.sin(A).astype(np.float32)

    u = (H @ weights[-1] + biases[-1]).astype(np.float32)  # [M,1]

    # Backprop du/dinput for scalar output.
    grad_H = np.repeat(weights[-1][:, 0][None, :], repeats=H.shape[0], axis=0).astype(np.float32)
    for i in range(n_layers - 2, -1, -1):
        grad_A = grad_H * np.cos(pre_acts[i])
        grad_H = grad_A @ weights[i].T
    grad_input = grad_H  # [M, D+1]
    du_dX_in = grad_input[:, 1 : 1 + D]
    du_dX = (du_dX_in / x_std).astype(np.float32)
    return u.astype(np.float32), du_dX


def _predict_stitched_numpy(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_initial: np.ndarray,
    params: Dict[str, Any],
    N_per_block: int,
    D: int,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    if len(block_blobs) != len(blocks):
        raise ValueError("block_blobs and blocks length mismatch")
    if Xi_initial.ndim != 2 or Xi_initial.shape[1] != int(D):
        raise ValueError(f"Xi_initial must have shape [M,{int(D)}]")

    s1 = float(params["s1"])
    s2 = float(params["s2"])
    s3 = float(params["s3"])
    sigma_diag = np.array([s1, s2, s3] + [0.0] * max(0, int(D) - 3), dtype=np.float32)

    Xi_curr = np.asarray(Xi_initial, dtype=np.float32).copy()
    t_segments: List[np.ndarray] = []
    X_segments: List[np.ndarray] = []
    Y_segments: List[np.ndarray] = []
    Z_segments: List[np.ndarray] = []

    for b, blob in enumerate(block_blobs):
        t_b, W_b = rollout_inputs[b]
        t_b = np.asarray(t_b, dtype=np.float32)
        W_b = np.asarray(W_b, dtype=np.float32)

        M = Xi_curr.shape[0]
        X_block = np.zeros((M, int(N_per_block) + 1, int(D)), dtype=np.float32)
        Y_block = np.zeros((M, int(N_per_block) + 1, 1), dtype=np.float32)
        Z_block = np.zeros((M, int(N_per_block) + 1, int(D)), dtype=np.float32)

        t0 = t_b[:, 0, :]
        X0 = Xi_curr
        Y0, Du0 = _nn_u_du_np(t0, X0, blob)
        Z0 = Du0 * sigma_diag.reshape(1, int(D))
        X_block[:, 0, :] = X0
        Y_block[:, 0, :] = Y0
        Z_block[:, 0, :] = Z0

        for n in range(int(N_per_block)):
            t1 = t_b[:, n + 1, :]
            dW = W_b[:, n + 1, :] - W_b[:, n, :]
            dt = t1 - t0

            mu0 = _mu_np(X0, Z0, params=params)
            sigma_dW = dW * sigma_diag.reshape(1, int(D))
            X1 = X0 + mu0 * dt + sigma_dW

            Y1, Du1 = _nn_u_du_np(t1, X1, blob)
            Z1 = Du1 * sigma_diag.reshape(1, int(D))

            X_block[:, n + 1, :] = X1
            Y_block[:, n + 1, :] = Y1
            Z_block[:, n + 1, :] = Z1

            t0 = t1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

        start_idx = 0 if b == 0 else 1
        t_segments.append(t_b[:, start_idx:, :].astype(np.float32))
        X_segments.append(X_block[:, start_idx:, :].astype(np.float32))
        Y_segments.append(Y_block[:, start_idx:, :].astype(np.float32))
        Z_segments.append(Z_block[:, start_idx:, :].astype(np.float32))
        Xi_curr = X_block[:, -1, :].astype(np.float32)

    return {
        "t": np.concatenate(t_segments, axis=1),
        "X": np.concatenate(X_segments, axis=1),
        "Y": np.concatenate(Y_segments, axis=1),
        "Z": np.concatenate(Z_segments, axis=1),
    }


def _build_exact_solution_functions(
    solution_name: str,
    params: Dict[str, Any],
    D: int,
) -> Optional[Dict[str, Any]]:
    name = str(solution_name or "none").strip().lower()
    if name in {"", "none", "off", "false", "0"}:
        return None
    if name not in {"quadratic_coupled", "quadratic", "qc4d"}:
        raise ValueError(f"Unsupported exact solution profile: {solution_name}")
    if int(D) != 4:
        raise ValueError(f"exact_solution='{solution_name}' requires D=4, found D={int(D)}")

    gamma = float(params["gamma"])
    s1 = float(params["s1"])
    s3 = float(params["s3"])

    def u_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
        _ = t_arr
        Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
        S = Xi_arr[:, 0:1]
        V = Xi_arr[:, 2:3]
        X_state = Xi_arr[:, 3:4]
        return (-gamma * np.exp(S) * X_state + V ** 2 + V * X_state).astype(np.float32)

    def z_exact(t_arr: np.ndarray, Xi_arr: np.ndarray) -> np.ndarray:
        _ = t_arr
        Xi_arr = np.asarray(Xi_arr, dtype=np.float32)
        S = Xi_arr[:, 0:1]
        V = Xi_arr[:, 2:3]
        X_state = Xi_arr[:, 3:4]
        z_s = -gamma * np.exp(S) * X_state * s1
        z_h = np.zeros_like(z_s)
        z_v = (2.0 * V + X_state) * s3
        z_x = np.zeros_like(z_s)
        return np.concatenate([z_s, z_h, z_v, z_x], axis=1).astype(np.float32)

    return {"name": "quadratic_coupled", "u_exact": u_exact, "z_exact": z_exact}


def _compute_exact_bundle(stitched: Dict[str, np.ndarray], exact_solution: Dict[str, Any]) -> Dict[str, Any]:
    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]

    M_paths, T_points, D = X_all.shape
    X_flat = X_all.reshape(-1, D)
    t_flat = t_all.reshape(-1, 1)
    Y_exact = exact_solution["u_exact"](t_flat, X_flat).reshape(M_paths, T_points, 1).astype(np.float32)
    Z_exact = exact_solution["z_exact"](t_flat, X_flat).reshape(M_paths, T_points, D).astype(np.float32)

    abs_err_Y = np.abs(Y_pred - Y_exact)
    abs_err_Z = np.abs(Z_pred - Z_exact)
    y0_pred = Y_pred[:, 0, 0]
    y0_exact = Y_exact[:, 0, 0]
    summary = {
        "mean_abs_error_y": float(np.mean(abs_err_Y)),
        "mean_abs_error_z": float(np.mean(abs_err_Z)),
        "abs_error_mean_y0": float(np.mean(np.abs(y0_pred - y0_exact))),
        "rmse_y": float(np.sqrt(np.mean((Y_pred - Y_exact) ** 2))),
        "rmse_y0": float(np.sqrt(np.mean((y0_pred - y0_exact) ** 2))),
    }
    return {
        "summary": summary,
        "Y_exact": Y_exact.astype(np.float32),
        "Z_exact": Z_exact.astype(np.float32),
    }


def _discover_pass_block_files(models_dir: str) -> Dict[int, Dict[int, str]]:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    by_pass: Dict[int, Dict[int, str]] = {}
    for name in sorted(os.listdir(models_dir)):
        m = PASS_DIR_RE.match(name)
        if m is None:
            continue
        pass_id = int(m.group(1))
        pass_dir = os.path.join(models_dir, name)
        if not os.path.isdir(pass_dir):
            continue
        block_map: Dict[int, str] = {}
        for fpath in glob.glob(os.path.join(pass_dir, "block_*.npz")):
            fname = os.path.basename(fpath)
            mb = BLOCK_NPZ_RE.match(fname)
            if mb is None:
                continue
            block_idx = int(mb.group(1))
            block_map[block_idx] = fpath
        if len(block_map) > 0:
            by_pass[pass_id] = block_map
    return by_pass


def _load_run_config_for_recursive(recursive_dir: str) -> Dict[str, Any]:
    run_root = os.path.dirname(recursive_dir.rstrip(os.sep))
    cfg = os.path.join(run_root, "run_config.json")
    if not os.path.isfile(cfg):
        return {}
    with open(cfg, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_expected_block_indices(
    run_config: Dict[str, Any],
    pass_blocks: Dict[int, Dict[int, str]],
) -> List[int]:
    t_total = _to_float(run_config.get("T_total", np.nan), default=np.nan)
    block_size = _to_float(run_config.get("block_size", np.nan), default=np.nan)
    if np.isfinite(t_total) and np.isfinite(block_size) and block_size > 0:
        n_blocks = int(round(float(t_total) / float(block_size)))
        if n_blocks >= 1:
            return list(range(n_blocks))

    max_blocks = 0
    for bmap in pass_blocks.values():
        max_blocks = max(max_blocks, len(bmap))
    if max_blocks < 1:
        raise RuntimeError("Could not infer expected number of blocks from models")
    return list(range(max_blocks))


def _build_blocks_from_reference(
    reference_block_files: Dict[int, str],
    expected_indices: List[int],
) -> List[Dict[str, float]]:
    blocks: List[Dict[str, float]] = []
    for idx in expected_indices:
        path = reference_block_files[idx]
        blob = _load_blob_npz(path)
        t_start = _to_float(blob.get("t_start", np.nan), default=np.nan)
        t_end = _to_float(blob.get("t_end", np.nan), default=np.nan)
        if not np.isfinite(t_start) or not np.isfinite(t_end):
            raise RuntimeError(f"Missing t_start/t_end in blob: {path}")
        blocks.append(
            {
                "t_start": float(t_start),
                "t_end": float(t_end),
                "T_block": float(t_end - t_start),
            }
        )
    blocks = sorted(blocks, key=lambda b: b["t_start"])
    return blocks


def _prepare_eval_bundle(
    recursive_dir: str,
    blocks: List[Dict[str, float]],
    N_per_block: int,
    D: int,
    eval_paths: int,
    eval_seed: int,
    save_eval_bundle: bool,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], str]:
    eval_bundle_path = os.path.join(recursive_dir, "evaluation_bundle.npz")
    if os.path.isfile(eval_bundle_path):
        Xi, rollout_inputs = _load_eval_bundle(
            path=eval_bundle_path,
            n_blocks_expected=len(blocks),
            N_per_block_expected=N_per_block,
            D_expected=D,
        )
        return Xi, rollout_inputs, eval_bundle_path

    m_paths = max(64, int(eval_paths))
    Xi = _make_deterministic_xi_default(m_paths, D, seed=int(eval_seed))

    rollout_inputs = _build_rollout_inputs(
        blocks=blocks,
        M=int(Xi.shape[0]),
        N_per_block=int(N_per_block),
        D=int(D),
        seed=int(eval_seed),
    )
    if bool(save_eval_bundle):
        _save_eval_bundle(
            path=eval_bundle_path,
            Xi_initial=Xi,
            rollout_inputs=rollout_inputs,
            blocks=blocks,
        )
    return Xi, rollout_inputs, eval_bundle_path


def _resolve_exact_solution_name(mode_arg: str, run_config: Dict[str, Any]) -> str:
    mode_arg = str(mode_arg or "auto").strip().lower()
    if mode_arg != "auto":
        return mode_arg
    cfg = str(run_config.get("exact_solution", "none")).strip().lower()
    return cfg if cfg != "" else "none"


def _build_heavy_training_plots_from_logs(
    recursive_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    pass_rows_by_id = _load_pass_logs(recursive_dir)
    pass_ids = sorted(pass_rows_by_id.keys())
    zero_based = _infer_zero_based(pass_ids)
    id_to_index = {pid: _pass_index(pid, zero_based=zero_based) for pid in pass_ids}
    pass_rows_by_index = {id_to_index[pid]: rows for pid, rows in pass_rows_by_id.items()}

    results_summary = _load_results_summary(recursive_dir)
    selected_index = _resolve_selected_pass_index(results_summary, id_to_index)
    exact_by_index = _load_exact_metrics(recursive_dir, id_to_index=id_to_index, zero_based=zero_based)
    loss_key = _pick_loss_key(pass_rows_by_index)

    os.makedirs(output_dir, exist_ok=True)
    _plot_block_metric(
        pass_rows_by_index=pass_rows_by_index,
        key=loss_key,
        title=f"Heavy training - block {loss_key} by pass",
        ylabel=f"{loss_key} (log scale)",
        out_path=os.path.join(output_dir, "heavy_blocks_loss.png"),
        ylog=True,
    )
    _plot_block_metric(
        pass_rows_by_index=pass_rows_by_index,
        key="eval_mean_y0",
        title="Heavy training - block eval_mean_y0 by pass",
        ylabel="eval_mean_y0",
        out_path=os.path.join(output_dir, "heavy_blocks_y0.png"),
        ylog=False,
    )
    _plot_pass_summary(
        pass_rows_by_index=pass_rows_by_index,
        loss_key=loss_key,
        selected_index=selected_index,
        out_path=os.path.join(output_dir, "heavy_pass_summary.png"),
    )
    _plot_refine_rounds(
        pass_rows_by_index=pass_rows_by_index,
        selected_index=selected_index,
        out_path=os.path.join(output_dir, "heavy_refine_rounds.png"),
    )
    _plot_exact_summary(
        exact_by_index=exact_by_index,
        selected_index=selected_index,
        out_path=os.path.join(output_dir, "heavy_exact_summary.png"),
    )

    summary_csv = os.path.join(output_dir, "heavy_pass_summary.csv")
    _write_summary_csv_logs(
        pass_rows_by_id=pass_rows_by_id,
        id_to_index=id_to_index,
        exact_by_index=exact_by_index,
        loss_key=loss_key,
        out_path=summary_csv,
    )

    return {
        "source_mode": "logs",
        "recursive_dir": recursive_dir,
        "output_dir": output_dir,
        "pass_ids": pass_ids,
        "pass_indices": sorted(pass_rows_by_index.keys()),
        "loss_key": loss_key,
        "selected_pass_index": selected_index,
        "has_exact_metrics": len(exact_by_index) > 0,
        "summary_csv": summary_csv,
        "plotting_available": bool(PLOTTING_AVAILABLE),
    }


def _build_heavy_training_plots_from_models(
    recursive_dir: str,
    output_dir: str,
    models_dir: str,
    exact_solution_mode: str,
    eval_paths: int,
    eval_seed: int,
    save_eval_bundle: bool,
) -> Dict[str, Any]:
    run_config = _load_run_config_for_recursive(recursive_dir)
    if len(run_config) == 0:
        raise RuntimeError(
            "run_config.json not found: required to infer params/layers/N/D for models-only evaluation"
        )

    N = _safe_int(run_config.get("N", np.nan), default=-1)
    D = _safe_int(run_config.get("D", np.nan), default=-1)
    layers = run_config.get("layers", None)
    params = run_config.get("params", None)
    T_total = _to_float(run_config.get("T_total", np.nan), default=np.nan)
    if N < 1 or D < 1 or not isinstance(layers, list) or not isinstance(params, dict) or not np.isfinite(T_total):
        raise RuntimeError("run_config.json is missing required fields (N,D,layers,params,T_total)")
    if int(D) != 4:
        raise RuntimeError(
            f"models mode currently supports D=4 (quadratic coupled setup), found D={int(D)}"
        )

    pass_blocks = _discover_pass_block_files(models_dir)
    if len(pass_blocks) == 0:
        raise RuntimeError(f"No pass_*/block_*.npz found in models_dir={models_dir}")

    expected_indices = _infer_expected_block_indices(run_config, pass_blocks)
    complete_pass_ids: List[int] = []
    incomplete_info: Dict[int, List[int]] = {}
    for pass_id, bmap in sorted(pass_blocks.items()):
        missing = [idx for idx in expected_indices if idx not in bmap]
        if len(missing) == 0:
            complete_pass_ids.append(pass_id)
        else:
            incomplete_info[pass_id] = missing
    if len(complete_pass_ids) == 0:
        raise RuntimeError(
            "No complete pass found (all required block_XX.npz missing). "
            f"Expected block indices={expected_indices}"
        )

    ref_pass_id = min(complete_pass_ids)
    blocks = _build_blocks_from_reference(pass_blocks[ref_pass_id], expected_indices=expected_indices)

    Xi, rollout_inputs, eval_bundle_path = _prepare_eval_bundle(
        recursive_dir=recursive_dir,
        blocks=blocks,
        N_per_block=N,
        D=D,
        eval_paths=eval_paths,
        eval_seed=eval_seed,
        save_eval_bundle=save_eval_bundle,
    )

    zero_based = _infer_zero_based(complete_pass_ids)
    id_to_index = {pid: _pass_index(pid, zero_based=zero_based) for pid in complete_pass_ids}

    exact_solution_name = _resolve_exact_solution_name(exact_solution_mode, run_config=run_config)
    exact_solution = _build_exact_solution_functions(
        solution_name=exact_solution_name,
        params=params,
        D=D,
    )

    stitched_by_index: Dict[int, Dict[str, np.ndarray]] = {}
    exact_by_index: Dict[int, Dict[str, Any]] = {}
    exact_y_by_index: Dict[int, np.ndarray] = {}
    stats_rows: List[Dict[str, Any]] = []

    prev_idx: Optional[int] = None
    prev_stitched: Optional[Dict[str, np.ndarray]] = None
    for pass_id in sorted(complete_pass_ids):
        pass_idx = id_to_index[pass_id]
        blobs = [_load_blob_npz(pass_blocks[pass_id][idx]) for idx in expected_indices]
        stitched = _predict_stitched_numpy(
            block_blobs=blobs,
            blocks=blocks,
            Xi_initial=Xi,
            params=params,
            N_per_block=N,
            D=D,
            rollout_inputs=rollout_inputs,
        )
        stitched_by_index[pass_idx] = stitched

        row: Dict[str, Any] = {
            "source_mode": "models",
            "pass_id": int(pass_id),
            "pass_index": int(pass_idx),
            "n_blocks": int(len(blocks)),
            "n_paths": int(stitched["Y"].shape[0]),
            "n_time_points": int(stitched["Y"].shape[1]),
            "mean_y0": float(np.mean(stitched["Y"][:, 0, 0])),
            "std_y0": float(np.std(stitched["Y"][:, 0, 0])),
            "mean_abs_y": float(np.mean(np.abs(stitched["Y"]))),
            "mean_abs_z": float(np.mean(np.abs(stitched["Z"]))),
            "drift_y_vs_prev": np.nan,
            "drift_z_vs_prev": np.nan,
            "prev_pass_index": prev_idx if prev_idx is not None else "",
        }
        if prev_stitched is not None:
            row["drift_y_vs_prev"] = float(np.mean(np.abs(stitched["Y"] - prev_stitched["Y"])))
            row["drift_z_vs_prev"] = float(np.mean(np.abs(stitched["Z"] - prev_stitched["Z"])))

        if exact_solution is not None:
            exact_bundle = _compute_exact_bundle(stitched=stitched, exact_solution=exact_solution)
            summary = exact_bundle["summary"]
            exact_by_index[int(pass_idx)] = summary
            exact_y_by_index[int(pass_idx)] = np.asarray(exact_bundle["Y_exact"], dtype=np.float32)
            row["exact_mean_abs_error_y"] = _to_float(summary.get("mean_abs_error_y", np.nan))
            row["exact_mean_abs_error_z"] = _to_float(summary.get("mean_abs_error_z", np.nan))
            row["exact_abs_error_mean_y0"] = _to_float(summary.get("abs_error_mean_y0", np.nan))
            row["exact_rmse_y"] = _to_float(summary.get("rmse_y", np.nan))
            row["exact_rmse_y0"] = _to_float(summary.get("rmse_y0", np.nan))

        stats_rows.append(row)
        prev_idx = pass_idx
        prev_stitched = stitched

    os.makedirs(output_dir, exist_ok=True)
    _plot_model_mean_y(
        stitched_by_index=stitched_by_index,
        out_path=os.path.join(output_dir, "heavy_models_mean_y.png"),
    )
    _plot_model_y0(
        stats_rows=stats_rows,
        selected_index=None,
        out_path=os.path.join(output_dir, "heavy_models_y0.png"),
    )
    _plot_model_drift(
        stats_rows=stats_rows,
        out_path=os.path.join(output_dir, "heavy_models_drift.png"),
    )
    _plot_exact_summary(
        exact_by_index=exact_by_index,
        selected_index=None,
        out_path=os.path.join(output_dir, "heavy_models_exact_summary.png"),
    )
    _plot_model_y_vs_exact_per_pass(
        stitched_by_index=stitched_by_index,
        exact_y_by_index=exact_y_by_index,
        out_dir=output_dir,
        sample_paths=6,
    )
    _plot_model_y_vs_exact_all_passes_mean(
        stitched_by_index=stitched_by_index,
        exact_y_by_index=exact_y_by_index,
        out_path=os.path.join(output_dir, "heavy_models_y_vs_exact_all_passes.png"),
    )

    summary_csv = os.path.join(output_dir, "heavy_pass_summary.csv")
    _write_summary_csv_models(stats_rows=stats_rows, out_path=summary_csv)

    return {
        "source_mode": "models",
        "recursive_dir": recursive_dir,
        "models_dir": models_dir,
        "output_dir": output_dir,
        "pass_ids": sorted(complete_pass_ids),
        "pass_indices": sorted(stitched_by_index.keys()),
        "loss_key": "n/a (models-only)",
        "selected_pass_index": None,
        "has_exact_metrics": len(exact_by_index) > 0,
        "summary_csv": summary_csv,
        "plotting_available": bool(PLOTTING_AVAILABLE),
        "exact_solution_name": exact_solution_name,
        "evaluation_bundle_path": eval_bundle_path,
        "incomplete_passes": {str(k): v for k, v in incomplete_info.items()},
    }


def build_heavy_training_plots(
    recursive_dir: str,
    output_dir: str,
    mode: str,
    models_dir: str,
    has_logs: bool,
    has_models: bool,
    exact_solution_mode: str,
    eval_paths: int,
    eval_seed: int,
    save_eval_bundle: bool,
) -> Dict[str, Any]:
    mode = str(mode or "auto").strip().lower()
    if mode not in {"auto", "logs", "models"}:
        raise ValueError(f"Unsupported mode='{mode}'. Use auto|logs|models")

    chosen = mode
    if mode == "auto":
        if has_logs:
            chosen = "logs"
        elif has_models:
            chosen = "models"
        else:
            raise RuntimeError("No logs and no models available to build plots")

    if chosen == "logs":
        if not has_logs:
            raise RuntimeError("logs mode requested but pass_*_logs.csv are missing")
        return _build_heavy_training_plots_from_logs(
            recursive_dir=recursive_dir,
            output_dir=output_dir,
        )

    if not has_models:
        raise RuntimeError("models mode requested but recursive/models/pass_* artifacts are missing")
    return _build_heavy_training_plots_from_models(
        recursive_dir=recursive_dir,
        output_dir=output_dir,
        models_dir=models_dir,
        exact_solution_mode=exact_solution_mode,
        eval_paths=eval_paths,
        eval_seed=eval_seed,
        save_eval_bundle=save_eval_bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostics plots for recursive heavy training. "
            "Works with either pass logs or models-only artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Path to recursive dir, run dir, models dir, or a parent folder "
            "that contains run_xxx/recursive."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output folder for plots/summary. Default: "
            "<recursive_dir>/plots/heavy_training_diagnostics"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "logs", "models"],
        help="Data source mode: auto prefers logs, otherwise falls back to models.",
    )
    parser.add_argument(
        "--exact-solution",
        type=str,
        default="auto",
        help=(
            "Exact solution profile used in models mode. "
            "Use 'auto' (from run_config), 'none', or e.g. 'quadratic_coupled'."
        ),
    )
    parser.add_argument(
        "--eval-paths",
        type=int,
        default=256,
        help="Number of evaluation paths for models mode when bundle is created.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1234,
        help="Seed used for deterministic evaluation bundle creation.",
    )
    parser.add_argument(
        "--save-eval-bundle",
        action="store_true",
        help="Save generated evaluation_bundle.npz in recursive dir (models mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = _resolve_artifact_context(args.input_dir)
    recursive_dir = ctx["recursive_dir"]
    if str(args.output_dir).strip() == "":
        output_dir = os.path.join(recursive_dir, "plots", "heavy_training_diagnostics")
    else:
        output_dir = os.path.abspath(os.path.expanduser(str(args.output_dir)))

    summary = build_heavy_training_plots(
        recursive_dir=recursive_dir,
        output_dir=output_dir,
        mode=args.mode,
        models_dir=ctx["models_dir"],
        has_logs=bool(ctx["has_logs"]),
        has_models=bool(ctx["has_models"]),
        exact_solution_mode=str(args.exact_solution),
        eval_paths=int(args.eval_paths),
        eval_seed=int(args.eval_seed),
        save_eval_bundle=bool(args.save_eval_bundle),
    )

    print(f"[HeavyPlot] source_mode={summary['source_mode']}")
    print(f"[HeavyPlot] recursive_dir={summary['recursive_dir']}")
    print(f"[HeavyPlot] output_dir={summary['output_dir']}")
    print(f"[HeavyPlot] pass_ids={summary['pass_ids']} -> pass_indices={summary['pass_indices']}")
    print(f"[HeavyPlot] loss_key={summary['loss_key']}")
    print(f"[HeavyPlot] selected_pass_index={summary['selected_pass_index']}")
    print(f"[HeavyPlot] has_exact_metrics={summary['has_exact_metrics']}")
    print(f"[HeavyPlot] summary_csv={summary['summary_csv']}")
    if "exact_solution_name" in summary:
        print(f"[HeavyPlot] exact_solution_name={summary['exact_solution_name']}")
    if "evaluation_bundle_path" in summary:
        print(f"[HeavyPlot] evaluation_bundle_path={summary['evaluation_bundle_path']}")
    if "incomplete_passes" in summary and len(summary["incomplete_passes"]) > 0:
        print(f"[HeavyPlot] incomplete_passes_skipped={summary['incomplete_passes']}")
    if not summary["plotting_available"]:
        print("[HeavyPlot][WARN] matplotlib unavailable: CSV generated, plots skipped.")


if __name__ == "__main__":
    main()
