import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def setup_logger(name: str = "time_stitching", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _as_blob_dict(blob_or_path: Union[Dict[str, np.ndarray], str, None]) -> Optional[Dict[str, np.ndarray]]:
    if blob_or_path is None:
        return None
    if isinstance(blob_or_path, dict):
        return blob_or_path
    if isinstance(blob_or_path, str):
        with np.load(blob_or_path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    raise TypeError("blob_or_path must be dict, str path, or None")


def _ensure_parent_dir(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)


def save_blob_npz(blob: Dict[str, np.ndarray], path: str) -> None:
    _ensure_parent_dir(path)
    np.savez(path, **blob)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data, path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2)


def save_rows_csv(rows: List[Dict], path: str) -> None:
    _ensure_parent_dir(path)
    if rows is None or len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    keys = []
    keys_set = set()
    for row in rows:
        for key in row.keys():
            if key not in keys_set:
                keys_set.add(key)
                keys.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_serializable(row.get(key, None)) for key in keys})


def export_standard_parameter_blob(model) -> Dict[str, np.ndarray]:
    values = model.net.get_weights()
    n_layers = len(values) // 2
    blob = {
        "n_layers": np.array(n_layers, dtype=np.int32),
        "layers": np.asarray(model.layers_dims, dtype=np.int32),
    }
    for i in range(n_layers):
        blob[f"W_{i}"] = np.asarray(values[2 * i], dtype=np.float32)
        blob[f"b_{i}"] = np.asarray(values[2 * i + 1], dtype=np.float32)
    return blob


def _pass_index(pass_id: int) -> int:
    pid = int(pass_id)
    return pid - 1 if pid >= 1 else pid


def _pass_label(pass_id: int) -> str:
    return f"pass{_pass_index(pass_id)}"


def _pass_tag(pass_id: int, width: int = 2) -> str:
    return f"pass{_pass_index(pass_id):0{int(width)}d}"


def score_pass_logs(
    rows: List[Dict],
    loss_key: str = "eval_mean_loss_per_sample",
    worst_block_weight: float = 0.35,
) -> float:
    losses = np.array([float(r.get(loss_key, np.nan)) for r in (rows or [])], dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return float("inf")
    return float(np.mean(losses) + worst_block_weight * np.max(losses))


def build_blocks(T_total: float, block_size: float) -> List[Dict[str, float]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    n_blocks = int(np.ceil(T_total / block_size))
    edges = [0.0]
    for i in range(1, n_blocks + 1):
        edges.append(min(float(i * block_size), float(T_total)))
    blocks = []
    for i in range(n_blocks):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        blocks.append({"idx": i, "t_start": t0, "t_end": t1, "T_block": (t1 - t0)})
    return blocks


def Xi_generator_default(M, D):
    assert D == 4
    Xi = np.zeros((M, 4), dtype=np.float32)
    Xi[:, 0] = np.random.normal(1.0, 1.0, M)
    Xi[:, 1] = np.random.normal(1.0, 1.0, M)
    Xi[:, 2] = np.random.normal(0.0, 1.0, M)
    Xi[:, 3] = np.random.uniform(1.0, 9.0, M)
    return Xi.astype(np.float32)


def make_deterministic_xi_default(M: int, D: int, seed: int = 1234) -> np.ndarray:
    if int(D) != 4:
        raise ValueError(f"make_deterministic_xi_default currently supports D=4, got D={int(D)}")
    rng = np.random.RandomState(int(seed))
    Xi = np.zeros((int(M), int(D)), dtype=np.float32)
    Xi[:, 0] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 1] = rng.normal(1.0, 1.0, int(M))
    Xi[:, 2] = rng.normal(0.0, 1.0, int(M))
    Xi[:, 3] = rng.uniform(1.0, 9.0, int(M))
    return Xi.astype(np.float32)


def make_empirical_generator(samples: np.ndarray, jitter_scale: float = 0.0):
    samples = np.asarray(samples, dtype=np.float32)
    _ = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-3)

    def _gen(M, D):
        idx = np.random.randint(0, samples.shape[0], size=M)
        Xi = samples[idx].copy()
        if jitter_scale > 0.0:
            Xi += jitter_scale * std * np.random.normal(size=Xi.shape).astype(np.float32)
        return Xi.astype(np.float32)

    return _gen


def estimate_generator_stats(generator_fn, D, n_samples=4096):
    x = generator_fn(n_samples, D).astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(np.std(x, axis=0, keepdims=True), 1.0e-3).astype(np.float32)
    return mean, std


def save_evaluation_bundle(
    path: str,
    Xi_initial: np.ndarray,
    rollout_inputs: List[Tuple[np.ndarray, np.ndarray]],
    blocks: List[Dict[str, float]],
) -> None:
    _ensure_parent_dir(path)
    t_stack = np.stack([pair[0] for pair in rollout_inputs], axis=0).astype(np.float32)
    w_stack = np.stack([pair[1] for pair in rollout_inputs], axis=0).astype(np.float32)
    t_start = np.array([float(block["t_start"]) for block in blocks], dtype=np.float32)
    t_end = np.array([float(block["t_end"]) for block in blocks], dtype=np.float32)
    np.savez(
        path,
        Xi_initial=np.asarray(Xi_initial, dtype=np.float32),
        t_bundle=t_stack,
        W_bundle=w_stack,
        block_t_start=t_start,
        block_t_end=t_end,
    )


def load_evaluation_bundle(
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
        raise ValueError(
            f"Invalid Xi_initial shape in evaluation bundle: {Xi.shape}, expected [M, {int(D_expected)}]"
        )
    if t_bundle.ndim != 4 or W_bundle.ndim != 4:
        raise ValueError(
            f"Invalid rollout bundle rank: t={t_bundle.shape}, W={W_bundle.shape}; expected rank-4"
        )
    if t_bundle.shape[0] != int(n_blocks_expected) or W_bundle.shape[0] != int(n_blocks_expected):
        raise ValueError(
            f"Evaluation bundle blocks mismatch: got {t_bundle.shape[0]}, expected {int(n_blocks_expected)}"
        )
    if t_bundle.shape[2] != int(N_per_block_expected) + 1:
        raise ValueError(
            "Evaluation bundle N_per_block mismatch: "
            f"got {t_bundle.shape[2] - 1}, expected {int(N_per_block_expected)}"
        )
    if W_bundle.shape[3] != int(D_expected):
        raise ValueError(
            f"Evaluation bundle D mismatch in W: got {W_bundle.shape[3]}, expected {int(D_expected)}"
        )
    if t_bundle.shape[1] != Xi.shape[0] or W_bundle.shape[1] != Xi.shape[0]:
        raise ValueError(
            "Evaluation bundle M mismatch between Xi and rollout tensors: "
            f"Xi={Xi.shape[0]}, t_bundle={t_bundle.shape[1]}, W_bundle={W_bundle.shape[1]}"
        )

    rollout_inputs = []
    for i in range(int(n_blocks_expected)):
        rollout_inputs.append((t_bundle[i], W_bundle[i]))
    return Xi, rollout_inputs


def _z_component_labels(n_components: int) -> List[str]:
    base = ["Z_S", "Z_H", "Z_V", "Z_X"]
    labels = []
    for i in range(int(n_components)):
        labels.append(base[i] if i < len(base) else f"Z_{i}")
    return labels


def build_exact_solution_functions(
    solution_name: str,
    params: Dict[str, Any],
    D: int,
) -> Optional[Dict[str, Any]]:
    name = str(solution_name or "none").strip().lower()
    if name in ("", "none", "off", "false", "0"):
        return None

    if name in ("quadratic_coupled", "quadratic", "qc4d"):
        if int(D) != 4:
            raise ValueError(f"exact_solution='quadratic_coupled' requires D=4, found D={int(D)}")

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

        return {
            "name": "quadratic_coupled",
            "u_exact": u_exact,
            "z_exact": z_exact,
        }

    raise ValueError(
        "Unknown exact_solution profile "
        f"'{solution_name}'. Supported: none, quadratic_coupled"
    )


def compute_stitched_exact_bundle(
    stitched: Dict[str, np.ndarray],
    exact_solution: Dict[str, Any],
    eps: float = 1.0e-8,
) -> Dict[str, Any]:
    t_all = stitched["t"]
    X_all = stitched["X"]
    Y_pred = stitched["Y"]
    Z_pred = stitched["Z"]

    M_paths = int(X_all.shape[0])
    T_points = int(X_all.shape[1])
    D = int(X_all.shape[2])

    X_flat = X_all.reshape(-1, D)
    t_flat = t_all.reshape(-1, 1)

    Y_exact = exact_solution["u_exact"](t_flat, X_flat).reshape(M_paths, T_points, 1).astype(np.float32)
    Z_exact = exact_solution["z_exact"](t_flat, X_flat).reshape(M_paths, T_points, D).astype(np.float32)

    abs_err_Y = np.abs(Y_pred - Y_exact)
    abs_err_Z = np.abs(Z_pred - Z_exact)

    rel_err_Z_legacy = abs_err_Z / (np.abs(Z_exact) + float(eps))

    valid_mask = np.abs(Z_exact) > float(eps)
    rel_err_Z = np.zeros_like(abs_err_Z, dtype=np.float32)
    np.divide(
        abs_err_Z,
        np.abs(Z_exact) + float(eps),
        out=rel_err_Z,
        where=valid_mask,
    )

    y0_pred = Y_pred[:, 0, 0]
    y0_exact = Y_exact[:, 0, 0]

    mean_abs_err_Y_t = np.mean(abs_err_Y[:, :, 0], axis=0)
    mean_abs_err_Z_t = np.mean(abs_err_Z, axis=0)
    mean_rel_err_Z_legacy_t = np.mean(rel_err_Z_legacy, axis=0)
    valid_count_t = np.maximum(np.sum(valid_mask, axis=0), 1.0).astype(np.float32)
    mean_rel_err_Z_t = (np.sum(rel_err_Z, axis=0) / valid_count_t).astype(np.float32)
    valid_count_all = float(max(np.sum(valid_mask), 1.0))
    valid_count_comp = np.maximum(np.sum(valid_mask, axis=(0, 1)), 1.0).astype(np.float32)

    summary = {
        "solution_name": exact_solution.get("name", "unknown"),
        "n_paths": int(M_paths),
        "n_time_points": int(T_points),
        "mean_pred_y0": float(np.mean(y0_pred)),
        "mean_exact_y0": float(np.mean(y0_exact)),
        "abs_error_mean_y0": float(np.mean(np.abs(y0_pred - y0_exact))),
        "rmse_y0": float(np.sqrt(np.mean((y0_pred - y0_exact) ** 2))),
        "mean_abs_error_y": float(np.mean(abs_err_Y)),
        "rmse_y": float(np.sqrt(np.mean((Y_pred - Y_exact) ** 2))),
        "mean_abs_error_z": float(np.mean(abs_err_Z)),
        "mean_rel_error_z": float(np.sum(rel_err_Z) / valid_count_all),
        "mean_rel_error_z_legacy": float(np.mean(rel_err_Z_legacy)),
        "mean_abs_error_z_by_component": np.mean(abs_err_Z, axis=(0, 1)).astype(np.float32),
        "mean_rel_error_z_by_component": (np.sum(rel_err_Z, axis=(0, 1)) / valid_count_comp).astype(np.float32),
        "mean_rel_error_z_by_component_legacy": np.mean(rel_err_Z_legacy, axis=(0, 1)).astype(np.float32),
        "valid_rel_error_fraction_z_by_component": np.mean(valid_mask.astype(np.float32), axis=(0, 1)).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    timeseries = {
        "t": t_all[0, :, 0].astype(np.float32),
        "mean_abs_error_y": mean_abs_err_Y_t.astype(np.float32),
        "mean_abs_error_z": mean_abs_err_Z_t.astype(np.float32),
        "mean_rel_error_z": mean_rel_err_Z_t.astype(np.float32),
        "mean_rel_error_z_legacy": mean_rel_err_Z_legacy_t.astype(np.float32),
        "valid_rel_error_fraction_z": np.mean(valid_mask.astype(np.float32), axis=0).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    return {
        "summary": summary,
        "timeseries": timeseries,
        "Y_exact": Y_exact,
        "Z_exact": Z_exact,
    }


def save_exact_error_timeseries_csv(timeseries: Dict[str, np.ndarray], path: str) -> None:
    t = np.asarray(timeseries["t"])
    abs_y = np.asarray(timeseries["mean_abs_error_y"])
    abs_z = np.asarray(timeseries["mean_abs_error_z"])
    rel_z = np.asarray(timeseries["mean_rel_error_z"])
    rel_z_legacy = (
        np.asarray(timeseries["mean_rel_error_z_legacy"])
        if "mean_rel_error_z_legacy" in timeseries
        else None
    )
    labels = timeseries.get("z_component_labels", _z_component_labels(abs_z.shape[1]))

    rows = []
    for i in range(int(t.shape[0])):
        row = {
            "t": float(t[i]),
            "mean_abs_error_y": float(abs_y[i]),
        }
        for d, label in enumerate(labels):
            row[f"mean_abs_error_{label}"] = float(abs_z[i, d])
            row[f"mean_rel_error_{label}"] = float(rel_z[i, d])
            if rel_z_legacy is not None:
                row[f"mean_rel_error_{label}_legacy"] = float(rel_z_legacy[i, d])
        rows.append(row)

    save_rows_csv(rows, path)


def resolve_pass_selection(
    pass_scores_by_loss: Dict[int, float],
    exact_summary_by_pass: Dict[int, Dict[str, Any]],
    selection_metric: str,
    loss_metric_label: str = "eval_mean_loss_per_sample",
) -> Tuple[int, str, float, Dict[str, float]]:
    if len(pass_scores_by_loss) == 0:
        raise RuntimeError("resolve_pass_selection called with empty pass_scores_by_loss")

    metric = str(selection_metric or "auto").strip().lower()
    selected_by_loss = int(min(pass_scores_by_loss, key=pass_scores_by_loss.get))

    if metric in ("", "auto"):
        if len(exact_summary_by_pass) > 0:
            metric = "exact_mae_y"
        else:
            metric = "loss"

    if metric == "loss":
        return (
            selected_by_loss,
            f"{loss_metric_label}+0.35*worst_block",
            float(pass_scores_by_loss[selected_by_loss]),
            {str(k): float(v) for k, v in pass_scores_by_loss.items()},
        )

    metric_extractors = {
        "exact_mae_y": ("exact.mean_abs_error_y", lambda s: float(s["mean_abs_error_y"])),
        "exact_rmse_y": ("exact.rmse_y", lambda s: float(s["rmse_y"])),
        "exact_abs_y0": ("exact.abs_error_mean_y0", lambda s: float(s["abs_error_mean_y0"])),
    }
    if metric not in metric_extractors:
        raise ValueError(
            f"Unsupported selection_metric='{selection_metric}'. "
            "Supported: auto, loss, exact_mae_y, exact_rmse_y, exact_abs_y0"
        )
    if len(exact_summary_by_pass) == 0:
        raise RuntimeError(
            f"selection_metric='{metric}' requires exact_solution metrics, but none are available"
        )

    label, extractor = metric_extractors[metric]
    scores = {}
    for pass_id, summary in exact_summary_by_pass.items():
        scores[int(pass_id)] = float(extractor(summary))
    selected_pass = int(min(scores, key=scores.get))
    return (
        selected_pass,
        label,
        float(scores[selected_pass]),
        {str(k): float(v) for k, v in scores.items()},
    )
