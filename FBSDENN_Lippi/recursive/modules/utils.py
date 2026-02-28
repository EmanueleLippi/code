import os
import json
import csv
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np


def setup_logger(name: str = "time_stitching", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
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


def save_blob_npz(blob: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2)


def save_rows_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rows is None or len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    keys = []
    keys_set = set()
    for row in rows:
        for k in row.keys():
            if k not in keys_set:
                keys_set.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k, None)) for k in keys})


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


def make_empirical_generator(samples: np.ndarray, jitter_scale: float = 0.0):
    samples = np.asarray(samples, dtype=np.float32)
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


def _z_component_labels(n_components: int) -> List[str]:
    base = ["Z_S", "Z_H", "Z_V", "Z_X"]
    labels = []
    for i in range(int(n_components)):
        labels.append(base[i] if i < len(base) else f"Z_{i}")
    return labels


def build_exact_solution_functions(
    solution_name: str,
    params: Dict,
    D: int,
) -> Optional[Dict[str, Any]]:
    name = str(solution_name or "none").strip().lower()
    if name in ("", "none", "off", "false", "0"):
        return None

    if name in ("quadratic_coupled", "quadratic", "qc4d"):
        if int(D) != 4:
            raise ValueError(
                f"exact_solution='quadratic_coupled' requires D=4, found D={int(D)}"
            )

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
    rel_err_Z = abs_err_Z / (np.abs(Z_exact) + float(eps))

    y0_pred = Y_pred[:, 0, 0]
    y0_exact = Y_exact[:, 0, 0]

    mean_abs_err_Y_t = np.mean(abs_err_Y[:, :, 0], axis=0)
    mean_abs_err_Z_t = np.mean(abs_err_Z, axis=0)
    mean_rel_err_Z_t = np.mean(rel_err_Z, axis=0)

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
        "mean_rel_error_z": float(np.mean(rel_err_Z)),
        "mean_abs_error_z_by_component": np.mean(abs_err_Z, axis=(0, 1)).astype(np.float32),
        "mean_rel_error_z_by_component": np.mean(rel_err_Z, axis=(0, 1)).astype(np.float32),
        "z_component_labels": _z_component_labels(D),
    }

    timeseries = {
        "t": t_all[0, :, 0].astype(np.float32),
        "mean_abs_error_y": mean_abs_err_Y_t.astype(np.float32),
        "mean_abs_error_z": mean_abs_err_Z_t.astype(np.float32),
        "mean_rel_error_z": mean_rel_err_Z_t.astype(np.float32),
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
        rows.append(row)

    save_rows_csv(rows, path)
