import os
import json
import csv
import logging
from typing import Dict, List, Optional, Union, Tuple
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
