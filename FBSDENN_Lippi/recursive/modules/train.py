import os
import time
import csv
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

from utils import (
    _as_blob_dict, save_blob_npz, estimate_generator_stats,
    make_empirical_generator, build_blocks
)
from models import NN_Quadratic_Coupled, NN_Quadratic_Coupled_Recursive


def train_model(
    model,
    N_Iter,
    learning_rate,
    const_value=None,
    eval_every=50,
    val_batches=8,
    patience=None,
    min_delta=1e-3,
    restore_best=False
):
    if const_value is not None:
        model.const_val.assign(const_value)
    
    best_score = float('inf')
    best_iter = 0
    best_weights = None
    no_improve_iters = 0
    stopped_early = False
    last_loss = 0.0

    for it in range(1, int(N_Iter) + 1):
        t_batch, W_batch, Xi_batch = model.fetch_minibatch()
        loss, _, _, _ = model.train_step(t_batch, W_batch, Xi_batch, learning_rate)
        last_loss = float(loss)

        if it % eval_every == 0 or it == 1 or it == N_Iter:
            eval_stats = model.evaluate_model(const_value=const_value, n_batches=val_batches)
            current_score = eval_stats["mean_loss"]

            if current_score < best_score - min_delta:
                best_score = current_score
                best_iter = it
                if restore_best:
                    best_weights = model.net.get_weights()
                no_improve_iters = 0
            else:
                no_improve_iters += eval_every

            if patience is not None and no_improve_iters >= patience:
                # print(f"[EarlyStop] it={it}, best_it={best_iter}, best_score={best_score:.6e}")
                stopped_early = True
                break

    if restore_best and best_weights is not None:
        model.net.set_weights(best_weights)

    return {
        "const": float(model.const_val.numpy()),
        "learning_rate": float(learning_rate),
        "n_iter": int(N_Iter),
        "last_loss": last_loss,
        "best_iter": int(best_iter),
        "best_score": float(best_score),
        "stopped_early": bool(stopped_early),
    }


def train_with_standard_schedule(
    model,
    stage_plan: List[Tuple[int, float]],
    final_plan: List[Tuple[int, float]],
    eval_batches=5,
    precision_target: Optional[float] = None,
    max_refine_rounds: int = 3,
    refine_plan: Optional[List[Tuple[int, float]]] = None,
    label: str = "",
    logger=None
):
    stage_logs = []
    coupling_step = 1.0
    coupling_levels = np.arange(1.0, 1.0 + coupling_step, coupling_step, dtype=np.float32)

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    for level in coupling_levels:
        model.const_val.assign(level)
        log(f"=== [{label}] Coupling stage: const={float(level):.1f} ===")
        for n_iter, lr in stage_plan:
            t0 = time.time()
            train_stats = train_model(model, N_Iter=n_iter, learning_rate=lr, const_value=level)
            eval_stats = model.evaluate_model(const_value=level, n_batches=eval_batches)
            elapsed = time.time() - t0
            stage_logs.append({
                "phase": "curriculum",
                "const": float(level),
                "lr": float(lr),
                "n_iter": int(n_iter),
                "train_last_loss": train_stats["last_loss"],
                "eval_mean_loss": eval_stats["mean_loss"],
                "eval_std_loss": eval_stats["std_loss"],
                "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
                "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
                "eval_mean_y0": eval_stats["mean_y0"],
                "eval_std_y0": eval_stats["std_y0"],
                "elapsed_sec": float(elapsed),
            })
            log(f"[StageSummary] {label} const={level:.1f}, lr={lr:.1e}, iters={n_iter}, "
                f"eval_loss={eval_stats['mean_loss']:.3e}±{eval_stats['std_loss']:.2e}, time={elapsed:.1f}s")

    model.const_val.assign(1.0)
    log(f"=== [{label}] Final fine-tuning at const=1.0 ===")
    for n_iter, lr in final_plan:
        t0 = time.time()
        train_stats = train_model(
            model, N_Iter=n_iter, learning_rate=lr, const_value=1.0,
            eval_every=25, val_batches=8, patience=150, min_delta=1e-3, restore_best=True
        )
        eval_stats = model.evaluate_model(const_value=1.0, n_batches=eval_batches)
        elapsed = time.time() - t0
        stage_logs.append({
            "phase": "final_finetune",
            "const": 1.0, "lr": float(lr), "n_iter": int(n_iter),
            "train_last_loss": train_stats["last_loss"],
            "best_iter": train_stats["best_iter"], "best_score": train_stats["best_score"],
            "stopped_early": train_stats["stopped_early"],
            "eval_mean_loss": eval_stats["mean_loss"], "eval_std_loss": eval_stats["std_loss"],
            "eval_mean_loss_per_sample": eval_stats["mean_loss_per_sample"],
            "eval_std_loss_per_sample": eval_stats["std_loss_per_sample"],
            "eval_mean_y0": eval_stats["mean_y0"], "eval_std_y0": eval_stats["std_y0"],
            "elapsed_sec": float(elapsed),
        })
        log(f"[FinalSummary] {label} const=1.0, lr={lr:.1e}, iters={n_iter}, "
            f"best_score={train_stats['best_score']:.3e}, eval_loss={eval_stats['mean_loss']:.3e}, time={elapsed:.1f}s")

    eval_stats = model.evaluate_model(const_value=1.0, n_batches=eval_batches)
    refine_rounds = 0
    local_refine_plan = refine_plan if refine_plan is not None else [(50, 1e-5), (50, 5e-6)]

    while (precision_target is not None and eval_stats["mean_loss"] > precision_target and refine_rounds < max_refine_rounds):
        refine_rounds += 1
        log(f"[Refine] {label} round={refine_rounds}, loss={eval_stats['mean_loss']:.3e} > target={precision_target:.3e}")
        for n_iter, lr in local_refine_plan:
            train_model(
                model, N_Iter=n_iter, learning_rate=lr, const_value=1.0,
                eval_every=25, val_batches=8, patience=100, min_delta=1e-3, restore_best=True
            )
        eval_stats = model.evaluate_model(const_value=1.0, n_batches=eval_batches)

    return {
        "stage_logs": stage_logs,
        "eval_stats": eval_stats,
        "refine_rounds": int(refine_rounds),
        "precision_target": None if precision_target is None else float(precision_target),
    }


def run_standard_reference(
    Xi_generator, params, M, N, D, T, layers, stage_plan, final_plan, logger=None
):
    model = NN_Quadratic_Coupled(Xi_generator, T, M, N, D, layers, params)
    logs = train_with_standard_schedule(
        model=model, stage_plan=stage_plan, final_plan=final_plan,
        eval_batches=5, precision_target=None, label="standard", logger=logger
    )
    return model, logs


# ------------------------------------------------------------------------------
# Routine per training Ricorsivo e risoluzione parametri
# ------------------------------------------------------------------------------

def detect_available_recursive_passes(models_dir: str) -> List[int]:
    if not os.path.isdir(models_dir):
        return []
    pass_ids = []
    for name in os.listdir(models_dir):
        full = os.path.join(models_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("pass_"):
            continue
        token = name.split("pass_", 1)[1]
        if token.isdigit():
            pass_ids.append(int(token))
    return sorted(pass_ids)


def resolve_resume_models_dir(resume_path: str) -> str:
    candidate_paths = [
        resume_path,
        os.path.join(resume_path, "models"),
        os.path.join(resume_path, "recursive", "models"),
    ]
    for c in candidate_paths:
        if len(detect_available_recursive_passes(c)) > 0:
            return c
    return resume_path


def load_pass_blobs_from_models_dir(models_dir: str, pass_id: int, blocks: List[Dict[str, float]]) -> List[Dict[str, np.ndarray]]:
    pass_dir = os.path.join(models_dir, f"pass_{pass_id}")
    if not os.path.isdir(pass_dir):
        raise FileNotFoundError(f"Missing directory for pass={pass_id}: {pass_dir}")

    loaded = []
    for b, block in enumerate(blocks):
        blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
        if not os.path.exists(blob_path):
            raise FileNotFoundError(f"Missing blob for pass={pass_id}, block={b}: {blob_path}")
        blob = _as_blob_dict(blob_path)
        loaded.append(blob)
    return loaded


def rollout_boundaries(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_generator,
    params,
    M_rollout,
    N_per_block,
    D,
    layers,
    T_total,
):
    boundary_samples = []
    Xi_curr = Xi_generator(M_rollout, D).astype(np.float32)
    boundary_samples.append(Xi_curr.copy())

    for b, block in enumerate(blocks):
        model = NN_Quadratic_Coupled_Recursive(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"], M=M_rollout, N=N_per_block, D=D, layers=layers, parameters=params,
            t_start=block["t_start"], t_end=block["t_end"], T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(block_blobs[b].get("normalize_time_input", 1))),
            x_norm_mean=block_blobs[b].get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=block_blobs[b].get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )
        model.import_parameter_blob(block_blobs[b], strict=True)
        t_b, W_b, _ = model.fetch_minibatch()
        X_pred, _, _ = model.predict_model(Xi_curr, t_b, W_b, const_value=1.0)
        Xi_curr = X_pred[:, -1, :].astype(np.float32)
        boundary_samples.append(Xi_curr.copy())

    return boundary_samples


def build_stitched_rollout_inputs(
    blocks: List[Dict[str, float]], M: int, N_per_block: int, D: int, seed: int = 1234,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    rollout_inputs = []
    for block in blocks:
        dt = float(block["T_block"]) / float(N_per_block)
        Dt = np.zeros((M, N_per_block + 1, 1), dtype=np.float32)
        DW = np.zeros((M, N_per_block + 1, D), dtype=np.float32)
        Dt[:, 1:, :] = dt

        if M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * rng.normal(size=(half_M, N_per_block, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * rng.normal(size=(N_per_block, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * rng.normal(size=(M, N_per_block, D))

        t_abs = float(block["t_start"]) + np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        rollout_inputs.append((t_abs.astype(np.float32), W.astype(np.float32)))
    return rollout_inputs


def predict_recursive_stitched(
    block_blobs: List[Dict[str, np.ndarray]],
    blocks: List[Dict[str, float]],
    Xi_initial: np.ndarray,
    params,
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    rollout_inputs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    Xi_curr = Xi_initial.astype(np.float32)
    t_segments, X_segments, Y_segments, Z_segments = [], [], [], []

    for b, block in enumerate(blocks):
        blob = block_blobs[b]

        model = NN_Quadratic_Coupled_Recursive(
            Xi_generator=make_empirical_generator(Xi_curr, jitter_scale=0.0),
            T=block["T_block"], M=Xi_curr.shape[0], N=N_per_block, D=D,
            layers=layers, parameters=params,
            t_start=block["t_start"], t_end=block["t_end"], T_total=T_total,
            terminal_blob=None,
            normalize_time_input=bool(int(blob.get("normalize_time_input", 1))),
            x_norm_mean=blob.get("x_norm_mean", np.zeros((1, D), dtype=np.float32)),
            x_norm_std=blob.get("x_norm_std", np.ones((1, D), dtype=np.float32)),
        )

        model.import_parameter_blob(blob, strict=True)
        if rollout_inputs is None:
            t_b, W_b, _ = model.fetch_minibatch()
        else:
            t_b, W_b = rollout_inputs[b]
        
        X_b, Y_b, Z_b = model.predict_model(Xi_curr, t_b, W_b, const_value=1.0)

        start_idx = 0 if b == 0 else 1
        t_segments.append(t_b[:, start_idx:, :])
        X_segments.append(X_b[:, start_idx:, :].astype(np.float32))
        Y_segments.append(Y_b[:, start_idx:, :].astype(np.float32))
        Z_segments.append(Z_b[:, start_idx:, :].astype(np.float32))

        Xi_curr = X_b[:, -1, :].astype(np.float32)

    return {
        "t": np.concatenate(t_segments, axis=1),
        "X": np.concatenate(X_segments, axis=1),
        "Y": np.concatenate(Y_segments, axis=1),
        "Z": np.concatenate(Z_segments, axis=1),
    }


def run_recursive_training(
    Xi_generator, params, M, N_per_block, D, T_total, block_size, layers,
    stage_plan, final_plan, output_dir, precision_margin=0.10, max_refine_rounds=3,
    rollout_M=2000, n_passes=2, resume_models_dir="", resume_from_pass=0,
    empirical_jitter_scale=0.02, logger=None
):
    blocks = build_blocks(T_total=T_total, block_size=block_size)
    if logger:
        logger.info(f"[Recursive] blocks={len(blocks)} -> {[(b['t_start'], b['t_end']) for b in blocks]}, n_passes={n_passes}")

    def _run_pass(pass_id, generators_per_block, warm_start_blobs=None, prev_pass_loss_by_block=None):
        pass_dir = os.path.join(output_dir, f"pass_{pass_id}")
        os.makedirs(pass_dir, exist_ok=True)
        next_blob = None
        block_blobs = [None] * len(blocks)
        logs = []
        reference_loss = None

        for b in range(len(blocks) - 1, -1, -1):
            block = blocks[b]
            label = f"pass{pass_id}:block{b}"
            if logger:
                logger.info(f"\n[RecursiveBlock] {label} t=[{block['t_start']:.2f},{block['t_end']:.2f}] T_block={block['T_block']:.2f}")

            x_mean, x_std = estimate_generator_stats(generators_per_block[b], D=D, n_samples=max(4096, M))

            model = NN_Quadratic_Coupled_Recursive(
                Xi_generator=generators_per_block[b], T=block["T_block"], M=M, N=N_per_block, D=D,
                layers=layers, parameters=params, t_start=block["t_start"], t_end=block["t_end"],
                T_total=T_total, terminal_blob=next_blob, normalize_time_input=True,
                x_norm_mean=x_mean, x_norm_std=x_std,
            )

            if warm_start_blobs is not None and warm_start_blobs[b] is not None:
                model.import_parameter_blob(warm_start_blobs[b], strict=False)

            precision_target = None
            if prev_pass_loss_by_block is not None and b in prev_pass_loss_by_block:
                precision_target = float(prev_pass_loss_by_block[b]) * (1.0 + precision_margin)
            elif reference_loss is not None:
                precision_target = reference_loss * (1.0 + precision_margin)

            block_stats = train_with_standard_schedule(
                model=model, stage_plan=stage_plan, final_plan=final_plan, eval_batches=5,
                precision_target=precision_target, max_refine_rounds=max_refine_rounds, label=label, logger=logger
            )

            eval_loss = block_stats["eval_stats"]["mean_loss"]
            if reference_loss is None:
                reference_loss = eval_loss

            blob = model.export_parameter_blob()
            blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
            save_blob_npz(blob, blob_path)
            model.save_weights(os.path.join(pass_dir, f"block_{b:02d}.weights.h5"))

            log_row = {
                "pass": int(pass_id), "block": int(b), "t_start": float(block["t_start"]),
                "t_end": float(block["t_end"]), "T_block": float(block["T_block"]),
                "eval_mean_loss": float(eval_loss), "eval_mean_y0": float(block_stats["eval_stats"]["mean_y0"]),
                "precision_target": float(precision_target) if precision_target else None,
                "refine_rounds": int(block_stats["refine_rounds"]),
                "blob_path": blob_path,
            }
            logs.append(log_row)
            block_blobs[b] = blob
            next_blob = blob

        logs = sorted(logs, key=lambda x: x["block"])
        return block_blobs, logs, float(reference_loss), pass_dir

    pass_results, prev_blobs, prev_boundary_samples, prev_pass_loss_by_block = [], None, None, None
    resumed_from = None
    start_pass_id = 1

    if resume_models_dir:
        resume_models_dir = resolve_resume_models_dir(resume_models_dir)
        loaded_pass_id = int(max(detect_available_recursive_passes(resume_models_dir)))
        prev_blobs = load_pass_blobs_from_models_dir(resume_models_dir, loaded_pass_id, blocks)
        prev_boundary_samples = rollout_boundaries(prev_blobs, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total)
        start_pass_id = loaded_pass_id + 1
        resumed_from = {"models_dir": resume_models_dir, "loaded_pass_id": loaded_pass_id}

    for pass_id in range(start_pass_id, int(n_passes) + 1):
        if pass_id == 1:
            generators = [Xi_generator for _ in blocks]
            warm_start = None
        else:
            if prev_boundary_samples is None:
                prev_boundary_samples = rollout_boundaries(prev_blobs, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total)
            generators = [make_empirical_generator(prev_boundary_samples[b], jitter_scale=empirical_jitter_scale) for b in range(len(blocks))]
            warm_start = prev_blobs

        blobs_i, logs_i, ref_loss_i, pass_dir_i = _run_pass(
            pass_id=pass_id, generators_per_block=generators, warm_start_blobs=warm_start,
            prev_pass_loss_by_block=prev_pass_loss_by_block
        )

        prev_blobs = blobs_i
        prev_pass_loss_by_block = {int(r["block"]): float(r["eval_mean_loss"]) for r in logs_i}
        prev_boundary_samples = rollout_boundaries(blobs_i, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total)

        pass_results.append({
            "pass_id": int(pass_id), "reference_loss": float(ref_loss_i),
            "logs": logs_i, "blobs": blobs_i, "models_dir": pass_dir_i
        })

    result = {
        "blocks": blocks, "passes": pass_results,
        "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
        "resumed_from": resumed_from,
    }
    return result

