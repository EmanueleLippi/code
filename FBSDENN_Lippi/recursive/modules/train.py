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


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def load_training_plan_csv(csv_path: Optional[str]) -> List[Dict]:
    if csv_path is None or str(csv_path).strip() == "":
        return []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training plan CSV not found: {csv_path}")

    rules = []
    required = {"pass_scope", "block_scope", "phase", "n_iter", "lr"}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Training plan CSV is empty or has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Training plan CSV missing required columns: {sorted(missing)}")

        for i_row, row in enumerate(reader, start=2):
            if row is None:
                continue
            pass_scope = str(row.get("pass_scope", "")).strip()
            block_scope = str(row.get("block_scope", "")).strip().lower()
            phase = str(row.get("phase", "")).strip().lower()
            if pass_scope == "" or block_scope == "" or phase == "":
                continue
            if phase not in ("stage", "final", "refine"):
                raise ValueError(
                    f"Invalid phase '{phase}' in {csv_path}:{i_row} (allowed: stage, final, refine)"
                )

            enabled_raw = str(row.get("enabled", "1")).strip().lower()
            enabled = enabled_raw not in ("0", "false", "no", "off", "")
            if not enabled:
                continue

            order_raw = str(row.get("order", "0")).strip()
            order = int(order_raw) if order_raw != "" else 0
            n_iter = int(str(row.get("n_iter", "")).strip())
            lr = float(str(row.get("lr", "")).strip())
            if n_iter <= 0:
                raise ValueError(f"n_iter must be > 0 in {csv_path}:{i_row}")
            if lr <= 0:
                raise ValueError(f"lr must be > 0 in {csv_path}:{i_row}")

            rules.append(
                {
                    "pass_scope": pass_scope,
                    "block_scope": block_scope,
                    "phase": phase,
                    "order": int(order),
                    "n_iter": int(n_iter),
                    "lr": float(lr),
                    "source_row": int(i_row),
                }
            )

    return rules


def _pass_scope_priority(pass_scope: str, pass_id: int) -> int:
    ps = str(pass_scope).strip().lower()
    if ps in ("*", "all"):
        return 1

    if ps.endswith("+"):
        base = ps[:-1].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.startswith(">="):
        base = ps[2:].strip()
        if base.isdigit() and pass_id >= int(base):
            return 2

    if ps.isdigit() and pass_id == int(ps):
        return 3

    return -1


def _block_scope_priority(block_scope: str, block_idx: int, n_blocks: int) -> int:
    bs = str(block_scope).strip().lower()
    is_terminal = block_idx == (n_blocks - 1)

    if bs in ("*", "all"):
        return 1
    if bs == "terminal" and is_terminal:
        return 2
    if bs == "other" and (not is_terminal):
        return 2

    if bs.startswith("block:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3
    if bs.startswith("idx:"):
        token = bs.split(":", 1)[1].strip()
        if token.isdigit() and block_idx == int(token):
            return 3

    if bs.isdigit() and block_idx == int(bs):
        return 3
    return -1


def _resolve_phase_plan(
    rules: List[Dict],
    phase: str,
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_plan: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    matched = []
    for r in rules:
        if r["phase"] != phase:
            continue
        p_prio = _pass_scope_priority(r["pass_scope"], pass_id)
        if p_prio < 0:
            continue
        b_prio = _block_scope_priority(r["block_scope"], block_idx, n_blocks)
        if b_prio < 0:
            continue
        matched.append((p_prio, b_prio, r["order"], r))

    if len(matched) == 0:
        return list(default_plan)

    best_scope = max((x[0], x[1]) for x in matched)
    selected = [x for x in matched if (x[0], x[1]) == best_scope]
    selected.sort(key=lambda x: x[2])
    return [(int(x[3]["n_iter"]), float(x[3]["lr"])) for x in selected]


def resolve_training_plan_for_block(
    rules: List[Dict],
    pass_id: int,
    block_idx: int,
    n_blocks: int,
    default_stage: List[Tuple[int, float]],
    default_final: List[Tuple[int, float]],
    default_refine: List[Tuple[int, float]],
) -> Dict[str, List[Tuple[int, float]]]:
    if rules is None:
        rules = []
    return {
        "stage_plan": _resolve_phase_plan(
            rules=rules,
            phase="stage",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_stage,
        ),
        "final_plan": _resolve_phase_plan(
            rules=rules,
            phase="final",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_final,
        ),
        "refine_plan": _resolve_phase_plan(
            rules=rules,
            phase="refine",
            pass_id=pass_id,
            block_idx=block_idx,
            n_blocks=n_blocks,
            default_plan=default_refine,
        ),
    }


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
        model = None
        try:
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
        finally:
            del model
            tf.keras.backend.clear_session()

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
    if len(blocks) == 0:
        raise ValueError("blocks must contain at least one block")
    if Xi_initial.ndim != 2 or Xi_initial.shape[1] != D:
        raise ValueError(f"Xi_initial must have shape [M, {D}]")
    if rollout_inputs is not None and len(rollout_inputs) != len(blocks):
        raise ValueError("rollout_inputs must have one (t, W) pair per block")

    Xi_curr = Xi_initial.astype(np.float32)
    t_segments, X_segments, Y_segments, Z_segments = [], [], [], []

    for b, block in enumerate(blocks):
        blob = block_blobs[b]

        model = None
        try:
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
        finally:
            del model
            tf.keras.backend.clear_session()

    return {
        "t": np.concatenate(t_segments, axis=1),
        "X": np.concatenate(X_segments, axis=1),
        "Y": np.concatenate(Y_segments, axis=1),
        "Z": np.concatenate(Z_segments, axis=1),
    }


def run_recursive_training(
    Xi_generator, params, M, N_per_block, D, T_total, block_size, layers,
    stage_plan, final_plan, output_dir, precision_margin=0.10, max_refine_rounds=3,
    rollout_M=2000, save_tf_checkpoints=True, training_plan_rules: Optional[List[Dict]] = None,
    pass1_warm_start_from_next=False, n_passes=2, resume_models_dir="", resume_from_pass=0,
    empirical_jitter_scale=0.02, logger=None
):
    if int(n_passes) < 1:
        raise ValueError("n_passes must be >= 1")
    if int(resume_from_pass) < 0:
        raise ValueError("resume_from_pass must be >= 0")

    blocks = build_blocks(T_total=T_total, block_size=block_size)
    _log(
        logger,
        f"[Recursive] blocks={len(blocks)} -> {[(b['t_start'], b['t_end']) for b in blocks]}, "
        f"n_passes={int(n_passes)}",
    )

    def _run_pass(
        pass_id,
        generators_per_block,
        warm_start_blobs=None,
        warm_start_from_next=False,
        prev_pass_loss_by_block=None,
    ):
        pass_dir = os.path.join(output_dir, f"pass_{pass_id}")
        os.makedirs(pass_dir, exist_ok=True)
        next_blob = None
        block_blobs = [None] * len(blocks)
        logs = []
        reference_loss = None

        for b in range(len(blocks) - 1, -1, -1):
            block = blocks[b]
            label = f"pass{pass_id}:block{b}"
            _log(
                logger,
                f"\n[RecursiveBlock] {label} t=[{block['t_start']:.2f},{block['t_end']:.2f}] "
                f"T_block={block['T_block']:.2f}",
            )

            x_mean, x_std = estimate_generator_stats(generators_per_block[b], D=D, n_samples=max(4096, M))

            model = None
            try:
                model = NN_Quadratic_Coupled_Recursive(
                    Xi_generator=generators_per_block[b], T=block["T_block"], M=M, N=N_per_block, D=D,
                    layers=layers, parameters=params, t_start=block["t_start"], t_end=block["t_end"],
                    T_total=T_total, terminal_blob=next_blob, normalize_time_input=True,
                    x_norm_mean=x_mean, x_norm_std=x_std,
                )

                if warm_start_from_next and next_blob is not None:
                    model.import_parameter_blob(next_blob, strict=False)

                if warm_start_blobs is not None and warm_start_blobs[b] is not None:
                    model.import_parameter_blob(warm_start_blobs[b], strict=False)

                precision_target = None
                if prev_pass_loss_by_block is not None and b in prev_pass_loss_by_block:
                    precision_target = float(prev_pass_loss_by_block[b]) * (1.0 + precision_margin)
                elif reference_loss is not None:
                    precision_target = reference_loss * (1.0 + precision_margin)

                default_refine_plan = [(50, 1e-5), (50, 5e-6)]
                resolved_plan = resolve_training_plan_for_block(
                    rules=training_plan_rules or [],
                    pass_id=int(pass_id),
                    block_idx=int(b),
                    n_blocks=len(blocks),
                    default_stage=stage_plan,
                    default_final=final_plan,
                    default_refine=default_refine_plan,
                )

                block_stats = train_with_standard_schedule(
                    model=model,
                    stage_plan=resolved_plan["stage_plan"],
                    final_plan=resolved_plan["final_plan"],
                    eval_batches=5,
                    precision_target=precision_target,
                    max_refine_rounds=max_refine_rounds,
                    refine_plan=resolved_plan["refine_plan"],
                    label=label,
                    logger=logger,
                )

                eval_stats = block_stats["eval_stats"]
                eval_loss = float(eval_stats["mean_loss"])
                if reference_loss is None:
                    reference_loss = eval_loss

                blob = model.export_parameter_blob()
                blob_path = os.path.join(pass_dir, f"block_{b:02d}.npz")
                save_blob_npz(blob, blob_path)

                weights_path = None
                if save_tf_checkpoints:
                    weights_path = os.path.join(pass_dir, f"block_{b:02d}.weights.h5")
                    model.save_weights(weights_path)

                log_row = {
                    "pass": int(pass_id),
                    "block": int(b),
                    "t_start": float(block["t_start"]),
                    "t_end": float(block["t_end"]),
                    "T_block": float(block["T_block"]),
                    "eval_mean_loss": float(eval_stats["mean_loss"]),
                    "eval_std_loss": float(eval_stats["std_loss"]),
                    "eval_mean_loss_per_sample": float(eval_stats["mean_loss_per_sample"]),
                    "eval_std_loss_per_sample": float(eval_stats["std_loss_per_sample"]),
                    "eval_mean_y0": float(eval_stats["mean_y0"]),
                    "precision_target": None if precision_target is None else float(precision_target),
                    "refine_rounds": int(block_stats["refine_rounds"]),
                    "stage_plan_used": resolved_plan["stage_plan"],
                    "final_plan_used": resolved_plan["final_plan"],
                    "refine_plan_used": resolved_plan["refine_plan"],
                    "blob_path": blob_path,
                    "weights_path": weights_path,
                }
                logs.append(log_row)
                block_blobs[b] = blob
                next_blob = blob
            finally:
                del model
                tf.keras.backend.clear_session()

        logs = sorted(logs, key=lambda x: x["block"])
        return block_blobs, logs, float(reference_loss), pass_dir

    pass_results, prev_blobs, prev_boundary_samples, prev_pass_loss_by_block = [], None, None, None
    resumed_from = None
    start_pass_id = 1

    resume_models_dir = str(resume_models_dir or "").strip()
    if resume_models_dir != "":
        resume_models_dir = resolve_resume_models_dir(resume_models_dir)
        available_passes = detect_available_recursive_passes(resume_models_dir)
        if len(available_passes) == 0:
            raise FileNotFoundError(
                f"No pass_* directories found in resume_models_dir: {resume_models_dir}"
            )

        if int(resume_from_pass) > 0:
            loaded_pass_id = int(resume_from_pass)
            if loaded_pass_id not in available_passes:
                raise ValueError(
                    f"Requested resume_from_pass={loaded_pass_id} not found in "
                    f"{resume_models_dir}. Available: {available_passes}"
                )
        else:
            loaded_pass_id = int(max(available_passes))

        if loaded_pass_id >= int(n_passes):
            raise ValueError(
                f"Loaded pass={loaded_pass_id} but n_passes={int(n_passes)}. "
                "Set n_passes > loaded pass to continue training."
            )

        prev_blobs = load_pass_blobs_from_models_dir(resume_models_dir, loaded_pass_id, blocks)
        prev_boundary_samples = rollout_boundaries(
            prev_blobs, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total
        )
        start_pass_id = loaded_pass_id + 1
        resumed_from = {
            "models_dir": resume_models_dir,
            "loaded_pass_id": int(loaded_pass_id),
            "available_passes": available_passes,
        }
        _log(
            logger,
            f"[Resume] loaded pass={loaded_pass_id} from {resume_models_dir}, "
            f"continuing from pass={start_pass_id}",
        )

    for pass_id in range(start_pass_id, int(n_passes) + 1):
        if pass_id == 1:
            generators = [Xi_generator for _ in blocks]
            warm_start = None
            warm_from_next = bool(pass1_warm_start_from_next)
        else:
            if prev_boundary_samples is None:
                if prev_blobs is None:
                    raise RuntimeError("Internal error: missing previous blobs for pass>=2")
                prev_boundary_samples = rollout_boundaries(
                    prev_blobs, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total
                )
            generators = [
                make_empirical_generator(prev_boundary_samples[b], jitter_scale=empirical_jitter_scale)
                for b in range(len(blocks))
            ]
            warm_start = prev_blobs
            warm_from_next = False

        blobs_i, logs_i, ref_loss_i, pass_dir_i = _run_pass(
            pass_id=pass_id,
            generators_per_block=generators,
            warm_start_blobs=warm_start,
            warm_start_from_next=warm_from_next,
            prev_pass_loss_by_block=prev_pass_loss_by_block,
        )

        prev_blobs = blobs_i
        prev_pass_loss_by_block = {int(r["block"]): float(r["eval_mean_loss"]) for r in logs_i}
        prev_boundary_samples = rollout_boundaries(
            blobs_i, blocks, Xi_generator, params, rollout_M, N_per_block, D, layers, T_total
        )

        pass_results.append({
            "pass_id": int(pass_id), "reference_loss": float(ref_loss_i),
            "logs": logs_i, "blobs": blobs_i, "models_dir": pass_dir_i
        })

    result = {
        "blocks": blocks, "passes": pass_results,
        "boundary_samples": prev_boundary_samples if prev_boundary_samples is not None else [],
        "resumed_from": resumed_from,
    }
    for item in pass_results:
        if item["pass_id"] == 1:
            result["pass1"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }
        if item["pass_id"] == 2:
            result["pass2"] = {
                "logs": item["logs"],
                "reference_loss": item["reference_loss"],
                "blobs": item["blobs"],
            }
    return result
