import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf

from config import ModelParams, TrainingConfig
from utils import setup_logger, save_json, save_rows_csv, Xi_generator_default
from train import run_standard_reference, run_recursive_training, predict_recursive_stitched, build_stitched_rollout_inputs
from visualization import (
    _PLOTTING_AVAILABLE, plot_stage_logs, plot_recursive_pass_logs_multi,
    plot_recursive_stitched_predictions, plot_recursive_stitched_y_convergence
)

def score_pass_logs(rows, loss_key="eval_mean_loss_per_sample", worst_block_weight=0.35):
    losses = np.array([float(r.get(loss_key, np.nan)) for r in (rows or [])], dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return float("inf")
    return float(np.mean(losses) + worst_block_weight * np.max(losses))


def main():
    parser = argparse.ArgumentParser(description="Recursive time-stitching experiment (TF2)")
    parser.add_argument("--mode", type=str, default="recursive", choices=["standard", "recursive", "both"])
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--N", type=int, default=100, help="N steps per block")
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--T_standard", type=float, default=12.0)
    parser.add_argument("--T_total", type=float, default=48.0)
    parser.add_argument("--block_size", type=float, default=12.0)
    parser.add_argument("--output_dir", type=str, default="recursive1_outputs")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--resume_models_dir", type=str, default="")
    parser.add_argument("--resume_from_pass", type=int, default=0)
    parser.add_argument("--empirical_jitter_scale", type=float, default=0.02)
    parser.add_argument("--pass1_warm_start_from_next", action="store_true")
    
    args = parser.parse_args()

    np.random.seed(1234)
    tf.random.set_seed(1234)

    config = TrainingConfig(
        M=args.M,
        N=args.N,
        D=args.D,
        T_standard=args.T_standard,
        T_total=args.T_total,
        block_size=args.block_size,
        passes=args.passes,
        empirical_jitter_scale=args.empirical_jitter_scale,
        pass1_warm_start_from_next=args.pass1_warm_start_from_next
    )
    
    params = ModelParams()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)
    
    logger = setup_logger(log_file=os.path.join(run_root, "execution.log"))
    logger.info(f"Avvio esperimento con mode={args.mode}")

    run_config = {
        "timestamp": run_id,
        "mode": args.mode,
        "config": config.__dict__,
        "params": params.to_dict(),
        "plotting_available": _PLOTTING_AVAILABLE,
    }
    save_json(run_config, os.path.join(run_root, "run_config.json"))
    logger.info(f"[Artifacts] run directory: {run_root}")

    if args.mode in ("standard", "both"):
        logger.info("\n==================== STANDARD ====================")
        std_dir = os.path.join(run_root, "standard")
        os.makedirs(std_dir, exist_ok=True)
        
        model_std, logs_std = run_standard_reference(
            Xi_generator=Xi_generator_default,
            params=params,
            M=config.M,
            N=config.N,
            D=config.D,
            T=config.T_standard,
            layers=config.layers,
            stage_plan=config.stage_plan,
            final_plan=config.final_plan,
            logger=logger
        )

        std_blob_path = os.path.join(std_dir, "model_weights.h5")
        model_std.save_weights(std_blob_path)

        save_rows_csv(logs_std.get("stage_logs", []), os.path.join(std_dir, "stage_logs.csv"))
        plot_stage_logs(logs_std.get("stage_logs", []), out_prefix=os.path.join(std_dir, "standard"), title="Standard")

        std_summary = {
            "final_eval": logs_std.get("eval_stats", {}),
            "refine_rounds": logs_std.get("refine_rounds", 0),
            "weights_h5_path": std_blob_path,
        }
        save_json(std_summary, os.path.join(std_dir, "results.json"))

        logger.info(f"[STANDARD] final eval: {logs_std['eval_stats']}")

    if args.mode in ("recursive", "both"):
        logger.info("\n==================== RECURSIVE ====================")
        rec_dir = os.path.join(run_root, "recursive")
        os.makedirs(rec_dir, exist_ok=True)
        rec = run_recursive_training(
            Xi_generator=Xi_generator_default,
            params=params,
            M=config.M,
            N_per_block=config.N,
            D=config.D,
            T_total=config.T_total,
            block_size=config.block_size,
            layers=config.layers,
            stage_plan=config.stage_plan,
            final_plan=config.final_plan,
            output_dir=os.path.join(rec_dir, "models"),
            n_passes=config.passes,
            resume_models_dir=args.resume_models_dir,
            resume_from_pass=args.resume_from_pass,
            empirical_jitter_scale=config.empirical_jitter_scale,
            logger=logger
        )

        pass_entries = sorted(rec.get("passes", []), key=lambda x: int(x["pass_id"]))
        pass_logs_by_pass = {}
        for p in pass_entries:
            pass_id = int(p["pass_id"])
            logs = p.get("logs", [])
            pass_logs_by_pass[pass_id] = logs
            save_rows_csv(logs, os.path.join(rec_dir, f"pass_{pass_id:02d}_logs.csv"))

        plot_recursive_pass_logs_multi(pass_logs_by_pass, os.path.join(rec_dir, "plots"))
        
        score_key = "eval_mean_loss_per_sample"
        pass_scores = {
            int(pass_id): score_pass_logs(rows, loss_key=score_key)
            for pass_id, rows in pass_logs_by_pass.items() if len(rows) > 0
        }
        
        if len(pass_scores) > 0:
            best_pass_id = int(min(pass_scores, key=pass_scores.get))
            logger.info(f"[Selection] metric={score_key}, best_pass={best_pass_id}, score={pass_scores[best_pass_id]:.6e}")

            Xi_stitched = Xi_generator_default(max(64, config.M), config.D).astype(np.float32)
            rollout_inputs = build_stitched_rollout_inputs(
                blocks=rec["blocks"], M=Xi_stitched.shape[0], N_per_block=config.N, D=config.D, seed=1234,
            )
            
            stitched_by_pass = {}
            for p in pass_entries:
                pass_id = int(p["pass_id"])
                stitched_pred = predict_recursive_stitched(
                    block_blobs=p["blobs"], blocks=rec["blocks"], Xi_initial=Xi_stitched, params=params,
                    N_per_block=config.N, D=config.D, layers=config.layers, T_total=config.T_total, rollout_inputs=rollout_inputs,
                )
                stitched_by_pass[pass_id] = stitched_pred

                np.savez(os.path.join(rec_dir, f"stitched_predictions_pass{pass_id:02d}.npz"),
                         t=stitched_pred["t"], X=stitched_pred["X"], Y=stitched_pred["Y"], Z=stitched_pred["Z"])
                
                plot_recursive_stitched_predictions(
                    stitched=stitched_pred, blocks=rec["blocks"], out_dir=os.path.join(rec_dir, "plots"), file_suffix=f"_pass{pass_id:02d}"
                )

            selected_stitched = stitched_by_pass[best_pass_id]
            np.savez(os.path.join(rec_dir, "stitched_predictions_final.npz"),
                     t=selected_stitched["t"], X=selected_stitched["X"], Y=selected_stitched["Y"], Z=selected_stitched["Z"])
            plot_recursive_stitched_predictions(selected_stitched, rec["blocks"], os.path.join(rec_dir, "plots"), file_suffix="")

            plot_recursive_stitched_y_convergence(stitched_by_pass, rec["blocks"], os.path.join(rec_dir, "plots"))

if __name__ == "__main__":
    main()
