import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf

from config import ModelParams, TrainingConfig
from utils import (
    setup_logger, save_json, save_rows_csv, Xi_generator_default,
    build_exact_solution_functions, compute_stitched_exact_bundle,
    save_exact_error_timeseries_csv
)
from train import (
    run_standard_reference, run_recursive_training, predict_recursive_stitched,
    build_stitched_rollout_inputs, load_training_plan_csv,
)
from visualization import (
    _PLOTTING_AVAILABLE, plot_stage_logs, plot_recursive_pass_logs_multi,
    plot_recursive_stitched_predictions, plot_recursive_stitched_y_convergence,
    plot_recursive_exact_comparison
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
    parser.add_argument(
        "--training_plan_csv",
        type=str,
        default="",
        help=(
            "CSV opzionale con piano training per blocco/pass. "
            "Colonne richieste: pass_scope,block_scope,phase,n_iter,lr "
            "(opzionali: order,enabled)."
        ),
    )
    parser.add_argument("--pass1_warm_start_from_next", action="store_true")
    parser.add_argument("--exact_solution", type=str, default="none")
    
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
        pass1_warm_start_from_next=args.pass1_warm_start_from_next,
        exact_solution=args.exact_solution,
        training_plan_csv=args.training_plan_csv,
    )
    
    params = ModelParams()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)
    
    logger = setup_logger(log_file=os.path.join(run_root, "execution.log"))
    logger.info(f"Avvio esperimento con mode={args.mode}")

    training_plan_rules = load_training_plan_csv(config.training_plan_csv)
    if len(training_plan_rules) > 0:
        logger.info(
            f"[TrainingPlan] loaded {len(training_plan_rules)} rules from {config.training_plan_csv}"
        )

    run_config = {
        "timestamp": run_id,
        "mode": args.mode,
        "config": config.__dict__,
        "training_plan_rules_count": len(training_plan_rules),
        "training_plan_rules": training_plan_rules,
        "params": params.to_dict(),
        "plotting_available": _PLOTTING_AVAILABLE,
        "exact_solution": config.exact_solution,
    }
    save_json(run_config, os.path.join(run_root, "run_config.json"))
    logger.info(f"[Artifacts] run directory: {run_root}")

    exact_solution = build_exact_solution_functions(
        solution_name=config.exact_solution,
        params=params.to_dict(),
        D=config.D,
    )
    if exact_solution is None:
        logger.info("[ExactSolution] disabled")
    else:
        logger.info(f"[ExactSolution] enabled profile='{exact_solution['name']}'")

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

        if exact_solution is not None:
            t_test, W_test, Xi_test = model_std.fetch_minibatch()
            X_pred, Y_pred, Z_pred = model_std.predict_model(Xi_test, t_test, W_test, const_value=1.0)
            stitched_std = {
                "t": np.array(t_test, dtype=np.float32),
                "X": np.array(X_pred, dtype=np.float32),
                "Y": np.array(Y_pred, dtype=np.float32),
                "Z": np.array(Z_pred, dtype=np.float32),
            }
            exact_std = compute_stitched_exact_bundle(
                stitched=stitched_std,
                exact_solution=exact_solution,
            )
            logger.info(
                "[Exact][Standard] "
                f"mean_pred_Y0={exact_std['summary']['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_std['summary']['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_std['summary']['abs_error_mean_y0']:.6e}"
            )

            save_json(
                {
                    "summary": exact_std["summary"],
                    "timeseries": exact_std["timeseries"],
                },
                os.path.join(std_dir, "exact_metrics.json"),
            )
            save_exact_error_timeseries_csv(
                exact_std["timeseries"],
                os.path.join(std_dir, "exact_errors.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_std,
                Y_exact=exact_std["Y_exact"],
                Z_exact=exact_std["Z_exact"],
                blocks=[{"t_start": 0.0, "t_end": float(config.T_standard), "T_block": float(config.T_standard)}],
                out_dir=os.path.join(std_dir, "plots"),
                sample_paths=8,
                file_suffix="",
            )
            std_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "summary": exact_std["summary"],
            }
        else:
            std_summary["exact_solution"] = {"enabled": False, "profile": "none"}

        save_json(std_summary, os.path.join(std_dir, "results.json"))

        logger.info(f"[STANDARD] final eval: {logs_std['eval_stats']}")
        del model_std
        tf.keras.backend.clear_session()

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
            save_tf_checkpoints=True,
            training_plan_rules=training_plan_rules,
            pass1_warm_start_from_next=bool(config.pass1_warm_start_from_next),
            n_passes=config.passes,
            resume_models_dir=args.resume_models_dir,
            resume_from_pass=args.resume_from_pass,
            empirical_jitter_scale=config.empirical_jitter_scale,
            logger=logger
        )

        pass_entries = sorted(rec.get("passes", []), key=lambda x: int(x["pass_id"]))
        if len(pass_entries) == 0:
            raise RuntimeError("No pass results available after recursive training")

        pass_logs_by_pass = {}
        for p in pass_entries:
            pass_id = int(p["pass_id"])
            logs = p.get("logs", [])
            pass_logs_by_pass[pass_id] = logs
            save_rows_csv(logs, os.path.join(rec_dir, f"pass_{pass_id:02d}_logs.csv"))
            if pass_id == 1:
                save_rows_csv(logs, os.path.join(rec_dir, "pass1_logs.csv"))
            if pass_id == 2:
                save_rows_csv(logs, os.path.join(rec_dir, "pass2_logs.csv"))

        plot_recursive_pass_logs_multi(pass_logs_by_pass, os.path.join(rec_dir, "plots"))

        score_key = "eval_mean_loss_per_sample"
        all_rows = [row for rows in pass_logs_by_pass.values() for row in rows]
        if not all(score_key in row for row in all_rows):
            score_key = "eval_mean_loss"

        pass_scores = {
            int(pass_id): score_pass_logs(rows, loss_key=score_key)
            for pass_id, rows in pass_logs_by_pass.items()
            if len(rows) > 0
        }
        if len(pass_scores) == 0:
            raise RuntimeError("No pass logs available for pass selection")

        best_pass_id = int(min(pass_scores, key=pass_scores.get))
        logger.info(
            f"[Selection] metric={score_key}, best_pass={best_pass_id}, "
            f"score={pass_scores[best_pass_id]:.6e}"
        )

        Xi_stitched = Xi_generator_default(max(64, config.M), config.D).astype(np.float32)
        rollout_inputs = build_stitched_rollout_inputs(
            blocks=rec["blocks"],
            M=Xi_stitched.shape[0],
            N_per_block=config.N,
            D=config.D,
            seed=1234,
        )

        stitched_by_pass = {}
        exact_summary_by_pass = {}
        selected_exact_bundle = None
        for p in pass_entries:
            pass_id = int(p["pass_id"])
            stitched_pred = predict_recursive_stitched(
                block_blobs=p["blobs"],
                blocks=rec["blocks"],
                Xi_initial=Xi_stitched,
                params=params,
                N_per_block=config.N,
                D=config.D,
                layers=config.layers,
                T_total=config.T_total,
                rollout_inputs=rollout_inputs,
            )
            stitched_by_pass[pass_id] = stitched_pred

            np.savez(
                os.path.join(rec_dir, f"stitched_predictions_pass{pass_id:02d}.npz"),
                t=stitched_pred["t"],
                X=stitched_pred["X"],
                Y=stitched_pred["Y"],
                Z=stitched_pred["Z"],
            )
            plot_recursive_stitched_predictions(
                stitched=stitched_pred,
                blocks=rec["blocks"],
                out_dir=os.path.join(rec_dir, "plots"),
                file_suffix=f"_pass{pass_id:02d}",
            )

            if exact_solution is not None:
                exact_bundle = compute_stitched_exact_bundle(
                    stitched=stitched_pred,
                    exact_solution=exact_solution,
                )
                exact_summary = exact_bundle["summary"]
                exact_summary_by_pass[pass_id] = exact_summary
                logger.info(
                    f"[Exact] pass{pass_id} "
                    f"mean_pred_Y0={exact_summary['mean_pred_y0']:.6f}, "
                    f"mean_exact_Y0={exact_summary['mean_exact_y0']:.6f}, "
                    f"abs_err_Y0={exact_summary['abs_error_mean_y0']:.6e}, "
                    f"mean_abs_err_Y={exact_summary['mean_abs_error_y']:.6e}, "
                    f"mean_abs_err_Z={exact_summary['mean_abs_error_z']:.6e}"
                )

                save_json(
                    {
                        "summary": exact_summary,
                        "timeseries": exact_bundle["timeseries"],
                    },
                    os.path.join(rec_dir, f"exact_metrics_pass{pass_id:02d}.json"),
                )
                save_exact_error_timeseries_csv(
                    exact_bundle["timeseries"],
                    os.path.join(rec_dir, f"exact_errors_pass{pass_id:02d}.csv"),
                )
                plot_recursive_exact_comparison(
                    stitched=stitched_pred,
                    Y_exact=exact_bundle["Y_exact"],
                    Z_exact=exact_bundle["Z_exact"],
                    blocks=rec["blocks"],
                    out_dir=os.path.join(rec_dir, "plots"),
                    sample_paths=8,
                    file_suffix=f"_pass{pass_id:02d}",
                )

                if pass_id == best_pass_id:
                    selected_exact_bundle = exact_bundle

        selected_stitched = stitched_by_pass[best_pass_id]
        np.savez(
            os.path.join(rec_dir, "stitched_predictions_final.npz"),
            t=selected_stitched["t"],
            X=selected_stitched["X"],
            Y=selected_stitched["Y"],
            Z=selected_stitched["Z"],
        )
        plot_recursive_stitched_predictions(
            selected_stitched,
            rec["blocks"],
            os.path.join(rec_dir, "plots"),
            sample_paths=8,
            file_suffix="",
        )

        if exact_solution is not None and selected_exact_bundle is not None:
            save_json(
                {
                    "summary": selected_exact_bundle["summary"],
                    "timeseries": selected_exact_bundle["timeseries"],
                },
                os.path.join(rec_dir, "exact_metrics_final.json"),
            )
            save_exact_error_timeseries_csv(
                selected_exact_bundle["timeseries"],
                os.path.join(rec_dir, "exact_errors_final.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=selected_stitched,
                Y_exact=selected_exact_bundle["Y_exact"],
                Z_exact=selected_exact_bundle["Z_exact"],
                blocks=rec["blocks"],
                out_dir=os.path.join(rec_dir, "plots"),
                sample_paths=8,
                file_suffix="",
            )

        plot_recursive_stitched_y_convergence(
            stitched_by_pass,
            rec["blocks"],
            os.path.join(rec_dir, "plots"),
            sample_paths=8,
        )

        boundary_stats = []
        for i, arr in enumerate(rec.get("boundary_samples", [])):
            boundary_stats.append(
                {
                    "boundary_idx": int(i),
                    "n_samples": int(arr.shape[0]),
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                    "min": np.min(arr, axis=0),
                    "max": np.max(arr, axis=0),
                }
            )

        passes_summary = []
        for p in pass_entries:
            passes_summary.append(
                {
                    "pass_id": int(p["pass_id"]),
                    "reference_loss": float(p["reference_loss"]),
                    "logs": p.get("logs", []),
                    "models_dir": p.get("models_dir", None),
                }
            )
        pass_summary_by_id = {int(p["pass_id"]): p for p in passes_summary}

        rec_summary = {
            "blocks": rec["blocks"],
            "passes": passes_summary,
            "resumed_from": rec.get("resumed_from", None),
            "boundary_stats": boundary_stats,
            "models_dir": os.path.join(rec_dir, "models"),
            "selected_pass_id": int(best_pass_id),
            "selected_score_metric": score_key,
            "selected_score": float(pass_scores[best_pass_id]),
            "pass_scores": {str(k): float(v) for k, v in pass_scores.items()},
        }
        if exact_solution is None:
            rec_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        else:
            rec_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "by_pass": {str(k): v for k, v in exact_summary_by_pass.items()},
                "selected_pass_summary": exact_summary_by_pass.get(int(best_pass_id), None),
            }
        if 1 in pass_summary_by_id:
            rec_summary["pass1"] = {
                "reference_loss": pass_summary_by_id[1]["reference_loss"],
                "logs": pass_summary_by_id[1]["logs"],
            }
        if 2 in pass_summary_by_id:
            rec_summary["pass2"] = {
                "reference_loss": pass_summary_by_id[2]["reference_loss"],
                "logs": pass_summary_by_id[2]["logs"],
            }
        save_json(rec_summary, os.path.join(rec_dir, "results.json"))

if __name__ == "__main__":
    main()
