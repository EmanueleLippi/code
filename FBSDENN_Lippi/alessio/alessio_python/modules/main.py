import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from config import ModelParams, TrainingConfig
from train import (
    build_stitched_rollout_inputs,
    find_resume_eval_bundle_path,
    load_training_plan_csv,
    load_training_plan_rules_from_resume_run,
    predict_recursive_stitched,
    resolve_resume_models_dir,
    run_recursive_training,
    run_standard_reference,
)
from utils import (
    _pass_index,
    _pass_label,
    _pass_tag,
    Xi_generator_default,
    build_exact_solution_functions,
    compute_stitched_exact_bundle,
    export_standard_parameter_blob,
    load_evaluation_bundle,
    make_deterministic_xi_default,
    resolve_pass_selection,
    save_blob_npz,
    save_evaluation_bundle,
    save_exact_error_timeseries_csv,
    save_json,
    save_rows_csv,
    score_pass_logs,
    setup_logger,
)
from visualization import (
    _PLOTTING_AVAILABLE,
    plot_recursive_exact_comparison,
    plot_recursive_pass_logs_multi,
    plot_recursive_stitched_predictions,
    plot_recursive_stitched_y_convergence,
    plot_stage_logs,
)


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _configure_tensorflow_runtime(logger=None) -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        _log(logger, f"[TensorFlow] unable to query GPUs: {exc}")
        return

    if len(gpus) == 0:
        _log(logger, "[TensorFlow] no visible GPUs")
        return

    gpu_labels = []
    for gpu in gpus:
        label = str(getattr(gpu, "name", gpu))
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            label = f"{label} (memory_growth=on)"
        except Exception as exc:
            label = f"{label} (memory_growth=off: {exc})"
        gpu_labels.append(label)

    _log(logger, "[TensorFlow] visible GPUs: " + ", ".join(gpu_labels))


def print_recursive_pass(
    pass_entries: List[Dict[str, Any]],
    blocks: List[Dict[str, float]],
    rec_dir: str,
    params: ModelParams,
    N_per_block: int,
    D: int,
    layers: List[int],
    T_total: float,
    exact_solution: Optional[Dict[str, Any]],
    selection_metric: str = "auto",
    exact_regression_tolerance: float = 0.20,
    exact_regression_action: str = "warn",
    eval_bundle_path: str = "",
    eval_seed: int = 1234,
    eval_min_paths: int = 64,
    sample_paths: int = 8,
    enforce_exact_regression_guardrail: bool = True,
    print_compact_logs: bool = True,
    logger=None,
) -> Dict[str, Any]:
    if pass_entries is None or len(pass_entries) == 0:
        raise RuntimeError("print_recursive_pass called with empty pass_entries")

    pass_entries = sorted(pass_entries, key=lambda x: int(x["pass_id"]))
    os.makedirs(rec_dir, exist_ok=True)
    plots_dir = os.path.join(rec_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    pass_logs_by_pass = {}
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_idx = _pass_index(pass_id)
        logs = p.get("logs", [])
        pass_logs_by_pass[pass_id] = logs

        if print_compact_logs:
            _log(logger, f"\n=== Recursive Log {_pass_label(pass_id)} (compact) ===")
            for row in logs:
                norm_msg = ""
                if "eval_mean_loss_per_sample" in row:
                    norm_msg = f", eval_loss/M={row['eval_mean_loss_per_sample']:.3e}"
                _log(
                    logger,
                    f"block={row['block']}, t=[{row['t_start']:.1f},{row['t_end']:.1f}], "
                    f"eval_loss={row['eval_mean_loss']:.3e}{norm_msg}, eval_y0={row['eval_mean_y0']:.3f}, "
                    f"target={row['precision_target']}, refine={row['refine_rounds']}",
                )

        save_rows_csv(logs, os.path.join(rec_dir, f"pass_{pass_idx:02d}_logs.csv"))
        if pass_idx == 0:
            save_rows_csv(logs, os.path.join(rec_dir, "pass0_logs.csv"))
        if pass_idx == 1:
            save_rows_csv(logs, os.path.join(rec_dir, "pass1_logs.csv"))

    plot_recursive_pass_logs_multi(pass_logs_by_pass, plots_dir)

    score_key = "eval_mean_loss_per_sample"
    all_rows = [row for rows in pass_logs_by_pass.values() for row in rows]
    if not all(score_key in row for row in all_rows):
        score_key = "eval_mean_loss"
    pass_scores_loss = {
        int(pass_id): score_pass_logs(rows, loss_key=score_key)
        for pass_id, rows in pass_logs_by_pass.items()
        if len(rows) > 0
    }
    if len(pass_scores_loss) == 0:
        raise RuntimeError("No pass logs available for pass selection")
    best_pass_by_loss = int(min(pass_scores_loss, key=pass_scores_loss.get))
    _log(
        logger,
        f"[Selection:loss] metric={score_key}, best={_pass_label(best_pass_by_loss)}, "
        f"score={pass_scores_loss[best_pass_by_loss]:.6e}",
    )

    eval_bundle_path = str(eval_bundle_path or "").strip()
    if eval_bundle_path == "":
        eval_bundle_path = os.path.join(rec_dir, "evaluation_bundle.npz")
    eval_bundle_path = os.path.abspath(os.path.expanduser(eval_bundle_path))

    if os.path.isfile(eval_bundle_path):
        Xi_stitched, rollout_inputs = load_evaluation_bundle(
            path=eval_bundle_path,
            n_blocks_expected=len(blocks),
            N_per_block_expected=N_per_block,
            D_expected=D,
        )
        _log(
            logger,
            f"[EvalBundle] loaded path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"blocks={len(rollout_inputs)}",
        )
    else:
        Xi_stitched = make_deterministic_xi_default(
            max(1, int(eval_min_paths)),
            D,
            seed=int(eval_seed),
        )
        rollout_inputs = build_stitched_rollout_inputs(
            blocks=blocks,
            M=Xi_stitched.shape[0],
            N_per_block=N_per_block,
            D=D,
            seed=int(eval_seed),
        )
        save_evaluation_bundle(
            path=eval_bundle_path,
            Xi_initial=Xi_stitched,
            rollout_inputs=rollout_inputs,
            blocks=blocks,
        )
        _log(
            logger,
            f"[EvalBundle] created path={eval_bundle_path}, M={Xi_stitched.shape[0]}, "
            f"seed={int(eval_seed)}",
        )

    stitched_by_pass = {}
    exact_summary_by_pass = {}
    exact_bundle_by_pass = {}
    for p in pass_entries:
        pass_id = int(p["pass_id"])
        pass_tag = _pass_tag(pass_id)
        stitched_pred = predict_recursive_stitched(
            block_blobs=p["blobs"],
            blocks=blocks,
            Xi_initial=Xi_stitched,
            params=params,
            N_per_block=N_per_block,
            D=D,
            layers=layers,
            T_total=T_total,
            rollout_inputs=rollout_inputs,
        )
        stitched_by_pass[pass_id] = stitched_pred

        np.savez(
            os.path.join(rec_dir, f"stitched_predictions_{pass_tag}.npz"),
            t=stitched_pred["t"],
            X=stitched_pred["X"],
            Y=stitched_pred["Y"],
            Z=stitched_pred["Z"],
        )
        plot_recursive_stitched_predictions(
            stitched=stitched_pred,
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix=f"_{pass_tag}",
        )

        if exact_solution is not None:
            exact_bundle = compute_stitched_exact_bundle(
                stitched=stitched_pred,
                exact_solution=exact_solution,
            )
            exact_summary = exact_bundle["summary"]
            exact_summary_by_pass[pass_id] = exact_summary
            exact_bundle_by_pass[pass_id] = exact_bundle
            _log(
                logger,
                f"[Exact] {_pass_label(pass_id)} "
                f"mean_pred_Y0={exact_summary['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_summary['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_summary['abs_error_mean_y0']:.6e}, "
                f"mean_abs_err_Y={exact_summary['mean_abs_error_y']:.6e}, "
                f"mean_abs_err_Z={exact_summary['mean_abs_error_z']:.6e}",
            )

            save_json(
                {
                    "summary": exact_summary,
                    "timeseries": exact_bundle["timeseries"],
                },
                os.path.join(rec_dir, f"exact_metrics_{pass_tag}.json"),
            )
            save_exact_error_timeseries_csv(
                exact_bundle["timeseries"],
                os.path.join(rec_dir, f"exact_errors_{pass_tag}.csv"),
            )
            plot_recursive_exact_comparison(
                stitched=stitched_pred,
                Y_exact=exact_bundle["Y_exact"],
                Z_exact=exact_bundle["Z_exact"],
                blocks=blocks,
                out_dir=plots_dir,
                sample_paths=sample_paths,
                file_suffix=f"_{pass_tag}",
            )

    if (
        enforce_exact_regression_guardrail
        and exact_solution is not None
        and len(exact_summary_by_pass) >= 2
        and str(exact_regression_action) != "ignore"
    ):
        tol = float(exact_regression_tolerance)
        if tol > 0.0:
            sorted_pass_ids = sorted(exact_summary_by_pass.keys())
            prev_id = sorted_pass_ids[0]
            prev_val = float(exact_summary_by_pass[prev_id]["mean_abs_error_y"])
            for pass_id in sorted_pass_ids[1:]:
                curr_val = float(exact_summary_by_pass[pass_id]["mean_abs_error_y"])
                if prev_val > 0.0 and curr_val > prev_val * (1.0 + tol):
                    msg = (
                        "[ExactGuardrail] Regression detected on mean_abs_error_y: "
                        f"{_pass_label(prev_id)}={prev_val:.6e} -> {_pass_label(pass_id)}={curr_val:.6e} "
                        f"(+{(curr_val / prev_val - 1.0) * 100.0:.2f}%, tol={tol * 100.0:.2f}%)"
                    )
                    if str(exact_regression_action) == "error":
                        raise RuntimeError(msg)
                    _log(logger, msg)
                prev_id = pass_id
                prev_val = curr_val

    selected_pass_id, selected_score_metric, selected_score, selected_score_by_pass = resolve_pass_selection(
        pass_scores_by_loss=pass_scores_loss,
        exact_summary_by_pass=exact_summary_by_pass,
        selection_metric=str(selection_metric),
        loss_metric_label=score_key,
    )
    _log(
        logger,
        f"[Selection:final] metric={selected_score_metric}, best={_pass_label(selected_pass_id)}, "
        f"score={selected_score:.6e}",
    )

    selected_stitched = stitched_by_pass[selected_pass_id]
    selected_exact_bundle = exact_bundle_by_pass.get(selected_pass_id, None)
    np.savez(
        os.path.join(rec_dir, "stitched_predictions_final.npz"),
        t=selected_stitched["t"],
        X=selected_stitched["X"],
        Y=selected_stitched["Y"],
        Z=selected_stitched["Z"],
    )
    plot_recursive_stitched_predictions(
        stitched=selected_stitched,
        blocks=blocks,
        out_dir=plots_dir,
        sample_paths=sample_paths,
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
            blocks=blocks,
            out_dir=plots_dir,
            sample_paths=sample_paths,
            file_suffix="",
        )

    plot_recursive_stitched_y_convergence(
        stitched_by_pass=stitched_by_pass,
        blocks=blocks,
        out_dir=plots_dir,
        sample_paths=sample_paths,
    )

    return {
        "processed_pass_ids": sorted(pass_logs_by_pass.keys()),
        "processed_pass_indices": sorted(_pass_index(pid) for pid in pass_logs_by_pass.keys()),
        "score_key": score_key,
        "pass_scores_loss": pass_scores_loss,
        "pass_scores_loss_by_index": {
            str(_pass_index(k)): float(v) for k, v in pass_scores_loss.items()
        },
        "selected_pass_id": int(selected_pass_id),
        "selected_pass_index": int(_pass_index(selected_pass_id)),
        "selected_score_metric": selected_score_metric,
        "selected_score": float(selected_score),
        "selected_scores_by_pass": selected_score_by_pass,
        "selected_scores_by_pass_index": {
            str(_pass_index(int(k))): float(v)
            for k, v in selected_score_by_pass.items()
        },
        "exact_summary_by_pass": exact_summary_by_pass,
        "exact_summary_by_pass_index": {
            str(_pass_index(k)): v for k, v in exact_summary_by_pass.items()
        },
        "eval_bundle_path": eval_bundle_path,
        "evaluation_bundle_M": int(Xi_stitched.shape[0]),
    }


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
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Numero totale di pass ricorsive da eseguire (>=1).",
    )
    parser.add_argument(
        "--resume_models_dir",
        type=str,
        default="",
        help=(
            "Directory con pass_*/block_XX.npz di una run precedente "
            "da cui riprendere (es. .../recursive/models)."
        ),
    )
    parser.add_argument(
        "--resume_from_pass",
        type=int,
        default=0,
        help="Pass di partenza nel resume. 0=auto (massima disponibile in resume_models_dir).",
    )
    parser.add_argument(
        "--empirical_jitter_scale",
        type=float,
        default=0.02,
        help="Rumore relativo usato nel generatore empirico per pass >= 2.",
    )
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
    parser.add_argument(
        "--pass1_warm_start_from_next",
        action="store_true",
        help=(
            "Se attivo, in pass1 il blocco i viene inizializzato coi pesi del blocco i+1 "
            "(quando disponibile). Le passate successive possono usare warm-start dal pass "
            "precedente (default attivo, disattivabile con --disable_cross_pass_warm_start)."
        ),
    )
    parser.add_argument(
        "--disable_cross_pass_warm_start",
        action="store_true",
        help=(
            "Se attivo, disabilita il warm-start dalle passate precedenti "
            "(warm_start=prev_blobs) per pass>=2."
        ),
    )
    parser.add_argument(
        "--exact_solution",
        type=str,
        default="none",
        help=(
            "Profilo opzionale per confronto con soluzione esatta. "
            "Valori supportati: none, quadratic_coupled"
        ),
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="auto",
        choices=["auto", "loss", "exact_mae_y", "exact_rmse_y", "exact_abs_y0"],
        help=(
            "Metrica di selezione della pass finale: "
            "auto usa exact_mae_y se exact_solution è attiva, altrimenti loss."
        ),
    )
    parser.add_argument(
        "--exact_regression_tolerance",
        type=float,
        default=0.20,
        help=(
            "Tolleranza regressione relativa tra pass consecutive su mean_abs_error_y "
            "(es. 0.20 = +20%). <=0 disabilita il guardrail."
        ),
    )
    parser.add_argument(
        "--exact_regression_action",
        type=str,
        default="warn",
        choices=["warn", "error", "ignore"],
        help="Azione quando il guardrail exact rileva regressione oltre soglia.",
    )
    parser.add_argument(
        "--eval_bundle_path",
        type=str,
        default="",
        help=(
            "Percorso opzionale a evaluation_bundle.npz da riusare per confronto path-by-path "
            "tra pass/run."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=1234,
        help="Seed usato per costruire un evaluation bundle nuovo quando non viene caricato.",
    )
    parser.add_argument(
        "--allow_resume_without_plan",
        action="store_true",
        help=(
            "Permette resume con training_plan assente/non ereditabile. "
            "Di default il resume fallisce per evitare mismatch di schedule."
        ),
    )
    args = parser.parse_args()

    np.random.seed(1234)

    config = TrainingConfig(
        M=args.M,
        N=args.N,
        D=args.D,
        T_standard=args.T_standard,
        T_total=args.T_total,
        block_size=args.block_size,
        passes=int(args.passes),
        resume_models_dir=str(args.resume_models_dir or "").strip(),
        resume_from_pass=int(args.resume_from_pass),
        empirical_jitter_scale=float(args.empirical_jitter_scale),
        pass1_warm_start_from_next=bool(args.pass1_warm_start_from_next),
        cross_pass_warm_start=not bool(args.disable_cross_pass_warm_start),
        exact_solution=str(args.exact_solution),
        selection_metric=str(args.selection_metric),
        exact_regression_tolerance=float(args.exact_regression_tolerance),
        exact_regression_action=str(args.exact_regression_action),
        eval_bundle_path=str(args.eval_bundle_path or "").strip(),
        eval_seed=int(args.eval_seed),
        allow_resume_without_plan=bool(args.allow_resume_without_plan),
        training_plan_csv=str(args.training_plan_csv or "").strip(),
    )
    params = ModelParams()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)

    logger = setup_logger(log_file=os.path.join(run_root, "execution.log"))
    _log(logger, f"Avvio esperimento con mode={args.mode}")
    _configure_tensorflow_runtime(logger)
    tf.random.set_seed(1234)

    training_plan_rules = load_training_plan_csv(config.training_plan_csv)
    training_plan_effective_source = str(config.training_plan_csv or "").strip()
    training_plan_inherited_from_resume = False
    training_plan_inherited_run_config = None
    training_plan_inherited_csv = None

    resume_models_dir_resolved = (
        resolve_resume_models_dir(config.resume_models_dir) if config.resume_models_dir != "" else ""
    )

    if resume_models_dir_resolved != "" and len(training_plan_rules) == 0:
        inherited_rules, resume_cfg_path, resume_plan_csv = load_training_plan_rules_from_resume_run(
            resume_models_dir_resolved
        )
        if len(inherited_rules) > 0:
            training_plan_rules = inherited_rules
            training_plan_effective_source = f"inherited_from_resume:{resume_cfg_path}"
            training_plan_inherited_from_resume = True
            training_plan_inherited_run_config = resume_cfg_path
            training_plan_inherited_csv = resume_plan_csv
            _log(
                logger,
                f"[TrainingPlan] inherited {len(training_plan_rules)} rules from resume run config: "
                f"{resume_cfg_path}",
            )
        else:
            msg = (
                "Resume requested but no training plan was provided and no reusable "
                "training_plan_rules were found in the resumed run config. "
                "Pass --training_plan_csv (recommended) or --allow_resume_without_plan to proceed."
            )
            if bool(config.allow_resume_without_plan):
                _log(logger, f"[TrainingPlan][WARN] {msg}")
            else:
                raise ValueError(msg)

    if len(training_plan_rules) > 0:
        _log(
            logger,
            f"[TrainingPlan] loaded {len(training_plan_rules)} rules from {training_plan_effective_source}",
        )

    exact_solution = build_exact_solution_functions(
        solution_name=config.exact_solution,
        params=params.to_dict(),
        D=config.D,
    )
    if exact_solution is None:
        _log(logger, "[ExactSolution] disabled")
    else:
        _log(logger, f"[ExactSolution] enabled profile='{exact_solution['name']}'")

    run_config = {
        "timestamp": run_id,
        "mode": args.mode,
        "M": config.M,
        "N": config.N,
        "D": config.D,
        "T_standard": config.T_standard,
        "T_total": config.T_total,
        "block_size": config.block_size,
        "passes": int(config.passes),
        "resume_models_dir": config.resume_models_dir,
        "resume_models_dir_resolved": resume_models_dir_resolved,
        "resume_from_pass": int(config.resume_from_pass),
        "empirical_jitter_scale": float(config.empirical_jitter_scale),
        "layers": config.layers,
        "stage_plan": config.stage_plan,
        "final_plan": config.final_plan,
        "training_plan_csv": config.training_plan_csv,
        "training_plan_effective_source": training_plan_effective_source,
        "training_plan_rules_count": len(training_plan_rules),
        "training_plan_rules": training_plan_rules,
        "training_plan_inherited_from_resume": bool(training_plan_inherited_from_resume),
        "training_plan_inherited_run_config": training_plan_inherited_run_config,
        "training_plan_inherited_csv": training_plan_inherited_csv,
        "pass1_warm_start_from_next": bool(config.pass1_warm_start_from_next),
        "cross_pass_warm_start": bool(config.cross_pass_warm_start),
        "selection_metric": str(config.selection_metric),
        "exact_regression_tolerance": float(config.exact_regression_tolerance),
        "exact_regression_action": str(config.exact_regression_action),
        "eval_bundle_path": str(config.eval_bundle_path),
        "eval_seed": int(config.eval_seed),
        "allow_resume_without_plan": bool(config.allow_resume_without_plan),
        "exact_solution": "none" if exact_solution is None else exact_solution["name"],
        "params": params.to_dict(),
        "plotting_available": _PLOTTING_AVAILABLE,
    }
    save_json(run_config, os.path.join(run_root, "run_config.json"))
    _log(logger, f"[Artifacts] run directory: {run_root}")

    if args.mode in ("standard", "both"):
        _log(logger, "\n==================== STANDARD ====================")
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
            logger=logger,
        )

        std_ckpt_path = os.path.join(std_dir, "model.weights.h5")
        model_std.save_model(std_ckpt_path)

        std_blob = export_standard_parameter_blob(model_std)
        std_blob_path = os.path.join(std_dir, "model_weights.npz")
        save_blob_npz(std_blob, std_blob_path)

        save_rows_csv(logs_std.get("stage_logs", []), os.path.join(std_dir, "stage_logs.csv"))
        plot_stage_logs(
            logs_std.get("stage_logs", []),
            out_prefix=os.path.join(std_dir, "standard"),
            title="Standard",
        )

        std_summary = {
            "final_eval": logs_std.get("eval_stats", {}),
            "refine_rounds": logs_std.get("refine_rounds", 0),
            "checkpoint_path": std_ckpt_path,
            "weights_npz_path": std_blob_path,
        }
        if exact_solution is not None:
            t_test, W_test, Xi_test = model_std.fetch_minibatch()
            X_pred, Y_pred, Z_pred = model_std.predict_model(Xi_test, t_test, W_test, const_value=1.0)
            stitched_std = {
                "t": np.asarray(t_test, dtype=np.float32),
                "X": X_pred.astype(np.float32),
                "Y": Y_pred.astype(np.float32),
                "Z": Z_pred.astype(np.float32),
            }
            exact_std = compute_stitched_exact_bundle(
                stitched=stitched_std,
                exact_solution=exact_solution,
            )
            _log(
                logger,
                "[Exact][Standard] "
                f"mean_pred_Y0={exact_std['summary']['mean_pred_y0']:.6f}, "
                f"mean_exact_Y0={exact_std['summary']['mean_exact_y0']:.6f}, "
                f"abs_err_Y0={exact_std['summary']['abs_error_mean_y0']:.6e}",
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

        _log(logger, f"[STANDARD] final eval: {logs_std['eval_stats']}")
        del model_std
        tf.keras.backend.clear_session()

    if args.mode in ("recursive", "both"):
        _log(logger, "\n==================== RECURSIVE ====================")
        rec_dir = os.path.join(run_root, "recursive")
        os.makedirs(rec_dir, exist_ok=True)

        explicit_eval_bundle = str(config.eval_bundle_path or "").strip()
        resume_eval_bundle = find_resume_eval_bundle_path(resume_models_dir_resolved)
        if explicit_eval_bundle != "":
            eval_bundle_path = os.path.abspath(os.path.expanduser(explicit_eval_bundle))
        elif resume_eval_bundle is not None:
            eval_bundle_path = os.path.abspath(os.path.expanduser(resume_eval_bundle))
        else:
            eval_bundle_path = os.path.abspath(os.path.join(rec_dir, "evaluation_bundle.npz"))

        pass_plot_summary_holder = {"summary": None}

        def _on_recursive_pass_end(progress: Dict[str, Any]) -> None:
            passes_so_far = sorted(progress.get("passes", []), key=lambda x: int(x["pass_id"]))
            if len(passes_so_far) == 0:
                return
            pass_id = int(progress.get("pass_id", passes_so_far[-1]["pass_id"]))
            is_last_requested_pass = pass_id >= int(config.passes)
            _log(
                logger,
                f"\n[RecursivePlot] completed {_pass_label(pass_id)}: "
                f"updating cumulative plots up to {_pass_label(pass_id)}",
            )
            pass_plot_summary_holder["summary"] = print_recursive_pass(
                pass_entries=passes_so_far,
                blocks=progress.get("blocks", []),
                rec_dir=rec_dir,
                params=params,
                N_per_block=config.N,
                D=config.D,
                layers=config.layers,
                T_total=config.T_total,
                exact_solution=exact_solution,
                selection_metric=str(config.selection_metric),
                exact_regression_tolerance=float(config.exact_regression_tolerance),
                exact_regression_action=str(config.exact_regression_action),
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(config.eval_seed),
                eval_min_paths=max(64, config.M),
                sample_paths=8,
                enforce_exact_regression_guardrail=is_last_requested_pass,
                print_compact_logs=is_last_requested_pass,
                logger=logger,
            )

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
            precision_margin=0.10,
            max_refine_rounds=3,
            rollout_M=max(2000, config.M),
            save_tf_checkpoints=True,
            training_plan_rules=training_plan_rules,
            pass1_warm_start_from_next=bool(config.pass1_warm_start_from_next),
            cross_pass_warm_start=bool(config.cross_pass_warm_start),
            n_passes=int(config.passes),
            resume_models_dir=resume_models_dir_resolved,
            resume_from_pass=int(config.resume_from_pass),
            empirical_jitter_scale=float(config.empirical_jitter_scale),
            on_pass_end=_on_recursive_pass_end,
            logger=logger,
        )

        pass_entries = sorted(rec.get("passes", []), key=lambda x: int(x["pass_id"]))
        if len(pass_entries) == 0:
            raise RuntimeError("No pass results available after recursive training")

        expected_pass_ids = sorted(int(p["pass_id"]) for p in pass_entries)
        plot_summary = pass_plot_summary_holder.get("summary", None)
        if plot_summary is None or plot_summary.get("processed_pass_ids", []) != expected_pass_ids:
            plot_summary = print_recursive_pass(
                pass_entries=pass_entries,
                blocks=rec["blocks"],
                rec_dir=rec_dir,
                params=params,
                N_per_block=config.N,
                D=config.D,
                layers=config.layers,
                T_total=config.T_total,
                exact_solution=exact_solution,
                selection_metric=str(config.selection_metric),
                exact_regression_tolerance=float(config.exact_regression_tolerance),
                exact_regression_action=str(config.exact_regression_action),
                eval_bundle_path=eval_bundle_path,
                eval_seed=int(config.eval_seed),
                eval_min_paths=max(64, config.M),
                sample_paths=8,
                enforce_exact_regression_guardrail=True,
                print_compact_logs=True,
                logger=logger,
            )

        exact_summary_by_pass = plot_summary["exact_summary_by_pass"]
        exact_summary_by_pass_index = plot_summary.get("exact_summary_by_pass_index", {})

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
            pass_id = int(p["pass_id"])
            passes_summary.append(
                {
                    "pass_id": pass_id,
                    "pass_index": _pass_index(pass_id),
                    "reference_loss": float(p["reference_loss"]),
                    "logs": p.get("logs", []),
                    "models_dir": p.get("models_dir", None),
                }
            )
        pass_summary_by_index = {int(p["pass_index"]): p for p in passes_summary}

        rec_summary = {
            "blocks": rec["blocks"],
            "passes": passes_summary,
            "resumed_from": rec.get("resumed_from", None),
            "boundary_stats": boundary_stats,
            "models_dir": os.path.join(rec_dir, "models"),
            "evaluation_bundle_path": plot_summary["eval_bundle_path"],
            "evaluation_bundle_M": int(plot_summary["evaluation_bundle_M"]),
            "selected_pass_id": int(plot_summary["selected_pass_id"]),
            "selected_pass_index": int(plot_summary["selected_pass_index"]),
            "selected_score_metric": plot_summary["selected_score_metric"],
            "selected_score": float(plot_summary["selected_score"]),
            "selected_scores_by_pass": plot_summary["selected_scores_by_pass"],
            "selected_scores_by_pass_index": plot_summary["selected_scores_by_pass_index"],
            "loss_score_metric": plot_summary["score_key"],
            "loss_pass_scores": {str(k): float(v) for k, v in plot_summary["pass_scores_loss"].items()},
            "loss_pass_scores_by_index": {
                str(k): float(v) for k, v in plot_summary["pass_scores_loss_by_index"].items()
            },
        }
        if exact_solution is None:
            rec_summary["exact_solution"] = {"enabled": False, "profile": "none"}
        else:
            rec_summary["exact_solution"] = {
                "enabled": True,
                "profile": exact_solution["name"],
                "by_pass": {str(k): v for k, v in exact_summary_by_pass.items()},
                "by_pass_index": exact_summary_by_pass_index,
                "selected_pass_summary": exact_summary_by_pass.get(
                    int(plot_summary["selected_pass_id"]),
                    None,
                ),
            }
        if 0 in pass_summary_by_index:
            rec_summary["pass0"] = {
                "reference_loss": pass_summary_by_index[0]["reference_loss"],
                "logs": pass_summary_by_index[0]["logs"],
            }
        if 1 in pass_summary_by_index:
            rec_summary["pass1"] = {
                "reference_loss": pass_summary_by_index[1]["reference_loss"],
                "logs": pass_summary_by_index[1]["logs"],
            }
        if 2 in pass_summary_by_index:
            rec_summary["pass2"] = {
                "reference_loss": pass_summary_by_index[2]["reference_loss"],
                "logs": pass_summary_by_index[2]["logs"],
            }
        save_json(rec_summary, os.path.join(rec_dir, "results.json"))


if __name__ == "__main__":
    main()
