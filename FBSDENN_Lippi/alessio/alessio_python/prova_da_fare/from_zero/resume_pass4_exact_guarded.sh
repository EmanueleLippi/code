#!/bin/bash -l

# Resume from an existing 3-pass run and add only pass 4 with exact-aware checks.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=FBSNN_recursive
#SBATCH --mail-type=NONE
#SBATCH --mem=100gb
#SBATCH --time=00:00:00
#SBATCH --output=FBSNN_recursive_%j.log

echo
echo "CUDA"
module load cuda
nvcc -V

echo "anaconda tensorflow"
module load anaconda
conda init bash
conda activate tf-gpu

ROOT="${ROOT:-[TUO_PATH]/code/FBSDENN_Lippi}"
RESUME_RUN="${RESUME_RUN:-$ROOT/alessio/recursive1_outputs/prova_da_fare/run_20260304_155408}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/alessio/recursive1_outputs/heavy_from_zero_exact_guarded}"

cd /export/home/alessio.rondelli2/python/recursive
python "$ROOT/recursive/recursive1.py" \
  --mode recursive \
  --M 4000 \
  --N 80 \
  --D 4 \
  --T_total 48 \
  --block_size 12 \
  --passes 4 \
  --resume_models_dir "$RESUME_RUN/recursive/models" \
  --resume_from_pass 3 \
  --empirical_jitter_scale 0.0 \
  --training_plan_csv "$ROOT/recursive/prova_da_fare/from_zero/training_plan_heavy_from_zero_exact_guarded.csv" \
  --exact_solution quadratic_coupled \
  --selection_metric exact_abs_y0 \
  --exact_regression_tolerance 0.05 \
  --exact_regression_action warn \
  --eval_bundle_path "$RESUME_RUN/recursive/evaluation_bundle.npz" \
  --output_dir "$OUTPUT_DIR"
