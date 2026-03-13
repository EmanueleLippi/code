#!/bin/bash -l

# Exact-aware variant for the 48h from-zero recursive run.
# It keeps the original schedule for passes 1-3 and uses a softer pass 4.

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

cd /export/home/alessio.rondelli2/python/recursive
python "$ROOT/recursive/recursive1.py" \
  --mode recursive \
  --M 4000 \
  --N 80 \
  --D 4 \
  --T_total 48 \
  --block_size 12 \
  --passes 4 \
  --empirical_jitter_scale 0.0 \
  --training_plan_csv "$ROOT/recursive/prova_da_fare/from_zero/training_plan_heavy_from_zero_exact_guarded.csv" \
  --exact_solution quadratic_coupled \
  --selection_metric exact_mae_y \
  --exact_regression_tolerance 0.05 \
  --exact_regression_action warn \
  --output_dir "$ROOT/alessio/recursive1_outputs/heavy_from_zero_exact_guarded"
