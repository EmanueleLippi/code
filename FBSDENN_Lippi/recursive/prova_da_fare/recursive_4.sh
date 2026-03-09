#!/bin/bash -l

set -euo pipefail

#

#SBATCH --partition=gpu                      # cpu / gpu (gpu in case of cuda p>

#SBATCH --nodes=1                            # do not to change

#SBATCH --cpus-per-task=24                   # Run on 24 CPUS (https://slurm.sc>

#SBATCH --job-name=FBSNN_recursive                    # Job name (do not include spaces)

#SBATCH --mail-type=NONE                     # Mail events (NONE, BEGIN, END, F>

#SBATCH --mem=100gb                           # Job memory request

#SBATCH --time=00:00:00                      # Time limit hrs:min:sec

#SBATCH --output=FBSNN_recursive_%j.log               # Standard output and error log

echo

echo "CUDA"

# call if need cuda

module load cuda

nvcc -V

 

# anaconda pytorch

echo "anaconda tensorflow"

module load anaconda

# Initialize Conda explicitly because Slurm batch shells are non-interactive.
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3.22/etc/profile.d/conda.sh}"
CONDA_BIN="${CONDA_BIN:-/usr/local/anaconda3.22/bin/conda}"
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -x "$CONDA_BIN" ]]; then
  eval "$("$CONDA_BIN" shell.bash hook)"
else
  echo "Unable to initialize conda after 'module load anaconda'." >&2
  exit 1
fi
conda activate tf-gpu

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

T_TOTAL="${T_TOTAL:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-0.5}"
M="${M:-2000}"
N="${N:-20}"
PASSES="${PASSES:-3}"
SELECTION_METRIC="${SELECTION_METRIC:-loss}"
TRAINING_PLAN="${TRAINING_PLAN:-$ROOT/recursive/training_plan_example.csv}"
LEGACY_OUT="${LEGACY_OUT:-$ROOT/recursive/recursive1_outputs/qc_recursive1}"
COMMON_EVAL_BUNDLE="${COMMON_EVAL_BUNDLE:-$ROOT/recursive/recursive1_outputs/qc_compare_eval_bundle_T4_B05_M2000_N20.npz}"

echo "ROOT=$ROOT"
echo "python=$(command -v python)"
echo "TF_GPU_ALLOCATOR=$TF_GPU_ALLOCATOR"
echo "training_plan=$TRAINING_PLAN"
echo "GPU snapshot"
nvidia-smi || true

echo "Running recursive1 only"

python "$ROOT/recursive/recursive1.py" \
  --mode recursive \
  --exact_solution quadratic_coupled \
  --selection_metric "$SELECTION_METRIC" \
  --T_total "$T_TOTAL" \
  --block_size "$BLOCK_SIZE" \
  --M "$M" \
  --N "$N" \
  --passes "$PASSES" \
  --training_plan_csv "$TRAINING_PLAN" \
  --output_dir "$LEGACY_OUT" \
  --disable_cross_pass_warm_start \
  --eval_bundle_path "$COMMON_EVAL_BUNDLE"

echo "Recursive1 finished"
