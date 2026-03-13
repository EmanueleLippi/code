#!/bin/bash -l

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

#conda info --envs
conda init bash
#conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu

#(echo "import tensorflow as tf" ; echo "print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))") | python

cd /export/home/alessio.rondelli2/python/recursive
python recursive/recursive1.py \
  --mode recursive \
  --exact_solution quadratic_coupled \
  --selection_metric loss \
  --T_total 12 \
  --block_size 1.0 \
  --M 2000 \
  --N 40 \
  --passes 3 \
  --training_plan_csv recursive/training_recursive.csv \
  --pass1_init base \
  --output_dir recursive/recursive1_outputs/t12_b10_base_xpass_ws \
  --eval_bundle_path recursive/recursive1_outputs/eval_bundle_T12_B10_M2000_N40.npz

echo "base done"
echo "coarse"

python recursive/recursive1.py \
  --mode recursive \
  --exact_solution quadratic_coupled \
  --selection_metric loss \
  --T_total 12 \
  --block_size 1.0 \
  --M 2000 \
  --N 40 \
  --passes 3 \
  --training_plan_csv recursive/training_recursive.csv \
  --pass1_init coarse \
  --coarse_prepass_iter_scale 0.15 \
  --output_dir recursive/recursive1_outputs/t12_b10_coarse_xpass_ws \
  --eval_bundle_path recursive/recursive1_outputs/eval_bundle_T12_B10_M2000_N40.npz

echo "coarse done"
echo "warm start isolation"

python recursive/recursive1.py \
  --mode recursive \
  --exact_solution quadratic_coupled \
  --selection_metric loss \
  --T_total 12 \
  --block_size 1.0 \
  --M 2000 \
  --N 40 \
  --passes 3 \
  --training_plan_csv recursive/training_recursive.csv \
  --pass1_init base \
  --disable_cross_pass_warm_start \
  --output_dir recursive/recursive1_outputs/t12_b10_base_no_xpass_ws \
  --eval_bundle_path recursive/recursive1_outputs/eval_bundle_T12_B10_M2000_N40.npz