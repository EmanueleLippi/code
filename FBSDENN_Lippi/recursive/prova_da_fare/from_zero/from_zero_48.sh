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
python "[TUO_PATH]/code/FBSDENN_Lippi/recursive/recursive1.py" \
  --mode recursive \
  --M 2048 \
  --N 120 \
  --D 4 \
  --T_total 48 \
  --block_size 12 \
  --passes 6 \
  --empirical_jitter_scale 0.015 \
  --pass1_warm_start_from_next \
  --training_plan_csv recursive/prova_da_fare/training_plan_heavy_from_zero.csv \
  --output_dir alessio/recursive1_outputs/heavy_from_zero