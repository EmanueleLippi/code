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
#python recursive1.py --mode recursive --T_total 36 --block_size 12 --M 12000 --N 100 --training_plan_csv training_plan_1.csv --pass1_warm_start_from_next
python final_model3.py --mode standard --T_standard 24 --M 9000 --N 150 --output_dir final_outputs/T_24_initial_data --date 2025settembre1
python final_model3.py --mode standard --T_standard 24 --M 9000 --N 150 --output_dir final_outputs/T_24_initial_data --date 2025dicembre2
