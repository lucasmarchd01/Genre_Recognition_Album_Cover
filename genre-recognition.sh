#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

SOURCEDIR=~/Genre_Recognition_Album_Cover

# Prepare virtualenv
source ~/tensorflow/bin/activate

# Prepare data
mkdir $SLURM_TMPDIR/data
tar xf ~/2024-10-08.tar -C $SLURM_TMPDIR/data

# Start training
tensorboard --logdir=/tmp/logs --host 0.0.0.0 --load_fast false &
python $SOURCEDIR/train.py $SLURM_TMPDIR/data/final_top_5_discogs.csv $SLURM_TMPDIR/data/
