#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=2-00:00     # DD-HH:MM:SS
#SBATCH --account=def-ichiro

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

SOURCEDIR=~/Genre_Recognition_Album_Cover

# Prepare virtualenv
source ~/tensorflow/bin/activate

# Prepare data
tar xf ~/2024-10-10.tar -C $SLURM_TMPDIR

# Start training
tensorboard --logdir=/tmp/logs --host 0.0.0.0 --load_fast false &
python $SOURCEDIR/classifier/train.py --csv_file $SLURM_TMPDIR/data/csv_discogs/final_top_5_discogs.csv --directory $SLURM_TMPDIR
