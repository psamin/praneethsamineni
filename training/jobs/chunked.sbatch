#!/bin/bash
# Diffusion Policy and ACT families: BC + 3 DAgger rounds each. Trains on the GPU.
#SBATCH -J push-chunked
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o %x-%j.out
source jobs/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
uv run python train_chunked.py diffusion 64,128 3
uv run python train_chunked.py act 32,64 3
echo ALL_DONE
