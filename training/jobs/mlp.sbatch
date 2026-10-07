#!/bin/bash
# MLP family: physics tests, DART data, BC + 6 DAgger rounds. CPU only.
#SBATCH -J push-mlp
#SBATCH -p general
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o %x-%j.out
source jobs/env.sh
uv run python test_physics.py
uv run python generate_data.py
uv run python train.py abs 32,64 6
echo ALL_DONE
