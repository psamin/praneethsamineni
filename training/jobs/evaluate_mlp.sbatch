#!/bin/bash
# Evaluate only the MLP family (so it can ship before the chunked job ends),
# then cross-check the 64-wide MLP in Pymunk. CPU only.
#SBATCH -J push-eval-mlp
#SBATCH -p general
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o %x-%j.out
source jobs/env.sh
uv run python evaluate.py 2000 "abs_*"
uv run python evaluate.py pymunk abs_64 1000
uv run python evaluate.py pymunk abs_32 1000
echo ALL_DONE
