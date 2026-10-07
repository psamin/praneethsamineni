#!/bin/bash
# Closed-loop evaluation of every model on 2,000 unseen scenes. CPU only.
#SBATCH -J push-eval
#SBATCH -p general
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o %x-%j.out
source jobs/env.sh
uv run python evaluate.py 2000
echo ALL_DONE
