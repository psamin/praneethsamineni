#!/bin/bash
# Ship one model:  MODEL=abs_32 sbatch jobs/finalize.sbatch
# Evaluates it, exports it to the web, writes the parity fixture and the list
# of verified starting scenes. CPU only.
#SBATCH -J push-finalize
#SBATCH -p general
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0:45:00
#SBATCH -o %x-%j.out
source jobs/env.sh
uv run python evaluate.py 2000 "$MODEL"
uv run python export.py "$MODEL"
uv run python parity.py
uv run python verified_scenes.py 400
echo ALL_DONE
