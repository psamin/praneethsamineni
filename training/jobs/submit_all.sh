#!/bin/bash
# Run from training/ on a Futurama login node:  bash jobs/submit_all.sh
# Starts a fresh run: clears old data and models, trains both families in
# parallel, then evaluates once both have succeeded.
set -euo pipefail
rm -rf data models
mlp=$(sbatch --parsable jobs/mlp.sbatch)
chunked=$(sbatch --parsable jobs/chunked.sbatch)
ev=$(sbatch --parsable --dependency=afterok:$mlp:$chunked jobs/evaluate.sbatch)
echo "mlp=$mlp chunked=$chunked evaluate=$ev"
