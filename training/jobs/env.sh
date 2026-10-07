# Shared job setup: sourced by every jobs/*.sbatch. Puts uv, Python and the
# venv in RAM-backed /dev/shm (home is 15 GB) and cleans up on exit.
set -euo pipefail
REPO="$SLURM_SUBMIT_DIR"
[ -d "$REPO" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"   # /nethome symlink loop on compute nodes
cd "$REPO"
W="/dev/shm/$USER-$SLURM_JOB_ID"
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT TERM INT
export HOME="$W" UV_CACHE_DIR="$W/uv-cache" UV_PYTHON_INSTALL_DIR="$W/py" UV_PROJECT_ENVIRONMENT="$W/venv"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W/bin" INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$W/bin:$PATH"
uv sync
