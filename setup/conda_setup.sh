#!/bin/bash
# Creates (or updates) the rapids-26.02 conda environment with RAPIDS + PyTorch.
#
# Usage:
#   First time:  bash setup/conda_setup.sh
#   Update:      bash setup/conda_setup.sh --update
#
# After setup, install tg-gnn itself:
#   pip install .

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="tg-gnn"
CONDA_PKGS=(
    # cudf pulls in cupy, rmm, pylibcudf transitively.
    # cugraph pulls in pylibcugraph transitively.
    "cudf=26.02"
    "cugraph=26.02"
    "pylibwholegraph=26.02"
    "cugraph-pyg=26.02"
    "python=3.12"
    "cuda-version>=12.2,<=12.9"
    "pytorch=*=*cuda*"
)

if [ "$1" == "--update" ]; then
  conda install -n "$ENV_NAME" -c rapidsai -c conda-forge -y "${CONDA_PKGS[@]}"
else
  conda create -n "$ENV_NAME" -c rapidsai -c conda-forge -y "${CONDA_PKGS[@]}"
fi

# Activate to apply changes (caller must source this script or re-activate manually)
conda activate "$ENV_NAME"

echo ""
echo "Setup complete. Run: cd ~/tg-gnn && pip install ."
