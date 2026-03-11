#!/bin/bash
# Sets up a pip-based Python 3.12 venv with RAPIDS + PyTorch (pip alternative to conda_setup.sh).
#
# Usage:
#   bash setup/python_setup.sh
#
# After setup, install tg-gnn itself:
#   cd ~/tg-gnn && pip install .

set -e

sudo apt update -y
sudo apt install -y python3.12-venv
python3 -m venv venv
source venv/bin/activate

# System CUDA 12.9 libraries take precedence over pip-bundled ones
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# torch must be installed from the PyTorch CUDA index — plain PyPI only has CPU wheels.
pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu126
# Remove pip's bundled CUDA 12.6 runtime so the system CUDA 12.9 in LD_LIBRARY_PATH
# is used by both torch and RAPIDS (rmm/cugraph), avoiding the version mismatch that
# causes cudaErrorInsufficientDriver in rmm.reinitialize().
pip uninstall nvidia-cuda-runtime-cu12 -y || true
# Only the RAPIDS packages actually used by tg-gnn.
# cudf pulls in cupy, rmm, pylibcudf transitively.
# cugraph pulls in pylibcugraph transitively.
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.2.*" \
    "cugraph-cu12==26.2.*" \
    "pylibwholegraph-cu12==26.2.*" \
    "cugraph-pyg==26.2.*"

echo ""
echo "Setup complete. Run: cd ~/tg-gnn && pip install ."
