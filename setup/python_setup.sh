#!/bin/bash

cd ~
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
# Pin cuda-python before RAPIDS install; cupy is pulled in transitively by cudf-cu12.
pip install "cuda-python>=13.0.1,<14.0"

# torch_geometric>=2.5 for PyTorch 2.5+ compatibility
pip install "torch_geometric>=2.5" pytigergraph scikit-learn tqdm

# RAPIDS 26.2 stack from NVIDIA's PyPI index
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.2.*" "dask-cudf-cu12==26.2.*" "cuml-cu12==26.2.*" \
    "cugraph-cu12==26.2.*" "nx-cugraph-cu12==26.2.*" "cuxfilter-cu12==26.2.*" \
    "cucim-cu12==26.2.*" "pylibraft-cu12==26.2.*" "raft-dask-cu12==26.2.*" \
    "cuvs-cu12==26.2.*" "pylibwholegraph-cu12==26.2.*" \
    "cugraph-pyg==26.2.*"

#git clone https://github.com/tigergraph/tg-gnn.git
cd tg-gnn
pip install --extra-index-url=https://pypi.nvidia.com .
