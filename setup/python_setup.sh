#!/bin/bash

cd ~
sudo apt update -y
sudo apt install -y python3.12-venv
python3 -m venv venv
source venv/bin/activate

# System CUDA 12.9 libraries take precedence over pip-bundled ones
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Install torch (cu126 wheel, the CUDA 12 pip wheel from official rapidsai dependencies.yaml)
pip install torch>=2.3 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Remove pip's bundled CUDA 12.6 runtime so the system CUDA 12.9 in LD_LIBRARY_PATH
# is used by both torch and RAPIDS (rmm/cugraph), avoiding the version mismatch that
# causes cudaErrorInsufficientDriver in rmm.reinitialize().
pip uninstall nvidia-cuda-runtime-cu12 -y || true

# cupy-cuda12x>=13.6.0 matches the official RAPIDS pip spec for CUDA 12.x
# (conda uses cupy>=13.6.0; pip uses cupy-cuda12x>=13.6.0 per cugraph-gnn dependencies.yaml)
pip install "cupy-cuda12x>=13.6.0" "cuda-python>=13.0.1,<14.0" "numpy>=1.23,<2.3"

# torch_geometric>=2.7 fixes the DiagnosticOptions ImportError with PyTorch 2.6+
pip install "torch_geometric>=2.7,<2.8" tensordict pytigergraph pytest scikit-learn tqdm

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cugraph-cu12>=26.02.0a0,<=26.02" \
    "cugraph-pyg-cu12>=26.02.0a0,<=26.02" \
    "cudf-cu12>=26.02.0a0,<=26.02" \
    "dask-cuda>=26.02.0a0,<=26.02" \
    "pylibwholegraph-cu12>=26.02.0a0,<=26.02" \
    "raft-dask-cu12>=26.02.0a0,<=26.02"

#git clone https://github.com/tigergraph/tg-gnn.git
cd tg-gnn
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
