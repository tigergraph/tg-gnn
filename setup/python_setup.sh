#!/bin/bash

cd ~
sudo apt update -y
sudo apt install -y python3.12-venv
python3 -m venv venv
source venv/bin/activate

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

pip install torch torchvision torchaudio torch_geometric pytigergraph pytest tensordict scikit-learn

pip install \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cugraph-cu12>=26.02.0a0,<=26.02" \
    "cugraph-pyg-cu12>=26.02.0a0,<=26.02" \
    "cudf-cu12>=26.02.0a0,<=26.02" \
    "dask-cuda>=26.02.0a0,<=26.02" \
    "pylibwholegraph-cu12>=26.02.0a0,<=26.02" \
    "raft-dask-cu12>=26.02.0a0,<=26.02"

git clone https://github.com/rapidsai/cugraph-gnn.git
cd cugraph-gnn/python/cugraph-pyg/
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
cd -

#git clone https://github.com/tigergraph/tg-gnn.git
cd tg-gnn
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
