#!/bin/bash

cd ~
sudo apt update -y
sudo apt install -y python3.12-venv
python3 -m venv venv
source venv/bin/activate

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

pip install torch torchvision torchaudio torch_geometric pytigergraph pytest tensordict pylibwholegraph-cu12

pip install \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cudf-cu12>=26.02.0a0,<=26.02" "dask-cudf-cu12>=26.02.0a0,<=26.02" \
    "cuml-cu12>=26.02.0a0,<=26.02" "cugraph-cu12>=26.02.0a0,<=26.02" \
    "nx-cugraph-cu12>=26.02.0a0,<=26.02" "cuspatial-cu12>=26.02.0a0,<=26.02" \
    "cuproj-cu12>=26.02.0a0,<=26.02" "cuxfilter-cu12>=26.02.0a0,<=26.02" \
    "cucim-cu12>=26.02.0a0,<=26.02" "pylibraft-cu12>=26.02.0a0,<=26.02" \
    "raft-dask-cu12>=26.02.0a0,<=26.02" "cuvs-cu12>=26.02.0a0,<=26.02" \
    "pylibraft-cu12>=26.02.0a0,<=26.02" "nx-cugraph-cu12>=26.02.0a0,<=26.02" \
    "dask-cuda>=26.02.0a0,<=26.02" "cugraph-cu12>=26.02.0a0,<=26.02" \
    "pylibwholegraph-cu12>=26.02.0a0,<=26.02" "cugraph-pyg-cu12>=26.02.0a0,<=26.02" \
    "cudf-cu12>=26.02.0a0,<=26.02"

git clone https://github.com/rapidsai/cugraph-gnn.git
cd cugraph-gnn/python/cugraph-pyg/
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
cd -

#git clone https://github.com/tigergraph/tg-gnn.git
cd tg-gnn
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
