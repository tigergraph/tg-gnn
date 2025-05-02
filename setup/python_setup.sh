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
    "cudf-cu12>=25.6.0a0,<=25.6" "dask-cudf-cu12>=25.6.0a0,<=25.6" \
    "cuml-cu12>=25.6.0a0,<=25.6" "cugraph-cu12>=25.6.0a0,<=25.6" \
    "nx-cugraph-cu12>=25.6.0a0,<=25.6" "cuspatial-cu12>=25.6.0a0,<=25.6" \
    "cuproj-cu12>=25.6.0a0,<=25.6" "cuxfilter-cu12>=25.6.0a0,<=25.6" \
    "cucim-cu12>=25.6.0a0,<=25.6" "pylibraft-cu12>=25.6.0a0,<=25.6" \
    "raft-dask-cu12>=25.6.0a0,<=25.6" "cuvs-cu12>=25.6.0a0,<=25.6" \
    "pylibraft-cu12>=25.6.0a0,<=25.6" "nx-cugraph-cu12>=25.6.0a0,<=25.6" \
    "dask-cuda>=25.6.0a0,<=25.6"

git clone https://github.com/rapidsai/cugraph-gnn.git
cd cugraph-gnn/python/cugraph-pyg/
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
cd -

#git clone https://github.com/tigergraph/tg-gnn.git
cd tg-gnn
pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
