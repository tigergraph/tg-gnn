#!/bin/bash

#bash ./Miniforge3-Linux-x86_64.sh -b -u -p ~/miniforge3
#source ~/miniforge3/bin/activate
#conda init
#exit and relogin
#conda create --name conda-forge-gnn python=3.12 --channel conda-forge --override-channels -y
#conda activate conda-forge-gnn

if conda info | grep "active environment" | grep "base" >/dev/null; then
  conda config --add channels conda-forge
  conda config --add channels rapidsai
  conda config --set channel_priority strict
else
  conda config --env --add channels conda-forge
  conda config --env --add channels rapidsai
  conda config --env --set channel_priority strict
fi

conda install -y python-abi3=3.12 cucim=26.02 cudf=26.02 cuml=26.02 cuproj=26.02 cuspatial=26.02 cuvs=26.02 cuxfilter=26.02 dask-cuda=26.02 dask-cudf=26.02 nx-cugraph=26.02 pylibraft=26.02 raft-dask=26.02 cugraph=26.02 cugraph-pyg=26.02 cudf=26.02 pylibwholegraph=26.02
