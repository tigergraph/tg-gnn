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

conda install -y python-abi3=3.12 cucim=25.04 cudf=25.04 cuml=25.04 cuproj=25.04 cuspatial=25.04 cuvs=25.04 cuxfilter=25.04 dask-cuda=25.04 dask-cudf=25.04 nx-cugraph=25.04 pylibraft=25.04 raft-dask=25.04 cugraph=25.04 cugraph-pyg=25.04 cudf=25.04 pylibwholegraph=25.04
