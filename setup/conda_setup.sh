#!/bin/bash

#bash ./Miniforge3-Linux-x86_64.sh -b -u -p ~/miniforge3
#source ~/miniforge3/bin/activate
#conda init
#exit and relogin
#conda create --name tg-gnn python=3.12 --channel conda-forge --override-channels -y
#conda activate tg-gnn

if conda info | grep "active environment" | grep "base" >/dev/null; then
  conda config --add channels conda-forge
  conda config --add channels nvidia
  conda config --add channels rapidsai
  conda config --set channel_priority strict
else
  conda config --env --add channels conda-forge
  conda config --env --add channels nvidia
  conda config --env --add channels rapidsai
  conda config --env --set channel_priority strict
fi

# Install RAPIDS + CUDA 12.9 environment following the official rapidsai/cugraph-gnn
# dependencies.yaml (conda section for CUDA 12.9):
#   - cuda-version=12.9: meta-package that pins all CUDA 12.9 library versions
#   - cupy>=13.6.0: cupy version 13.x compiled for CUDA 12.9 (version != CUDA version)
#   - pytorch-gpu>=2.3: conda pytorch built against same CUDA as RAPIDS (avoids
#     the CUDA 12.6 vs 12.9 runtime mismatch that occurs with pip's cu126 wheels)
#   - pytorch_geometric>=2.7,<2.8: conda package for torch_geometric (>=2.7 fixes
#     the DiagnosticOptions import error with PyTorch 2.6+)
conda install -y python-abi3=3.12 cuda-version=12.9 \
  cugraph=26.02 cugraph-pyg=26.02 cudf=26.02 dask-cuda=26.02 pylibwholegraph=26.02 raft-dask=26.02 \
  libcugraph=26.02 libcudf=26.02 libraft=26.02 librmm=26.02 "libucxx>=0.48,<0.49" \
  "cupy>=13.6.0" "pytorch-gpu>=2.3" "pytorch_geometric>=2.7,<2.8" \
  torchvision torchaudio \
  "numpy>=1.23,<2.3" nccl "libstdcxx-ng>=12"
conda install -y pytest scikit-learn

pip install "cuda-python>=13.0.1,<14.0" tensordict pytigergraph

# Ensure conda's lib path is first so CUDA 12.9 libraries take precedence
# over any system CUDA libraries in /usr/local/cuda/lib64.
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/libstdcxx_path.sh <<'EOF'
case ":$LD_LIBRARY_PATH:" in
  *":$CONDA_PREFIX/lib:"*) ;;
  *) export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ;;
esac
EOF
