# TG-GNN

This project is intended for using [TigerGraph](https://www.tigergraph.com/) as a graph data storage solution in combination with [cuGraph](https://github.com/rapidsai/cugraph) and [cuGraph-PyG](https://github.com/rapidsai/cugraph-pyg). By leveraging these tools, you can train a GNN model on multi-GPU setups at any scale.

---

## Getting Started
### Cuda setup
- Required Cuda version >= 12.8;

### Environment Setup

1. **Download and install Conda, for example Miniforge**
   ```bash
   bash ./Miniforge3-Linux-x86_64.sh -b -u -p ~/miniforge3
   source ~/miniforge3/bin/activate
   conda init
   ```
2. **Create a new conda environment (optional, recommended):**
   ```bash
   conda create --name conda-forge-gnn python=3.12 --channel conda-forge --override-channels -y
   conda activate conda-forge-gnn
   ```
3. **Install tg-gnn project**
    ```bash
    git clone https://github.com/tigergraph/tg-gnn.git
    cd tg-gnn
    bash setup/conda_setup.sh
    ```
4. **Install CUDA Toolkit (Optional)**
   ```bash
   bash setup/cuda_setup.sh
   ```
---

## Changes Required to Run example GCN Model code on Any TigerGraph Graph

To integrate your TigerGraph graph with the GCN model, a few schema updates and metadata configurations are required.

### 1. Load data on Tigergraph
- To load please follow instructions available in benchmark/Benchmark.md 

### 2. Configure Metadata 

Below is a sample `metadata` dictionary that illustrates how to define the mapping between your TigerGraph entities and the model’s requirements:

```python
metadata = {
    "nodes": {
        "<vertex_name 1>": {
            "vertex_name>" : "<vertex_name 1>" 
            "features_list" (optional): {
                "feature1": "<feature_datatype (should be LIST or FLOAT)>",
                "feature2": "LIST",
                "feature3": "FLOAT"
            },
            "label" (optional): "<label_attr_name>",
            "split" (optional): "<split_attr_name>",
            "num_nodes": <number of nodes>
        },

        "<vertex_name 2>": {
            "vertex_name>" : "<vertex_name 2>" 
            "features_list" (optional): {
                "feature1": "<feature_datatype (should be LIST or FLOAT)>",
                "feature2": "LIST",
            },
            "num_nodes": <number of nodes>
        },
        
        "<vertex_name 3>": {
            "vertex_name>" : "<vertex_name 3>",
            "num_nodes": <number of nodes>
        }

    },
    "edges": {
        "<rel_name>": {
            "rel_name": "<rel_name">,
            "undirected": True,
            "src": "<src_name>", 
            "dst": "<dst_name>",
            "features_list" (optional): {
                "feature1": "<feature_datatype (should be LIST or FLOAT)>",
                "feature2": "LIST",
            }
        }
    },
    "data_dir": "<path to export the data from tigergraph; should be accessible by tg as well as torchrun processes>",
    "fs_type": "<data_dir type; it would be either shared or local>",
    "num_tg_nodes" (optional): "<number of tg nodes>"
}
```

- **`vertex_name`**: The name of the node/vertex in TigerGraph (e.g., `paper`).
- **`features_list`**: Specifies which node attributes are used as input features. feature_datatype could be either "INT", "FLOAT" or "LIST" (list should only contain FLOAT/INT values for GNN training) 
- **`label`**: Points to the attribute name that holds the label for training/validation/testing.  
- **`split`**: Identifies the attribute that indicates whether a node belongs to the training, validation, or test set.  
- **`num_nodes`**: Indicates the number of vertices/nodes for given vertex/node type.  
- **`rel_name`**: The name of the edge in TigerGraph (e.g., `uses`).
- **`undirected`**: The direction of the edge in TigerGraph. Reverse edge (e.g., `rev_uses`) will be generated when it's True.
- **`src`**: The `from` vertex type of the edge in TigerGraph.
- **`dst`**: The `to` vertex type of the edge in TigerGraph.
- **`data_dir`**: The temporary directory where TigerGraph will export the data and data would be assesed by GNN model training process.

---

## Running the Example

Use the following command to run the GCN example. Make sure to replace the placeholders (`<tg-graph-name>`, `<tg-host-ip>`, `<tg-username>`, `<tg-password>`) with the correct values for your TigerGraph environment.

```bash
torchrun --nnodes 1 --nproc-per-node 4 --rdzv-id 4RANDOM --rdzv-backend c10d --rdzv-endpoint localhost:29500 examples/tg_ogbn_products_mnmg.py \
    -g <tg-graph-name> --host <tg-host-ip> --restppPort <tg-port> -u <tg-username> -p <tg-password> --tg_nodes 1 --data_dir /tg_export
```

- `--nnodes` **and** `--nproc-per-node`: Control the number of nodes and GPUs per node for multi-GPU training.  
- `--rdzv-id`, `--rdzv-backend`, `--rdzv-endpoint`: Arguments for PyTorch’s distributed runtime.  
- `-g` (`--graph`): Name of the TigerGraph graph to be used.  
- `--host`, `--restppPort`, `-u`, `-p`: Host IP, port for rest++ queries, username, and password for connecting to your TigerGraph instance for data export.
- `--tg_nodes`: The number of nodes in TigerGraph cluster
- `--data_dir`: The location of the data files stored
- `--skip_tg_export`, `-s`: Control data exporting from TigerGraph, omit the argument to perform data exporting step when provided


## Expected Result from running above example

```
Beginning training...
Epoch: 0, Iteration: 0, Loss: tensor(3.8706, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 0, Iteration: 10, Loss: tensor(0.8285, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 0, Iteration: 20, Loss: tensor(0.7156, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 0, Iteration: 30, Loss: tensor(0.6421, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 0, Iteration: 40, Loss: tensor(0.6386, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.013461819716862269 s/iter
Validation Accuracy: 86.8490%
Epoch: 1, Iteration: 0, Loss: tensor(0.5380, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 1, Iteration: 10, Loss: tensor(0.5128, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 1, Iteration: 20, Loss: tensor(0.4820, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 1, Iteration: 30, Loss: tensor(0.5696, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 1, Iteration: 40, Loss: tensor(0.5411, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.01327134030205863 s/iter
Validation Accuracy: 87.7387%
Epoch: 2, Iteration: 0, Loss: tensor(0.5010, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 2, Iteration: 10, Loss: tensor(0.4989, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 2, Iteration: 20, Loss: tensor(0.4815, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 2, Iteration: 30, Loss: tensor(0.4908, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 2, Iteration: 40, Loss: tensor(0.4771, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.013151185853140695 s/iter
Validation Accuracy: 87.6302%
Epoch: 3, Iteration: 0, Loss: tensor(0.4845, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 3, Iteration: 10, Loss: tensor(0.4576, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 3, Iteration: 20, Loss: tensor(0.4722, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 3, Iteration: 30, Loss: tensor(0.5193, device='cuda:0', grad_fn=<NllLossBackward0>)
Epoch: 3, Iteration: 40, Loss: tensor(0.4854, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.013145617076328822 s/iter
Validation Accuracy: 88.0100%
Test Accuracy: 73.2040%
Total Program Runtime (total_time) = 118.15 seconds
total_time - prep_time = 28.16000000000001 seconds
```

---


**Happy Graphing and GNN Training!**


