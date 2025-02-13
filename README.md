# TG-GNN

This project is intended for using [TigerGraph](https://www.tigergraph.com/) as a graph data storage solution in combination with [cuGraph](https://github.com/rapidsai/cugraph) and [cuGraph-PyG](https://github.com/rapidsai/cugraph-pyg). By leveraging these tools, you can train a GNN model on multi-GPU setups at any scale.

---

## Getting Started
### Cuda setup
- Required Cuda version >= 11.8; 
- Note: support for 11.8 will be drop very soon. so, use cuda >= 12.0

### Environment Setup

1. **Create a new virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```
3. **Install tg-gnn project**
    ```bash
    git clone https://github.com/tigergraph/tg-gnn.git
    cd tg-gnn
    pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
    ```
4. **Fix some bugs which is not available yet in nightly build**
    ```bash
    git clone https://github.com/alexbarghi-nv/cugraph-gnn.git
    cd cugraph-gnn
    git checkout taobao-add-timing
    cd python/cugraph-pyg
    pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple .
---

## Changes Required to Run a GCN Model on Any TigerGraph Graph

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
            "src": "<src_name>", 
            "dst": "<dst_name>",
            "features_list" (optional): {
                "feature1": "<feature_datatype (should be LIST or FLOAT)>",
                "feature2": "LIST",
            }
        }
    },
    "data_dir": "<path to export the data from tigergraph; should be accessible by tg as well as torchrun processes>",
}
```

- **`vertex_name`**: The name of the node/vertex in TigerGraph (e.g., `paper`).  
- **`features_list`**: Specifies which node attributes are used as input features. feature_datatype could be either "INT", "FLOAT" or "LIST" (list should only contain FLOAT/INT values for GNN training) 
- **`label`**: Points to the attribute name that holds the label for training/validation/testing.  
- **`split`**: Identifies the attribute that indicates whether a node belongs to the training, validation, or test set.  
- **`num_nodes`**: Indicates the number of vertices/nodes for given vertex/node type.  
- **`data_dir`**: The temporary directory where TigerGraph will export the data and data would be assesed by GNN model training process.

---

## Running the Example

Use the following command to run the GCN example. Make sure to replace the placeholders (`<tg-graph-name>`, `<tg-host-ip>`, `<tg-username>`, `<tg-password>`) with the correct values for your TigerGraph environment.

```bash
torchrun \
    --nnodes 1 \
    --nproc-per-node 4 \
    --rdzv-id 4RANDOM \
    --rdzv-backend c10d \
    --rdzv-endpoint localhost:29500 \
    tg_gcn_mnmg.py \
    -g <tg-graph-name> \
    --host <tg-host-ip> \
    -u <tg-username> \
    -p <tg-password>
    --port <tg-port>
```

- `--nnodes` **and** `--nproc-per-node`: Control the number of nodes and GPUs per node for multi-GPU training.  
- `--rdzv-id`, `--rdzv-backend`, `--rdzv-endpoint`: Arguments for PyTorch’s distributed runtime.  
- `-g` (`--graph`): Name of the TigerGraph graph to be used.  
- `--host`, `--port`, `-u`, `-p`: Host IP, port, username, and password for connecting to your TigerGraph instance.

---


**Happy Graphing and GNN Training!**


