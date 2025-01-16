# TG-GNN

This project is intended for using [TigerGraph](https://www.tigergraph.com/) as a graph data storage solution in combination with [cuGraph](https://github.com/rapidsai/cugraph) and [cuGraph-PyG](https://github.com/rapidsai/cugraph-pyg). By leveraging these tools, you can train a GNN model on multi-GPU setups at any scale.

---

## Getting Started

### 1. Environment Setup

1. **Create a new virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```
2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
3. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Changes Required to Run a GCN Model on Any TigerGraph Graph

To integrate your TigerGraph graph with the GCN model, a few schema updates and metadata configurations are required.

### 1. Update the Schema

- Ensure that your TigerGraph schema is compatible with your model’s needs.  
- The vertex schema (e.g., “paper” as shown below) should include attributes for node features, labels, and data splits (train/val/test).

### 2. Configure Metadata

Below is a sample `metadata` dictionary that illustrates how to define the mapping between your TigerGraph entities and the model’s requirements:

```python
metadata = {
    "nodes": [
        {
            "vertex_name": "paper",
            "features_list": {
                "feature": "LIST",
            },
            "label": "label",
            "split": "split"
        }
    ],
    "edges": [
        {
            "rel_name": "rel",
            "src": "paper",
            "dst": "paper"
        }
    ],
    "data_dir": "/tg/tmp/ogbn_paper",
}
```

- **`vertex_name`**: The name of the vertex in TigerGraph (e.g., `paper`).  
- **`features_list`**: Specifies which node attributes are used as input features.  
- **`label`**: Points to the attribute name that holds the label for training/validation/testing.  
- **`split`**: Identifies the attribute that indicates whether a node belongs to the training, validation, or test set.  
- **`data_dir`**: The temporary directory where TigerGraph will export the data.

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
    tg_gcn.py \
    -g <tg-graph-name> \
    --host <tg-host-ip> \
    -u <tg-username> \
    -p <tg-password>
```

- `--nnodes` **and** `--nproc-per-node`: Control the number of nodes and GPUs per node for multi-GPU training.  
- `--rdzv-id`, `--rdzv-backend`, `--rdzv-endpoint`: Arguments for PyTorch’s distributed runtime.  
- `-g` (`--graph`): Name of the TigerGraph graph to be used.  
- `--host`, `-u`, `-p`: Host IP, username, and password for connecting to your TigerGraph instance.

---


**Happy Graphing and GNN Training!**


