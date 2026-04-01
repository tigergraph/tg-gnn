# TG-GNN

This project is intended for using [TigerGraph](https://www.tigergraph.com/) as a graph data storage solution in combination with [cuGraph](https://github.com/rapidsai/cugraph) and [cuGraph-PyG](https://docs.rapids.ai/api/cugraph/stable/graph_support/pyg_support/). By leveraging these tools, you can train a GNN model on multi-GPU setups at any scale.

---

## Getting Started
### CUDA Setup
- Required CUDA version >= 12.8;

### Environment Setup

1. **Install CUDA Toolkit (Optional)**
   ```bash
   bash setup/cuda_setup.sh
   ```
2. **Download and install Conda, for example Miniforge**
   ```bash
   curl -OL  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash ./Miniforge3-Linux-x86_64.sh -b -u -p ~/miniforge3
   source ~/miniforge3/bin/activate
   conda init
   ```
3. **Install tg-gnn project**
    ```bash
    git clone https://github.com/tigergraph/tg-gnn.git
    cd tg-gnn
    source setup/conda_setup.sh
    pip install .
    ```
---

## Changes Required to Run example GCN Model code on Any TigerGraph Graph

To integrate your TigerGraph graph with the GCN model, a few schema updates and metadata configurations are required.

### 1. Load Data on TigerGraph
- To load please follow instructions available in benchmark/Benchmark.md 

### 2. Configure Metadata 

Below is a sample `metadata` dictionary that illustrates how to define the mapping between your TigerGraph entities and the model’s requirements:

```python
metadata = {
    "nodes": {
        "<vertex_name 1>": {
            "vertex_name" : "<vertex_name 1>",
            "features_list" (optional): {
                "feature1": "<feature_datatype (INT, FLOAT, or LIST)>",
                "feature2": "LIST",
                "feature3": "FLOAT"
            },
            "label" (optional): "<label_attr_name>",
            "split" (optional): "<split_attr_name>",
            "num_nodes": <number of nodes>
        },

        "<vertex_name 2>": {
            "vertex_name" : "<vertex_name 2>",
            "features_list" (optional): {
                "feature1": "<feature_datatype (INT, FLOAT, or LIST)>",
                "feature2": "LIST",
            },
            "num_nodes": <number of nodes>
        },

        "<vertex_name 3>": {
            "vertex_name" : "<vertex_name 3>",
            "num_nodes": <number of nodes>
        }

    },
    "edges": {
        "<rel_name>": {
            "rel_name": "<rel_name>",
            "undirected": True,
            "src": "<src_name>", 
            "dst": "<dst_name>",
            "time_attr" (optional): "<time_attr_name>",
            "features_list" (optional): {
                "feature1": "<feature_datatype (INT, FLOAT, or LIST)>",
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
- **`features_list`**: Specifies which node attributes are used as input features. `feature_datatype` can be `"INT"`, `"FLOAT"`, or `"LIST"` (lists should only contain FLOAT/INT values for GNN training).
- **`label`**: Points to the attribute name that holds the label for training/validation/testing.  
- **`split`**: Identifies the attribute that indicates whether a node belongs to the training, validation, or test set.  
- **`num_nodes`**: Indicates the number of vertices/nodes for the given vertex/node type.  
- **`rel_name`**: The name of the edge in TigerGraph (e.g., `uses`).
- **`undirected`**: The direction of the edge in TigerGraph. Reverse edge (e.g., `rev_uses`) will be generated when it's True.
- **`src`**: The `from` vertex type of the edge in TigerGraph.
- **`dst`**: The `to` vertex type of the edge in TigerGraph.
- **`time_attr`**: The attribute name of the time attribute of the edge in TigerGraph. Required for temporal split and temporal neighbor sampling (see below).
- **`data_dir`**: The temporary directory where TigerGraph will export the data and data will be accessed by the GNN model training process.
- **`fs_type`**: The filesystem type of `data_dir` — either `"shared"` (all ranks access the same directory) or `"local"` (each rank has its own copy).
- **`num_tg_nodes`**: The number of nodes in the TigerGraph cluster. Used to partition the data export across TigerGraph nodes.

---

## Temporal Split and Temporal Neighborhoods

When edges carry a timestamp (configured via `time_attr` in metadata), tg-gnn supports **temporal data splitting** and **temporal neighbor sampling**. These techniques are critical for link-prediction tasks on dynamic graphs where using future information during training leads to data leakage and unrealistic evaluation.

### Temporal Split

Instead of assigning edges to train/val/test sets randomly, a **temporal split** partitions edges chronologically based on their timestamps:

1. All edges are sorted by their timestamp in ascending order.
2. The oldest 80% of edges form the **training set**.
3. The newest 20% form the **test set**.

```
Timeline:  ─────────────────────────────────────────────────►
           │◄──── oldest 80% (train) ────►│◄── newest 20% (test) ──►│
```

This mirrors real-world scenarios where a model is trained on historical data and evaluated on future events. It prevents the model from "seeing the future" during training, which random splits cannot guarantee.

**How it works in code** (see `examples/tg_movielens_mnmg.py`):

```python
edge_time = data["user", "rates", "movie"].time
perm = edge_time.argsort()
# Reorder edges and timestamps chronologically
data["user", "rates", "movie"].edge_index = edge_index[:, perm]
data["user", "rates", "movie"].time = edge_time[perm]
# 80/20 split
off = int(0.8 * perm.numel())
splits = {
    "train": edge_index[:, :off],
    "test":  edge_index[:, off:],
}
```

### Temporal Neighbor Sampling

Standard neighbor sampling selects neighboring nodes/edges without regard to when those edges occurred. For temporal graphs, this allows future edges to influence the representation of a node at an earlier point in time — a form of information leakage.

**Temporal neighbor sampling** restricts the sampled neighborhood to edges whose timestamps are strictly **before** the seed edge's timestamp. This ensures that when the model computes embeddings for a training edge at time *t*, it only aggregates information from edges that occurred before *t*.

```
                    seed edge time (t)
                           │
Timeline:  ────────────────┼──────────────────────►
           ◄── eligible ──►│  ✗ future (excluded)
```

This is implemented by passing `time_attr` and `edge_label_time` to cuGraph-PyG's `LinkNeighborLoader`:

```python
train_loader = LinkNeighborLoader(
    edge_label_index=(("user", "rates", "movie"), train_edges),
    edge_label_time=train_times - 1,  # -1 ensures strict "before" semantics
    time_attr="time",                 # name of the time field in FeatureStore
    num_neighbors={
        ("user", "rates", "movie"): [5, 5, 5],
        ("movie", "rev_rates", "user"): [5, 5, 5],
    },
    ...
)
```

**Key parameters:**

| Parameter | Description |
|---|---|
| `time_attr` (metadata) | TigerGraph edge attribute name for timestamps (e.g., `"timestamp"`). Used during data export. |
| `time_attr` (loader) | Name of the time field in the FeatureStore (e.g., `"time"`). Tells the sampler which attribute to use for temporal filtering. |
| `edge_label_time` | Per-seed-edge timestamp tensor. The sampler only selects neighbors with `time < edge_label_time`. The `-1` offset enforces strict ordering. |
| `num_neighbors` | Fan-out per hop per edge type (e.g., `[5, 5, 5]` for 3-hop, 5 neighbors each). |

**Setup requirements for temporal features:**

1. **TigerGraph schema**: Edges must have an integer or unsigned-integer timestamp attribute (e.g., `timestamp UINT` on the `rates` edge type).
2. **Metadata**: Set `"time_attr": "<attribute_name>"` in the edge's metadata entry so tg-gnn exports the timestamps during data export.
3. **FeatureStore**: After loading, store per-edge timestamps under a key (e.g., `"time"`) in the FeatureStore. Node types that don't have natural timestamps should be assigned `time = 0`.
4. **Loader**: Pass `time_attr="time"` and `edge_label_time=<timestamps> - 1` to `LinkNeighborLoader`.

For a complete working example, see [`examples/tg_movielens_mnmg.py`](examples/tg_movielens_mnmg.py) and its companion documentation [`examples/tg_movielens_mnmg.md`](examples/tg_movielens_mnmg.md).

---

## Running the Example

Use the following command to run the GCN example. Make sure to replace the placeholders with the correct values for your TigerGraph environment.

```bash
torchrun --nnodes 1 --nproc-per-node 4 --rdzv-id 4RANDOM --rdzv-backend c10d --rdzv-endpoint localhost:29500 examples/tg_ogbn_products_mnmg.py \
    -g <tg-graph-name> --host http://<tg-host-ip> --restppPort <tg-port> \
    -u <tg-username> -p <tg-password> --tg_nodes 1 --data_dir /tg_export
```

- `--nnodes` **and** `--nproc-per-node`: Control the number of nodes and GPUs per node for multi-GPU training.  
- `--rdzv-id`, `--rdzv-backend`, `--rdzv-endpoint`: Arguments for PyTorch’s distributed runtime.  
- `-g` (`--graph`): Name of the TigerGraph graph to be used.  
- `--host`: The TigerGraph server URL, **including the protocol** (e.g., `http://172.31.43.164`). Defaults to `http://localhost`.
- `--restppPort`: The port for REST++ queries. Defaults to `9000`.
- `-u` (`--username`), `-p` (`--password`): TigerGraph credentials. Both default to `tigergraph`.
- `--tg_nodes`: The number of nodes in the TigerGraph cluster. Defaults to `1`.
- `--data_dir`: The directory where TigerGraph exports data for GNN training.
- `--file_system`: The filesystem type — `shared` (default) or `local`.
- `-s` (`--skip_tg_export`): Set to `True` to skip the data export step (e.g., `-s True`). Defaults to `False`.

## Expected Result

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
