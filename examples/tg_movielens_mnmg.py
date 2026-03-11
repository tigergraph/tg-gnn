"""MovieLens temporal link-prediction example with TigerGraph + cuGraph-PyG.

This example demonstrates temporal-aware GNN training for link prediction
on the MovieLens dataset.  Edges carry a ``timestamp`` attribute that is
used for:

1. **Temporal train/test splitting** — the oldest 80 % of ratings form the
   training set while the newest 20 % are held out for evaluation.
2. **Temporal neighbor sampling** — ``LinkNeighborLoader`` with
   ``time_attr='time'`` ensures that only neighbors whose edges occurred
   *before* the target edge's timestamp are sampled, preventing future
   information leakage.

Usage (multi-GPU via torchrun)::

    torchrun --nproc_per_node=<N_GPUS> tg_movielens_mnmg.py \\
        --host http://<TG_HOST> --data_dir /tmp/tg

Prerequisites:
    1. Run ``examples/movielens/prepare_movielens.sh`` to download data and
       generate embeddings.
    2. Create the graph schema using
       ``examples/movielens/movielens_schema.gsql``.
    3. Load data using ``examples/movielens/load_movielens.gsql``.
"""

import os
import subprocess
import warnings
from argparse import ArgumentParser
from datetime import timedelta

import torch
import torch.nn.functional as F

from torch.nn import Linear

from tqdm import tqdm

from torch_geometric import EdgeIndex
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

from sklearn.metrics import roc_auc_score

#### TigerGraph connection and data pipeline ####
from pyTigerGraph import TigerGraphConnection
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from tg_gnn.data import export_tg_data, load_tg_data
from tg_gnn.utils import redistribute_splits


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=False,
        pool_allocator=False,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    from pylibcugraph.comms import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


#### Load exported data, apply temporal split, and build cuGraph stores ####
# Uses load_tg_data to read the TG exported data into PyG HeteroData.
# Edges are sorted by timestamp and split 80/20 for train/test.
# Time attributes are stored in the feature store for temporal neighbor sampling.
def load_partitions(metadata, wg_mem_type):
    from cugraph_pyg.data import GraphStore, FeatureStore

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    undirected = metadata.get("edges").get("rates").get("undirected", False)

    data = load_tg_data(metadata, renumber=True)
    print(f"Exported tg data loaded successfully.")
    print(f"TG data: {data}")

    # user has no features in TG — use identity matrix
    data["user"].x = (
        torch.tensor_split(
            torch.eye(data["user"].num_nodes, dtype=torch.float32), world_size
        )[rank]
        .detach()
        .clone()
    )

    # --- Temporal split: sort by timestamp, 80 % train / 20 % test ---------
    edge_time = data["user", "rates", "movie"].time
    perm = edge_time.argsort()

    data["user", "rates", "movie"].edge_index = (
        data["user", "rates", "movie"].edge_index[:, perm]
    )
    data["user", "rates", "movie"].time = edge_time[perm]

    off = int(0.8 * perm.numel())
    splits = {
        "train": data["user", "rates", "movie"].edge_index[:, :off],
        "test":  data["user", "rates", "movie"].edge_index[:, off:],
    }
    time_splits = {
        "train": data["user", "rates", "movie"].time[:off],
        "test":  data["user", "rates", "movie"].time[off:],
    }

    splits = redistribute_splits(splits)
    time_splits = redistribute_splits(time_splits)

    # --- Build cuGraph stores -----------------------------------------------
    graph_store = GraphStore()
    feature_store = FeatureStore()

    graph_store[
        ("user", "rates", "movie"),
        "coo",
        False,
        (data["user"].num_nodes, data["movie"].num_nodes),
    ] = data["user", "rates", "movie"].edge_index

    if undirected:
        graph_store[
            ("movie", "rev_rates", "user"),
            "coo",
            False,
            (data["movie"].num_nodes, data["user"].num_nodes),
        ] = data["movie", "rev_rates", "user"].edge_index

    feature_store["user", "x", None] = data["user"].x
    feature_store["movie", "x", None] = data["movie"].x

    # Store time attributes — nodes get t=0, edges keep their timestamps
    feature_store["user", "time", None] = torch.tensor_split(
        torch.zeros(data["user"].num_nodes, dtype=torch.int64), world_size
    )[rank]
    feature_store["movie", "time", None] = torch.tensor_split(
        torch.zeros(data["movie"].num_nodes, dtype=torch.int64), world_size
    )[rank]
    feature_store[("user", "rates", "movie"), "time", None] = (
        data["user", "rates", "movie"].time
    )
    if undirected:
        feature_store[("movie", "rev_rates", "user"), "time", None] = (
            data["movie", "rev_rates", "user"].time
        )

    return feature_store, graph_store, splits, time_splits


class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        user_x = self.conv1(
            (x_dict["movie"], x_dict["user"]),
            edge_index_dict["movie", "rev_rates", "user"],
        ).relu()

        movie_x = self.conv2(
            (x_dict["user"], x_dict["movie"]),
            edge_index_dict["user", "rates", "movie"],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict["movie", "rev_rates", "user"],
        ).relu()

        return {
            "user": self.lin1(user_x),
            "movie": self.lin2(movie_x),
        }


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat(
            [
                x_dict["user"][row],
                x_dict["movie"][col],
            ],
            dim=-1,
        )
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = Encoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, num_samples):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(
            x_dict,
            edge_index_dict["user", "rates", "movie"][:, :num_samples],
        )


def train(train_loader, model, optimizer):
    model.train()
    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "rates", "movie"].edge_label.shape[0],
        )
        y = batch["user", "rates", "movie"].edge_label

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * y.numel()
        total_examples += y.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(test_loader, model):
    model.eval()
    preds, targets = [], []
    for batch in test_loader:
        batch = batch.to(device)
        pred = (
            model(
                batch.x_dict,
                batch.edge_index_dict,
                batch["user", "rates", "movie"].edge_label.shape[0],
            )
            .sigmoid()
            .view(-1)
            .cpu()
        )
        target = batch["user", "rates", "movie"].edge_label.long().cpu()
        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    return roc_auc_score(target, pred)


#### Graph metadata: vertex types, edge types, features, and export config ####
# Please update the metadata as per your Graph attributes and features.
# Make sure to have all the required features in features list
# and num of nodes for each node type.
# data_dir path is used to export the data from TG database.
# time_attr on the edge tells the framework which TG edge attribute holds the timestamp.
metadata = {
    "nodes": {
        "user": {
            "vertex_name": "user",
            "num_nodes": 610,
        },
        "movie": {
            "vertex_name": "movie",
            "features_list": {
                "movie_embedding": "LIST"
            },
            "num_nodes": 9742,
        },
    },
    "edges": {
        "rates": {
            "undirected": True,
            "src": "user",
            "dst": "movie",
            "time_attr": "timestamp",
        },
    },
    "data_dir": "/data/movielens",
    "fs_type": "shared",
    "num_tg_nodes": 1,
}

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--wg_mem_type", type=str, default="distributed")
    parser.add_argument("-g", "--graph", default="movielens",
        help="The default graph for running queries.")
    parser.add_argument("--host", default="http://172.17.0.3",
        help=("The host name or IP address of the TigerGraph server."
            "Make sure to include the protocol (http:// or https://)."
            "If certPath is None and the protocol is https, a self-signed certificate will be used.")
    )
    parser.add_argument("--restppPort", default="9000", help="The port for REST++ queries.")
    parser.add_argument("--username", "-u", default="tigergraph",
        help="The username on the TigerGraph server.")
    parser.add_argument("--password", "-p", default="tigergraph",
        help="The password for that user.")
    parser.add_argument("--skip_tg_export", "-s", type=bool, default=False,
        help="Whether to skip the data export from TG. Default value (False) will fetch the data.")
    parser.add_argument("--data_dir", type=str, default="/tmp/tg",
        help="The directory to store the data exported from TG.")
    parser.add_argument("--file_system", type=str, default="shared",
        help="The type of file system to use. Options are 'local' or 'shared'.")
    parser.add_argument("--tg_nodes", type=int, default=1,
        help="The number of TigerGraph nodes in your cluster. Default value is 1.")

    args = parser.parse_args()
    metadata["data_dir"] = args.data_dir
    metadata["fs_type"] = args.file_system
    metadata["num_tg_nodes"] = args.tg_nodes
    subprocess.run(['sudo', 'chmod', '-R', '0777', args.data_dir])

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600), device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"))
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    print(f"[rank{global_rank}] Starting on {os.uname().nodename} (local_rank={local_rank}, world_size={world_size})")

    if global_rank == 0:
        from rmm.allocators.torch import rmm_torch_allocator
        torch.cuda.change_current_allocator(rmm_torch_allocator)

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from pylibcugraph.comms import (
            cugraph_comms_create_unique_id,
        )
        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    if not args.skip_tg_export and global_rank == 0:
        #### Export graph data from TigerGraph to partitioned CSV files ####
        # Only rank 0 exports; other ranks wait at the barrier below.
        conn = TigerGraphConnection(
            host=args.host,
            restppPort=args.restppPort,
            graphname=args.graph,
            username=args.username,
            password=args.password
        )
        conn.getToken(conn.createSecret())
        export_tg_data(conn, metadata, force=True)

    torch.distributed.barrier()
    feature_store, graph_store, splits, time_splits = load_partitions(
        metadata, args.wg_mem_type
    )
    torch.distributed.barrier()

    eli_train = splits["train"]
    eli_test = splits["test"]

    kwargs = dict(
        data=(feature_store, graph_store),
        num_neighbors={
            ("user", "rates", "movie"): [5, 5, 5],
            ("movie", "rev_rates", "user"): [5, 5, 5],
        },
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    from cugraph_pyg.loader import LinkNeighborLoader

    # time_attr + edge_label_time enable temporal neighbor sampling on train only;
    # -1 offset prevents future information leakage.
    train_loader = LinkNeighborLoader(
        edge_label_index=(("user", "rates", "movie"), eli_train),
        edge_label_time=time_splits["train"] - 1,
        time_attr="time",
        neg_sampling=dict(mode="binary", amount=2),
        **kwargs,
    )

    # Test loader uses temporal sampling for consistent evaluation:
    # negatives and neighbors are restricted to edges before each seed's timestamp.
    test_loader = LinkNeighborLoader(
        edge_label_index=(("user", "rates", "movie"), eli_test),
        edge_label_time=time_splits["test"] - 1,
        time_attr="time",
        neg_sampling=dict(mode="binary", amount=1),
        **kwargs,
    )

    sparse_size = (metadata["nodes"]["user"]["num_nodes"], metadata["nodes"]["movie"]["num_nodes"])
    test_edge_label_index = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]
    test_exclude_links = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]

    model = Model(hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_loader, model, optimizer)
        print(f"Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        auc = test(test_loader, model)
        print(f"Test AUC: {auc:.4f} ")

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()
    wm_finalize()
