# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import warnings

import gc

from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch.nn.parallel import DistributedDataParallel

import torch_geometric.transforms as T
from torch_geometric.datasets import Taobao
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.data import HeteroData

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)
# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ["CUDF_SPILL"] = "1"

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from sklearn.metrics import roc_auc_score

#### TG changes 1: import changes ####
from pyTigerGraph import TigerGraphConnection
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from tg_gnn.data import export_tg_data, load_tg_data
from tg_gnn.utils import get_local_world_size



def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    from cugraph.gnn import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


# GNN model 
class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict["item"],
            edge_index_dict[("item", "to", "item")],
        ).relu()

        user_x = self.conv2(
            (x_dict["item"], x_dict["user"]),
            edge_index_dict[("item", "rev_to", "user")],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[("item", "rev_to", "user")],
        ).relu()

        return self.lin(user_x)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.item_emb = Embedding(num_items, hidden_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict["user"] = self.user_emb(x_dict["user"])
        x_dict["item"] = self.item_emb(x_dict["item"])
        z_dict["item"] = self.item_encoder(
            x_dict["item"],
            edge_index_dict[("item", "to", "item")],
        )
        z_dict["user"] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict["user"], z_dict["item"], edge_label_index)


def cugraph_pyg_from_heterodata(data, wg_mem_type, return_edge_label=True):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)
    print(f"data user size: {data['user'].num_nodes}.")
    print(f"data item size: {data['item'].num_nodes}.")

    graph_store[
        ("user", "to", "item"),
        "coo",
        False,
        (data["user"].num_nodes, data["item"].num_nodes),
    ] = data["user", "user_to_item", "item"].edge_index
    graph_store[
        ("item", "rev_to", "user"),
        "coo",
        False,
        (data["item"].num_nodes, data["user"].num_nodes),
    ] = data["item", "rev_to", "user"].edge_index

    graph_store[
        ("item", "to", "item"),
        "coo",
        False,
        (data["item"].num_nodes, data["item"].num_nodes),
    ] = data["item", "item_to_item", "item"].edge_index
    graph_store[
        ("item", "rev_to", "item"),
        "coo",
        False,
        (data["item"].num_nodes, data["item"].num_nodes),
    ] = data["item", "rev_to", "item"].edge_index

    feature_store["item", "x", None] = data["item"].x
    feature_store["user", "x", None] = data["user"].x

    out = (
        (feature_store, graph_store),
        data["user", "user_to_item", "item"].edge_label_index,
        (data["user", "user_to_item", "item"].edge_label if return_edge_label else None),
    )

    return out

#### TG changes 2: load partitions ####
# use load_tg_data to read the TG exported data
# load_tg_data will returned Data or HeteroData object of PyG
# using Data or HeteroData object you can create GraphStore and FeatureStore
def load_partitions(metadata, wg_mem_type):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data = load_tg_data(metadata, renumber=True)
    print(f"Exported tg data loaded successfully.")
    print(f"TG data: {data}")

    # adding node features as we don't have any
    data["user"].x = torch.tensor_split(
        torch.arange(data["user"].num_nodes), world_size
    )[rank]

    data["item"].x = torch.tensor_split(
        torch.arange(data["item"].num_nodes), world_size
    )[rank]

    # adding rev edges
    data["item", "rev_to", "item"].edge_index = torch.stack(
        [
            data["item", "item_to_item", "item"].edge_index[1],
            data["item", "item_to_item", "item"].edge_index[0],
        ]
    )
    data["item", "rev_to", "user"].edge_index = torch.stack(
        [
            data["user", "user_to_item", "item"].edge_index[1],
            data["user", "user_to_item", "item"].edge_index[0],
        ]
    )

    print(f"data with reverse edges and feature {data}")

    # create splits using RandomLinkSplit
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=[("user", "user_to_item", "item")],
        rev_edge_types=[("item", "rev_to", "user")],
    )(data)

    print(train_data, test_data, val_data)

    return {
        "train": cugraph_pyg_from_heterodata(
            train_data, wg_mem_type, return_edge_label=False
        ),
        "test": cugraph_pyg_from_heterodata(test_data, wg_mem_type),
        "val": cugraph_pyg_from_heterodata(val_data, wg_mem_type),
    }

def train(model, optimizer, loader, epoch):
    rank = torch.distributed.get_rank()
    model.train()

    total_loss = total_examples = 0
    for i, batch in enumerate(loader):
        batch = batch.to(rank)
        optimizer.zero_grad()

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "item"].edge_label_index,
        )
        loss = F.binary_cross_entropy_with_logits(
            pred, batch["user", "item"].edge_label
        )

        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_examples += pred.numel()
        if global_rank == 0 and i % 100 == 0:
            print(
                "Epoch: "
                + str(epoch)
                + ", Iteration: "
                + str(i)
                + ", Loss: "
                + str(loss)
            )

    return total_loss / total_examples

@torch.no_grad()
def test(model, loader):
    rank = torch.distributed.get_rank()

    model.eval()
    preds, targets = [], []
    for i, batch in enumerate(loader):
        batch = batch.to(rank)

        pred = (
            model(
                batch.x_dict,
                batch.edge_index_dict,
                batch["user", "item"].edge_label_index,
            )
            .sigmoid()
            .view(-1)
            .cpu()
        )
        target = batch["user", "item"].edge_label.long().cpu()

        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()

    return roc_auc_score(target, pred)

#### TG changes 3: define the metadata ####
# Please update the metadata as per your Graph attributes and features
# make sure to have all the required features in features list
# and num of nodes for each node type
# data_dir path is used to export the data from TG database
metadata = {
    "nodes": {
        "item": {
            "vertex_name": "item",
            "num_nodes": 4161138,
        },
        "user": {
            "vertex_name": "user",
            "num_nodes": 987991
        }
    }, 
    "edges": {
        "item_to_item": {
            "src": "item",
            "dst": "item"
        },
        "user_to_item": {
            "src": "user",
            "dst": "item"
        }
    },
    "data_dir": "/data/taobao",
}

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--wg_mem_type", type=str, default="distributed")
    parser.add_argument("-g", "--graph", default="taobao", 
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
        help="Wheather to skip the data export from TG. Default value (False) will fetch the data.")

    args = parser.parse_args()


    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600))
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    if global_rank == 0:
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from cugraph.gnn import (
            cugraph_comms_create_unique_id,
        )

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)


    if not args.skip_tg_export and global_rank == 0:
        #### TG changes 4: Export TG data ####
        # write data from TG database to tmp path
        # need to call only once so global_rank 0 is used.

        # tg connection
        conn = TigerGraphConnection(
            host=args.host,
            restppPort=args.restppPort,
            graphname=args.graph,
            username=args.username,
            password=args.password
        )
        conn.getToken(conn.createSecret())
        export_tg_data(conn, metadata, force=True)

    print(f"local world size: {get_local_world_size()}")

    torch.distributed.barrier()
    data_dict = load_partitions(metadata, args.wg_mem_type) 
    torch.distributed.barrier()

    from cugraph_pyg.loader import LinkNeighborLoader

    def create_loader(data_l):
        return LinkNeighborLoader(
            data=data_l[0],
            edge_label_index=data_l[1],
            edge_label=data_l[2],
            neg_sampling="binary" if data_l[2] is None else None,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_neighbors={
                ("user", "to", "item"): [8, 4],
                ("item", "rev_to", "user"): [8, 4],
                ("item", "to", "item"): [8, 4],
                ("item", "rev_to", "item"): [8, 4],
            },
            local_seeds_per_call=16384,
        )

    print("Creating train loader...")
    train_loader = create_loader(
        data_dict["train"],
    )
    print(f"Created train loader on rank {global_rank}")

    torch.distributed.barrier()

    print("Creating validation loader...")
    val_loader = create_loader(
        data_dict["val"],
    )
    print(f"Created validation loader on rank {global_rank}")

    torch.distributed.barrier()

    model = Model(
        num_users=metadata["nodes"]["user"]["num_nodes"],
        num_items=metadata["nodes"]["item"]["num_nodes"],
        hidden_channels=64,
        out_channels=64,
    ).to(local_rank)
    print(f"Created model on rank {global_rank}")

    # Initialize lazy modules
    # FIXME DO NOT DO THIS!!!!  Use set parameters
    for batch in train_loader:
        batch = batch.to(local_rank)
        _ = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "item"].edge_label_index,
        )
        break
    print(f"Initialized model on rank {global_rank}")

    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = 0
    for epoch in range(1, args.epochs + 1):
        print("Train")
        loss = train(model, optimizer, train_loader, epoch)

        if global_rank == 0:
            print("Val")
        val_auc = test(model, val_loader)
        best_val_auc = max(best_val_auc, val_auc)

        if global_rank == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss:4f}, Val AUC: {val_auc:.4f}")

    del train_loader
    del val_loader
    gc.collect()
    print("Creating test loader...")
    test_loader = create_loader(data_dict["test"])

    if global_rank == 0:
        print("Test")
    test_auc = test(model, test_loader)
    print(
        f"Total {args.epochs:02d} epochs: Final Loss: {loss:4f}, "
        f"Best Val AUC: {best_val_auc:.4f}, "
        f"Test AUC: {test_auc:.4f}"
    )

    wm_finalize()

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()
