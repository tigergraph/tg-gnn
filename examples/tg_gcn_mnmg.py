# Copyright (c) 2024, NVIDIA CORPORATION.
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

# Multi-node, multi-GPU example with WholeGraph feature storage.
# Can be run with torchrun.

import argparse
import os
import warnings
import tempfile
import time
import json


import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from pyTigerGraph import TigerGraphConnection


import torch_geometric

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
)

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

#### TG changes 1: import changes ####
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


from datetime import timedelta
from tg_gnn.data import load_tg_data, export_tg_data
from tg_gnn.utils import redistribute_splits


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    torch.cuda.set_device(local_rank)

    import rmm

    rmm.reinitialize(
        devices=[local_rank],
        managed_memory=True,
        pool_allocator=True,
    )

    torch.distributed.barrier()
    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def run_train(
    global_rank,
    data,
    split_idx,
    world_size,
    device,
    model,
    epochs,
    batch_size,
    fan_out,
    num_classes,
    wall_clock_start,
    tempdir=None,
    num_layers=3,
    in_memory=True,
    seeds_per_call=-1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx["train"].cuda()
    train_path = None if in_memory else os.path.join(tempdir, f"train_{global_rank}")
    if train_path:
        os.mkdir(train_path)
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        directory=train_path,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    ix_test = split_idx["test"].cuda()
    test_path = None if in_memory else os.path.join(tempdir, f"test_{global_rank}")
    if test_path:
        os.mkdir(test_path)
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        directory=test_path,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    ix_valid = split_idx["val"].cuda()
    valid_path = None if in_memory else os.path.join(tempdir, f"valid_{global_rank}")
    if valid_path:
        os.mkdir(valid_path)
    valid_loader = NeighborLoader(
        data,
        input_nodes=ix_valid,
        directory=valid_path,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    dist.barrier()

    eval_steps = 1000
    warmup_steps = 20
    dist.barrier()
    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time, "seconds")
        print("Beginning training...")

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()

            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.view(-1).to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if global_rank == 0 and i % 10 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Iteration: "
                    + str(i)
                    + ", Loss: "
                    + str(loss)
                )
        nb = i + 1.0

        if global_rank == 0:
            print(
                "Average Training Iteration Time:",
                (time.time() - start) / (nb - warmup_steps),
                "s/iter",
            )

        with torch.no_grad():
            total_correct = total_examples = 0
            for i, batch in enumerate(valid_loader):
                if i >= eval_steps:
                    break

                batch = batch.to(device)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_val = total_correct / total_examples
            if global_rank == 0:
                print(
                    f"Validation Accuracy: {acc_val * 100.0:.4f}%",
                )

        torch.cuda.synchronize()

    with torch.no_grad():
        total_correct = total_examples = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        acc_test = total_correct / total_examples
        if global_rank == 0:
            print(
                f"Test Accuracy: {acc_test * 100.0:.4f}%",
            )

    if global_rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    wm_finalize()
    cugraph_comms_shutdown()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--wg_mem_type", type=str, default="distributed")
    parser.add_argument("--in_memory", action="store_true", default=True)
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument("--tempdir_root", type=str, default="/tmp")
    parser.add_argument("-g", "--graph", default="ogbn_products_zetta", 
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


    return parser.parse_args()

#### TG changes 2: load partitions ####
# use load_tg_data to read the TG exported data
# load_tg_data will returned Data or HeteroData object of PyG
# using Data or HeteroData object you can create GraphStore and FeatureStore
def load_partitions(
    metadata: dict, 
    wg_mem_type: str, 
): 
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    # Load TG data and renumber the node ids
    # renumbering is required so keep it True
    data = load_tg_data(metadata, renumber=True)

    split_idx = {}

    # Load features
   
    feature_store["product", "x"] = data.x
    feature_store["product", "y"] = data.y

    # load splits
    # Note: (train/val/test)_mask will available if present in TG attr and 
    # the attr name is specified in metadata as split attribute
    # make sure to treat split attribute as enum of 3 int values 
    # (0 for train, 1 for valid and 2 for test)
    # other values are simply ignored
    split_idx["train"] = data.node_ids[data.train_mask] 
    split_idx["val"] = data.node_ids[data.val_mask] 
    split_idx["test"] = data.node_ids[data.test_mask] 

    # redistribute split to create unifrom size
    split_idx = redistribute_splits(split_idx)

    # create graph store
    graph_store[
        ("product", "rel", "product"), 
        "coo", 
        False,
        ( data.num_nodes, data.num_nodes)
    ] =  data.edge_index


    return (feature_store, graph_store), split_idx

#### TG changes 3: define the metadata ####
# Please update the metadata as per your Graph attributes and features
# make sure to have all the required features in features list
# and num of nodes for each node type
# data_dir path is used to export the data from TG database
metadata = {
    "nodes": {
        "product": {
            "vertex_name": "product",
            "features_list": {
                "embedding": "LIST"
            },
            "label": "node_label",
            "split": "train_val_test",
            "num_nodes": 2449029,
            "num_classes": 47,
            "num_features": 100,
        }
    }, 
    "edges": {
        "rel": {
            "rel_name": "rel",
            "src": "product",
            "dst": "product"
        }
    },
    "data_dir": "/data/ogbn_product",
    "num_classes": 47,
    "num_features": 100, 
    "num_nodes": 2449029 
}



if __name__ == "__main__":
    args = parse_args()
    wall_clock_start = time.perf_counter()
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(seconds=7200))
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        
        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        dist.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)
        
        if global_rank == 0 and not args.skip_tg_export:
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

        dist.barrier()
        data, split_idx = load_partitions(
            metadata,
            args.wg_mem_type
        )

        dist.barrier()

        model = torch_geometric.nn.models.GCN(
            metadata["num_features"],
            args.hidden_channels,
            args.num_layers,
            metadata["num_classes"],
        ).to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])

        with tempfile.TemporaryDirectory(dir=args.tempdir_root) as tempdir:
            run_train(
                global_rank,
                data,
                split_idx,
                world_size,
                device,
                model,
                args.epochs,
                args.batch_size,
                args.fan_out,
                metadata["num_classes"],
                wall_clock_start,
                tempdir,
                args.num_layers,
                args.in_memory,
                args.seeds_per_call,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
