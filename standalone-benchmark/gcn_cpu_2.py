import argparse
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

import torch_geometric
from torch_geometric.loader import NeighborLoader


def get_num_workers(world_size):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * world_size)
    return int(num_work)


def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes, wall_clock_start, tempdir=None,
              num_layers=3):

    # init pytorch worker
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    if world_size > 1:
        split_idx['train'] = split_idx['train'].split(
            split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
        split_idx['valid'] = split_idx['valid'].split(
            split_idx['valid'].size(0) // world_size, dim=0)[rank].clone()
        split_idx['test'] = split_idx['test'].split(
            split_idx['test'].size(0) // world_size, dim=0)[rank].clone()

    # No GPU, so no need to move the model to a specific device
    model = DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    num_work = get_num_workers(world_size)
    train_loader = NeighborLoader(data, input_nodes=split_idx['train'],
                                  num_workers=num_work, shuffle=True,
                                  drop_last=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=split_idx['valid'],
                                num_workers=num_work, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=split_idx['test'],
                                 num_workers=num_work, **kwargs)

    eval_steps = 1000
    warmup_steps = 20
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    dist.barrier()

    if rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time,
              "seconds")
        print("Beginning training...")
    
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                start = time.time()
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            
            if rank == 0 and i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        
        nb = i + 1.0
        dist.barrier()

        if rank == 0:
            print(f"Average Training Iteration Time: {(time.time() - start) / (nb - warmup_steps)} s/iter")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= eval_steps:
                    break
                out = model(batch.x, batch.edge_index)
                acc(out[:batch.batch_size].softmax(dim=-1), batch.y[:batch.batch_size])
            
            acc_sum = acc.compute()
            if rank == 0:
                print(f"Validation Accuracy: {acc_sum * 100.0:.4f}%")
        
        dist.barrier()
        acc.reset()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            out = model(batch.x, batch.edge_index)
            acc(out[:batch.batch_size].softmax(dim=-1), batch.y[:batch.batch_size])
        
        acc_sum = acc.compute()
        if rank == 0:
            print(f"Test Accuracy: {acc_sum * 100.0:.4f}%")
    
    dist.barrier()
    acc.reset()
    
    if rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print(f"Total Program Runtime (total_time): {total_time} seconds")
        print(f"Total_time - prep_time: {total_time - prep_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=30)
    parser.add_argument('--use_gat_conv', action='store_true', help="Use GATConv instead of GCNConv")
    parser.add_argument('--n_gat_conv_heads', type=int, default=4, help="Number of attention heads for GATConv")
    parser.add_argument('--n_devices', type=int, default=-1, help="Unused for CPU-only version")

    args = parser.parse_args()
    wall_clock_start = time.perf_counter()
    world_size = 1  # CPU only, so world size is 1

    dataset = PygNodePropPredDataset(name='ogbn-products', root='dataset/ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.y = data.y.reshape(-1)

    if args.use_gat_conv:
        model = torch_geometric.nn.models.GAT(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes,
                                              heads=args.n_gat_conv_heads)
    else:
        model = torch_geometric.nn.models.GCN(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes)

    print("Data =", data)

    with tempfile.TemporaryDirectory() as tempdir:
        run_train(0, data, world_size, model, args.epochs, args.batch_size,
                  args.fan_out, split_idx, dataset.num_classes,
                  wall_clock_start, tempdir, args.num_layers)
