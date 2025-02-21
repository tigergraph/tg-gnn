import pytest
from unittest import mock
import torch
from torch_geometric.data import Data, HeteroData
from tg_gnn.utils import renumber_data
import torch.distributed as dist

def run_homogeneous_test(rank, world_size):
    # Initialize distributed process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12345',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

    # Create per-rank data with node_ids and edges as original IDs
    if rank == 0:
        node_ids = torch.tensor([10, 20], dtype=torch.int64).cuda()
        edge_index = torch.tensor([[10, 20], [20, 10]], dtype=torch.int64).cuda()
    else:
        node_ids = torch.tensor([30, 40], dtype=torch.int64).cuda()
        edge_index = torch.tensor([[30, 40], [40, 30]], dtype=torch.int64).cuda()

    data = Data(node_ids=node_ids, edge_index=edge_index)
    metadata = {
        "nodes": {"_N": {"vertex_name": "_N"}},
        "edges": {"_E": {"src": "_N", "dst": "_N"}}
    }

    # Apply renumbering
    renumbered_data = renumber_data(data, metadata)

    expected_node_ids = torch.tensor([0, 1] if rank == 0 else [2, 3], dtype=torch.int64)
    assert torch.equal(renumbered_data.node_ids.cpu(), expected_node_ids), f"Node IDs mismatch on rank {rank}"

    expected_edges = torch.tensor([[0, 1], [1, 0]] if rank == 0 else [[2, 3], [3, 2]], dtype=torch.int64)
    assert torch.equal(renumbered_data.edge_index.cpu(), expected_edges), f"Edge indices mismatch on rank {rank}"

    dist.destroy_process_group()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_homogeneous_renumbering():
    world_size = 2
    torch.multiprocessing.spawn(run_homogeneous_test, args=(world_size,), nprocs=world_size, join=True)


def run_heterogeneous_test(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12346',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

    # Create heterogeneous data
    data = HeteroData()
    if rank == 0:
        data['user'].node_ids = torch.tensor([100, 200], dtype=torch.int64).cuda()
        data['item'].node_ids = torch.tensor([300, 400], dtype=torch.int64).cuda()
        data[('user', 'buys', 'item')].edge_index = torch.tensor([[100, 200], [300, 400]], dtype=torch.int64).cuda()
    else:
        data['user'].node_ids = torch.tensor([300, 400], dtype=torch.int64).cuda()
        data['item'].node_ids = torch.tensor([500, 600], dtype=torch.int64).cuda()
        data[('user', 'buys', 'item')].edge_index = torch.tensor([[300, 400], [500, 600]], dtype=torch.int64).cuda()
    print(data['user'])
    metadata = {
        "nodes": {
            "user": {"vertex_name": "user"},
            "item": {"vertex_name": "item"}
        },
        "edges": {
            "buys": {"src": "user", "dst": "item"}
        }
    }

    # Apply renumbering
    renumbered_data = renumber_data(data, metadata)

    # Check node IDs
    expected_user_ids = torch.tensor([0, 1] if rank == 0 else [2, 3], dtype=torch.int64)
    expected_item_ids = torch.tensor([0, 1] if rank == 0 else [2, 3], dtype=torch.int64)
    assert torch.equal(renumbered_data['user'].node_ids.cpu(), expected_user_ids), f"User node IDs mismatch on rank {rank}"
    assert torch.equal(renumbered_data['item'].node_ids.cpu(), expected_item_ids), f"Item node IDs mismatch on rank {rank}"

    # Check edges
    expected_edges = torch.tensor([[0, 1], [0, 1]] if rank == 0 else [[2, 3], [2, 3]], dtype=torch.int64)
    assert torch.equal(renumbered_data[('user', 'buys', 'item')].edge_index.cpu(), expected_edges), f"Edges mismatch on rank {rank}"

    dist.destroy_process_group()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_heterogeneous_renumbering():
    world_size = 2
    torch.multiprocessing.spawn(run_heterogeneous_test, args=(world_size,), nprocs=world_size, join=True)