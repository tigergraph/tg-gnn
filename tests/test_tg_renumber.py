import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import Data, HeteroData
from tg_gnn.tg_renumber import renumber_data


# Mock torch.distributed functions
@pytest.fixture
def mock_distributed():
    with patch("torch.distributed.all_gather_into_tensor") as mock_all_gather, \
         patch("torch.distributed.get_rank") as mock_get_rank, \
         patch("torch.distributed.get_world_size") as mock_get_world_size:
        
        # Setup default mock behavior
        mock_all_gather.side_effect = lambda tensor_list, tensor: [
            tensor_list[i].copy_(tensor) for i in range(len(tensor_list))
        ]
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        yield mock_all_gather, mock_get_rank, mock_get_world_size

# Mock CUDA-related functions and objects
@pytest.fixture
def mock_cuda():
    with patch("torch.device") as mock_device, \
         patch("torch.cuda.set_device") as mock_set_device, \
         patch("torch.cuda.empty_cache") as mock_empty_cache:
        
        mock_device.return_value = torch.device("cuda:0")
        yield mock_device, mock_set_device, mock_empty_cache

# Mock cudf and cupy
@pytest.fixture
def mock_cudf_cupy():
    with patch("cudf.DataFrame") as mock_cudf_df, \
         patch("cupy.asarray") as mock_cupy_asarray:
        
        # Mock cudf DataFrame
        mock_df = MagicMock()
        mock_cudf_df.return_value = mock_df

        # Mock cupy.asarray to return numpy arrays for simplicity
        mock_cupy_asarray.side_effect = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        yield mock_cudf_df, mock_cupy_asarray, mock_df

def test_renumber_data_single_rank_homogeneous(mock_distributed, mock_cuda, mock_cudf_cupy):
    mock_all_gather, mock_get_rank, mock_get_world_size = mock_distributed
    mock_device, mock_set_device, mock_empty_cache = mock_cuda
    mock_cudf_df, mock_cupy_asarray, mock_df = mock_cudf_cupy

    # Create a simple Data object
    data = Data()
    data.node_ids = torch.tensor([10, 20, 30], dtype=torch.int64)
    data.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)

    # Metadata
    metadata = {
        "nodes": [{"vertex_name": "node"}],
        "edges": [{"src": "node", "dst": "node", "rel_name": "connects"}]
    }

    # Call renumber_data
    renumbered_data = renumber_data(
        data=data,
        metadata=metadata,
        local_rank=0,
        world_size=1
    )

    # Assertions
    assert renumbered_data.node_ids.tolist() == [0, 1, 2], "Node IDs should be renumbered to [0, 1, 2]"
    assert renumbered_data.edge_index.tolist() == [[0, 1, 2], [1, 2, 0]], "Edge indices should remain the same"

    # Check if cudf DataFrame was called correctly
    mock_cudf_df.assert_called_once()
    mock_df.__setitem__.assert_called_once_with("id", [0, 1, 2])

def test_renumber_data_single_rank_heterogeneous(mock_distributed, mock_cuda, mock_cudf_cupy):
    mock_all_gather, mock_get_rank, mock_get_world_size = mock_distributed
    mock_device, mock_set_device, mock_empty_cache = mock_cuda
    mock_cudf_df, mock_cupy_asarray, mock_df = mock_cudf_cupy

    # Create a HeteroData object with two node types and one edge type
    data = HeteroData()
    data['user'].node_ids = torch.tensor([100, 200], dtype=torch.int64)
    data['item'].node_ids = torch.tensor([300, 400, 500], dtype=torch.int64)
    data['user', 'buys', 'item'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)

    # Metadata
    metadata = {
        "nodes": [
            {"vertex_name": "user"},
            {"vertex_name": "item"}
        ],
        "edges": [
            {"src": "user", "dst": "item", "rel_name": "buys"}
        ]
    }

    # Call renumber_data
    renumbered_data = renumber_data(
        data=data,
        metadata=metadata,
        local_rank=0,
        world_size=1
    )

    # Assertions for 'user' nodes
    assert renumbered_data['user'].node_ids.tolist() == [0, 1], "User node IDs should be renumbered to [0, 1]"

    # Assertions for 'item' nodes
    assert renumbered_data['item'].node_ids.tolist() == [2, 3, 4], "Item node IDs should be renumbered to [2, 3, 4]"

    # Assertions for edges
    expected_edge_index = [[0, 1], [3, 4]]  # Assuming user 0->item 3 and user 1->item 4
    assert renumbered_data['user', 'buys', 'item'].edge_index.tolist() == expected_edge_index, \
        f"Edge indices should be renumbered to {expected_edge_index}"

    # Check if cudf DataFrame was called correctly for both node types
    assert mock_cudf_df.call_count == 2
    mock_df.__setitem__.assert_any_call("id", [0, 1])
    mock_df.__setitem__.assert_any_call("id", [2, 3, 4])

def test_renumber_data_missing_node_ids(mock_distributed, mock_cuda, mock_cudf_cupy):
    mock_all_gather, mock_get_rank, mock_get_world_size = mock_distributed
    mock_device, mock_set_device, mock_empty_cache = mock_cuda
    mock_cudf_df, mock_cupy_asarray, mock_df = mock_cudf_cupy

    # Create a Data object without node_ids
    data = Data()
    data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    # Metadata
    metadata = {
        "nodes": [{"vertex_name": "node"}],
        "edges": [{"src": "node", "dst": "node", "rel_name": "connects"}]
    }

    # Call renumber_data
    renumbered_data = renumber_data(
        data=data,
        metadata=metadata,
        local_rank=0,
        world_size=1
    )

    # Assertions
    # Since node_ids are missing, they should remain unchanged (or handle as per implementation)
    assert renumbered_data.node_ids is None, "Node IDs should remain None when missing"

    # Edge indices should remain unchanged
    assert renumbered_data.edge_index.tolist() == [[0, 1], [1, 0]], "Edge indices should remain unchanged"

    # cudf DataFrame should not be called since node_ids are missing
    mock_cudf_df.assert_not_called()

def test_renumber_data_multiple_ranks_homogeneous(mock_cuda, mock_cudf_cupy):
    with patch("torch.distributed.all_gather_into_tensor") as mock_all_gather, \
         patch("torch.distributed.get_rank") as mock_get_rank, \
         patch("torch.distributed.get_world_size") as mock_get_world_size:

        # Simulate 2 ranks with different node_ids
        def all_gather_side_effect(tensor_list, tensor):
            if mock_get_rank.return_value == 0:
                tensor_list[0].copy_(tensor)
                tensor_list[1].copy_(torch.tensor([3], dtype=torch.int64, device=tensor.device))
            elif mock_get_rank.return_value == 1:
                tensor_list[0].copy_(tensor)
                tensor_list[1].copy_(torch.tensor([5], dtype=torch.int64, device=tensor.device))

        mock_all_gather.side_effect = all_gather_side_effect

        # Define mock rank and world size
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 2

        # Mock cudf DataFrame
        mock_df = MagicMock()
        mock_cudf_df, mock_cupy_asarray, _ = mock_cudf_cupy
        mock_cudf_df.return_value = mock_df
        mock_cupy_asarray.side_effect = lambda x: x.cpu().numpy()

        # Create a Data object
        data = Data()
        data.node_ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        data.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)

        # Metadata
        metadata = {
            "nodes": [{"vertex_name": "node"}],
            "edges": [{"src": "node", "dst": "node", "rel_name": "connects"}]
        }

        # Call renumber_data
        renumbered_data = renumber_data(
            data=data,
            metadata=metadata,
            local_rank=0,
            world_size=2
        )

        # Since world_size=2 and rank=0 has 3 nodes, global renumbering starts at 0
        # Rank 1 would have node_ids [3,4,5] but since we're testing rank=0, only its nodes are renumbered
        assert renumbered_data.node_ids.tolist() == [0, 1, 2], "Rank 0 node IDs should be renumbered to [0, 1, 2]"

        # Edge indices should be updated accordingly
        assert renumbered_data.edge_index.tolist() == [[0, 1, 2], [1, 2, 0]], "Edge indices should remain the same for single rank"

        # Check cudf DataFrame was called
        mock_cudf_df.assert_called_once()
        mock_df.__setitem__.assert_called_once_with("id", [0, 1, 2])

def test_renumber_data_missing_edge_index(mock_distributed, mock_cuda, mock_cudf_cupy):
    mock_all_gather, mock_get_rank, mock_get_world_size = mock_distributed
    mock_device, mock_set_device, mock_empty_cache = mock_cuda
    mock_cudf_df, mock_cupy_asarray, mock_df = mock_cudf_cupy

    # Create a Data object with node_ids but without edge_index
    data = Data()
    data.node_ids = torch.tensor([10, 20, 30], dtype=torch.int64)

    # Metadata
    metadata = {
        "nodes": [{"vertex_name": "node"}],
        "edges": [{"src": "node", "dst": "node", "rel_name": "connects"}]
    }

    # Call renumber_data
    renumbered_data = renumber_data(
        data=data,
        metadata=metadata,
        local_rank=0,
        world_size=1
    )

    # Assertions
    # Node IDs should be renumbered
    assert renumbered_data.node_ids.tolist() == [0, 1, 2], "Node IDs should be renumbered to [0, 1, 2]"

    # Edge indices should remain None
    assert renumbered_data.edge_index is None, "Edge indices should remain None when missing"

    # cudf DataFrame should be called for node renumbering
    mock_cudf_df.assert_called_once()
    mock_df.__setitem__.assert_called_once_with("id", [0, 1, 2])

def test_renumber_data_heterogeneous_missing_edge_type(mock_distributed, mock_cuda, mock_cudf_cupy):
    mock_all_gather, mock_get_rank, mock_get_world_size = mock_distributed
    mock_device, mock_set_device, mock_empty_cache = mock_cuda
    mock_cudf_df, mock_cupy_asarray, mock_df = mock_cudf_cupy

    # Create a HeteroData object with one node type and missing edge type
    data = HeteroData()
    data['user'].node_ids = torch.tensor([100, 200], dtype=torch.int64)
    # Note: 'item' node type is missing
    # Edge type 'buys' refers to 'item' which is missing

    # Metadata
    metadata = {
        "nodes": [
            {"vertex_name": "user"},
            {"vertex_name": "item"}
        ],
        "edges": [
            {"src": "user", "dst": "item", "rel_name": "buys"}
        ]
    }

    # Call renumber_data
    renumbered_data = renumber_data(
        data=data,
        metadata=metadata,
        local_rank=0,
        world_size=1
    )

    # Assertions for 'user' nodes
    assert renumbered_data['user'].node_ids.tolist() == [0, 1], "User node IDs should be renumbered to [0, 1]"

    # 'item' node type does not exist, so no renumbering
    assert 'item' not in renumbered_data, "'item' node type should be skipped"

    # Edge indices should remain None since edge type 'buys' is missing
    assert ('user', 'buys', 'item') not in renumbered_data.edge_types, "Missing edge type should be skipped"

    # Check if cudf DataFrame was called only for existing node type
    mock_cudf_df.assert_called_once()
    mock_df.__setitem__.assert_called_once_with("id", [0, 1])

