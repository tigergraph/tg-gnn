import os
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import Data, HeteroData
from tg_gnn.tg_data import load_tg_data

# Mock the renumber_data function
@pytest.fixture
def mock_renumber_data():
    with patch("tg_gnn.renumber_data") as mock_renumber:
        mock_renumber.side_effect = lambda data, metadata, local_rank, world_size: data
        yield mock_renumber

# Mock cudf and related functions
@pytest.fixture
def mock_cudf():
    with patch("tg_gnn.cudf") as mock_cudf, \
         patch("tg_gnn.cupy") as mock_cupy:
        
        # Mock cudf.read_csv to return a MagicMock DataFrame
        mock_df = MagicMock()
        mock_cudf.read_csv.return_value = mock_df
        
        # Mock DataFrame's iloc and shape
        mock_df.shape = (5, 4)  # Example shape
        mock_df.iloc.__getitem__.side_effect = lambda key: {
            0: torch.tensor([10, 20, 30, 40, 50]),
            1: torch.tensor([1, 2, 3, 4, 5]),
            2: torch.tensor([0, 1, 2, 0, 1]),
            3: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        }.get(key, None)
        
        # Mock cupy.asarray to return numpy arrays
        mock_cupy.asarray.side_effect = lambda x: x.numpy()
        
        yield mock_cudf, mock_cupy, mock_df

# Mock os.path.exists
@pytest.fixture
def mock_os_path_exists():
    with patch("tg_gnn.os.path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

# Mock torch.as_tensor to behave normally
@pytest.fixture
def mock_torch_as_tensor():
    with patch("tg_gnn.torch.as_tensor") as mock_as_tensor:
        mock_as_tensor.side_effect = torch.as_tensor
        yield mock_as_tensor

# Mock torch.cuda.empty_cache
@pytest.fixture
def mock_cuda_empty_cache():
    with patch("tg_gnn.torch.cuda.empty_cache") as mock_empty_cache:
        yield mock_empty_cache

def test_load_tg_data_single_node_single_edge(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata for single node and single edge
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "node", "label": True, "split": True}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects", "label": False, "split": False}
        ]
    }
    
    # Configure mock DataFrame for node CSV
    mock_df.shape = (5, 4)  # 5 nodes, columns: id, label, split, feature
    mock_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([10, 20, 30, 40, 50]),
        1: torch.tensor([0, 1, 0, 1, 0]),
        2: torch.tensor([0, 1, 2, 0, 1]),
        3: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
    }.get(key, None)
    
    # Configure mock DataFrame for edge CSV
    # Assuming edge CSV has columns: src, dst
    edge_df = MagicMock()
    edge_df.shape = (3, 2)  # 3 edges, columns: src, dst
    edge_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0, 1, 2]),
        1: torch.tensor([1, 2, 0]),
    }.get(key, None)
    mock_cudf.read_csv.side_effect = [mock_df, edge_df]
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions for node data
    assert isinstance(data, Data), "Data should be a homogeneous Data object"
    assert torch.equal(data.node_ids, torch.tensor([10, 20, 30, 40, 50])), "Node IDs mismatch"
    assert torch.equal(data.y, torch.tensor([0, 1, 0, 1, 0])), "Node labels mismatch"
    assert torch.equal(data.train_mask, torch.tensor([True, False, False, True, False])), "Train mask mismatch"
    assert torch.equal(data.val_mask, torch.tensor([False, True, False, False, True])), "Val mask mismatch"
    assert torch.equal(data.test_mask, torch.tensor([False, False, True, False, False])), "Test mask mismatch"
    assert torch.equal(data.x, torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])), "Node features mismatch"
    
    # Assertions for edge data
    assert torch.equal(data.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]])), "Edge indices mismatch"
    
    # Ensure renumber_data was called
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_multiple_nodes_multiple_edges(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata for multiple nodes and edges
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "user", "label": True, "split": True},
            {"vertex_name": "item", "label": False, "split": True}
        ],
        "edges": [
            {"src": "user", "dst": "item", "rel_name": "buys", "label": True, "split": True},
            {"src": "item", "dst": "user", "rel_name": "sold_to", "label": False, "split": False}
        ]
    }
    
    # Configure mock DataFrames for nodes and edges
    user_df = MagicMock()
    user_df.shape = (3, 4)  # 3 users
    user_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([100, 200, 300]),
        1: torch.tensor([1, 0, 1]),
        2: torch.tensor([0, 1, 2]),
        3: torch.tensor([0.5, 0.6, 0.7]),
    }.get(key, None)
    
    item_df = MagicMock()
    item_df.shape = (2, 3)  # 2 items, no labels
    item_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([400, 500]),
        1: torch.tensor([0, 1]),
        2: torch.tensor([0.8, 0.9]),
    }.get(key, None)
    
    buys_df = MagicMock()
    buys_df.shape = (2, 3)  # 2 edges, columns: src, dst, label
    buys_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0, 1]),
        1: torch.tensor([1, 0]),
        2: torch.tensor([1, 0]),
    }.get(key, None)
    
    sold_to_df = MagicMock()
    sold_to_df.shape = (1, 2)  # 1 edge, columns: src, dst
    sold_to_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0]),
        1: torch.tensor([1]),
    }.get(key, None)
    
    mock_cudf.read_csv.side_effect = [user_df, item_df, buys_df, sold_to_df]
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions for node data
    assert isinstance(data, HeteroData), "Data should be a heterogeneous HeteroData object"
    
    # Check 'user' node data
    assert torch.equal(data['user'].node_ids, torch.tensor([100, 200, 300])), "User node IDs mismatch"
    assert torch.equal(data['user'].y, torch.tensor([1, 0, 1])), "User labels mismatch"
    assert torch.equal(data['user'].train_mask, torch.tensor([True, False, False])), "User train mask mismatch"
    assert torch.equal(data['user'].val_mask, torch.tensor([False, True, False])), "User val mask mismatch"
    assert torch.equal(data['user'].test_mask, torch.tensor([False, False, True])), "User test mask mismatch"
    assert torch.equal(data['user'].x, torch.tensor([[0.5], [0.6], [0.7]])), "User features mismatch"
    
    # Check 'item' node data
    assert torch.equal(data['item'].node_ids, torch.tensor([400, 500])), "Item node IDs mismatch"
    assert not hasattr(data['item'], 'y'), "Item should not have labels"
    assert torch.equal(data['item'].train_mask, torch.tensor([True, False])), "Item train mask mismatch"
    assert torch.equal(data['item'].val_mask, torch.tensor([False, True])), "Item val mask mismatch"
    assert torch.equal(data['item'].test_mask, torch.tensor([False, False])), "Item test mask mismatch"
    assert torch.equal(data['item'].x, torch.tensor([[0.8], [0.9]])), "Item features mismatch"
    
    # Check 'buys' edge data
    assert ('user', 'buys', 'item') in data.edge_types, "'buys' edge type missing"
    buys_edge = data['user', 'buys', 'item']
    assert torch.equal(buys_edge.edge_index, torch.tensor([[0, 1], [1, 0]])), "Buys edge indices mismatch"
    assert torch.equal(buys_edge.edge_label, torch.tensor([1, 0])), "Buys edge labels mismatch"
    assert torch.equal(buys_edge.train_mask, torch.tensor([True, False])), "Buys edge train mask mismatch"
    assert torch.equal(buys_edge.val_mask, torch.tensor([False, True])), "Buys edge val mask mismatch"
    assert torch.equal(buys_edge.test_mask, torch.tensor([False, False])), "Buys edge test mask mismatch"
    
    # Check 'sold_to' edge data
    assert ('item', 'sold_to', 'user') in data.edge_types, "'sold_to' edge type missing"
    sold_to_edge = data['item', 'sold_to', 'user']
    assert torch.equal(sold_to_edge.edge_index, torch.tensor([[0], [1]])), "Sold_to edge indices mismatch"
    assert not hasattr(sold_to_edge, 'edge_label'), "Sold_to should not have labels"
    
    # Ensure renumber_data was called
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_missing_node_file(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata with one node and one edge
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "node", "label": True, "split": False}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects", "label": False, "split": False}
        ]
    }
    
    # Configure os.path.exists to return False for node file
    mock_os_path_exists.side_effect = lambda path: not path.endswith("node_p0.csv")  # Node file missing
    
    # Configure mock DataFrame for edge CSV
    edge_df = MagicMock()
    edge_df.shape = (2, 2)  # 2 edges
    edge_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0, 1]),
        1: torch.tensor([1, 0]),
    }.get(key, None)
    mock_cudf.read_csv.side_effect = [edge_df]
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions
    assert isinstance(data, Data), "Data should be a homogeneous Data object"
    assert not hasattr(data, 'node_ids'), "Node data should be missing"
    assert torch.equal(data.edge_index, torch.tensor([[0, 1], [1, 0]])), "Edge indices mismatch"
    
    # Ensure renumber_data was called even if node data is missing
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_missing_edge_file(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata with one node and one edge
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "node", "label": False, "split": False}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects", "label": False, "split": False}
        ]
    }
    
    # Configure os.path.exists to return False for edge file
    mock_os_path_exists.side_effect = lambda path: not path.endswith("node_connects_node_p0.csv")  # Edge file missing
    
    # Configure mock DataFrame for node CSV
    node_df = MagicMock()
    node_df.shape = (3, 3)  # 3 nodes, columns: id, feature1, feature2
    node_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([10, 20, 30]),
        1: torch.tensor([0.1, 0.2, 0.3]),
        2: torch.tensor([0.4, 0.5, 0.6]),
    }.get(key, None)
    mock_cudf.read_csv.side_effect = [node_df]
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions
    assert isinstance(data, Data), "Data should be a homogeneous Data object"
    assert torch.equal(data.node_ids, torch.tensor([10, 20, 30])), "Node IDs mismatch"
    assert torch.equal(data.x, torch.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])), "Node features mismatch"
    assert not hasattr(data, 'edge_index'), "Edge data should be missing"
    
    # Ensure renumber_data was called even if edge data is missing
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_with_renumbering_disabled(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata for single node and single edge
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "node", "label": False, "split": False}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects", "label": False, "split": False}
        ]
    }
    
    # Configure mock DataFrame for node CSV
    node_df = MagicMock()
    node_df.shape = (2, 2)  # 2 nodes, columns: id, feature
    node_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([100, 200]),
        1: torch.tensor([0.1, 0.2]),
    }.get(key, None)
    
    # Configure mock DataFrame for edge CSV
    edge_df = MagicMock()
    edge_df.shape = (1, 2)  # 1 edge
    edge_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0]),
        1: torch.tensor([1]),
    }.get(key, None)
    
    mock_cudf.read_csv.side_effect = [node_df, edge_df]
    
    # Call load_tg_data with renumber=False
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=False
    )
    
    # Assertions for node data
    assert isinstance(data, Data), "Data should be a homogeneous Data object"
    assert torch.equal(data.node_ids, torch.tensor([100, 200])), "Node IDs mismatch"
    assert torch.equal(data.x, torch.tensor([[0.1], [0.2]])), "Node features mismatch"
    
    # Assertions for edge data
    assert torch.equal(data.edge_index, torch.tensor([[0], [1]])), "Edge indices mismatch"
    
    # Ensure renumber_data was NOT called
    mock_renumber_data.assert_not_called()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_heterogeneous_missing_edge_type(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata with multiple nodes and edges, but one edge type missing
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [
            {"vertex_name": "user", "label": False, "split": False},
            {"vertex_name": "item", "label": False, "split": False}
        ],
        "edges": [
            {"src": "user", "dst": "item", "rel_name": "buys", "label": False, "split": False},
            {"src": "item", "dst": "user", "rel_name": "sold_to", "label": False, "split": False}
        ]
    }
    
    # Configure os.path.exists to return False for 'item_sold_to_user_p0.csv'
    def exists_side_effect(path):
        if "item_sold_to_user_p0.csv" in path:
            return False
        return True
    mock_os_path_exists.side_effect = exists_side_effect
    
    # Configure mock DataFrames for existing files
    user_df = MagicMock()
    user_df.shape = (2, 2)  # 2 users
    user_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([1, 2]),
        1: torch.tensor([0.1, 0.2]),
    }.get(key, None)
    
    item_df = MagicMock()
    item_df.shape = (3, 2)  # 3 items
    item_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([10, 20, 30]),
        1: torch.tensor([0.3, 0.4, 0.5]),
    }.get(key, None)
    
    buys_df = MagicMock()
    buys_df.shape = (2, 2)  # 2 buys edges
    buys_df.iloc.__getitem__.side_effect = lambda key: {
        0: torch.tensor([0, 1]),
        1: torch.tensor([1, 2]),
    }.get(key, None)
    
    mock_cudf.read_csv.side_effect = [user_df, item_df, buys_df]
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions
    assert isinstance(data, HeteroData), "Data should be a heterogeneous HeteroData object"
    
    # Check 'user' node data
    assert torch.equal(data['user'].node_ids, torch.tensor([1, 2])), "User node IDs mismatch"
    assert torch.equal(data['user'].x, torch.tensor([[0.1], [0.2]])), "User features mismatch"
    
    # Check 'item' node data
    assert torch.equal(data['item'].node_ids, torch.tensor([10, 20, 30])), "Item node IDs mismatch"
    assert torch.equal(data['item'].x, torch.tensor([[0.3], [0.4], [0.5]])), "Item features mismatch"
    
    # Check 'buys' edge data
    assert ('user', 'buys', 'item') in data.edge_types, "'buys' edge type missing"
    buys_edge = data['user', 'buys', 'item']
    assert torch.equal(buys_edge.edge_index, torch.tensor([[0, 1], [1, 2]])), "Buys edge indices mismatch"
    
    # 'sold_to' edge type should be missing
    assert ('item', 'sold_to', 'user') not in data.edge_types, "'sold_to' edge type should be missing"
    
    # Ensure renumber_data was called
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_tg_data_no_nodes_no_edges(
    mock_renumber_data, mock_cudf, mock_os_path_exists, mock_torch_as_tensor, mock_cuda_empty_cache
):
    mock_cudf, mock_cupy, mock_df = mock_cudf
    
    # Define metadata with no nodes and no edges
    metadata = {
        "data_dir": "/fake_dir",
        "nodes": [],
        "edges": []
    }
    
    # Call load_tg_data
    from tg_gnn import load_tg_data  # Adjust import as necessary
    data = load_tg_data(
        metadata=metadata,
        local_rank=0,
        world_size=1,
        renumber=True
    )
    
    # Assertions
    assert isinstance(data, Data), "Data should be a homogeneous Data object by default"
    assert len(data) == 0, "Data object should be empty"
    
    # Ensure renumber_data was called
    mock_renumber_data.assert_called_once()
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()
