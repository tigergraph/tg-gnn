import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import Data, HeteroData
from tg_gnn import load_partitioned_data

@pytest.fixture
def mock_load_tg_data():
    with patch("tg_gnn.load_tg_data") as mock_tg_data:
        yield mock_tg_data

@pytest.fixture
def mock_cugraph_pyg():
    with patch("tg_gnn.cugraph_pyg.data.GraphStore") as MockGraphStore, \
         patch("tg_gnn.cugraph_pyg.data.WholeFeatureStore") as MockWholeFeatureStore:
        
        # Create mock instances
        mock_graph_store = MagicMock()
        mock_whole_feature_store = MagicMock()
        
        MockGraphStore.return_value = mock_graph_store
        MockWholeFeatureStore.return_value = mock_whole_feature_store
        
        yield MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_whole_feature_store

@pytest.fixture
def mock_cuda_empty_cache():
    with patch("tg_gnn.torch.cuda.empty_cache") as mock_empty_cache:
        yield mock_empty_cache

def test_load_partitioned_data_homogeneous_all_attributes(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object with all attributes
    mock_data = Data()
    mock_data.node_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    mock_data.x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    mock_data.y = torch.tensor([0, 1, 0], dtype=torch.long)
    mock_data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)
    mock_data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata for a homogeneous graph
    metadata = {
        "nodes": [
            {"vertex_name": "node", "features_list": ["x"], "label": "y", "split": "split"}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=0,
        wg_mem_type="shared",
        world_size=1
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    feature_store.__setitem__.assert_any_call(("node", "x"), mock_data.x)
    feature_store.__setitem__.assert_any_call(("node", "y"), mock_data.y)
    
    # Assertions for Split Indices
    expected_split_idx = {
        "train": torch.tensor([10]),
        "val": torch.tensor([20]),
        "test": torch.tensor([30])
    }
    assert split_idx == expected_split_idx, "Split indices mismatch"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 0, 1, renumber=True)
    
    # Ensure empty_cache was called
    mock_cuda_empty_cache.assert_called_once()

def test_load_partitioned_data_heterogeneous_with_edge_attr(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock HeteroData object with edge attributes
    mock_data = HeteroData()
    
    # Node types
    mock_data['user'].node_ids = torch.tensor([100, 200], dtype=torch.long)
    mock_data['user'].x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    mock_data['user'].y = torch.tensor([0, 1], dtype=torch.long)
    mock_data['user'].train_mask = torch.tensor([True, False], dtype=torch.bool)
    mock_data['user'].val_mask = torch.tensor([False, True], dtype=torch.bool)
    mock_data['user'].test_mask = torch.tensor([False, False], dtype=torch.bool)
    
    mock_data['item'].node_ids = torch.tensor([300, 400, 500], dtype=torch.long)
    mock_data['item'].x = torch.tensor([[3.0], [4.0], [5.0]], dtype=torch.float32)
    mock_data['item'].train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_data['item'].val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_data['item'].test_mask = torch.tensor([False, False, True], dtype=torch.bool)
    
    # Edge types
    rel_buys = ('user', 'buys', 'item')
    rel_sold_to = ('item', 'sold_to', 'user')
    
    mock_data[rel_buys].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    mock_data[rel_buys].edge_attr = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    
    mock_data[rel_sold_to].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Configure load_tg_data to return the mock HeteroData object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata for a heterogeneous graph
    metadata = {
        "nodes": [
            {"vertex_name": "user", "features_list": ["x"], "label": "y", "split": "split"},
            {"vertex_name": "item", "features_list": ["x"], "split": "split"}
        ],
        "edges": [
            {"src": "user", "dst": "item", "rel_name": "buys"},
            {"src": "item", "dst": "user", "rel_name": "sold_to"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=1,
        wg_mem_type="device",
        world_size=2
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    feature_store.__setitem__.assert_any_call(("user", "x"), mock_data['user'].x)
    feature_store.__setitem__.assert_any_call(("user", "y"), mock_data['user'].y)
    feature_store.__setitem__.assert_any_call(("item", "x"), mock_data['item'].x)
    feature_store.__setitem__.assert_any_call((rel_buys, "edge_attr"), mock_data[rel_buys].edge_attr)
    
    # Assertions for Split Indices
    expected_split_idx = {
        "train": torch.tensor([100, 300]),
        "val": torch.tensor([200, 400]),
        "test": torch.tensor([500])
    }
    assert split_idx == expected_split_idx, "Split indices mismatch for heterogeneous data"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(rel_buys, "coo", False, mock_data[rel_buys].edge_index)
    graph_store.__setitem__.assert_any_call(rel_sold_to, "coo", False, mock_data[rel_sold_to].edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 1, 2, renumber=True)
    
    # Ensure empty_cache was called twice (once for nodes, once for edges)
    assert mock_cuda_empty_cache.call_count == 2, "empty_cache should be called twice for nodes and edges"

def test_load_partitioned_data_missing_features_and_labels(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object without features and labels
    mock_data = Data()
    mock_data.node_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    # No 'x' or 'y'
    mock_data.train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_data.val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_data.test_mask = torch.tensor([False, False, True], dtype=torch.bool)
    mock_data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata without features and labels
    metadata = {
        "nodes": [
            {"vertex_name": "node", "features_list": [], "label": "", "split": "split"}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=0,
        wg_mem_type="shared",
        world_size=1
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    # No features or labels should be set
    feature_store.__setitem__.assert_not_called()
    
    # Assertions for Split Indices
    expected_split_idx = {
        "train": torch.tensor([1]),
        "val": torch.tensor([2]),
        "test": torch.tensor([3])
    }
    assert split_idx == expected_split_idx, "Split indices mismatch when features and labels are missing"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 0, 1, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()

def test_load_partitioned_data_with_edge_features(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object with edge attributes
    mock_data = Data()
    mock_data.node_ids = torch.tensor([1, 2], dtype=torch.long)
    mock_data.x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    mock_data.y = torch.tensor([0, 1], dtype=torch.long)
    mock_data.train_mask = torch.tensor([True, False], dtype=torch.bool)
    mock_data.val_mask = torch.tensor([False, True], dtype=torch.bool)
    mock_data.test_mask = torch.tensor([False, False], dtype=torch.bool)
    mock_data.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    mock_data.edge_attr = torch.tensor([[0.5]], dtype=torch.float32)
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata for a homogeneous graph with edge attributes
    metadata = {
        "nodes": [
            {"vertex_name": "node", "features_list": ["x"], "label": "y", "split": "split"}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=0,
        wg_mem_type="device",
        world_size=1
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    feature_store.__setitem__.assert_any_call(("node", "x"), mock_data.x)
    feature_store.__setitem__.assert_any_call(("node", "y"), mock_data.y)
    feature_store.__setitem__.assert_any_call((("node", "connects", "node"), "edge_attr"), mock_data.edge_attr)
    
    # Assertions for Split Indices
    expected_split_idx = {
        "train": torch.tensor([1]),
        "val": torch.tensor([2]),
        "test": torch.tensor([0])  # Note: node_ids[2] = 3, but mock_data.node_ids = [1,2]
    }
    # Correction: In mock_data.node_ids = [1,2], test_mask is [False, False], so test should be empty
    expected_split_idx = {
        "train": torch.tensor([1]),
        "val": torch.tensor([2]),
        "test": torch.tensor([], dtype=torch.long)
    }
    assert split_idx == expected_split_idx, "Split indices mismatch with edge attributes"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 0, 1, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()

def test_load_partitioned_data_no_splits(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object without split masks
    mock_data = Data()
    mock_data.node_ids = torch.tensor([5, 10], dtype=torch.long)
    mock_data.x = torch.tensor([[5.0], [10.0]], dtype=torch.float32)
    mock_data.y = torch.tensor([1, 0], dtype=torch.long)
    # No train_mask, val_mask, test_mask
    mock_data.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata without splits
    metadata = {
        "nodes": [
            {"vertex_name": "node", "features_list": ["x"], "label": "y", "split": ""}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=2,
        wg_mem_type="shared",
        world_size=4
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    feature_store.__setitem__.assert_any_call(("node", "x"), mock_data.x)
    feature_store.__setitem__.assert_any_call(("node", "y"), mock_data.y)
    
    # Assertions for Split Indices
    expected_split_idx = {}
    assert split_idx == expected_split_idx, "Split indices should be empty when no splits are defined"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 2, 4, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()

def test_load_partitioned_data_empty_metadata(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object with no nodes and no edges
    mock_data = Data()
    # No node_ids, no features, no labels, no splits, no edge_index
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define empty metadata
    metadata = {
        "nodes": [],
        "edges": []
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=0,
        wg_mem_type="device",
        world_size=1
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    feature_store.__setitem__.assert_not_called()
    
    # Assertions for Split Indices
    expected_split_idx = {}
    assert split_idx == expected_split_idx, "Split indices should be empty for empty metadata"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_not_called()
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 0, 1, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()

def test_load_partitioned_data_with_missing_node_attributes(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock HeteroData object with some missing attributes
    mock_data = HeteroData()
    
    # Node types
    mock_data['user'].node_ids = torch.tensor([1, 2], dtype=torch.long)
    # No 'x' or 'y' for 'user'
    mock_data['user'].train_mask = torch.tensor([True, False], dtype=torch.bool)
    mock_data['user'].val_mask = torch.tensor([False, True], dtype=torch.bool)
    mock_data['user'].test_mask = torch.tensor([False, False], dtype=torch.bool)
    
    mock_data['item'].node_ids = torch.tensor([3, 4, 5], dtype=torch.long)
    mock_data['item'].x = torch.tensor([[3.0], [4.0], [5.0]], dtype=torch.float32)
    # No 'y' for 'item'
    mock_data['item'].train_mask = torch.tensor([True, False, False], dtype=torch.bool)
    mock_data['item'].val_mask = torch.tensor([False, True, False], dtype=torch.bool)
    mock_data['item'].test_mask = torch.tensor([False, False, True], dtype=torch.bool)
    
    # Edge types
    rel_follows = ('user', 'follows', 'user')
    rel_rated = ('user', 'rated', 'item')
    
    mock_data[rel_follows].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    # No edge_attr for 'follows'
    
    mock_data[rel_rated].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    mock_data[rel_rated].edge_attr = torch.tensor([[0.7], [0.8]], dtype=torch.float32)
    
    # Configure load_tg_data to return the mock HeteroData object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata with some missing attributes
    metadata = {
        "nodes": [
            {"vertex_name": "user", "features_list": [], "label": "", "split": "split"},
            {"vertex_name": "item", "features_list": ["x"], "label": "", "split": "split"}
        ],
        "edges": [
            {"src": "user", "dst": "user", "rel_name": "follows"},
            {"src": "user", "dst": "item", "rel_name": "rated"}
        ]
    }
    
    # Call load_partitioned_data
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph, split_idx = load_partitioned_data(
        metadata=metadata,
        local_rank=3,
        wg_mem_type="shared",
        world_size=4
    )
    
    feature_store, graph_store = feature_graph
    
    # Assertions for FeatureStore
    # 'user' has no features or labels
    # 'item' has features but no labels
    feature_store.__setitem__.assert_any_call(("item", "x"), mock_data['item'].x)
    feature_store.__setitem__.assert_any_call((("user", "rated", "item"), "edge_attr"), mock_data[rel_rated].edge_attr)
    
    # 'user' has no 'x' or 'y', so no corresponding feature_store entries
    
    # Assertions for Split Indices
    expected_split_idx = {
        "train": torch.tensor([1, 3]),
        "val": torch.tensor([2, 4]),
        "test": torch.tensor([5])
    }
    assert split_idx == expected_split_idx, "Split indices mismatch with some missing node attributes"
    
    # Assertions for GraphStore
    graph_store.__setitem__.assert_any_call(rel_follows, "coo", False, mock_data[rel_follows].edge_index)
    graph_store.__setitem__.assert_any_call(rel_rated, "coo", False, mock_data[rel_rated].edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_once_with(metadata, 3, 4, renumber=True)
    
    # Ensure empty_cache was called twice (once for nodes, once for edges)
    assert mock_cuda_empty_cache.call_count == 2, "empty_cache should be called twice for nodes and edges"

def test_load_partitioned_data_with_different_memory_types(
    mock_load_tg_data, mock_cugraph_pyg, mock_cuda_empty_cache
):
    MockGraphStore, MockWholeFeatureStore, mock_graph_store, mock_feature_store = mock_cugraph_pyg
    
    # Create a mock Data object with minimal attributes
    mock_data = Data()
    mock_data.node_ids = torch.tensor([7, 8], dtype=torch.long)
    mock_data.x = torch.tensor([[7.0], [8.0]], dtype=torch.float32)
    mock_data.y = torch.tensor([1, 0], dtype=torch.long)
    mock_data.train_mask = torch.tensor([True, False], dtype=torch.bool)
    mock_data.val_mask = torch.tensor([False, True], dtype=torch.bool)
    mock_data.test_mask = torch.tensor([False, False], dtype=torch.bool)
    mock_data.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Configure load_tg_data to return the mock Data object
    mock_load_tg_data.return_value = mock_data
    
    # Define metadata for a homogeneous graph
    metadata = {
        "nodes": [
            {"vertex_name": "node", "features_list": ["x"], "label": "y", "split": "split"}
        ],
        "edges": [
            {"src": "node", "dst": "node", "rel_name": "connects"}
        ]
    }
    
    # Test with wg_mem_type="shared"
    from tg_gnn import load_partitioned_data  # Adjust import as necessary
    feature_graph_shared, split_idx_shared = load_partitioned_data(
        metadata=metadata,
        local_rank=0,
        wg_mem_type="shared",
        world_size=1
    )
    
    feature_store_shared, graph_store_shared = feature_graph_shared
    
    # Assertions for Shared Memory Type
    feature_store_shared.__setitem__.assert_any_call(("node", "x"), mock_data.x)
    feature_store_shared.__setitem__.assert_any_call(("node", "y"), mock_data.y)
    graph_store_shared.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_with(metadata, 0, 1, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()
    
    # Reset mocks for next memory type
    mock_load_tg_data.reset_mock()
    mock_cugraph_pyg[2].reset_mock()
    mock_cugraph_pyg[3].reset_mock()
    mock_cuda_empty_cache.reset_mock()
    
    # Test with wg_mem_type="device"
    feature_graph_device, split_idx_device = load_partitioned_data(
        metadata=metadata,
        local_rank=1,
        wg_mem_type="device",
        world_size=2
    )
    
    feature_store_device, graph_store_device = feature_graph_device
    
    # Assertions for Device Memory Type
    feature_store_device.__setitem__.assert_any_call(("node", "x"), mock_data.x)
    feature_store_device.__setitem__.assert_any_call(("node", "y"), mock_data.y)
    graph_store_device.__setitem__.assert_any_call(("node", "connects", "node"), "coo", False, mock_data.edge_index)
    
    # Ensure renumber_data was called within load_tg_data
    mock_load_tg_data.assert_called_with(metadata, 1, 2, renumber=True)
    
    # Ensure empty_cache was called once
    mock_cuda_empty_cache.assert_called_once()

