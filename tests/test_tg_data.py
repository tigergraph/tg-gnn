import pytest
import os
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from unittest.mock import patch, MagicMock
from tg_gnn.tg_data import load_tg_data

@pytest.fixture
def tmp_data_dir(tmpdir):
    return tmpdir.mkdir("data")

def test_load_single_node_with_features_labels_splits(tmp_data_dir):
    """Test loading a single node type with features, labels, and splits into Data."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {
            "user": {
                "vertex_name": "user",
                "label": True,
                "split": True,
            }
        },
        "edges": {}
    }
    node_file = tmp_data_dir / "user_p0.csv"
    df = pd.DataFrame([
        [0, 1, 0, 0.1, 0.2],
        [1, 2, 1, 0.3, 0.4],
    ])
    df.to_csv(node_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    
    assert isinstance(data, Data)
    assert torch.equal(data.node_ids, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(data.x, torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32))
    assert torch.equal(data.y, torch.tensor([1, 2], dtype=torch.long))
    assert data.train_mask.tolist() == [True, False]
    assert data.val_mask.tolist() == [False, True]
    assert data.test_mask.tolist() == [False, False]

def test_hetero_data_multiple_node_types(tmp_data_dir):
    """Test HeteroData creation with multiple node types."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {
            "user": {"vertex_name": "user"},
            "item": {"vertex_name": "item"}
        },
        "edges": {}
    }
    user_file = tmp_data_dir / "user_p0.csv"
    pd.DataFrame([[0]]).to_csv(user_file, index=False, header=False)
    item_file = tmp_data_dir / "item_p0.csv"
    pd.DataFrame([[0]]).to_csv(item_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    
    assert isinstance(data, HeteroData)
    assert 'user' in data.node_types
    assert 'item' in data.node_types
    assert torch.equal(data['user'].node_ids, torch.tensor([0], dtype=torch.long))
    assert torch.equal(data['item'].node_ids, torch.tensor([0], dtype=torch.long))

def test_edge_loading_with_attributes(tmp_data_dir):
    """Test edge loading with indices, labels, splits, and features into Data."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {"user": {"vertex_name": "user"}},
        "edges": {
            "follows": {
                "src": "user",
                "dst": "user",
                "label": True,
                "split": True,
            }
        }
    }
    node_file = tmp_data_dir / "user_p0.csv"
    pd.DataFrame([[0]]).to_csv(node_file, index=False, header=False)
    edge_file = tmp_data_dir / "user_follows_user_p0.csv"
    df = pd.DataFrame([
        [0, 1, 5, 0, 0.1, 0.2],
        [1, 2, 6, 1, 0.3, 0.4],
    ])
    df.to_csv(edge_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    
    assert isinstance(data, Data)
    assert data.edge_index.shape == (2, 2)
    assert torch.equal(data.edge_label, torch.tensor([5, 6], dtype=torch.long))
    assert data.edge_attr.shape == (2, 2)
    assert data.train_mask.tolist() == [True, False]
    assert data.val_mask.tolist() == [False, True]



def test_local_rank_partitioning(tmp_data_dir):
    """Test correct partitioning based on local_rank."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {"user": {"vertex_name": "user"}},
        "edges": {}
    }
    node_file = tmp_data_dir / "user_p2.csv"
    pd.DataFrame([[0]]).to_csv(node_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=2, world_size=3, renumber=False)
    
    assert isinstance(data, Data)
    assert torch.equal(data.node_ids, torch.tensor([0], dtype=torch.long))

def test_empty_metadata_handling(tmp_data_dir):
    """Test handling of empty metadata."""
    metadata = {"data_dir": str(tmp_data_dir), "nodes": {}, "edges": {}}
    data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    assert isinstance(data, Data)
    assert len(data) == 0

def test_node_without_features_or_labels(tmp_data_dir):
    """Test node loading with only IDs."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {"user": {"vertex_name": "user"}},
        "edges": {}
    }
    node_file = tmp_data_dir / "user_p0.csv"
    pd.DataFrame([[0]]).to_csv(node_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    assert isinstance(data, Data)
    assert hasattr(data, 'node_ids')
    assert data.x is None
    assert data.y is None
    assert not hasattr(data, 'train_mask')

def test_hetero_edge_types(tmp_data_dir):
    """Test HeteroData edge type handling."""
    metadata = {
        "data_dir": str(tmp_data_dir),
        "nodes": {
            "user": {"vertex_name": "user"},
            "item": {"vertex_name": "item"}
        },
        "edges": {
            "follows": {"src": "user", "dst": "user"},
            "buys": {"src": "user", "dst": "item"}
        }
    }
    user_file = tmp_data_dir / "user_p0.csv"
    pd.DataFrame([[0]]).to_csv(user_file, index=False, header=False)
    item_file = tmp_data_dir / "item_p0.csv"
    pd.DataFrame([[0]]).to_csv(item_file, index=False, header=False)
    follows_file = tmp_data_dir / "user_follows_user_p0.csv"
    pd.DataFrame([[0, 1]]).to_csv(follows_file, index=False, header=False)
    buys_file = tmp_data_dir / "user_buys_item_p0.csv"
    pd.DataFrame([[0, 1]]).to_csv(buys_file, index=False, header=False)
    
    with patch('cudf.read_csv', pd.read_csv):
        data = load_tg_data(metadata, local_rank=0, world_size=1, renumber=False)
    
    assert isinstance(data, HeteroData)
    assert ('user', 'follows', 'user') in data.edge_types
    assert ('user', 'buys', 'item') in data.edge_types
    assert data['user', 'follows', 'user'].edge_index.shape == (2, 1)
    assert data['user', 'buys', 'item'].edge_index.shape == (2, 1)