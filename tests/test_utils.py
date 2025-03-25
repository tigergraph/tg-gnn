import pytest
from unittest import mock
import torch
from torch_geometric.data import Data, HeteroData
from tg_gnn.utils import renumber_data, _gather_files, _assign_files_to_rank, _check_nested, get_assigned_files, load_csv, read_file
import torch.distributed as dist
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from pytest_mock import MockerFixture 
from unittest.mock import MagicMock
import cudf
import pandas as pd


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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


def test_gather_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "product_p1.csv").touch()
        Path(tmpdir, "product_p2.csv").touch()
        data_path = Path(tmpdir)
        files = sorted(str(p) for p in data_path.glob("product*"))
        print(files)
        files = _gather_files(tmpdir, "product_p*.csv")
        assert len(files) == 2
        assert all(f.endswith('.csv') for f in files)
        

        nested_dir = Path(tmpdir, "nested")
        nested_dir.mkdir()

        # Test nested directory
        shutil.move(Path(tmpdir, "product_p1.csv"), nested_dir)
        shutil.move(Path(tmpdir, "product_p2.csv"), nested_dir)

        files = _gather_files(tmpdir, "product_p*.csv")
        assert len(files) == 2, "Should detect nested structure correctly"
        
        
def test_assign_files_to_rank():
    files = [f"file_{i}.csv" for i in range(8)]
    assigned_rank_0 = _assign_files_to_rank(files, rank=0, world_size=4)
    assert assigned_rank_0 == ['file_0.csv', 'file_4.csv'], "Rank 0 assignment incorrect"

    assigned_rank_3 = _assign_files_to_rank(files, 3, 4)
    assert assigned_rank_3 == ['file_3.csv', 'file_7.csv'], "Rank 3 assignment incorrect"
    
    
def test_check_nested_with_subdirs(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    assert _check_nested(tmp_path) is True


def test_check_nested_with_files(tmp_path: Path):
    (tmp_path / "file.txt").touch()
    assert _check_nested(tmp_path) is False


def test_gather_files_nested(tmp_path: Path):
    (tmp_path / "subdir1").mkdir()
    (tmp_path / "subdir1" / "file1.txt").touch()
    (tmp_path / "subdir2").mkdir()
    (tmp_path / "subdir2" / "file2.txt").touch()

    files = _gather_files(str(tmp_path), "*.txt")
    assert len(files) == 2
    assert all(f.endswith(".txt") for f in files)
    assert sorted(files) == [
        str(tmp_path / "subdir1" / "file1.txt"),
        str(tmp_path / "subdir2" / "file2.txt"),
    ]


def test_gather_files_flat(tmp_path: Path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    files = _gather_files(str(tmp_path), "*.txt")
    assert sorted(files) == [
        str(tmp_path / "file1.txt"),
        str(tmp_path / "file2.txt"),
    ]


def test_assign_files_even_distribution():
    files = ["f0", "f1", "f2", "f3"]
    assert _assign_files_to_rank(files, 0, 2) == ["f0", "f2"]
    assert _assign_files_to_rank(files, 1, 2) == ["f1", "f3"]


def test_assign_files_uneven_distribution():
    files = ["f0", "f1", "f2", "f3", "f4"]
    assert _assign_files_to_rank(files, 0, 2) == ["f0", "f2", "f4"]
    assert _assign_files_to_rank(files, 1, 2) == ["f1", "f3"]


def test_get_assigned_files_nested(mocker: MockerFixture):
    mocker.patch("tg_gnn.utils._check_nested", return_value=True)
    mocker.patch("tg_gnn.utils.dist.get_global_rank", return_value=1)
    mocker.patch("tg_gnn.utils.dist.get_world_size", return_value=3)
    mocker.patch(
        "tg_gnn.utils._gather_files",
        return_value=["f0", "f1", "f2", "f3", "f4", "f5"],
    )
    assigned = get_assigned_files("data_dir", "*.txt")
    assert assigned == ["f1", "f4"]


def test_get_assigned_files_flat(mocker: MockerFixture):
    mocker.patch("tg_gnn.utils._check_nested", return_value=False)
    mocker.patch("tg_gnn.utils.get_local_rank", return_value=0)
    mocker.patch("tg_gnn.utils.get_local_world_size", return_value=2)
    mocker.patch("tg_gnn.utils._gather_files", return_value=["f0", "f1", "f2", "f3"])

    assigned = get_assigned_files("data_dir", "*.txt")
    assert assigned == ["f0", "f2"]


def test_get_assigned_files_ignore_passed_args(mocker: MockerFixture):
    mocker.patch("tg_gnn.utils._check_nested", return_value=True)
    mocker.patch("torch.distributed.get_global_rank", return_value=0)
    mocker.patch("torch.distributed.get_world_size", return_value=2)
    mocker.patch("tg_gnn.utils._gather_files", return_value=["f0", "f1", "f2", "f3"])

    assigned = get_assigned_files("data_dir", "*.txt", rank=5, world_size=10)
    assert assigned == ["f0", "f2"]
    
    
def test_read_file_calls_cudf_read_csv(mocker):
    """Test that read_file correctly calls cudf.read_csv"""
    mock_read = mocker.patch("cudf.read_csv")
    test_path = "dummy.csv"
    
    result = read_file(test_path)
    
    mock_read.assert_called_once_with(test_path)
    assert result == mock_read.return_value

def test_load_csv_with_multiple_files(tmp_path):
    """Test load_csv with multiple files using ThreadPool"""
    # Create two temporary CSV files with sample data
    csv_content1 = "a,b\n1,2\n3,4"
    csv_content2 = "a,b\n5,6\n7,8"
    
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    file1.write_text(csv_content1)
    file2.write_text(csv_content2)
    
    file_paths = [str(file1), str(file2)]
    
    result = load_csv(file_paths)
    
    expected = cudf.DataFrame({
        "a": [1, 3, 5, 7],
        "b": [2, 4, 6, 8]
    })
    
    pd.testing.assert_frame_equal(result.to_pandas(), expected.to_pandas())

def test_load_csv_single_file(tmp_path):
    csv_content = "a,b\n1,2\n3,4"
    file = tmp_path / "single.csv"
    file.write_text(csv_content)
    
    result = load_csv([str(file)])
    
    expected = cudf.DataFrame({
        "a": [1, 3],
        "b": [2, 4]
    })
    
    pd.testing.assert_frame_equal(
        result.to_pandas(),
        expected.to_pandas()
    )

def test_load_csv_empty_files():
    with pytest.raises(ValueError):
        load_csv([])