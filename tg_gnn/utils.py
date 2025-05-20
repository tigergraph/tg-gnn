import time
import os
import torch
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData
import cudf
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import logging
logger = logging.getLogger(__name__)

def timeit(func):
    """
    Decorator to measure the runtime of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' executed in {runtime:.6f} seconds.")
        return result

    return wrapper

def get_local_world_size() -> int:
    try:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except KeyError:
        logger.warning(
            "Since 'LOCAL_WORLD_SIZE' is not set."
            "Calculating the local world size using torch.cuda.device_count()"
        )
        local_world_size = torch.cuda.device_count()
    return local_world_size

def get_local_rank() -> int:
    # torchrun set the LOCAL RANK which we could use
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        try:
            return int(local_rank_env)
        except ValueError:
            raise ValueError("The LOCAL_RANK environment variable must be an integer.")

    if dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = 0

    local_world_size = get_local_world_size()

    if local_world_size == 0:
        raise RuntimeError("No GPUs are available on this node.")

    local_rank = global_rank % local_world_size
    return local_rank

@timeit
def redistribute_splits(splits: dict, target: str = "cpu") -> dict:
    """
    Gathers and redistributes data for train, val, and test splits across ranks.
    This support redistributions of node indices of shape [n] and edge indices of shape [2,n]

    Args:
        splits (dict): A dictionary with keys "train", "val", "test", and tensors as values.
    
    Returns:
        dict: A dictionary with redistributed splits for this rank.
    """

    world_size = dist.get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    splits_shuffled = {}

    logger.info("Redistributing splits across ranks...")
    for key, local_indices in splits.items():
        # Determine splits are of nodes types or edges
        if local_indices.ndim == 1:
            split_dim = gather_dim = 0
            size_dim, log_type = 0, 'node'
        elif local_indices.ndim == 2 and local_indices.size(0) == 2:
            split_dim = gather_dim = 1
            size_dim, log_type = 1, 'edge'
        else:
            raise ValueError(f"Invalid {key} tensor shape: {local_indices.shape}")

        logger.info(f"Processing {log_type} indices for {key}...")

        # Gather sizes from all ranks
        local_size = torch.tensor([local_indices.size(size_dim)], dtype=torch.int64, device=device)
        all_sizes = torch.zeros(world_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(all_sizes, local_size)

        # Create tensor buffers for gathering
        if log_type == 'node':
            all_indices_list = [torch.zeros((s.item(),), dtype=torch.int64, device=device) for s in all_sizes]
        else:
            all_indices_list = [torch.zeros((2, s.item()), dtype=torch.int64, device=device) for s in all_sizes]

        # Collect indices from all ranks
        dist.all_gather(all_indices_list, local_indices.to(device))
        
        # Combine and split indices
        combined = torch.cat(all_indices_list, dim=gather_dim).to(target)
        splits_shuffled[key] = torch.tensor_split(combined, world_size, dim=split_dim)[local_rank]

        # Cleanup resources
        del combined, local_size, all_sizes, all_indices_list
        torch.cuda.empty_cache()
        logger.info(f"Completed {log_type} {key} redistribution")

    logger.info(f"All splits redistributed successfully on {target}")
    return splits_shuffled

@timeit
def renumber_data(
    data: Data | HeteroData,
    metadata: dict,
    target: str = "cpu"
) -> Data | HeteroData:
    """
    Renumbers node indices (and subsequently edge indices) across multiple ranks
    in a distributed environment. Uses PyTorch Distributed collectives
    (`all_gather_into_tensor`) to gather node IDs from each rank and create
    a global renumbering map.
    
    Args:
        data (Data | HeteroData): The PyG data object (homogeneous or heterogeneous).
        metadata (dict): Metadata containing node and edge information:
                         {
                           "nodes": {"vertex_name": ..., ...}, ...},
                           "edges": {"rel_name":{"src": ..., "dst": ...,}, ...}
                         }

    Returns:
        Data | HeteroData: The input data object with node and edge indices renumbered.
    """

    # import after rmm re-initialization
    import cudf
    import cupy
    world_size = torch.distributed.get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    global_renumber_map = {}

    is_hetero = isinstance(data, HeteroData)

    # Process Nodes
    for vertex_name, node_meta in metadata["nodes"].items():
        logger.info(f"Renumbering the {vertex_name} ids...")
        if is_hetero and data[vertex_name] is None:
            logger.info(f"Data for '{vertex_name}' not found. Skipping...")
            continue

        # Retrieve node_ids
        node_ids = data[vertex_name].node_ids if is_hetero else data.node_ids
        if node_ids is None:
            logger.info(f"'node_ids' not found for '{vertex_name}'. Skipping...")
            continue

        local_num_nodes = len(node_ids)

        # Gather node counts from all ranks
        current_num_nodes = torch.tensor([local_num_nodes], dtype=torch.int64, device=device)
        node_offsets = torch.zeros((world_size,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(node_offsets, current_num_nodes)

        # create local map with offsets 
        map_tensor = [
            torch.zeros((2, node_offsets[i].item()), device=device, dtype=torch.int64)
            for i in range(node_offsets.numel())
        ]

        node_offsets_cum = node_offsets.cumsum(0).to(target)
        this_rank = dist.get_rank() 
        local_offset = 0 if this_rank == 0 else int(node_offsets_cum[this_rank - 1])

        local_renumber_map = torch.stack(
            [
                torch.arange(
                    local_offset, 
                    local_offset + local_num_nodes, 
                    dtype=torch.int64, 
                    device=device
                ),
                node_ids.to(device)
            ],
            dim=0
        )

        # Gather local map from all ranks 
        dist.all_gather(map_tensor, local_renumber_map)
        map_tensor = torch.cat(map_tensor, dim=1).to(target)

        global_renumber_map[vertex_name] = cudf.DataFrame(
            data={"id": cupy.asarray(map_tensor[0])},
            index=cupy.asarray(map_tensor[1])
        )

        # update the node ids
        if is_hetero:
            data[vertex_name].node_ids = local_renumber_map[0].to(target)
        else:
            data.node_ids = local_renumber_map[0].to(target)

        del map_tensor, node_offsets, node_offsets_cum, current_num_nodes, local_renumber_map
        logger.info(f"Renumbering the {vertex_name} ids completed successfully on {target}.")
 
    # Process/Renumber edges
    for rel_name, edge_meta in metadata["edges"].items():
        logger.info(f"Renumbering the {rel_name} indices...")
        src_name = edge_meta["src"]
        dst_name = edge_meta["dst"]
        rel = (src_name, rel_name, dst_name)

        if is_hetero and rel not in data.edge_types:
            print(f"Edge data for '{rel}' not found. Skipping...")
            continue

        # Retrieve edge_index
        edge_index = data[rel].edge_index if is_hetero else data.edge_index
        if edge_index is None:
            print(f"'edge_index' not found for edge type '{rel}'. Skipping...")
            continue

        # Convert to CuPy for indexing
        srcs = cupy.asarray(edge_index[0].to(target))
        dsts = cupy.asarray(edge_index[1].to(target))

        # Retrieve the renumber maps for src and dst
        #   - map is a cudf DF: old_id => new_id
        if src_name not in global_renumber_map:
            logging.info(f"Renumber map not found for '{src_name}'. Skipping edge type '{rel}'.")
            continue
        
        if dst_name not in global_renumber_map:
            logging.info(f"Renumber map not found for '{dst_name}'. Skipping edge type '{rel}'.")
            continue
        
        src_map = global_renumber_map[src_name]["id"]
        dst_map = global_renumber_map[dst_name]["id"]

        if src_map is None or dst_map is None:
            logging.info(f"There is no data to do the mapping for either '{src_name}' or '{dst_name}'")
            continue

        # Renumber edges by looking up the new ID for each old ID
        new_edge_srcs = src_map.loc[srcs].values
        new_edge_dsts = dst_map.loc[dsts].values

        new_edge_index = torch.stack(
            [
                torch.as_tensor(new_edge_srcs, dtype=torch.int64, device=device),
                torch.as_tensor(new_edge_dsts, dtype=torch.int64, device=device)
            ],
            dim=0
        )

        # Store in data
        if is_hetero:
            data[rel].edge_index = new_edge_index.to(target)
        else:
            data.edge_index = new_edge_index.to(target)

        del src_map, dst_map, srcs, dsts
        logger.info(f"Renumbering of {rel_name} indices completed successfully on {target}.")

    del global_renumber_map
    torch.cuda.empty_cache()

    return data

def _check_nested(data_path: Path) -> bool:
    """Check if data dir has nested structure."""
    return any(p.is_dir() for p in data_path.iterdir())

def get_fs_type(data_dir: str) -> str:
    """Return shared filesystem in data dir is nested."""
    if _check_nested(Path(data_dir)):
        return "shared"
    else:
        return "local"

def _gather_files(data_dir: str, pattern: str) -> List[str]:
    """Gather all files matching the pattern from nested or flat structure."""
    data_path = Path(data_dir)
    files = []

    if _check_nested(data_path):
        # Nested structure for shared filesytem
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                files.extend(sorted(str(p) for p in subdir.glob(pattern)))
    else:
        # flat structure for local filesystem
        files = sorted(str(p) for p in data_path.glob(pattern))

    return files


def _assign_files_to_rank(files: List[str], rank: int, world_size: int) -> List[str]:
    """
    Evenly assigns files to GPU ranks. 
    For local rank use local world size and for global use global world size.
    """
    return files[rank::world_size]


def get_assigned_files(
    data_dir: str,
    pattern: str,
    fs_type: str
) -> List[str]:
    """Get assined files to this rank"""
    data_path = Path(data_dir)

    if fs_type.lower() == "local":
        # local filesystem write
        # only local files are visible and used local values to distribute. 
        rank = get_local_rank()
        world_size = get_local_world_size()
    else:
        # shared filesystem write
        # all files are visible and used global sizes to distribute. 
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    files = _gather_files(data_dir, pattern)
    assigned_files = _assign_files_to_rank(files, rank, world_size)
    return assigned_files 

@timeit
def read_file(fp):
    return cudf.read_csv(fp, header=None)

@timeit
def load_csv(file_paths: List[str]):
    if not file_paths:
        raise ValueError("No file paths provided.")
    with ThreadPoolExecutor() as executor:
        df_list = list(executor.map(read_file, file_paths))
    return cudf.concat(df_list, ignore_index=True)

def get_num_partitions(fs_type: str = "local", num_tg_nodes: int | None = None) -> int:
    if fs_type.lower() == "local":
        # if fs system is "local" then num of partitions is local world size.
        num_partitions = get_local_world_size()
    else:
        # else num of partitions is global world size. 
        num_partitions = dist.get_world_size()
        if num_tg_nodes is not None: 
            if num_partitions % num_tg_nodes == 0: 
                # if world size is divisible by tg node count then 
                # num partitions can be divided uniformoly.
                num_partitions = num_partitions // num_tg_nodes
            else:
                # in case world size is not divisible by num of tg nodes
                # we will keep 1 extra partitions so that each process has atleast
                # one file and few will have more than 1.
                num_partitions = (num_partitions // num_tg_nodes) + 1
    return num_partitions

def find_misaligned_tensors(data: HeteroData, expected_device: str | torch.device):
    misaligned = []

    if isinstance(expected_device, str):
        expected_device = torch.device(expected_device)
    for node_type in data.node_types:
        for key, tensor in data[node_type].items():
            if isinstance(tensor, torch.Tensor) and tensor.device != expected_device:
                misaligned.append((f"Node[{node_type}][{key}]", tensor.device))

    for edge_type in data.edge_types:
        for key, tensor in data[edge_type].items():
            if isinstance(tensor, torch.Tensor) and tensor.device != expected_device:
                misaligned.append((f"Edge[{edge_type}][{key}]", tensor.device))

    if misaligned:
        print("Tensors not on expected device:")
        for name, device in misaligned:
            print(f"{name} is on {device}")
    else:
        print("All tensors are on the correct device.")
    return misaligned

def check_heterodata_integrity(data: HeteroData):
    issues = []

    for node_type in data.node_types:
        x = data[node_type].get('x', None)
        if x is not None and not isinstance(x, torch.Tensor):
            issues.append(f"[{node_type}] 'x' is not a Tensor.")
    
    for edge_type in data.edge_types:
        if 'edge_index' not in data[edge_type]:
            issues.append(f"[{edge_type}] Missing 'edge_index'.")
            continue
        
        edge_index = data[edge_type]['edge_index']
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            issues.append(f"[{edge_type}] 'edge_index' must be of shape [2, num_edges].")

        src_type, _, dst_type = edge_type
        num_src_nodes = data[src_type].get('x', torch.empty(0)).size(0)
        num_dst_nodes = data[dst_type].get('x', torch.empty(0)).size(0)

        if edge_index.size(1) > 0:
            src_max = edge_index[0].max().item()
            dst_max = edge_index[1].max().item()

            if src_max >= num_src_nodes:
                issues.append(f"[{edge_type}] 'edge_index[0]' has index {src_max} >= num {src_type} nodes ({num_src_nodes})")
            if dst_max >= num_dst_nodes:
                issues.append(f"[{edge_type}] 'edge_index[1]' has index {dst_max} >= num {dst_type} nodes ({num_dst_nodes})")
    
    if not issues:
        print("HeteroData integrity check passed.")
    else:
        print("HeteroData integrity issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
