import torch
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData
import os

def renumber_data(
    data: Data | HeteroData,
    metadata: dict,
    local_rank: int,
    world_size: int
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
                           "nodes": [{"vertex_name": ..., ...}, ...],
                           "edges": [{"src": ..., "dst": ..., "rel_name": ...}, ...]
                         }
        local_rank (int): The local GPU rank on this node.
        world_size (int): The total number of ranks in the distributed setup.

    Returns:
        Data | HeteroData: The input data object with node and edge indices renumbered.
    """

    # import after rmm re-initialization
    import cudf
    import cupy

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    global_renumber_map = {}

    # Determine if the data is heterogeneous
    is_hetero = isinstance(data, HeteroData)

    # Process Nodes
    for node_meta in metadata["nodes"]:
        vertex_name = node_meta["vertex_name"]

        if is_hetero and vertex_name not in data:
            print(f"Data for '{vertex_name}' not found. Skipping...")
            continue

        # Retrieve node_ids
        node_ids = data[vertex_name].node_ids if is_hetero else data.node_ids
        if node_ids is None:
            print(f"'node_ids' not found for '{vertex_name}'. Skipping...")
            continue

        local_num_nodes = len(node_ids)

        # Gather node counts from all ranks
        current_num_nodes = torch.tensor([local_num_nodes], dtype=torch.int64, device=device)
        node_offsets = torch.zeros((world_size,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(node_offsets, current_num_nodes)

        # Prepare a list to gather each rank's renumber map
        # node_offsets[i] holds the #nodes for rank i
        map_tensor = [
            torch.zeros((2, node_offsets[i].item()), device=device, dtype=torch.int64)
            for i in range(node_offsets.numel())
        ]

        # Compute cumsum for offsets: global shift for the start index of each rank
        node_offsets_cum = node_offsets.cumsum(0).cpu()
        this_rank = dist.get_rank()  # Current rank (0-based)
        local_offset = 0 if this_rank == 0 else int(node_offsets_cum[this_rank - 1])

        # Create local renumber map for the node IDs on this rank
        # (new_id, old_id)
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

        dist.all_gather(map_tensor, local_renumber_map)

        # Concatenate all maps along dim=1 => shape [2, total_num_nodes]
        map_tensor = torch.cat(map_tensor, dim=1).cpu()

        # Build a cudf DataFrame to map old ID -> new ID
        #   - index = old IDs
        #   - column 'id' = new IDs
        global_renumber_map[vertex_name] = cudf.DataFrame(
            data={"id": cupy.asarray(map_tensor[0])},
            index=cupy.asarray(map_tensor[1])
        )

        # update the node ids
        if is_hetero:
            data[vertex_name].node_ids = local_renumber_map[0].cpu()
        else:
            data.node_ids = local_renumber_map[0].cpu()

        del map_tensor, node_offsets, node_offsets_cum, current_num_nodes, local_renumber_map

    # Process/Renumber edges
    for edge_meta in metadata["edges"]:
        src_name = edge_meta["src"]
        dst_name = edge_meta["dst"]
        rel_name = edge_meta["rel_name"]
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
        srcs = cupy.asarray(edge_index[0].cpu())
        dsts = cupy.asarray(edge_index[1].cpu())

        # Retrieve the renumber maps for src and dst
        #   - map is a cudf DF: old_id => new_id
        src_map = global_renumber_map[src_name]["id"]
        dst_map = global_renumber_map[dst_name]["id"]

        if src_map is None or dst_map is None:
            print(f"Renumber map not found for '{src_name}' or '{dst_name}'. Skipping edge type '{rel}'.")
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
            data[rel].edge_index = new_edge_index
        else:
            data.edge_index = new_edge_index

        # Clean up per-edge-type references
        if src_name == dst_name:
            del global_renumber_map[src_name]
        else:
            del global_renumber_map[src_name], global_renumber_map[dst_name]
        del src_map, dst_map, srcs, dsts

    torch.cuda.empty_cache()

    return data



def initialize_distributed():
    """
    Initialize the distributed environment.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        raise EnvironmentError("RANK and WORLD_SIZE environment variables are required for distributed training.")

    # Initialize the process group
    dist.init_process_group(backend='nccl')

    return local_rank, world_size

def create_sample_data(is_hetero: bool = False) -> tuple[Data | HeteroData, dict]:
    """
    Create sample Data or HeteroData objects along with metadata for demonstration.
    """
    if is_hetero:
        data = HeteroData()

        # Example for two node types: 'author' and 'paper'
        data['author'].node_ids = torch.tensor([100, 101, 102])  # Sample node IDs
        data['paper'].node_ids = torch.tensor([200, 201])

        # Example edge type: author writes paper
        data['author', 'writes', 'paper'].edge_index = torch.tensor([
            [0, 1],
            [0, 1]
        ])

        metadata = {
            "nodes": [
                {"vertex_name": "author"},
                {"vertex_name": "paper"}
            ],
            "edges": [
                {"src": "author", "dst": "paper", "rel_name": "writes"}
            ]
        }
    else:
        data = Data()

        data.node_ids = torch.tensor([100, 101, 102, 103])  # Sample node IDs
        data.edge_index = torch.tensor([
            [0, 1, 2],
            [1, 2, 3]
        ])

        metadata = {
            "nodes": [
                {"vertex_name": "node"}  # Single node type
            ],
            "edges": [
                {"src": "node", "dst": "node", "rel_name": "edge"}  # Single edge type
            ]
        }

    return data, metadata

def main():
    # Initialize distributed environment
    local_rank, world_size = initialize_distributed()
    rank = dist.get_rank()

    # Optionally, print the rank information
    print(f"Rank {rank}/{world_size} initialized on device cuda:{local_rank}")

    # Create sample data
    is_hetero = True  # Change to False to use Data instead of HeteroData
    data, metadata = create_sample_data(is_hetero=is_hetero)

    # Move data to the appropriate device
    device = torch.device(f"cuda:{local_rank}")
    data = data.to(device)

    # Call renumber_data
    renumbered_data = renumber_data(data, metadata, local_rank, world_size)

    # Optionally, print the renumbered data for verification
    if rank == 0:
        print("Renumbered Data:")
        print(renumbered_data)

    # Finalize the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()