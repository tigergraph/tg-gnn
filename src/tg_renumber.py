import os
import torch
import argparse
from tg_utils import timeit

@timeit
def renumber_data(data, metadata, local_rank, world_size):
    # TODO: check if this logic is ok in MNMG environment
    device = torch.device(local_rank)
    torch.cuda.set_device(local_rank)
    import cudf 
    import cupy

    nodes_metadata = metadata["nodes"]

    global_renumber_map = {}

    # create new nodes renumber map
    for node_meta in nodes_metadata:
        vertex_name = node_meta["vertex_name"]

        if vertex_name not in data:
            print(
                f"Data for {vertex_name} not found in the provided data dictionary. Skipping..."
            )
            continue

        node_ids = data[vertex_name]["node_ids"]
        local_num_nodes = len(node_ids)

        # Create local node offsets
        node_offset_tensor = torch.zeros(
            (world_size,), dtype=torch.int64, device=device
        )
        current_num_nodes = torch.tensor(
            [local_num_nodes], dtype=torch.int64, device=device
        )
        torch.distributed.all_gather_into_tensor(node_offset_tensor, current_num_nodes)

        map_tensor = [
            torch.zeros((2, node_offset_tensor[i]), device=device, dtype=torch.int64)
            for i in range(node_offset_tensor.numel())
        ]
        node_offset_tensor = node_offset_tensor.cumsum(0).cpu()

        local_node_offset = (
            0 if torch.distributed.get_rank() == 0
            else int(node_offset_tensor[torch.distributed.get_rank() - 1])
        )

        # Create local renumber map
        local_renumber_map = torch.stack(
            [
                torch.arange(
                    local_node_offset,
                    local_node_offset + local_num_nodes,
                    dtype=torch.int64,
                ),
                node_ids,
            ]
        )

        # Gather the global map
        torch.distributed.all_gather(
            map_tensor, local_renumber_map.to(device)
        )
        map_tensor = torch.concat(map_tensor, dim=1).cpu()

        # mapper dataframe
        global_renumber_map[vertex_name] = cudf.DataFrame(
            {
                "id": cupy.asarray(map_tensor[0]),
            },
            index=cupy.asarray(map_tensor[1]),
        )

        data[vertex_name]["node_ids_renumbered"] = local_renumber_map[0].to("cpu")

        # memory cleanup
        del map_tensor, node_offset_tensor, local_node_offset, current_num_nodes, local_renumber_map
        torch.cuda.empty_cache()

    # Renumber edge indices using above map
    edges_metadata = metadata["edges"]
    for edge_meta in edges_metadata:
        rel_name = edge_meta["rel_name"]
        src_name = edge_meta["src"]
        dst_name = edge_meta["dst"]
        rel_name_ext = f"{src_name}_{rel_name}_{dst_name}"


        if rel_name_ext not in data:
            print(
                f"Edge data for {rel_name_ext} not found. Skipping..."
            )
            continue

        edge_indices = data[rel_name_ext]["edge_indices"]
        srcs = cupy.asarray(edge_indices[0].cpu())
        dsts = cupy.asarray(edge_indices[1].cpu())

        src_map = global_renumber_map[src_name]["id"]
        dst_map = global_renumber_map[dst_name]["id"]

        # Create new edge tensor
        new_edge_indices = torch.stack(
            [
                torch.as_tensor(
                    src_map.loc[srcs].values, dtype=torch.int64, device=device
                ),
                torch.as_tensor(
                    dst_map.loc[dsts].values, dtype=torch.int64, device=device
                ),
            ],
            dim=0
        ).to("cpu")
        
        # add new edge indices
        data[rel_name_ext]["edge_indices_renumbered"] = new_edge_indices

        # clean memory
        if src_name == dst_name:
            del global_renumber_map[src_name]
        else:
            del global_renumber_map[src_name], global_renumber_map[dst_name]
        del edge_indices, src_map, dst_map
        torch.cuda.empty_cache()

    return data
