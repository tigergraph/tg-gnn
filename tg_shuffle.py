import torch
import torch.distributed as dist
from tg_utils import timeit


@timeit
def shuffle_splits(local_splits, local_rank, world_size):
    """
    Gathers and redistributes data for train, val, and test splits across ranks, and shuffles them.

    Args:
        local_splits (dict): A dictionary with keys "train", "val", "test", and tensors as values.
        local_rank (int): The rank of the current process.
        world_size (int): Total number of ranks.

    Returns:
        dict: A dictionary with redistributed and shuffled splits for this rank.
    """
    # TODO: this won't work on MNMG please fix this
    device = torch.device(local_rank)
    torch.cuda.set_device(local_rank)
    local_splits_shuffled = {}

    for key, local_indices in local_splits.items():
        # gather all indices
        local_size = torch.tensor([local_indices.size(0)], dtype=torch.int64, device=device)
        all_sizes = torch.zeros((world_size,), dtype=torch.int64, device=device)
        torch.distributed.all_gather_into_tensor(all_sizes, local_size)
        
        all_indices = [
                torch.zeros((all_sizes[i]), device=device, dtype=torch.int64)
                for i in range(all_sizes.numel())
        ]
        torch.distributed.all_gather(all_indices, local_indices.to(device))
        all_indices = torch.concat(all_indices, dim=0).cpu()
        local_splits_shuffled[key] = torch.tensor_split(all_indices, world_size)[local_rank]
        
        del all_indices, local_size, all_sizes
        torch.cuda.empty_cache()
 
    return local_splits_shuffled



if __name__ == "__main__":
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_splits = {
        "train": torch.arange(10 + rank * 5, device=torch.device(f"cpu")),
        "val": torch.arange(5 + rank * 3, device=torch.device(f"cpu")),
        "test": torch.arange(8 + rank * 2, device=torch.device(f"cpu")),
    }

    print(f"Rank {rank} local splits: {local_splits}")


    local_splits_shuffled = shuffle_splits(local_splits, rank, world_size)

    print(f"Rank {rank} shuffled splits: {local_splits_shuffled}")

    # Finalize distributed training
    dist.barrier()
    dist.destroy_process_group()

