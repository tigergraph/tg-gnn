import torch
import os
import subprocess
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from tg_gnn.tg_gsql import create_gsql_query, install_and_run_query
from tg_gnn.utils import timeit, get_local_world_size, renumber_data, load_csv, get_assigned_files, get_fs_type, get_num_partitions
import logging
logger = logging.getLogger(__name__)


@timeit
def export_tg_data(conn, metadata, force=False, timeout=2000000):
    """
    Install and run the GSQL query to export the data from TigerGraph.
    args:
        conn: TigerGraphConnection object
        metadata: metadata about the graph, contains info about required vertices and 
            features for GNN training
        force: force to reinstall the query
        timeout: timeout for the gsql query
    """
    try:
        # file sytem type
        fs_type = metadata.get("fs_type", "local")
        logger.info(f"File system is set to {fs_type}.")
        
        # caculate the num of partitions
        num_tg_nodes = metadata.get("num_tg_nodes", None)
        num_partitions = get_num_partitions(fs_type=fs_type, num_tg_nodes=num_tg_nodes)
        logger.info(f"Creating total {num_partitions} partitions for tg to export the data.")
        
        # create and run query 
        gsql_query = create_gsql_query(metadata, num_partitions=num_partitions)
        install_and_run_query(conn, gsql_query, force=force, timeout=timeout)
    except Exception as export_error:
        logger.exception(f"Error exporting TG data: {export_error}")
        raise RuntimeError(f"Error exporting TG data: {export_error}") from export_error

@timeit
def load_tg_data(
        metadata: dict, 
        renumber: bool = True,
        shuffle_splits: bool = False,
        mem_loc: str = "cpu",
        **kwargs) -> Data | HeteroData:
    """
    Loads partitioned graph data and prepares it for use with PyTorch Geometric.

    This function processes node and edge data based on the provided metadata, reads
    corresponding CSV files, and constructs either a `Data` or `HeteroData` object
    suitable for PyTorch Geometric operations. Optionally, it can renumber the data
    for distributed processing.

    Args:
        metadata (dict): A dictionary containing metadata about the graph, including:
            - "data_dir" (str): Directory path where node and edge CSV files are located.
            - "nodes" (list of dict): List containing metadata for each node type.
                Each dictionary should have keys like "vertex_name", "label", "split".
            - "edges" (list of dict): List containing metadata for each edge type.
                Each dictionary should have keys like "src", "rel_name", "dst", "label", "split".
        renumber (bool, optional): Whether to renumber node and edge indices.
            Defaults to True.

        shuffle_splits (bool, optional): Whether to shuffle splits indices.
            This is required for uniform distribution of splits across the ranks.
            Distributed training environment expect unifrom distribution of spits 
            across the processes (i.e. ranks).
            This is only applicable if you have nodes with splits.
            Defaults to True.
    Returns:
        Data | HeteroData: A PyTorch Geometric `Data` or `HeteroData` object containing the loaded graph data.
    """
    # import after rmm re-initialization
    import cudf
    data_dir = metadata.get("data_dir", "")
    nodes_meta = metadata.get("nodes", {})
    edges_meta = metadata.get("edges", {})
    fs_type = metadata.get("fs_type", "local")
    subprocess.run(['sudo', 'chmod', '-R', '0755', data_dir])
    
    # Determine whether to use HeteroData or Data based on the number of node and edge types
    is_hetero = len(nodes_meta) > 1 or len(edges_meta) > 1
    data = HeteroData() if is_hetero else Data()

    @timeit
    def load_node_csv(vertex_name: str, meta: dict) -> dict:
        """
        Reads a node CSV file and extracts node features, labels, and masks.

        Args:
            vertex_name (str): Name of the node type.
            meta (dict): Metadata for the node type, expected to contain info about
                features (dict), label and split info

        Returns:
            dict : A dictionary containing node data suitable for updating the `Data` or `HeteroData` object.
        """
        file_paths = get_assigned_files(data_dir, f"{vertex_name}_p*.csv", fs_type)

        df = load_csv(file_paths)
        
        has_label = bool(meta.get("label", False))
        has_split = bool(meta.get("split", False))
        features_start = 1 + int(has_label) + int(has_split)
        has_features = df.shape[1] > features_start

        node_data = {
            "node_ids": torch.as_tensor(df.iloc[:, 0].values, dtype=torch.long),
        }
        
        if has_features:
            node_data["x"] = torch.as_tensor(
                df.iloc[:, features_start:].values, dtype=torch.float32
            )
        
        if has_label:
            label_col_idx = 1
            node_data["y"] = torch.as_tensor(df.iloc[:, label_col_idx].values, dtype=torch.long)

        if has_split:
            split_col_idx = 2 if has_label else 1 
            splits = torch.as_tensor(df.iloc[:, split_col_idx].values, dtype=torch.long)
            node_data["train_mask"] = (splits == 0)
            node_data["val_mask"]   = (splits == 1)
            node_data["test_mask"]  = (splits == 2)

        node_data = {k: v.to(mem_loc) for k, v in node_data.items()}
        
        del df
        return node_data
    
    @timeit
    def load_edge_csv(rel_name: str, meta: dict) -> tuple | None:
        """
        Reads an edge CSV file and extracts edge indices, attributes, labels, and masks.

        Args:
            rel_name (str): Relationship name. 
            meta (dict): Metadata for the edge type, expected to contain:
                - "src" (str): Source node type.
                - "dst" (str): Destination node type.
                - "label" (str, optional): the edge label name.
                - "split" (str, optional): the edge info about train/val/test splits.

        Returns:
            tuple: A tuple containing the edge relationship tuple and a dictionary of edge data.
        """

        src, dst = (
            meta.get("src", ""), meta.get("dst", "")
        )
        rel = (src, rel_name, dst)
        
        filename_pattern = f"{src}_{rel_name}_{dst}_p*.csv"
        file_paths = get_assigned_files(data_dir, filename_pattern, fs_type)
        df = load_csv(file_paths)
        
        has_label = "label" in meta
        has_split = "split" in meta  # In case edges have splits
        features_start = 2 + int(has_label) + int(has_split)
        has_features = df.shape[1] > features_start

        edge_data = {
            "edge_index": torch.stack([
                torch.as_tensor(df.iloc[:, 0].values, dtype=torch.long),
                torch.as_tensor(df.iloc[:, 1].values, dtype=torch.long)
            ], dim=0)
        }

        if has_features:
            edge_data["edge_attr"] = torch.as_tensor(
                df.iloc[:, features_start:].values,
                dtype=torch.float32
            )

        if has_label:
            label_col_idx = 2
            edge_data["edge_label"] = torch.as_tensor(
                df.iloc[:, label_col_idx].values, dtype=torch.long
            )

        if has_split:
            split_col_idx = 3 if has_label else 2
            splits = torch.as_tensor(df.iloc[:, split_col_idx].values, dtype=torch.long)
            edge_data["train_mask"] = (splits == 0)
            edge_data["val_mask"]   = (splits == 1)
            edge_data["test_mask"]  = (splits == 2)

        edge_data = {k: v.to(mem_loc) for k, v in edge_data.items()}
        
        del df
        return rel, edge_data

    # Process each node
    for vertex_name, node_meta in nodes_meta.items():
        logger.info(f"Loading the data for {vertex_name}...")
        node_data = load_node_csv(vertex_name, node_meta)
        if node_data is None:
            continue
        
        # If hetero, store it under the node type key
        if isinstance(data, HeteroData):
            data[vertex_name].update(node_data)
            if node_meta["num_nodes"] is not None:
                data[vertex_name].num_nodes = node_meta["num_nodes"]
        else:
            data.update(node_data)
            if node_meta["num_nodes"] is not None:
                data.num_nodes = node_meta["num_nodes"]
        logger.info(f"Data loading for {vertex_name} completed successfully.")

    # Process each edge
    undirected_edge = []
    for rel_name, edge_meta in edges_meta.items():
        undirected = edge_meta.get("undirected", False)
        logger.info(f"Loading the data for {rel_name} with undirected={undirected}...")
        rel, edge_data = load_edge_csv(rel_name, edge_meta)
        if not edge_data:
            continue

        if undirected:
            undirected_edge.append(rel)

        if isinstance(data, HeteroData):
            data[rel].update(edge_data)
        else:
            data.update(edge_data)
        
        logger.info(f"Data loading for {rel_name} completed successfully.")
    
    # renumber the data
    # this is required as feature store does not support random ordering
    if renumber:
        data = renumber_data(data, metadata, mem_loc)

    if undirected_edge:
        if is_hetero:
            for rel in undirected_edge:
                rev_rel = (rel[2], f"rev_{rel[1]}", rel[0])
                rev_edge_index = torch.stack([data[rel].edge_index[1], data[rel].edge_index[0]], dim=0).to(mem_loc)
                data[rev_rel].edge_index = rev_edge_index
                logger.info(f"Creating data for rev_{rel_name} completed successfully.")
        else:
            data = ToUndirected()(data)
            logger.info(f"Creating data for rev_{rel_name} completed successfully.")

    torch.cuda.empty_cache()

    return data

