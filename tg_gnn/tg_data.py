import torch
import os
import argparse
import argparse
import os
from pyTigerGraph import TigerGraphConnection
from torch_geometric.data import Data, HeteroData
from .tg_gsql import create_gsql_query, install_and_run_query
from .tg_utils import timeit
from .tg_renumber import renumber_data



@timeit
def export_tg_data(conn, metadata, local_world_size, force=False, timeout=2000000):
    gsql_query = create_gsql_query(metadata, num_partitions=local_world_size)
    # print(gsql_query)
    install_and_run_query(conn, gsql_query, force=force, timeout=timeout)


def load_tg_data(
        metadata: dict, 
        local_rank: int, 
        world_size: int, 
        renumber: bool = True) -> Data | HeteroData:
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
        local_rank (int): The rank of the local process, used for partition-specific data loading.
        world_size (int): Total number of processes involved in data loading.
        renumber (bool, optional): Whether to renumber node and edge indices for distributed processing.
                                    Defaults to True.

    Returns:
        Data | HeteroData: A PyTorch Geometric `Data` or `HeteroData` object containing the loaded graph data.
    """
    # import after rmm re-initialization
    import cudf
    data_dir = metadata.get("data_dir", "")
    nodes_meta = metadata.get("nodes", [])
    edges_meta = metadata.get("edges", [])
    
    # Determine whether to use HeteroData or Data based on the number of node and edge types
    is_hetero = len(nodes_meta) > 1 or len(edges_meta) > 1
    data = HeteroData() if is_hetero else Data()


    def load_node_csv(meta: dict) -> dict | None:
        """
        Reads a node CSV file and extracts node features, labels, and masks.

        Args:
            meta (dict): Metadata for the node type, expected to contain:
                - "vertex_name" (str): Name of the node type.
                - "label" (bool, optional): Whether the node has labels.
                - "split" (bool, optional): Whether the node data includes train/val/test splits.

        Returns:
            dict | None: A dictionary containing node data suitable for updating the `Data` or `HeteroData` object.
                          Returns None if the file does not exist.
        """
        vertex_name = meta.get("vertex_name", "")
        file_path = os.path.join(data_dir, f"{vertex_name}_p{local_rank}.csv")
        if not os.path.exists(file_path):
            print(f"Node file not found: {file_path}")
            return None

        df = cudf.read_csv(file_path, header=None)
        
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
            node_data["y"] = torch.as_tensor(df.iloc[:, 1].values, dtype=torch.long)

        if has_split:
            splits = torch.as_tensor(df.iloc[:, 2].values, dtype=torch.long)
            node_data["train_mask"] = (splits == 0)
            node_data["val_mask"]   = (splits == 1)
            node_data["test_mask"]  = (splits == 2)

        node_data = {k: v.to("cpu") for k, v in node_data.items()}
        
        del df
        return node_data

    def load_edge_csv(meta: dict) -> tuple | None:
        """
        Reads an edge CSV file and extracts edge indices, attributes, labels, and masks.

        Args:
            meta (dict): Metadata for the edge type, expected to contain:
                - "src" (str): Source node type.
                - "rel_name" (str): Relationship name.
                - "dst" (str): Destination node type.
                - "label" (bool, optional): Whether the edge has labels.
                - "split" (bool, optional): Whether the edge data includes train/val/test splits.

        Returns:
            tuple | None: A tuple containing the edge relationship tuple and a dictionary of edge data.
                           Returns None if the file does not exist.
        """
        src, rel_name, dst = (
            meta.get("src", ""), meta.get("rel_name", ""), meta.get("dst", "")
        )
        rel = (src, rel_name, dst)
        file_name = f"{src}_{rel_name}_{dst}_p{local_rank}.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Edge file not found: {file_path}")
            return rel, None

        df = cudf.read_csv(file_path, header=None)

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
        
        # feature columns, store them in edge_attr
        # TODO: feature attributes without edges info?
        # is that okay? confirm with Alex
        if has_features:
            edge_data["edge_attr"] = torch.as_tensor(
                df.iloc[:, features_start:].values,
                dtype=torch.float32
            )
        
        if has_label:
            edge_data["edge_label"] = torch.as_tensor(
                df.iloc[:, 2].values, dtype=torch.long
            )

        if has_split:
            split_col_idx = 2 + int(has_label) 
            splits = torch.as_tensor(
                df.iloc[:, split_col_idx].values, dtype=torch.long
            )
            edge_data["train_mask"] = (splits == 0)
            edge_data["val_mask"]   = (splits == 1)
            edge_data["test_mask"]  = (splits == 2)
            del splits

        edge_data = {k: v.to("cpu") for k, v in edge_data.items()}
        
        del df
        return rel, edge_data

    # Process each node
    for node_meta in nodes_meta:
        node_data = load_node_csv(node_meta)
        if node_data is None:
            continue
        
        # If hetero, store it under the node type key
        if isinstance(data, HeteroData):
            data[node_meta["vertex_name"]].update(node_data)
        else:
            data.update(node_data)

    # Process each edge
    for edge_meta in edges_meta:
        rel, edge_data = load_edge_csv(edge_meta)
        if edge_data is None:
            continue
        
        if isinstance(data, HeteroData):
            data[rel].update(edge_data)
        else:
            data.update(edge_data)
    
    # renumber the data
    # this is currently required as feature store does not support random ordering
    if renumber:
        data = renumber_data(data, metadata, local_rank, world_size)

    torch.cuda.empty_cache()

    return data


def load_partitioned_data(
    metadata: dict, 
    local_rank: int, 
    wg_mem_type: str, 
    world_size: int
):
    """
    Loads partitioned graph data and prepares it for use with PyTorch Geometric and
    cuGraph-PyG data structures.

    This function processes node and edge data from metadata and stores them in feature 
    and graph stores. It supports both homogeneous and heterogeneous graph representations 
    using PyTorch Geometric's `Data` and `HeteroData` objects.

    Args:
        metadata (Dict): A dictionary containing metadata about the graph, including nodes and edges.
        local_rank (int): The rank of the local process, used for partition-specific data loading.
        wg_mem_type (str): Memory type for the WholeFeatureStore (e.g., "shared" or "device").
        world_size (int): Total number of processes involved in data loading.

    Returns:
        tuple: A tuple containing the feature store, graph store, splits indices dict.
            - feature_store: WholeFeatureStore object containing node features and labels.
            - graph_store: GraphStore object containing edge indices.
            - split_idx: A dictionary mapping split names ("train", "val", "test") to node IDs.
    """
    from cugraph_pyg.data import GraphStore, WholeFeatureStore
    from torch_geometric.data import Data, HeteroData

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    # Load TG data and renumber the node ids
    data = load_tg_data(metadata, local_rank, world_size, renumber=True)

    split_idx = {}

    # Load nodes
    for node_meta in metadata["nodes"]:
        vertex_name = node_meta["vertex_name"]
        features_list = node_meta.get("features_list", {})
        labels = node_meta.get("label", "")
        splits = node_meta.get("split", "")
        node_data = data[vertex_name] if isinstance(data, HeteroData) else data

        # Load features
        if features_list:
            features_tensor = node_data.x
            if features_tensor is not None:
                feature_store[vertex_name, "x"] = features_tensor

        # Load labels
        if labels:
            labels_tensor = node_data.y
            if labels_tensor is not None:
                feature_store[vertex_name, "y"] = labels_tensor

        # Load splits
        if splits:
            if "train_mask" in node_data : 
                train_mask = node_data.train_mask
                split_idx["train"] = (
                    node_data.node_ids[train_mask] 
                    if hasattr(node_data, 'node_ids') 
                    else None
                )
            
            if "val_mask" in node_data :
                val_mask = node_data.val_mask
                split_idx["val"] = (
                    node_data.node_ids[val_mask] 
                    if hasattr(node_data, 'node_ids') 
                    else None
                )
            
            if "test_mask" in node_data : 
                test_mask = node_data.test_mask
                split_idx["test"] = (
                    node_data.node_ids[test_mask] 
                    if hasattr(node_data, 'node_ids') 
                    else None
                )


        del labels_tensor, split_tensor, features_tensor
        torch.cuda.empty_cache()

    # Load edges
    for edge_meta in metadata["edges"]:
        rel_name = edge_meta["rel_name"]
        src_name = edge_meta["src"]
        dst_name = edge_meta["dst"]
        rel = (src_name, rel_name, dst_name)

        edge_data = data[rel] if isinstance(data, HeteroData) else data

        # Load edge indices
        if isinstance(data, HeteroData):
            edge_indices_tensor = edge_data.edge_index
        else:
            edge_indices_tensor = data.edge_index

        if edge_indices_tensor is not None:
            if metadata[src_name]["num_nodes"] is None or metadata[dst_name]["num_nodes"] is None:
                graph_store[
                    (src_name, rel_name, dst_name), "coo", False
                ] = edge_indices_tensor
            else:
                graph_store[
                    (src_name, rel_name, dst_name), "coo", False,
                    (metadata[src_name]["num_nodes"], metadata[dst_name]["num_nodes"])
                ] = edge_indices_tensor
        
        # TODO: Need to confirm the logic with Alex about 
        # edge features population
        # edge_attr_tensor contains src and dst id? shape
        # (#edges, #edges_features)
        # and how to handle the label
        if "edge_attr" in edge_data:
            edge_attr_tensor = edge_data.edge_attr
            if edge_attr_tensor is not None:
                feature_store[(rel, "edge_attr")] = edge_attr_tensor

        del edge_indices_tensor
        torch.cuda.empty_cache()

        # TODO: edge labels and edge split mask handling

    return (feature_store, graph_store), split_idx


metadata = {
    "nodes": [ 
        {
            "vertex_name": "product",
            "features_list": { 
                "feature": "LIST"
            },
            "label": "label",
            "split": "split"
        }
    ], 
    "edges": [
        {
            "rel_name": "rel",
            "src": "product",
            "dst": "product"
        }
    ],
    "data_dir": "/tg/tmp"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--graph", required=True)
    parser.add_argument("--host", default="http://172.17.0.2")
    parser.add_argument("--username", "-u", default="tigergraph")
    parser.add_argument("--password", "-p", default="tigergraph")
    args = parser.parse_args()
    
    # tg connection
    conn = TigerGraphConnection(
        host=args.host,
        graphname=args.graph,
        username=args.username,
        password=args.password
    )
    conn.getToken(conn.createSecret())
    
    print("exporting tg data...")
    export_tg_data(conn, metadata, 4, force=True)


