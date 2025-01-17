import torch
import os
import argparse
import argparse
import os
from pyTigerGraph import TigerGraphConnection
from tg_gsql import create_gsql_query, install_and_run_query
from tg_utils import timeit
from tg_renumber import renumber_data



@timeit
def export_tg_data(conn, metadata, local_world_size, force=False, timeout=2000000):
    gsql_query = create_gsql_query(metadata, num_partitions=local_world_size)
    # print(gsql_query)
    install_and_run_query(conn, gsql_query, force=force, timeout=timeout)


@timeit
def load_tg_data(metadata, local_rank):
    import cudf
    data_dir = metadata["data_dir"]
    nodes_metadata = metadata["nodes"]
    edges_metadata = metadata["edges"]

    data = {}

    # Process nodes
    for node_meta in nodes_metadata:
        vertex_name = node_meta["vertex_name"]
        label_col = node_meta.get("label", None)
        split_col = node_meta.get("split", None)

        # Load node file
        node_file_path = os.path.join(data_dir, f"{vertex_name}_p{local_rank}.csv")
        if not os.path.exists(node_file_path):
            print(f"Node file {node_file_path} not found. Skipping...")
            continue

        node_df = cudf.read_csv(node_file_path, header=None)

        # Extract node IDs
        node_tensor = torch.as_tensor(
            node_df.iloc[:, 0].values, dtype=torch.long
        ).to("cpu")

        # Extract label if present
        label_tensor = None
        if label_col:
            label_tensor = torch.as_tensor(
                node_df.iloc[:, 1].values, dtype=torch.long
            ).to("cpu")

        # Extract split if present
        split_tensor = None
        if split_col:
            split_col_idx = 2 if label_col else 1
            split_tensor = torch.as_tensor(
                node_df.iloc[:, split_col_idx].values, dtype=torch.long
            ).to("cpu")

        # Extract features
        feature_start_idx = (
            3 if label_col and split_col else 2 if label_col or split_col else 1
        )
        features_tensor = torch.as_tensor(
            node_df.iloc[:, feature_start_idx:].values, dtype=torch.float32
        ).to("cpu")

        # Store in data dictionary
        data[vertex_name] = {
            "node_ids": node_tensor,
            "features": features_tensor,
            "label": label_tensor,
            "split": split_tensor,
        }
        
        # Clean up GPU memory
        del node_df
        torch.cuda.empty_cache()

    # Process edges
    for edge_meta in edges_metadata:
        rel_name_ext = (
            f"{edge_meta['src']}_{edge_meta['rel_name']}_{edge_meta['dst']}"
        )
        edge_file_path = os.path.join(data_dir, f"{rel_name_ext}_p{local_rank}.csv")

        if not os.path.exists(edge_file_path):
            print(f"Edge file {edge_file_path} not found. Skipping...")
            continue

        edge_df = cudf.read_csv(edge_file_path, header=None)

        # Extract edge indices
        src_ids = torch.as_tensor(
            edge_df.iloc[:, 0].values, dtype=torch.long
        ).to("cpu")
        
        dst_ids = torch.as_tensor(
            edge_df.iloc[:, 1].values, dtype=torch.long
        ).to("cpu")
        
        edge_indices_tensor = torch.stack([src_ids, dst_ids], dim=0).to("cpu")

        # Extract edge labels
        edge_label_tensor = None
        if "label" in edge_meta:
            label_col_idx = 2
            edge_label_tensor = torch.as_tensor(
                edge_df.iloc[:, label_col_idx].values, dtype=torch.long
            ).to("cpu")

        # Extract edge splits
        edge_split_tensor = None
        if "split" in edge_meta:
            split_col_idx = 3 if "label" in edge_meta else 2
            edge_split_tensor = torch.as_tensor(
                edge_df.iloc[:, split_col_idx].values, dtype=torch.long
            )

        # Extract edge features if present
        feature_start_idx = (
            4
            if "label" in edge_meta and "split" in edge_meta
            else 3
            if "label" in edge_meta or "split" in edge_meta
            else 2
        )
        edge_features_tensor = None
        if edge_df.shape[1] > feature_start_idx:
            edge_features_tensor = torch.as_tensor(
                edge_df.iloc[:, feature_start_idx:].values, dtype=torch.float32
            ).to("cpu")

        data[rel_name_ext] = {
            "edge_indices": edge_indices_tensor,
            "features": edge_features_tensor,
            "labels": edge_label_tensor,
            "splits": edge_split_tensor,
        }

        # Clean up GPU memory
        del edge_df
        torch.cuda.empty_cache()

    return data


@timeit
def load_partitioned_data(metadata, local_rank, wg_mem_type, world_size):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore
    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    # load tg data 
    data = load_tg_data(metadata, local_rank)

    # renumber the nodes and map the edges 
    data = renumber_data(
        data, metadata, local_rank, world_size
    )

    split_idx = {}

    # Load nodes
    for node_meta in metadata["nodes"]:
        vertex_name = node_meta["vertex_name"]
        features_list = node_meta.get("features_list", {})
        labels = node_meta.get("label", "")
        splits = node_meta.get("split", "")
        node_data = data[vertex_name]
        
        # Load features
        if features_list:
            features_tensor = node_data.get("features")
            if features_tensor is not None:
                feature_store[vertex_name, "x"] = features_tensor

        # Load labels
        if labels:
            labels_tensor = node_data.get("label")
            if labels_tensor is not None:
                feature_store[vertex_name, "y"] = labels_tensor

        # Load splits
        if splits:
            split_tensor = node_data.get("split")
            if split_tensor is not None:
                split_names = ["train", "val", "test"]
                split_masks = [(split_tensor == i) for i in range(len(split_names))]

                for split_name, mask in zip(split_names, split_masks):
                    split_idx[split_name] = node_data["node_ids_renumbered"][mask]

        del labels_tensor, split_tensor, features_tensor
        torch.cuda.empty_cache()

    # Load edges
    for edge_meta in metadata["edges"]:
        rel_name = edge_meta["rel_name"]
        src_name = edge_meta["src"]
        dst_name = edge_meta["dst"]
        rel_name_ext = f"{src_name}_{rel_name}_{dst_name}"

        # Load edge indices
        edge_indices_tensor = data[rel_name_ext].get("edge_indices_renumbered")
        # TODO: update the metadata to handle each src and dst num_nodes
        # otherwise it will break for multi-graph
        if edge_indices_tensor is not None:
            graph_store[
                (src_name, rel_name, dst_name), "coo", False,
                (metadata["num_nodes"], metadata["num_nodes"])
            ] = edge_indices_tensor

        del edge_indices_tensor
        torch.cuda.empty_cache()

    return (feature_store, graph_store), split_idx, metadata


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


