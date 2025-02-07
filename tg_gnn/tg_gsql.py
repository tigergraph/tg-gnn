import argparse
from pyTigerGraph import TigerGraphConnection
import logging
import re
from tg_gnn.tg_utils import timeit

logger = logging.getLogger(__name__)

def is_query_installed(
    conn, query_name: str, return_status: bool = False
) -> bool:
    # If the query already installed return true
    target = "GET /query/{}/{}".format(conn.graphname, query_name)
    queries = conn.getInstalledQueries()
    is_installed = target in queries
    if return_status:
        if is_installed:
            is_enabled = queries[target]["enabled"]
        else:
            is_enabled = None
        return is_installed, is_enabled
    else:
        return is_installed

def install_query(
    conn,
    query: str = None,
    file_path: str = None,
    replace: dict = None,
    distributed: bool = False,
    force: bool = False,
) -> str:
    # Read the first line of the file to get query name. The first line should be
    # something like CREATE QUERY query_name (...
    if file_path:
        with open(file_path) as infile:
            firstline = infile.readline()
        try:
            query_name = re.search(r"QUERY (.+?)\(", firstline).group(1).strip()
        except:
            raise ValueError(
                "Cannot parse the query file. It should start with CREATE QUERY ... "
            )
    else:
        try:
            query_name = re.search(r"QUERY (.+?)\(", query).group(1).strip()
        except:
            raise ValueError(
                "Cannot parse the query file. It should start with CREATE QUERY ... "
            )
        
    # If a suffix is to be added to query name
    if replace and ("{QUERYSUFFIX}" in replace):
        query_name = query_name.replace(
            "{QUERYSUFFIX}", replace["{QUERYSUFFIX}"])
    
    # If query is already installed, skip unless force install.
    is_installed, is_enabled = is_query_installed(
        conn, query_name, return_status=True)
    if is_installed:
        if force or (not is_enabled):
            drop_query = "USE GRAPH {}\nDROP QUERY {}\n".format(
                conn.graphname, query_name)
            resp = conn.gsql(drop_query)
            if "Successfully dropped queries" not in resp:
                raise ConnectionError(resp)
        else:
            print("Query is already installed.")
            return query_name
    # Otherwise, install the query from file
    if file_path:
        with open(file_path) as infile:
            query = infile.read()
    # Replace placeholders with actual content if given
    if replace:
        for placeholder in replace:
            query = query.replace(placeholder, replace[placeholder])
    if distributed:
        query = query.replace("CREATE QUERY", "CREATE DISTRIBUTED QUERY")

    query = (
        "USE GRAPH {}\n".format(conn.graphname)
        + query
        + "\nInstall Query {}\n".format(query_name)
    )
    print(
        "Installing and optimizing queries. It might take a minute or two."
    )
    resp = conn.gsql(query)
    if "Query installation finished" not in resp:
        raise ConnectionError(resp)
    else:
        print("Query installation finished.")
    return query_name

@timeit
def create_gsql_query(metadata, num_partitions):
    data_dir = metadata["data_dir"]
    nodes = metadata["nodes"]
    edges = metadata["edges"]

    query = "CREATE DISTRIBUTED QUERY data_load_gen_query_dist() {\n\n"

    # Loop through nodes to generate node file and features 
    for i, (vertex_name, node) in enumerate(nodes.items()):
        features_list = node.get("features_list", {})
        label = node.get("label", "")
        split = node.get("split", "")
        
        # process node
        query += f"  // Process node: {vertex_name}\n"
        for p in range(num_partitions):
            query += f"  FILE node_file{i}_p{p}(\"{data_dir}/{vertex_name}_p{p}.csv\");\n"
        query += f"\n"

        # iterate through node list
        query += f"  node_list = SELECT s FROM {vertex_name}:s \n"
        query += f"    ACCUM \n"
        
        # add node vid
        query += f"      STRING node_data_list = to_string(getvid(s))"
        
        # add label  
        if label:
            query += f",\n        node_data_list = node_data_list + \",\" + to_string(s.{label})"
        
        # add split
        if split:
            query += f",\n        node_data_list = node_data_list + \",\" + to_string(s.{split})"

        # add features
        if features_list:
            for feature_name, feature_type in features_list.items():
                if feature_type.upper() == "LIST":
                    query += f",\n      FOREACH feature IN s.{feature_name} DO"
                    query += f"\n        node_data_list = node_data_list + \",\" + to_string(feature)"
                    query += f"\n      END"
                else:
                    query += f",\n      node_data_list = node_data_list + \",\" + to_string(s.{feature_name})"
        query += f",\n" 
        # write data 
        for p in range(num_partitions):
            query += f"      IF getvid(s) % {num_partitions} == {p} THEN \n"
            query += f"        node_file{i}_p{p}.println(node_data_list)"
            if p == num_partitions-1: 
                query += f"\n      END;"
            else:
                query += f"\n      END,\n"
            
        query += f"\n"

    # Loop through edges to generate edge file, features, and labels
    for i, (rel_name, edge) in enumerate(edges.items()):
        src = edge["src"]
        dst = edge["dst"]
        edge_features = edge.get("features_list", {})
        edge_split = edge.get("split", "")
        edge_label = edge.get("label", "")
        
        # process edge
        query += f"  // Process edge: {rel_name}\n"
        for p in range(num_partitions):
            query += f"  FILE edge_file{i}_p{p}(\"{data_dir}/{src}_{rel_name}_{dst}_p{p}.csv\");\n"
        query += f"\n"

        query += f"  edge_list = SELECT s FROM {src}:s - ({rel_name}:e) -> {dst}:t \n"
        query += f"    ACCUM \n"

        # add edge indices
        query += f"      STRING edge_data_list = to_string(getvid(s)) + \",\" + to_string(getvid(t))"
        
        # add label 
        if edge_label:
            query += f",\n        edge_data_list = edge_data_list + \",\" + to_string(e.{edge_label})"
        
        # add split 
        if edge_split:
            query += f",\n        edge_data_list = edge_data_list + \",\" + to_string(e.{edge_split})"

        # add features 
        if edge_features:
            for feature_name, feature_type in edge_features.items():
                if feature_type.upper() == "LIST":
                    query += f",\n      FOREACH feature IN e.{feature_name} DO"
                    query += f"\n        edge_data_list = edge_data_list + \",\" + to_string(feature)"
                    query += f"\n      END"
                else:
                    query += f",\n      edge_data_list = edge_data_list + \",\" + to_string(e.{feature_name})"
        query += f",\n" 
        
        # write data 
        for p in range(num_partitions):
            query += f"      IF getvid(s) % {num_partitions} == {p} THEN \n"
            query += f"        edge_file{i}_p{p}.println(edge_data_list)"
            if p == num_partitions-1: 
                query += f"\n      END;"
            else:
                query += f"\n      END,\n"
            
        query += f"\n"

    query += "}\n"

    return query

@timeit
def install_and_run_query(conn, gsql_query, timeout=200000, force=True):
    try:
        logger.info("Installing the GSQL query...")
        query_name = install_query(conn, gsql_query, force=force)
    except Exception as install_error:
        logger.exception("Error installing the GSQL query: %s", install_error)
        raise  

    try:
        logger.info("Running the GSQL query to export the data...")
        conn.runInstalledQuery(query_name, timeout=timeout)
    except Exception as run_error:
        logger.exception("Error running the installed GSQL query: %s", run_error)
        raise