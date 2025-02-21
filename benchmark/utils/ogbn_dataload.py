import argparse
from pyTigerGraph import TigerGraphConnection
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load data into TigerGraph using a loading job.")

    parser.add_argument('--host', type=str, default="http://172.17.0.3",
                        help='TigerGraph server host URL (default: http://localhost)')
    parser.add_argument('--graph_name', '-g', type=str, required=True,
                        help='Name of the TigerGraph graph')
    parser.add_argument('--username', type=str, default="tigergraph",
                        help='Username for TigerGraph (default: tigergraph)')
    parser.add_argument('--password', type=str, default="tigergraph",
                        help='Password for TigerGraph (default: tigergraph)')
    parser.add_argument('--node_feature_file', '-f', type=str, default="",
                        help='Path to node data file')
    parser.add_argument('--node_split_file', '-s', type=str, default="",
                        help='Path to node data file')
    parser.add_argument('--node_label_file', '-l', type=str, default="",
                        help='Path to node data file')

    parser.add_argument('--edge_file', '-e', type=str, default="",
                        help='Path to the edge file')

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Establish connection to TigerGraph
    try:
        conn = TigerGraphConnection(
            host=args.host,
            graphname=args.graph_name,
            username=args.username,
            password=args.password
        )
        conn.getToken(conn.createSecret())
        print("Successfully connected to TigerGraph.")
    except Exception as e:
        print(f"Error connecting to TigerGraph: {e}")
        sys.exit(1)

    # Define the GSQL script with the provided parameters
    if args.node_feature_file != "":
        node_feature_load = f"""
            USE GRAPH {args.graph_name}
            BEGIN
            CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
                DEFINE FILENAME f="{args.node_feature_file}";
                LOAD f TO VERTEX product VALUES ($1, SPLIT($0, ','), _, _) USING header="false", separator="|";
            }}
            END
            RUN LOADING JOB load_data
            DROP JOB load_data
            set exit_on_error = "false"
        """
    if args.node_split_file != "":
        node_split_load = f"""
        USE GRAPH {args.graph_name}
        BEGIN
        CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
            DEFINE FILENAME f="{args.node_split_file}";
            LOAD f TO VERTEX product VALUES ($0, _, _, $1) USING header="true", separator=",";
        }}
        END
        RUN LOADING JOB load_data
        DROP JOB load_data
        set exit_on_error = "false"
        """
    if args.node_label_file != "":
        node_label_load = f"""
        USE GRAPH {args.graph_name}
        BEGIN
        CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
            DEFINE FILENAME f="{args.node_label_file}";
            LOAD f TO VERTEX product VALUES ($0, _,$1, _) USING header="false", separator=",";
        }}
        END
        RUN LOADING JOB load_data
        DROP JOB load_data
        set exit_on_error = "false"
        """

    if args.edge_file != "":
        edge_data_load = f"""
        USE GRAPH {args.graph_name}
        BEGIN
        CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
            DEFINE FILENAME f="{args.edge_file}";
            LOAD f TO EDGE rel VALUES ($0, $1) USING header="false", separator=",";
        }}
        END
        RUN LOADING JOB load_data
        DROP JOB load_data
        set exit_on_error = "false"
        """

    # Execute the GSQL script
    try:
        print("Executing GSQL script...")
        if args.node_feature_file != "":
            response = conn.gsql(node_feature_load)
        if args.node_split_file != "":
            response = conn.gsql(node_split_load)
        if args.edge_file != "":
            response = conn.gsql(edge_data_load)
        if args.node_label_file != "":
            response = conn.gsql(node_label_load)
        # response = conn.gsql(edge_data_load)
        print("GSQL script executed successfully.")
        print("Response:", response)
    except Exception as e:
        print(f"Error executing GSQL script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

