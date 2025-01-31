import argparse
from pyTigerGraph import TigerGraphConnection
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load data into TigerGraph using a loading job.")

    parser.add_argument('--host', type=str, default="http://localhost",
                        help='TigerGraph server host URL (default: http://localhost)')
    parser.add_argument('--graph_name', '-g', type=str, required=True,
                        help='Name of the TigerGraph graph')
    parser.add_argument('--username', type=str, default="tigergraph",
                        help='Username for TigerGraph (default: tigergraph)')
    parser.add_argument('--password', type=str, default="tigergraph",
                        help='Password for TigerGraph (default: tigergraph)')
    parser.add_argument('--node_file', '-n', type=str, required=True,
                        help='Path to node data file')
    parser.add_argument('--edge_file', '-e', type=str, required=True,
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
    gsql_script = f"""
    USE GRAPH {args.graph_name}
    BEGIN
    CREATE LOADING JOB load_social FOR GRAPH {args.graph_name} {{
        DEFINE FILENAME file1="{args.file1}";
        DEFINE FILENAME file2="{args.file2}";
        LOAD file2 TO VERTEX product VALUES ($0, _) USING header="false", separator=",";
        LOAD file2 TO EDGE rel VALUES ($0, $1, _) USING header="false", separator=",";
    }}
    END
    RUN LOADING JOB load_social
    DROP JOB load_social
    set exit_on_error = "false"
    """

    # Execute the GSQL script
    try:
        print("Executing GSQL script...")
        response = conn.gsql(gsql_script)
        print("GSQL script executed successfully.")
        print("Response:", response)
    except Exception as e:
        print(f"Error executing GSQL script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
