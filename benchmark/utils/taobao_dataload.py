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
    parser.add_argument('--user_item_file', '-u', type=str, default="",
                        help='Path to the edge file')
    parser.add_argument('--item_item_file', '-i', type=str, default="",
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

 
    if args.user_item_file != "":
        user_item_load = f"""
        USE GRAPH {args.graph_name}
        BEGIN
        CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
            DEFINE FILENAME f="{args.user_item_file}";
            LOAD f TO EDGE user_to_item VALUES ($0, $1) USING header="true", separator=",";
        }}
        END
        RUN LOADING JOB load_data
        DROP JOB load_data
        set exit_on_error = "false"
        """
    
    if args.item_item_file != "":
        item_item_load = f"""
        USE GRAPH {args.graph_name}
        BEGIN
        CREATE LOADING JOB load_data FOR GRAPH {args.graph_name} {{
            DEFINE FILENAME f="{args.item_item_file}";
            LOAD f TO EDGE item_to_item VALUES ($0, $1) USING header="true", separator=",";
        }}
        END
        RUN LOADING JOB load_data
        DROP JOB load_data
        set exit_on_error = "false"
        """

    # Execute the GSQL script
    try:
        print("Executing GSQL script...")
        if args.user_item_file != "":
            response = conn.gsql(user_item_load)
        if args.item_item_file != "":
            response = conn.gsql(item_item_load)
        print("GSQL script executed successfully.")
        print("Response:", response)
    except Exception as e:
        print(f"Error executing GSQL script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

