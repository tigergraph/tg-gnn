#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
import sys

def main():
    # Set up command line argument parsing.
    parser = argparse.ArgumentParser(
        description="Convert a .pt file containing an edge_index tensor (shape: (2, n)) to a CSV file with columns 'src' and 'dst'."
    )
    parser.add_argument("--input", help="Path to the input .pt file.")
    parser.add_argument("--output", help="Path to the output CSV file.")
    args = parser.parse_args()

    # Load the tensor from the input .pt file.
    try:
        edge_index = torch.load(args.input, weights_only=True)
    except Exception as e:
        print(f"Error loading file {args.input}: {e}")
        sys.exit(1)

    # Check if the tensor has the expected shape.
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        print("Error: The loaded tensor does not have the expected shape (2, n).")
        sys.exit(1)

    # Convert the tensor to a NumPy array and transpose it so that each row is an edge.
    edges = edge_index.numpy().T  # Now edges is an array of shape (n, 2)

    # Create a DataFrame with two columns: 'src' and 'dst'.
    df = pd.DataFrame(edges, columns=["src", "dst"])

    # Save the DataFrame to a CSV file.
    try:
        df.to_csv(args.output, index=False)
    except Exception as e:
        print(f"Error writing CSV file {args.output}: {e}")
        sys.exit(1)

    print(f"CSV file has been successfully written to {args.output}.")

if __name__ == "__main__":
    main()
