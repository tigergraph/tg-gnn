import numpy as np
import pandas as pd
from multiprocessing import Pool

# Load the .npz file
file_path = '/tg/gnn/cugraph/python/cugraph-pyg/cugraph_pyg/examples/dataset/ogbn_papers100M/raw/data.npz'  # Replace with your file path
data = np.load(file_path)

def write_to_csv(args):
    """Write a single key's data to a CSV file."""
    key, array = args
    # Convert the array to a pandas DataFrame
    df = pd.DataFrame(array)
    
    # Define the output CSV file name (use key as part of the filename)
    output_file = f"/tg/himanshu/data/ogbn_papers100M/{key}.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data under key '{key}' has been written to {output_file}.")

if False:
    # Prepare data for multiprocessing (pass as a list of tuples)
    tasks = [(key, data[key]) for key in data.keys()]

    # Use a multiprocessing pool to write CSVs in parallel
    if __name__ == "__main__":
        with Pool() as pool:
            pool.map(write_to_csv, tasks)

    print("All CSV files have been written.")




# Load the .npz file
# file_path = "/tg/gnn/cugraph/python/cugraph-pyg/cugraph_pyg/examples/dataset/ogbn_papers100M/raw/node-label.npz"
# data = np.load(file_path)


import polars as pl


# Extract the array
node_labels = data['edge_index']  # Assuming the key is 'node_label'

# Flatten the array to 1D if it's multidimensional
# node_labels = node_labels.flatten()
# node_labels = np.nan_to_num(node_labels, nan=-1).astype(int)
# # Create a Polars DataFrame with index and values
# df = pl.DataFrame({
#     "Index": range(len(node_labels)),
#     "Value": node_labels
# })



# Transpose the data if needed to ensure rows represent pairs
data = node_labels.T  # Transpose the 2D array if it's in the format [columns x rows]

# Create a Polars DataFrame with two columns
df = pl.DataFrame({
    "Column1": data[:, 0],
    "Column2": data[:, 1]
})

# Write to CSV
output_csv = "output.csv"
df.write_csv(output_csv)
# Write to CSV
output_csv = "/tg/himanshu/data/ogbn_papers100M/edges.csv"
df.write_csv(output_csv)

print(f"CSV file written to {output_csv}")





