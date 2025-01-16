import numpy as np
import pandas as pd
from multiprocessing import Pool

# Load the .npz file
file_path = '/tg/gnn/cugraph/python/cugraph-pyg/cugraph_pyg/examples/dataset/ogbn_papers100M/raw/data.npz'  # Replace with your file path
data = np.load(file_path)

def write_partition_to_csv(args):
    """Write a partition of the data to a CSV file."""
    key, partition_index, partition_data = args
    # Convert the partition to a pandas DataFrame
    df = pd.DataFrame(partition_data)
    
    # Define the output CSV file name (use key and partition_index as part of the filename)
    output_file = f"/tg/himanshu/data/ogbn_papers100M/{key}_part{partition_index}.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Partition {partition_index} of key '{key}' has been written to {output_file}.")

def split_and_prepare_tasks(key, array, partition_size):
    """Split the data into partitions and prepare tasks."""
    tasks = []
    num_partitions = (len(array) + partition_size - 1) // partition_size  # Calculate the number of partitions
    for i in range(num_partitions):
        start = i * partition_size
        end = min((i + 1) * partition_size, len(array))
        partition_data = array[start:end]
        tasks.append((key, i, partition_data))
    return tasks

# Define the partition size (e.g., 100,000 rows per partition)
partition_size = 1000000

# Prepare data for multiprocessing (split each key's data into partitions)
tasks = []
for key in data.keys():
    array = data[key]
    tasks.extend(split_and_prepare_tasks(key, array, partition_size))

# Use a multiprocessing pool to write CSVs in parallel
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(write_partition_to_csv, tasks)

print("All partitions have been written to CSV files.")
