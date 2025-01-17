import polars as pl

def add_incremental_counter_with_delimiter(file_path, output_path, delimiter="|"):
    # Read the CSV file (assuming the original file uses a comma as a delimiter)
    df = pl.read_csv(file_path, has_header=False)
    
    # Add a counter column
    df = df.with_columns(pl.Series("NodeId", range(len(df))))
    
    # Reorder the columns to place "NodeId" as the first column
    df = df.select(["NodeId", *df.columns[:-1]])
    
    # Write the updated DataFrame to the output file with a custom delimiter
    df.write_csv(output_path, separator=delimiter)
    print(f"Updated file written to: {output_path} with delimiter '{delimiter}'")



def add_incremental_counter_with_custom_delimiter(file_path, output_path, delimiter="|"):
    # Read the CSV file
    df = pl.read_csv(file_path, has_header=False)
    
    # Add the incremental counter column
    df = df.with_columns(pl.Series("NodeId", range(len(df))))
    
    # Create the `NodeId_with_delimiter` column and drop `NodeId` immediately after
    df = df.with_columns(
        (df["column_128"].cast(pl.Utf8) + delimiter + df["NodeId"].cast(pl.Utf8)).alias("NodeId_with_delimiter")
    )
    df = df.drop(["NodeId", "column_128"])
    # Write the updated DataFrame to the output file
    df.write_csv(output_path)
    print(f"Updated file written to: {output_path}")






# Example usage
file_path = "/tg/himanshu/data/ogbn_papers100M/node_feat.csv" 
output_path = "/tg/himanshu/data/ogbn_papers100M/node_feat_with_id.csv" 

add_incremental_counter_with_custom_delimiter(file_path, output_path, delimiter="|")