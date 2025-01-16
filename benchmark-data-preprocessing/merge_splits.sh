#!/bin/bash

# Define the split codes
train_split=0
valid_split=1
test_split=2

# Define input files
train_file="train.csv"
valid_file="valid.csv"
test_file="test.csv"

# Define output file
output_file="merged.csv"

# Initialize output file with header
echo "node,split" > "$output_file"

# Process train file
while IFS=, read -r node_id; do
    echo "$node_id,$train_split" >> "$output_file"
done < "$train_file"

# Process valid file
while IFS=, read -r node_id; do
    echo "$node_id,$valid_split" >> "$output_file"
done < "$valid_file"

# Process test file
while IFS=, read -r node_id; do
    echo "$node_id,$test_split" >> "$output_file"
done < "$test_file"

echo "Merging completed. Output saved to $output_file."
