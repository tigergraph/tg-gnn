#!/bin/bash

# Input files
merged_file="merged.csv"
features_file="features.csv"
labels_file="labels.csv"

# Output file
output_file="merged_with_features_and_labels.csv"

# Temporary files
merged_no_header="merged_no_header.csv"
features_temp="features_temp.csv"
temp_output="temp_output.csv"
header_file="header.csv"

# Remove header from merged.csv (if it has one)
if head -1 "$merged_file" | grep -q "node"; then
    tail -n +2 "$merged_file" > "$merged_no_header"
else
    cp "$merged_file" "$merged_no_header"
fi

# Transform features.csv: Combine columns into a single column with ';' delimiter
# awk -F',' '{OFS=";"; print $0}' "$features_file" > "$features_temp"

# Create the header
echo "node,split,feature,label" > "$header_file"

# Merge files line by line using a loop to ensure semicolon-separated features are handled correctly
{
    paste -d '|' "$merged_no_header" "$features_file" "$labels_file" > "$temp_output"
} || {
    echo "Error merging files. Please check input file alignment."
    exit 1
}

# Combine header and data
cat "$header_file" "$temp_output" > "$output_file"

# Clean up temporary files
rm "$merged_no_header" "$features_temp" "$temp_output" "$header_file"

echo "Merging completed. Output saved to $output_file."
