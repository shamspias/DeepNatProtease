import os
import pandas as pd

# Root directory
root_dir = "../data/virus_activity/raw"

# List to collect all DataFrames
all_dfs = []

# Loop through all subdirectories
for subdir in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, subdir, "data.csv")
    if os.path.isfile(dataset_path):
        try:
            df = pd.read_csv(dataset_path, sep='\t')
            df["source_dataset"] = subdir  # optional: track source
            all_dfs.append(df)
            print(f"Loaded: {dataset_path} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {dataset_path}: {e}")

# Merge all data
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✅ Merged total: {len(merged_df)} rows from {len(all_dfs)} datasets.")

    # Ensure output directory exists
    output_dir = "../data/virus_activity"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_all.csv")

    # Save merged file
    merged_df.to_csv(output_path, index=False, sep='\t')
    print(f"💾 Saved merged file to: {output_path}")
else:
    print("⚠️ No CSV files found in subdirectories.")
