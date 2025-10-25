import pandas as pd

# Load CSV
df = pd.read_csv("../data/virus_activity/raw/chembl36/data.csv", sep='\t')

# Show number of rows and columns
print(f"Total rows (data entries): {len(df)}")
print(f"Total columns: {len(df.columns)}")
