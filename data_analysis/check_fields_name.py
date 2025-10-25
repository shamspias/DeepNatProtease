import pandas as pd

# Load CSV
df = pd.read_csv("../data/virus_activity/raw/chembl36/data.csv", sep='\t')

# Print all field (column) names
print("Field names:")
for col in df.columns:
    print("-", col)
