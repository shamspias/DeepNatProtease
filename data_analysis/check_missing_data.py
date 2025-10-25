import pandas as pd

# Load CSV
df = pd.read_csv("../data/virus_activity/raw/chembl36/data.csv", sep='\t')

# Check for missing values
missing_counts = df.isnull().sum()

print("Missing data per field:")
print(missing_counts)

# Optionally show only columns with missing data
missing_only = missing_counts[missing_counts > 0]
if not missing_only.empty:
    print("\nColumns with missing data:")
    print(missing_only)
else:
    print("\n✅ No missing data found!")
