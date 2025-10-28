import pandas as pd

# Load CSV
df = pd.read_csv("../data/virus_activity/raw/merged_all.csv", sep='\t')

# Group by virus_name and count occurrences
virus_counts = df["virus_name"].value_counts().head(10)

print("Top 10 viruses by data count:")
print(virus_counts)
