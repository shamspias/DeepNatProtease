# This is a lightweight data collection stub. 
# Replace with your real data ingestion to populate train/val/test CSVs per virus.
from pathlib import Path
import pandas as pd

VIRUSES = ["dengue", "zika", "hiv_1", "sars_cov_2", "hcv", "west_nile"]

for v in VIRUSES:
    vd = Path(f"data/{v}")
    vd.mkdir(parents=True, exist_ok=True)
    # Create minimal placeholder CSVs with headers expected by ChemProp
    for split in ["train", "val", "test"]:
        f = vd / f"{split}.csv"
        if not f.exists():
            df = pd.DataFrame({
                "smiles": [],   # fill with SMILES strings
                "active": []    # 1 for active, 0 for inactive
            })
            df.to_csv(f, index=False)
            print(f"Created empty placeholder: {f}")
print("✅ virus_specific_pipeline.py completed (placeholders).")
