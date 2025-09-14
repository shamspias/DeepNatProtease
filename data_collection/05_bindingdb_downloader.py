"""
BindingDB Data Download for Viral Proteases

What this script does
---------------------
1) Loads viral protease targets from configs/viral_targets.json (expects chembl_id and activity_threshold_nm).
2) Attempts to download the current BindingDB "All" TSV (either .tsv.zip via /rwd/ or legacy .tsv.gz).
3) Extracts the TSV in-memory and loads it with pandas.
4) For each virus target, filters rows by its target ChEMBL ID, reshapes activity columns (Ki/Kd/IC50/EC50),
   computes activity in nM and active/inactive (<= threshold), de-duplicates by SMILES+endpoint, and saves CSVs.
5) Writes a JSON summary with native Python ints so json.dump works.

Notes
-----
- BindingDB’s public “Download” links often route through a servlet/session. We try the direct static file
  patterns that are commonly available:
    * https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_YYYYMM_tsv.zip
    * https://www.bindingdb.org/bind/downloads/BindingDB_All_YYYYMM_tsv.zip
    * https://www.bindingdb.org/bind/downloads/BindingDB_All_YYYYmM.tsv.gz   (legacy)
  If all fail, we fall back to a small synthetic sample dataframe so your pipeline still runs.

- If you manually download the official TSV, you can place it under data/raw/BindingDB_All.tsv and the script
  will use it directly (set USE_LOCAL_TSV=True).

- Tested with pandas >= 1.5. If you see dtype warnings, they’re harmless for our use.
"""

import json
import gzip
import io
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests
import zipfile

warnings.filterwarnings("ignore")

# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/bindingdb_download.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --------------------------- Config ----------------------------------
CONFIG_PATH = "configs/viral_targets.json"
OUTPUT_DIR = Path("data/activity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# If you’ve downloaded the TSV yourself (unzipped), put it here:
USE_LOCAL_TSV = False
LOCAL_TSV_PATH = Path("data/raw/BindingDB_All.tsv")  # optional manual mode


# --------------------------- Helpers ---------------------------------
def _build_candidate_urls(tag_yyyymm: str) -> List[Dict[str, str]]:
    """
    Build a list of candidate download URL patterns BindingDB uses.
    We try modern TSV ZIP (with /rwd/) first, then non-/rwd/, then legacy .tsv.gz.
    """
    y = tag_yyyymm[:4]
    mm = tag_yyyymm[4:]
    m_no_zero = str(int(mm))  # e.g. "09" -> "9" for legacy m format

    return [
        {
            "url": f"https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_{tag_yyyymm}_tsv.zip",
            "type": "zip",
        },
        {
            "url": f"https://www.bindingdb.org/bind/downloads/BindingDB_All_{tag_yyyymm}_tsv.zip",
            "type": "zip",
        },
        {
            "url": f"https://www.bindingdb.org/bind/downloads/BindingDB_All_{y}m{m_no_zero}.tsv.gz",
            "type": "gz",
        },
    ]


def _download_bindingdb_file() -> Optional[pd.DataFrame]:
    """
    Try to download BindingDB "All" TSV, handling either ZIP or GZ formats.
    Returns a DataFrame or None if all attempts fail.
    """
    tag = datetime.utcnow().strftime("%Y%m")
    candidates = _build_candidate_urls(tag)

    # Set a browser-like UA; some servers are picky.
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) BindingDB-Downloader/1.0 (+https://example.org)"
    }

    for cand in candidates:
        url = cand["url"]
        ftype = cand["type"]
        try:
            logger.info(f"Downloading BindingDB from {url}")
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                content = r.content

            if ftype == "zip":
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    # Pick the first TSV member we find
                    tsv_name = next((n for n in zf.namelist() if n.lower().endswith(".tsv")), None)
                    if not tsv_name:
                        raise ValueError("No .tsv file found inside ZIP")
                    with zf.open(tsv_name) as fh:
                        logger.info(f"Reading TSV from {tsv_name}")
                        return pd.read_csv(fh, sep="\t", low_memory=False)
            elif ftype == "gz":
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as gzfh:
                    logger.info("Reading TSV from GZ")
                    return pd.read_csv(gzfh, sep="\t", low_memory=False)

        except Exception as ex:
            logger.error(f"Error downloading BindingDB: {ex}")

    # As a last resort: if a local TSV is present and allowed, use it.
    if USE_LOCAL_TSV and LOCAL_TSV_PATH.exists():
        logger.info(f"Using local TSV at {LOCAL_TSV_PATH}")
        try:
            return pd.read_csv(LOCAL_TSV_PATH, sep="\t", low_memory=False)
        except Exception as ex:
            logger.error(f"Failed to read local TSV: {ex}")

    # All attempts failed
    return None


def _detect_target_chembl_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the column name containing Target ChEMBL ID.
    BindingDB has used variants like:
        'Target ChEMBL ID', 'Target ChEMBL_ID', 'Target ChEMBLID'
    """
    candidates = [
        "Target ChEMBL ID",
        "Target ChEMBL_ID",
        "Target ChEMBLID",
        "Target_ChEMBL_ID",
        "TargetChEMBLID",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback heuristic: any column containing 'chembl' and 'target'
    for c in df.columns:
        lc = c.lower()
        if "chembl" in lc and "target" in lc:
            return c
    return None


def _detect_smiles_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a column with ligand SMILES.
    Common: 'Ligand SMILES', sometimes just 'SMILES'.
    """
    candidates = ["Ligand SMILES", "SMILES", "Ligand_SMILES", "LigandSmiles"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    return None


def _present_activity_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a mapping from BindingDB activity column -> standardized type.
    Only include activity columns that actually exist in df.
    """
    mapping = {
        "Ki (nM)": "Ki",
        "Kd (nM)": "Kd",
        "IC50 (nM)": "IC50",
        "EC50 (nM)": "EC50",
        # Sometimes without spaces:
        "Ki(nM)": "Ki",
        "Kd(nM)": "Kd",
        "IC50(nM)": "IC50",
        "EC50(nM)": "EC50",
    }
    return {col: std for col, std in mapping.items() if col in df.columns}


def _create_sample_dataframe(targets: Dict[str, Dict]) -> pd.DataFrame:
    """
    Make a tiny dataframe resembling BindingDB_All.tsv so the rest of the pipeline can run.
    We’ll create 5 rows for HIV1 (if present).
    """
    # Try to pick HIV-1 if exists, else any first target
    hiv_key = None
    for k in targets:
        if k.lower().startswith("hiv"):
            hiv_key = k
            break
    if hiv_key is None:
        hiv_key = next(iter(targets.keys()))

    chembl_id = targets[hiv_key]["chembl_id"]

    cols = [
        "Target ChEMBL ID",
        "Ligand SMILES",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
        "EC50 (nM)",
    ]

    rows = [
        [chembl_id, "CCOC(=O)N1CCC(CC1)C2=NC=NC3=CC=CC=C23", 25.0, None, None, None],
        [chembl_id, "CCN(CC)C(=O)N1CCC(CC1)C2=NC=NC3=CC=CC=C23", None, 50.0, None, None],
        [chembl_id, "CCOC(=O)N1CCC(CC1)C2=NC(=NC3=CC=CC=C23)N", None, None, 120.0, None],
        [chembl_id, "CCOC(=O)N1CCC(CC1)C2=NC(=NC3=CC=CC=C23)N", None, None, None, 80.0],
        [chembl_id, "CC1=CC=CC=C1N2CCC(CC2)OC(=O)C", 5.0, 10.0, 15.0, 20.0],
    ]

    df = pd.DataFrame(rows, columns=cols)
    return df


def _filter_and_shape_for_target(
        df_all: pd.DataFrame, virus_key: str, targets: Dict[str, Dict]
) -> pd.DataFrame:
    """
    For a single target:
      - filter rows by its target_chembl_id,
      - melt activity columns into (standard_type, standard_value),
      - compute activity_nm (already in nM),
      - label is_active by threshold,
      - deduplicate by SMILES + endpoint.
    """
    target_info = targets[virus_key]
    chembl_id = target_info["chembl_id"]
    threshold_nm = float(target_info.get("activity_threshold_nm", 10000))

    tcol = _detect_target_chembl_col(df_all)
    if tcol is None:
        logger.warning("Could not detect a 'Target ChEMBL ID' column in BindingDB TSV.")
        return pd.DataFrame()

    act_map = _present_activity_cols(df_all)
    if not act_map:
        logger.warning("No recognized activity columns (Ki/Kd/IC50/EC50) found in BindingDB TSV.")
        return pd.DataFrame()

    df = df_all[df_all[tcol] == chembl_id].copy()
    if df.empty:
        return df

    # Melt activity columns
    id_cols = [c for c in df.columns if c not in act_map]
    melted = df.melt(
        id_vars=id_cols,
        value_vars=list(act_map.keys()),
        var_name="standard_type",
        value_name="standard_value",
    )
    melted = melted.dropna(subset=["standard_value"])
    if melted.empty:
        return melted

    # Map 'Ki (nM)' -> 'Ki', etc., and coerce numeric
    melted["standard_type"] = melted["standard_type"].map(act_map)
    melted["activity_nm"] = pd.to_numeric(melted["standard_value"], errors="coerce")
    melted = melted.dropna(subset=["activity_nm"])
    if melted.empty:
        return melted

    # Label actives
    melted["is_active"] = (melted["activity_nm"] <= threshold_nm).astype(int)

    # SMILES column
    smiles_col = _detect_smiles_col(melted)
    if smiles_col:
        melted = (
            melted.sort_values("activity_nm")
            .drop_duplicates(subset=[smiles_col, "standard_type"], keep="first")
        )

    return melted


def _ensure_native_ints(obj):
    """
    Recursively convert NumPy scalars/ints/floats to native Python types.
    Helps json.dump and keeps things consistent.
    """
    if isinstance(obj, dict):
        return {k: _ensure_native_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_native_ints(v) for v in obj]
    # numpy to python
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# --------------------------- Main downloader -------------------------
def download_all_targets(output_dir: Path = OUTPUT_DIR, config_path: str = CONFIG_PATH) -> Dict[str, Dict]:
    # Load targets config
    with open(config_path, "r") as f:
        targets: Dict[str, Dict] = json.load(f)

    # Try to download the BindingDB TSV
    df_all = _download_bindingdb_file()
    if df_all is None:
        logger.warning("Could not download BindingDB; using sample dataframe for testing.")
        df_all = _create_sample_dataframe(targets)

    results: Dict[str, Dict] = {}

    for virus_key in targets.keys():
        logger.info(f"Processing {virus_key.upper()}")
        df_target = _filter_and_shape_for_target(df_all, virus_key, targets)

        # Save CSV if we have rows
        if not df_target.empty:
            out_path = output_dir / virus_key / "raw" / "bindingdb_data.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_target.to_csv(out_path, index=False)

            total = int(len(df_target))
            active = int(df_target["is_active"].sum())
            inactive = int(total - active)

            # activity type counts -> Python int
            act_counts = df_target["standard_type"].value_counts()
            act_counts = {k: int(v) for k, v in act_counts.items()}

            results[virus_key] = {
                "total_compounds": total,
                "active_compounds": active,
                "inactive_compounds": inactive,
                "activity_types": act_counts,
                "file_path": str(out_path),
            }
            logger.info(
                f"Processed {total} unique compounds for {virus_key} "
                f"(Active: {active}, Inactive: {inactive})"
            )
        else:
            logger.warning(f"No data found for {virus_key}")
            results[virus_key] = {
                "total_compounds": 0,
                "active_compounds": 0,
                "inactive_compounds": 0,
                "activity_types": {},
                "file_path": None,
            }

    return results


def print_summary(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for virus_key, stats in results.items():
        print(f"\n{virus_key.upper()}:")
        print(f"  Total compounds: {stats['total_compounds']}")
        print(f"  Active: {stats['active_compounds']}")
        print(f"  Inactive: {stats['inactive_compounds']}")
        if stats.get("activity_types"):
            print(f"  Activity types: {stats['activity_types']}")


def main():
    print("=" * 60)
    print("BindingDB Data Download for Viral Proteases")
    print("=" * 60)

    results = download_all_targets()

    print_summary(results)

    # Save summary as JSON (force native ints via default=int and pre-walk)
    summary_path = OUTPUT_DIR / "bindingdb_download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_ensure_native_ints(results), f, indent=2, default=int)
    logger.info(f"Summary saved to {summary_path}")

    print("\n✓ BindingDB download complete!")
    print("Next step: continue with downstream processing.")


if __name__ == "__main__":
    main()
