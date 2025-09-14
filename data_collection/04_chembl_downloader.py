"""
Download activity data from ChEMBL database for each viral protease
(Updated: exact assay_type filter, robust tqdm, joins SMILES, stricter target filters)
"""

import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chembl_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def chunked(iterable: List, size: int) -> Iterable[List]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def safe_set_page_size(ps: int = 1000):
    try:
        new_client.set_page_size(ps)
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Main downloader
# ------------------------------------------------------------------------------

class ChEMBLDownloader:
    """Download activity data from ChEMBL for viral proteases"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        self.targets = self._load_targets(config_path)
        self.activity = new_client.activity
        self.molecule = new_client.molecule
        self.assay = new_client.assay
        self.target = new_client.target
        safe_set_page_size(1000)

    def _load_targets(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    # ---------------------------- core fetchers ---------------------------- #

    def _iter_activities(self, chembl_target_id: str, limit: Optional[int] = None) -> Iterable[dict]:
        """
        Iterate activities with strict filters suited for viral protease potency mining.
        - assay_type: only Binding/Functional (no regex)
        - target_chembl_id: exact viral protease
        - relationship_type='D': direct target assignment
        - target_type='SINGLE PROTEIN': avoid NON-PROTEIN/complexes
        """
        query = self.activity.filter(
            target_chembl_id=chembl_target_id,
            standard_type__in=['IC50', 'Ki', 'Kd', 'EC50'],
            standard_relation__in=['=', '<', '<='],
            assay_type__in=['B', 'F'],
            relationship_type='D',
            target_type='SINGLE PROTEIN',
        )

        # Optional: require a numeric standardized potency present (comment out if too strict)
        # query = query.filter(pchembl_value__isnull=False)

        if limit is not None:
            query = query[:limit]

        return query

    def _fetch_smiles_map(self, mol_ids: List[str]) -> Dict[str, str]:
        """Lookup canonical SMILES for a list of molecule ChEMBL IDs."""
        smiles_map: Dict[str, str] = {}
        if not mol_ids:
            return smiles_map

        for batch in chunked(sorted(set(mol_ids)), 1000):
            try:
                recs = self.molecule.filter(molecule_chembl_id__in=batch).only(
                    ['molecule_chembl_id', 'molecule_structures']
                )
                for r in recs:
                    mid = r.get('molecule_chembl_id')
                    ms = r.get('molecule_structures') or {}
                    smi = ms.get('canonical_smiles')
                    if mid and smi:
                        smiles_map[mid] = smi
            except Exception as e:
                logger.warning(f"SMILES lookup failed for a batch of {len(batch)} molecules: {e}")
                continue

        return smiles_map

    def download_target_data(self, virus_key: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Download activity data for a specific viral protease
        """
        if virus_key not in self.targets:
            raise ValueError(f"Unknown virus: {virus_key}")

        target_info = self.targets[virus_key]
        chembl_id = target_info['chembl_id']

        logger.info(f"Downloading data for {target_info['name']} ({chembl_id})")
        logger.info(f"Fetching activities for {virus_key}...")

        activity_list: List[dict] = []
        attempts, max_attempts, delay = 0, 3, 2.0

        with tqdm(desc=f"Downloading {virus_key} activities", unit="act") as pbar:
            while attempts < max_attempts:
                try:
                    for act in self._iter_activities(chembl_id, limit=limit):
                        activity_list.append(act)
                        pbar.update(1)
                        time.sleep(0.005)  # gentle pacing
                    break
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Iteration error ({attempts}/{max_attempts}) for {virus_key}: {e}")
                    if attempts >= max_attempts:
                        logger.error(f"Giving up after {attempts} attempts for {virus_key}")
                        break
                    time.sleep(delay)
                    delay *= 2  # backoff

        logger.info(f"Downloaded {len(activity_list)} activities for {virus_key}")

        if not activity_list:
            logger.warning(f"No activities found for {virus_key}")
            return pd.DataFrame()

        df = pd.DataFrame.from_records(activity_list)

        # Post-filter (belt & suspenders): ensure rows are for the requested target id
        if 'target_chembl_id' in df.columns:
            before = len(df)
            df = df[df['target_chembl_id'] == chembl_id].copy()
            logger.info(f"Post-filter by target_chembl_id kept {len(df)}/{before} rows")

        # Ensure canonical_smiles exists by joining molecules if missing
        if 'canonical_smiles' not in df.columns or df['canonical_smiles'].isna().all():
            if 'molecule_chembl_id' in df.columns:
                mol_ids = df['molecule_chembl_id'].dropna().astype(str).tolist()
                smiles_map = self._fetch_smiles_map(mol_ids)
                if smiles_map:
                    df['canonical_smiles'] = df['molecule_chembl_id'].map(smiles_map)

        # Keep relevant columns that exist
        columns_to_keep = [
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_type',
            'standard_relation',
            'standard_value',
            'standard_units',
            'pchembl_value',
            'activity_comment',
            'assay_chembl_id',
            'assay_description',
            'assay_type',
            'target_chembl_id',
            'target_pref_name',
            'target_organism',
            'document_chembl_id',
            'document_year',
            'src_id'
        ]
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        df = df[columns_to_keep].copy()

        # Clean & standardize
        df = self._clean_activity_data(df, target_info['activity_threshold_nm'])
        return df

    # ---------------------------- cleaning ---------------------------- #

    def _convert_to_nm(self, row) -> Optional[float]:
        value = row.get('standard_value')
        units = (row.get('standard_units') or 'nM').strip()
        if pd.isna(value):
            return None
        conversions = {
            'nM': 1, 'nm': 1,
            'uM': 1000, 'µM': 1000,
            'mM': 1_000_000,
            'M': 1_000_000_000,
            'pM': 0.001
        }
        factor = conversions.get(units, None)
        return float(value) * (factor if factor is not None else 1.0)

    def _clean_activity_data(self, df: pd.DataFrame, threshold_nm: int) -> pd.DataFrame:
        if df.empty:
            return df

        logger.info("Cleaning activity data...")

        # Remove rows without SMILES if present
        if 'canonical_smiles' in df.columns:
            df = df.dropna(subset=['canonical_smiles'])

        # Require numeric activity
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])
        df = df[df['standard_value'] >= 0]

        # Standardize to nM
        df['activity_nm'] = df.apply(self._convert_to_nm, axis=1)
        df = df.dropna(subset=['activity_nm'])

        # Remove extreme outliers (>10 mM)
        df = df[df['activity_nm'] <= 10_000_000]

        # Active / inactive label
        df['is_active'] = (df['activity_nm'] <= float(threshold_nm)).astype(int)

        # Source tag
        df['source'] = 'ChEMBL'

        # De-duplicate (most potent per SMILES + endpoint)
        subset_cols = [c for c in ['canonical_smiles', 'standard_type'] if c in df.columns]
        if subset_cols:
            df = df.sort_values('activity_nm').drop_duplicates(subset=subset_cols, keep='first')

        logger.info(f"Cleaned data: {len(df)} unique compounds")
        logger.info(f"Active compounds (≤{threshold_nm} nM): {int(df['is_active'].sum())}")
        logger.info(f"Inactive compounds: {int((~df['is_active'].astype(bool)).sum())}")
        return df

    # ---------------------------- orchestration ---------------------------- #

    def download_all_targets(self, output_dir: str = "data/activity") -> Dict:
        results: Dict[str, Dict] = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                df = self.download_target_data(virus_key)

                if not df.empty:
                    output_path = Path(output_dir) / virus_key / "raw" / "chembl_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                    results[virus_key] = {
                        'total_compounds': int(len(df)),
                        'active_compounds': int(df['is_active'].sum()),
                        'inactive_compounds': int((~df['is_active'].astype(bool)).sum()),
                        'unique_assays': int(df['assay_chembl_id'].nunique()) if 'assay_chembl_id' in df.columns else 0,
                        'file_path': str(output_path)
                    }
                else:
                    logger.warning(f"No data retrieved for {virus_key}")
                    results[virus_key] = {
                        'total_compounds': 0,
                        'active_compounds': 0,
                        'inactive_compounds': 0,
                        'unique_assays': 0,
                        'file_path': None
                    }

            except Exception as e:
                logger.exception(f"Error downloading data for {virus_key}: {e}")
                results[virus_key] = {'error': str(e)}

            time.sleep(1.0)

        return results

    def get_summary_statistics(self, results: Dict) -> pd.DataFrame:
        summary_data = []
        for virus_key, stats in results.items():
            if 'error' not in stats:
                total = stats.get('total_compounds', 0) or 1
                active = stats.get('active_compounds', 0)
                summary_data.append({
                    'Virus': virus_key.upper(),
                    'Total Compounds': stats.get('total_compounds', 0),
                    'Active': active,
                    'Inactive': stats.get('inactive_compounds', 0),
                    'Active Ratio': f"{active / total * 100:.1f}%",
                    'Unique Assays': stats.get('unique_assays', 0)
                })
        return pd.DataFrame(summary_data)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ChEMBL Data Download for Viral Proteases")
    print("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path("data/activity").mkdir(parents=True, exist_ok=True)

    downloader = ChEMBLDownloader()

    results = downloader.download_all_targets()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    summary_df = downloader.get_summary_statistics(results)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    summary_path = "data/activity/chembl_download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    total_compounds = sum(r.get('total_compounds', 0) for r in results.values() if 'error' not in r)
    total_active = sum(r.get('active_compounds', 0) for r in results.values() if 'error' not in r)

    print(f"\n{'=' * 60}")
    print(f"Total compounds downloaded: {total_compounds:,}")
    print(f"Total active compounds: {total_active:,}")
    print(f"{'=' * 60}")

    print("\n✓ ChEMBL download complete!")
    print("\nNext step: Run data_collection/05_bindingdb_downloader.py to download BindingDB data")


if __name__ == "__main__":
    main()
