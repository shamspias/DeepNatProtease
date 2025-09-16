"""
Fast BindingDB extractor that stops after finding enough data
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import time
from typing_extensions import Dict

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bindingdb_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FastBindingDBExtractor:
    """Fast extraction with early stopping"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize extractor"""
        self.targets = self._load_targets(config_path)
        self.chunk_size = 10000

        # SPEED OPTIMIZATIONS
        self.max_rows_to_search = 1000000  # Stop after 1M rows (about 10 minutes)
        self.min_compounds_needed = 500  # Stop if we find at least this many
        self.max_compounds_needed = 5000  # Stop if we find this many

        # Simplified search patterns (faster)
        self.search_patterns = {
            'hiv1': ['HIV', 'immunodeficiency.*protease', 'HIV.*protease', 'P03366'],
            'hcv': ['hepatitis C.*NS3', 'HCV.*NS3', 'HCV.*protease', 'P26662'],
            'sars_cov2': ['SARS-CoV-2', 'COVID.*protease', 'coronavirus.*main', '3CL.*protease', 'Mpro'],
            'dengue': ['dengue.*NS3', 'dengue.*protease', 'DENV.*NS3'],
            'zika': ['zika.*NS3', 'zika.*protease', 'ZIKV.*NS3']
        }

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def extract_virus_data_fast(self, virus_key: str, tsv_path: str) -> pd.DataFrame:
        """
        Fast extraction with early stopping
        """
        logger.info(f"Fast extraction for {virus_key}")
        logger.info(f"Will stop after {self.max_rows_to_search:,} rows or {self.max_compounds_needed:,} compounds")

        patterns = self.search_patterns.get(virus_key, [])
        if not patterns:
            return pd.DataFrame()

        target_info = self.targets[virus_key]
        threshold_nm = target_info['activity_threshold_nm']

        all_data = []
        rows_searched = 0
        start_time = time.time()

        # Build regex pattern once (faster than multiple searches)
        import re
        pattern = '|'.join(patterns)
        regex = re.compile(pattern, re.IGNORECASE)

        with tqdm(desc=f"Fast search for {virus_key}", unit="rows") as pbar:
            for chunk in pd.read_csv(tsv_path, sep='\t', chunksize=self.chunk_size,
                                     low_memory=False, encoding='latin-1',
                                     on_bad_lines='skip'):

                rows_searched += len(chunk)
                pbar.update(len(chunk))

                # Quick search - combine all text columns and search once
                text_cols = chunk.select_dtypes(include=['object']).columns

                # Create combined text field for faster searching
                combined_text = chunk[text_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)

                # Single regex search
                mask = combined_text.str.contains(regex, na=False)

                if mask.any():
                    matches = chunk[mask]

                    # Find SMILES column
                    smiles_col = None
                    for col in chunk.columns:
                        if 'SMILES' in str(col).upper():
                            smiles_col = col
                            break

                    if not smiles_col:
                        continue

                    # Process activity data
                    activity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']

                    for _, row in matches.iterrows():
                        smiles = row.get(smiles_col)
                        if pd.isna(smiles) or not smiles:
                            continue

                        # Check activity columns
                        for col in activity_cols:
                            if col in row and pd.notna(row[col]):
                                try:
                                    value = float(row[col])
                                    if value > 0:
                                        all_data.append({
                                            'canonical_smiles': smiles,
                                            'standard_type': col.split()[0].replace('(', ''),
                                            'standard_value': value,
                                            'activity_nm': value,
                                            'is_active': int(value <= threshold_nm),
                                            'source': 'BindingDB'
                                        })
                                except:
                                    continue

                # Early stopping conditions
                if len(all_data) >= self.max_compounds_needed:
                    logger.info(f"✓ Found {len(all_data)} compounds - stopping (max reached)")
                    break

                if rows_searched >= self.max_rows_to_search:
                    logger.info(f"✓ Searched {rows_searched:,} rows - stopping (max rows reached)")
                    break

                # Log progress
                if rows_searched % 100000 == 0:
                    elapsed = time.time() - start_time
                    rate = rows_searched / elapsed
                    logger.info(f"  Progress: {rows_searched:,} rows, {len(all_data)} compounds, {rate:.0f} rows/sec")

                    if len(all_data) >= self.min_compounds_needed:
                        logger.info(f"✓ Found enough compounds ({len(all_data)}) - you can stop anytime")

        if not all_data:
            return pd.DataFrame()

        # Create DataFrame and remove duplicates
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=['canonical_smiles', 'standard_type'], keep='first')

        elapsed = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed:.1f} seconds")
        logger.info(f"Found {len(df)} unique compounds for {virus_key}")

        return df

    def process_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """Process all viral targets with fast extraction"""

        # Find BindingDB file
        tsv_files = list(Path("data/raw").glob("BindingDB_All*.tsv"))
        if not tsv_files:
            logger.error("No BindingDB file found")
            return {}

        tsv_path = str(tsv_files[0])
        logger.info(f"Using: {tsv_path}")

        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Fast extraction
                df = self.extract_virus_data_fast(virus_key, tsv_path)

                if not df.empty:
                    # Save to file
                    output_path = Path(output_dir) / virus_key / "raw" / "bindingdb_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                    results[virus_key] = {
                        'total_compounds': len(df),
                        'active_compounds': int(df['is_active'].sum()),
                        'inactive_compounds': int((~df['is_active'].astype(bool)).sum()),
                        'file_path': str(output_path)
                    }
                else:
                    results[virus_key] = {
                        'total_compounds': 0,
                        'active_compounds': 0,
                        'inactive_compounds': 0,
                        'file_path': None
                    }

            except Exception as e:
                logger.error(f"Error processing {virus_key}: {str(e)}")
                results[virus_key] = {'error': str(e)}

        return results


def main():
    """Main execution function"""

    print("=" * 60)
    print("BindingDB FAST Extraction")
    print("Stops after finding enough data or 1M rows")
    print("=" * 60)

    # Create directories
    Path("logs").mkdir(exist_ok=True)

    # Initialize extractor
    extractor = FastBindingDBExtractor()

    # Process all targets
    start = time.time()
    results = extractor.process_all_targets()
    total_time = time.time() - start

    # Generate summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    for virus, stats in results.items():
        if 'error' not in stats:
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {stats['total_compounds']:,}")
            if stats['total_compounds'] > 0:
                print(f"  Active: {stats['active_compounds']:,}")
                print(f"  Inactive: {stats['inactive_compounds']:,}")

    print(f"\n✓ Total extraction time: {total_time / 60:.1f} minutes")

    # Save summary
    summary_path = "data/activity/bindingdb_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    print("\n✓ BindingDB extraction complete!")
    print("\nNext: python data_collection/06_pubchem_downloader.py")


if __name__ == "__main__":
    main()
