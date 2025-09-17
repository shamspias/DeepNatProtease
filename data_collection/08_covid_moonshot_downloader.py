"""
Download COVID Moonshot data - crowd-sourced SARS-CoV-2 Mpro inhibitors
This is specifically for SARS-CoV-2 main protease
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
from rdkit import Chem
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/covid_moonshot_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class COVIDMoonshotDownloader:
    """Download COVID Moonshot SARS-CoV-2 Mpro inhibitor data"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize COVID Moonshot downloader"""
        self.targets = self._load_targets(config_path)

        # COVID Moonshot data sources (as of 2025)
        self.moonshot_urls = {
            'activity_data': 'https://covid.postera.ai/covid/activity_data.csv',
            'submissions': 'https://covid.postera.ai/covid/submissions.csv',
            'experimental': 'https://github.com/postera-ai/COVID_moonshot_submissions/raw/main/covid_submissions_all_info.csv'
        }

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def download_moonshot_data(self) -> pd.DataFrame:
        """
        Download COVID Moonshot activity data

        Returns:
            DataFrame with SARS-CoV-2 Mpro inhibitor data
        """
        all_data = []

        # Try to download activity data
        for source_name, url in self.moonshot_urls.items():
            logger.info(f"Downloading COVID Moonshot {source_name} from {url}")

            try:
                # Download CSV
                df = pd.read_csv(url)
                logger.info(f"Downloaded {len(df)} entries from {source_name}")

                # Process based on source type
                if source_name == 'activity_data':
                    # Main activity data with IC50 values
                    processed_df = self._process_activity_data(df)
                elif source_name == 'submissions':
                    # User submissions with predicted activities
                    processed_df = self._process_submissions(df)
                elif source_name == 'experimental':
                    # Experimental validation data
                    processed_df = self._process_experimental(df)
                else:
                    processed_df = pd.DataFrame()

                if not processed_df.empty:
                    all_data.append(processed_df)

            except Exception as e:
                logger.warning(f"Error downloading {source_name}: {e}")
                continue

        if not all_data:
            logger.error("No COVID Moonshot data could be downloaded")
            return pd.DataFrame()

        # Combine all sources
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (keep most potent)
        if 'activity_nm' in combined_df.columns:
            combined_df = combined_df.sort_values('activity_nm')
        combined_df = combined_df.drop_duplicates(subset=['canonical_smiles'], keep='first')

        logger.info(f"Total unique compounds from COVID Moonshot: {len(combined_df)}")

        return combined_df

    def _process_activity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the main activity data file

        Args:
            df: Raw activity data

        Returns:
            Processed DataFrame
        """
        processed_data = []

        # Expected columns: SMILES, CDD_ID, f_avg_IC50, f_avg_pIC50, etc.
        smiles_cols = [col for col in df.columns if 'SMILES' in col.upper()]
        ic50_cols = [col for col in df.columns if 'IC50' in col.upper() and 'pIC50' not in col.upper()]

        if not smiles_cols:
            logger.warning("No SMILES column found in activity data")
            return pd.DataFrame()

        smiles_col = smiles_cols[0]

        # Get SARS-CoV-2 threshold
        sars_cov2_info = self.targets.get('sars_cov2', {})
        threshold_nm = sars_cov2_info.get('activity_threshold_nm', 10000)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing activity data"):
            smiles = row[smiles_col]

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            # Extract IC50 if available
            ic50_nm = None
            for ic50_col in ic50_cols:
                if pd.notna(row[ic50_col]):
                    try:
                        # Convert to nM if needed (often in uM)
                        value = float(row[ic50_col])
                        if 'uM' in ic50_col or value < 1000:  # Likely in uM
                            ic50_nm = value * 1000
                        else:
                            ic50_nm = value
                        break
                    except (ValueError, TypeError):
                        continue

            # If no IC50, check for other activity indicators
            if ic50_nm is None:
                # Check if marked as active/inactive
                if 'active' in row and pd.notna(row['active']):
                    if row['active'] in [1, True, 'True', 'active']:
                        ic50_nm = threshold_nm / 10  # Pseudo-active
                    else:
                        ic50_nm = threshold_nm * 10  # Pseudo-inactive
                else:
                    continue  # Skip if no activity info

            processed_data.append({
                'canonical_smiles': canonical_smiles,
                'activity_nm': ic50_nm,
                'is_active': int(ic50_nm <= threshold_nm),
                'standard_type': 'IC50',
                'standard_value': ic50_nm,
                'source': 'COVID_Moonshot',
                'assay_type': 'Mpro_enzymatic'
            })

        return pd.DataFrame(processed_data)

    def _process_submissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process user submission data

        Args:
            df: Raw submission data

        Returns:
            Processed DataFrame
        """
        processed_data = []

        # Get SARS-CoV-2 threshold
        sars_cov2_info = self.targets.get('sars_cov2', {})
        threshold_nm = sars_cov2_info.get('activity_threshold_nm', 10000)

        # Find SMILES column
        smiles_cols = [col for col in df.columns if 'SMILES' in col.upper()]
        if not smiles_cols:
            return pd.DataFrame()

        smiles_col = smiles_cols[0]

        for _, row in df.iterrows():
            smiles = row[smiles_col]

            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            # Submissions are potential hits, assign optimistic pseudo-activity
            processed_data.append({
                'canonical_smiles': canonical_smiles,
                'activity_nm': threshold_nm * 2,  # Slightly inactive (unknown)
                'is_active': 0,
                'standard_type': 'predicted',
                'standard_value': threshold_nm * 2,
                'source': 'COVID_Moonshot_submission',
                'assay_type': 'computational'
            })

        return pd.DataFrame(processed_data)

    def _process_experimental(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process experimental validation data

        Args:
            df: Raw experimental data

        Returns:
            Processed DataFrame
        """
        # Similar to activity data processing
        return self._process_activity_data(df)

    def process_for_virus(self, virus_key: str) -> pd.DataFrame:
        """
        Process COVID Moonshot data for a specific virus
        Only applicable for SARS-CoV-2

        Args:
            virus_key: Virus identifier

        Returns:
            DataFrame with COVID Moonshot data or empty if not SARS-CoV-2
        """
        if virus_key != 'sars_cov2':
            logger.info(f"COVID Moonshot data not applicable for {virus_key}")
            return pd.DataFrame()

        logger.info("Downloading COVID Moonshot data for SARS-CoV-2")

        # Download all COVID Moonshot data
        df = self.download_moonshot_data()

        if df.empty:
            return df

        # Add virus identifier
        df['virus'] = virus_key

        # Ensure required columns
        if 'source' not in df.columns:
            df['source'] = 'COVID_Moonshot'

        logger.info(f"Processed {len(df)} COVID Moonshot compounds for SARS-CoV-2")
        logger.info(f"Active compounds: {df['is_active'].sum()}")
        logger.info(f"Inactive compounds: {(~df['is_active'].astype(bool)).sum()}")

        return df

    def download_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """
        Download COVID Moonshot data (only for SARS-CoV-2)
        """
        results = {}

        for virus_key in self.targets.keys():
            if virus_key != 'sars_cov2':
                # Skip non-SARS-CoV-2 viruses
                results[virus_key] = {
                    'total_compounds': 0,
                    'note': 'COVID Moonshot data only available for SARS-CoV-2'
                }
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing COVID Moonshot data for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Process COVID Moonshot data
                df = self.process_for_virus(virus_key)

                if not df.empty:
                    # Save to file
                    output_path = Path(output_dir) / virus_key / "raw" / "covid_moonshot_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                    # Store results
                    results[virus_key] = {
                        'total_compounds': len(df),
                        'active_compounds': df['is_active'].sum() if 'is_active' in df.columns else 0,
                        'inactive_compounds': (~df['is_active'].astype(bool)).sum() if 'is_active' in df.columns else 0,
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
    print("COVID Moonshot Data Download")
    print("Crowd-sourced SARS-CoV-2 Mpro inhibitors")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize downloader
    downloader = COVIDMoonshotDownloader()

    # Download all target data
    results = downloader.download_all_targets()

    # Generate summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for virus, stats in results.items():
        if 'error' not in stats:
            if stats.get('total_compounds', 0) > 0:
                print(f"\n{virus.upper()}:")
                print(f"  Total compounds: {stats['total_compounds']:,}")
                print(f"  Active: {stats['active_compounds']:,}")
                print(f"  Inactive: {stats['inactive_compounds']:,}")
            elif virus != 'sars_cov2':
                print(f"\n{virus.upper()}: Not applicable (COVID Moonshot is SARS-CoV-2 specific)")

    # Save summary
    summary_path = "data/activity/covid_moonshot_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    print("\n✓ COVID Moonshot download complete!")
    print("\nNext step: Run data_collection/09_data_integration.py")
    print("This will integrate all data sources (ChEMBL, BindingDB, PubChem, ZINC, COVID Moonshot)")


if __name__ == "__main__":
    main()
