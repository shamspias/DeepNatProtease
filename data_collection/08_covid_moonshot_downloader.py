"""
Download COVID Moonshot data (SARS-CoV-2 specific crowd-sourced inhibitors)
Fixed: JSON serialization for numpy types
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
    """Download COVID Moonshot data for SARS-CoV-2 Mpro inhibitors"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize downloader"""
        self.targets = self._load_targets(config_path)

        # COVID Moonshot URLs (updated for 2025)
        self.urls = {
            'activity_data': 'https://covid.postera.ai/covid/activity_data.csv',
            'submissions': 'https://covid.postera.ai/covid/submissions.csv',
            # Alternative/backup URLs
            'github_backup': 'https://raw.githubusercontent.com/postera-ai/COVID_moonshot_submissions/master/covid_submissions_all_info.csv'
        }

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def download_moonshot_data(self) -> pd.DataFrame:
        """
        Download and process COVID Moonshot data

        Returns:
            DataFrame with standardized activity data
        """
        logger.info("Downloading COVID Moonshot data for SARS-CoV-2")

        all_data = []

        # 1. Download activity data (primary source)
        try:
            logger.info(f"Downloading COVID Moonshot activity_data from {self.urls['activity_data']}")
            response = requests.get(self.urls['activity_data'], timeout=30)
            response.raise_for_status()

            # Read CSV from response
            import io
            activity_df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Downloaded {len(activity_df)} entries from activity_data")

            # Process activity data
            processed_activity = []
            for _, row in tqdm(activity_df.iterrows(), total=len(activity_df), desc="Processing activity data"):
                # Extract relevant fields
                smiles = row.get('SMILES') or row.get('smiles') or row.get('canonical_smiles')

                # Get activity values - COVID Moonshot uses different naming
                ic50 = row.get('f_avg_IC50') or row.get('IC50_(nM)') or row.get('IC50_nM')
                pIC50 = row.get('pIC50')

                if smiles:
                    # Validate SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Convert pIC50 to IC50 if needed
                        if pIC50 and pd.notna(pIC50) and not ic50:
                            try:
                                ic50 = 10 ** (9 - float(pIC50))  # Convert pIC50 to nM
                            except:
                                ic50 = None

                        # Determine activity
                        if ic50 and pd.notna(ic50):
                            try:
                                activity_nm = float(ic50)
                                if activity_nm > 0:
                                    processed_activity.append({
                                        'canonical_smiles': smiles,
                                        'standard_type': 'IC50',
                                        'standard_value': activity_nm,
                                        'activity_nm': activity_nm,
                                        'is_active': int(activity_nm <= 1000),  # 1 µM threshold
                                        'source': 'COVID_Moonshot_Activity',
                                        'compound_id': row.get('CID') or row.get('compound_id') or row.get('TITLE', '')
                                    })
                            except:
                                pass
                        else:
                            # If no IC50, treat as inactive (screening data)
                            processed_activity.append({
                                'canonical_smiles': smiles,
                                'standard_type': 'Screening',
                                'standard_value': 10000,
                                'activity_nm': 10000,
                                'is_active': 0,
                                'source': 'COVID_Moonshot_Activity',
                                'compound_id': row.get('CID') or row.get('compound_id') or row.get('TITLE', '')
                            })

            all_data.extend(processed_activity)
            logger.info(f"Processed {len(processed_activity)} compounds from activity data")

        except Exception as e:
            logger.warning(f"Error downloading activity data: {str(e)}")

        # 2. Download submissions data (additional compounds)
        try:
            logger.info(f"Downloading COVID Moonshot submissions from {self.urls['submissions']}")
            response = requests.get(self.urls['submissions'], timeout=30)
            response.raise_for_status()

            # Read CSV from response
            import io
            submissions_df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Downloaded {len(submissions_df)} entries from submissions")

            # Process submissions (these are mostly untested compounds)
            for _, row in submissions_df.iterrows():
                smiles = row.get('SMILES') or row.get('smiles')

                if smiles:
                    # Validate SMILES
                    try:
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol:
                            # Submissions are generally untested, so mark as inactive
                            all_data.append({
                                'canonical_smiles': str(smiles),
                                'standard_type': 'Submission',
                                'standard_value': 10000,
                                'activity_nm': 10000,
                                'is_active': 0,
                                'source': 'COVID_Moonshot_Submission',
                                'compound_id': row.get('CID') or row.get('compound_id', '')
                            })
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Error downloading submissions: {str(e)}")

        # 3. Try GitHub backup if needed
        if len(all_data) < 100:  # If we have very little data, try backup
            try:
                logger.info(f"Trying GitHub backup from {self.urls['github_backup']}")
                response = requests.get(self.urls['github_backup'], timeout=30)
                response.raise_for_status()

                import io
                backup_df = pd.read_csv(io.StringIO(response.text))

                for _, row in backup_df.iterrows():
                    smiles = row.get('SMILES') or row.get('smiles')
                    ic50 = row.get('IC50') or row.get('IC50_nM')

                    if smiles:
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol:
                            activity_nm = float(ic50) if ic50 and pd.notna(ic50) else 10000
                            all_data.append({
                                'canonical_smiles': str(smiles),
                                'standard_type': 'IC50',
                                'standard_value': activity_nm,
                                'activity_nm': activity_nm,
                                'is_active': int(activity_nm <= 1000),
                                'source': 'COVID_Moonshot_GitHub',
                                'compound_id': ''
                            })

            except Exception as e:
                logger.warning(f"Error downloading from GitHub backup: {str(e)}")

        if not all_data:
            logger.warning("No COVID Moonshot data could be downloaded")
            return pd.DataFrame()

        # Create DataFrame and remove duplicates
        df = pd.DataFrame(all_data)

        # Remove duplicates based on SMILES, keeping most potent
        df = df.sort_values('activity_nm')
        df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')

        logger.info(f"Total unique compounds from COVID Moonshot: {len(df)}")

        return df

    def process_virus_data(self, virus_key: str) -> pd.DataFrame:
        """
        Process COVID Moonshot data for a specific virus

        Args:
            virus_key: Virus identifier

        Returns:
            DataFrame with processed data
        """
        # COVID Moonshot is only for SARS-CoV-2
        if virus_key.lower() != 'sars_cov2':
            logger.info(f"COVID Moonshot data not applicable for {virus_key}")
            return pd.DataFrame()

        df = self.download_moonshot_data()

        if df.empty:
            return df

        # Add virus identifier
        df['virus'] = 'sars_cov2'

        # Ensure required columns
        target_info = self.targets.get('sars_cov2', {})
        threshold_nm = target_info.get('activity_threshold_nm', 1000)

        # Recalculate activity labels with virus-specific threshold
        df['is_active'] = (df['activity_nm'] <= threshold_nm).astype(int)

        logger.info(f"Processed {len(df)} COVID Moonshot compounds for SARS-CoV-2")
        logger.info(f"Active compounds: {df['is_active'].sum()}")
        logger.info(f"Inactive compounds: {(~df['is_active'].astype(bool)).sum()}")

        return df

    def download_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """
        Process COVID Moonshot data for all viral targets
        Note: Only SARS-CoV-2 will have data
        """
        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing COVID Moonshot data for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            # Process data
            df = self.process_virus_data(virus_key)

            if not df.empty:
                # Save to file
                output_path = Path(output_dir) / virus_key / "raw" / "covid_moonshot_data.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                df.to_csv(output_path, index=False)
                logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                # Convert numpy types to Python native types for JSON serialization
                results[virus_key] = {
                    'total_compounds': int(len(df)),
                    'active_compounds': int(df['is_active'].sum()),
                    'inactive_compounds': int((~df['is_active'].astype(bool)).sum()),
                    'file_path': str(output_path)
                }
            else:
                if virus_key.lower() == 'sars_cov2':
                    results[virus_key] = {
                        'total_compounds': 0,
                        'active_compounds': 0,
                        'inactive_compounds': 0,
                        'file_path': None,
                        'note': 'Failed to download data'
                    }
                else:
                    results[virus_key] = {
                        'note': 'COVID Moonshot is SARS-CoV-2 specific'
                    }

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

    # Download data for all targets (only SARS-CoV-2 will have data)
    results = downloader.download_all_targets()

    # Generate summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for virus in ['hiv1', 'hcv', 'sars_cov2', 'dengue', 'zika']:
        virus_upper = virus.upper()
        if virus in results:
            stats = results[virus]
            if 'total_compounds' in stats:
                print(f"\n{virus_upper}:")
                print(f"  Total compounds: {stats['total_compounds']:,}")
                print(f"  Active: {stats['active_compounds']:,}")
                print(f"  Inactive: {stats['inactive_compounds']:,}")
            else:
                print(f"\n{virus_upper}: {stats.get('note', 'No data')}")

    # Save summary - ensure all values are JSON serializable
    summary_path = "data/activity/covid_moonshot_summary.json"

    # Convert all numpy types to native Python types
    json_safe_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_safe_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.int64)):
                    json_safe_results[key][k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    json_safe_results[key][k] = float(v)
                elif isinstance(v, np.ndarray):
                    json_safe_results[key][k] = v.tolist()
                else:
                    json_safe_results[key][k] = v
        else:
            json_safe_results[key] = value

    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")

    print("\n✓ COVID Moonshot download complete!")
    print("\nNote: COVID Moonshot data is specific to SARS-CoV-2 Mpro")
    print("\nNext step: Run data_collection/09_data_integration.py to integrate all data sources")


if __name__ == "__main__":
    main()
