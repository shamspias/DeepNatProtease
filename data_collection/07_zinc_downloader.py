"""
Download FDA-approved and protease-focused compounds from ZINC database
These serve as positive controls and drug repurposing candidates
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
from io import BytesIO
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
        logging.FileHandler('logs/zinc_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ZINCDownloader:
    """Download curated compound sets from ZINC database"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize ZINC downloader"""
        self.targets = self._load_targets(config_path)

        # ZINC subset URLs (as of 2025)
        self.zinc_subsets = {
            'fda_approved': 'https://zinc.docking.org/substances/subsets/fda/fda.smi.gz',
            'world_drugs': 'https://zinc.docking.org/substances/subsets/world/world.smi.gz',
            'protease_inhibitors': 'https://zinc.docking.org/substances/subsets/protease-inhibitor/protease-inhibitor.smi.gz',
            'antivirals': 'https://zinc.docking.org/substances/subsets/antiviral/antiviral.smi.gz',
            'in_trials': 'https://zinc.docking.org/substances/subsets/in-trials/in-trials.smi.gz'
        }

        # Known protease inhibitor patterns (for activity assignment)
        self.protease_inhibitor_patterns = [
            'navir',  # HIV protease inhibitors
            'previr',  # HCV protease inhibitors
            'rupintrivir',  # Rhinovirus protease inhibitor
            'protease',
            'inhibitor'
        ]

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def download_zinc_subset(self, subset_name: str) -> pd.DataFrame:
        """
        Download a ZINC subset

        Args:
            subset_name: Name of the subset to download

        Returns:
            DataFrame with SMILES and ZINC IDs
        """
        if subset_name not in self.zinc_subsets:
            logger.error(f"Unknown subset: {subset_name}")
            return pd.DataFrame()

        url = self.zinc_subsets[subset_name]
        logger.info(f"Downloading ZINC {subset_name} subset from {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Decompress if gzipped
            if url.endswith('.gz'):
                content = gzip.decompress(response.content).decode('utf-8')
            else:
                content = response.text

            # Parse SMILES file (format: SMILES ZINC_ID)
            lines = content.strip().split('\n')
            data = []

            for line in tqdm(lines, desc=f"Processing {subset_name}"):
                if line and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        smiles = parts[0]
                        zinc_id = parts[1]

                        # Validate SMILES
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            data.append({
                                'canonical_smiles': Chem.MolToSmiles(mol, canonical=True),
                                'zinc_id': zinc_id,
                                'subset': subset_name
                            })

            logger.info(f"Processed {len(data)} compounds from {subset_name}")
            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error downloading {subset_name}: {e}")
            return pd.DataFrame()

    def assign_pseudo_activities(self, df: pd.DataFrame, virus_key: str) -> pd.DataFrame:
        """
        Assign pseudo-activities to ZINC compounds
        FDA-approved drugs and known protease inhibitors get favorable pseudo-activities

        Args:
            df: DataFrame with ZINC compounds
            virus_key: Virus identifier

        Returns:
            DataFrame with activity assignments
        """
        target_info = self.targets[virus_key]
        threshold_nm = target_info['activity_threshold_nm']

        # Initialize with conservative inactive assumption
        df['activity_nm'] = threshold_nm * 10  # Well above threshold
        df['is_active'] = 0
        df['activity_type'] = 'pseudo'

        # FDA-approved antivirals likely have some activity
        if 'subset' in df.columns:
            # Antivirals and protease inhibitors get active pseudo-values
            active_subsets = ['antivirals', 'protease_inhibitors']
            mask = df['subset'].isin(active_subsets)
            df.loc[mask, 'activity_nm'] = threshold_nm / 10  # Well below threshold
            df.loc[mask, 'is_active'] = 1

            # FDA-approved drugs get moderate values
            fda_mask = df['subset'] == 'fda_approved'
            df.loc[fda_mask, 'activity_nm'] = threshold_nm * 2  # Slightly inactive

        # Check for known protease inhibitor names
        if 'zinc_id' in df.columns:
            for pattern in self.protease_inhibitor_patterns:
                mask = df['zinc_id'].str.lower().str.contains(pattern, na=False)
                df.loc[mask, 'activity_nm'] = threshold_nm / 10
                df.loc[mask, 'is_active'] = 1

        logger.info(f"Assigned activities: {df['is_active'].sum()} active, "
                    f"{(~df['is_active'].astype(bool)).sum()} inactive")

        return df

    def process_for_virus(self, virus_key: str) -> pd.DataFrame:
        """
        Download and process ZINC data for a specific virus

        Args:
            virus_key: Virus identifier

        Returns:
            Combined DataFrame with all ZINC data
        """
        all_data = []

        # Determine which subsets to download based on virus
        if virus_key == 'hiv1':
            subsets = ['fda_approved', 'protease_inhibitors', 'antivirals']
        elif virus_key == 'hcv':
            subsets = ['fda_approved', 'protease_inhibitors', 'antivirals']
        elif virus_key == 'sars_cov2':
            subsets = ['fda_approved', 'antivirals', 'in_trials']
        else:
            subsets = ['fda_approved', 'antivirals']

        for subset in subsets:
            logger.info(f"Downloading {subset} for {virus_key}")
            df = self.download_zinc_subset(subset)

            if not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning(f"No ZINC data downloaded for {virus_key}")
            return pd.DataFrame()

        # Combine all subsets
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['canonical_smiles'], keep='first')

        # Assign pseudo-activities
        combined_df = self.assign_pseudo_activities(combined_df, virus_key)

        # Add metadata
        combined_df['source'] = 'ZINC'
        combined_df['virus'] = virus_key
        combined_df['standard_type'] = 'IC50'  # Pseudo-type
        combined_df['standard_value'] = combined_df['activity_nm']

        logger.info(f"Total ZINC compounds for {virus_key}: {len(combined_df)}")

        return combined_df

    def download_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """Download ZINC data for all viral targets"""

        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing ZINC data for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Process ZINC data for this virus
                df = self.process_for_virus(virus_key)

                if not df.empty:
                    # Save to file
                    output_path = Path(output_dir) / virus_key / "raw" / "zinc_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                    # Store results
                    results[virus_key] = {
                        'total_compounds': len(df),
                        'active_compounds': df['is_active'].sum(),
                        'inactive_compounds': (~df['is_active'].astype(bool)).sum(),
                        'subsets_used': df['subset'].unique().tolist() if 'subset' in df.columns else [],
                        'file_path': str(output_path)
                    }
                else:
                    logger.warning(f"No data retrieved for {virus_key}")
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
    print("ZINC Database Download for Viral Proteases")
    print("=" * 60)
    print("Downloading FDA-approved drugs and known inhibitors")
    print("These serve as positive controls and repurposing candidates")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize downloader
    downloader = ZINCDownloader()

    # Download all target data
    results = downloader.download_all_targets()

    # Generate summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for virus, stats in results.items():
        if 'error' not in stats:
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {stats['total_compounds']:,}")
            print(f"  Active (pseudo): {stats['active_compounds']:,}")
            print(f"  Inactive (pseudo): {stats['inactive_compounds']:,}")
            if stats.get('subsets_used'):
                print(f"  Subsets: {', '.join(stats['subsets_used'])}")

    # Save summary
    summary_path = "data/activity/zinc_download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    # Calculate totals
    total_compounds = sum(r.get('total_compounds', 0) for r in results.values() if 'error' not in r)

    print(f"\n{'=' * 60}")
    print(f"Total ZINC compounds downloaded: {total_compounds:,}")
    print(f"{'=' * 60}")

    print("\n✓ ZINC download complete!")
    print("\nNext step: Run data_collection/08_covid_moonshot_downloader.py")
    print("Note: ZINC data provides drug repurposing candidates")


if __name__ == "__main__":
    main()
