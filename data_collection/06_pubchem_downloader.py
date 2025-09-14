#!/usr/bin/env python3
"""
06_pubchem_downloader.py
Download bioassay data from PubChem for viral proteases
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Set
import pubchempy as pcp
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pubchem_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PubChemDownloader:
    """Download bioassay data from PubChem for viral proteases"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize PubChem downloader"""
        self.targets = self._load_targets(config_path)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.rate_limit_delay = 0.2  # 5 requests per second max

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_assay_data(self, aid: int) -> pd.DataFrame:
        """
        Get bioassay data for a specific assay ID (AID)

        Args:
            aid: PubChem Assay ID

        Returns:
            DataFrame with assay results
        """
        try:
            # Get assay description first
            desc_url = f"{self.base_url}/assay/aid/{aid}/description/JSON"
            response = requests.get(desc_url)
            time.sleep(self.rate_limit_delay)

            if response.status_code != 200:
                logger.warning(f"Could not get description for AID {aid}")
                return pd.DataFrame()

            assay_desc = response.json()

            # Get assay data (active and inactive compounds)
            data_url = f"{self.base_url}/assay/aid/{aid}/cids/JSON?cids_type=all"
            response = requests.get(data_url)
            time.sleep(self.rate_limit_delay)

            if response.status_code != 200:
                logger.warning(f"Could not get data for AID {aid}")
                return pd.DataFrame()

            assay_data = response.json()

            # Get active compounds
            active_url = f"{self.base_url}/assay/aid/{aid}/cids/JSON?cids_type=active"
            response = requests.get(active_url)
            time.sleep(self.rate_limit_delay)

            active_cids = set()
            if response.status_code == 200:
                active_data = response.json()
                if 'InformationList' in active_data:
                    for info in active_data['InformationList']['Information']:
                        if 'CID' in info:
                            active_cids.update(info['CID'])

            # Get compound details with activity data
            compounds_data = []

            # Get full assay data with activity values
            full_data_url = f"{self.base_url}/assay/aid/{aid}/JSON"
            response = requests.get(full_data_url)
            time.sleep(self.rate_limit_delay)

            if response.status_code == 200:
                full_assay = response.json()

                # Parse the data table if available
                if 'PC_AssayContainer' in full_assay:
                    for container in full_assay['PC_AssayContainer']:
                        if 'assay' in container and 'data' in container:
                            data = container['data']

                            for item in data:
                                if 'sid' in item and 'data' in item:
                                    cid = item.get('cid', None)
                                    if cid:
                                        compound_info = {
                                            'cid': cid,
                                            'sid': item['sid'],
                                            'is_active': 1 if cid in active_cids else 0,
                                            'aid': aid
                                        }

                                        # Extract activity values
                                        for data_point in item['data']:
                                            if 'value' in data_point:
                                                tid = data_point.get('tid', 0)
                                                value = data_point['value']

                                                # Map common test IDs to meaningful names
                                                if tid == 1:  # Usually IC50
                                                    compound_info['ic50_um'] = value.get('fval', None)
                                                elif tid == 2:  # Usually % inhibition
                                                    compound_info['percent_inhibition'] = value.get('fval', None)

                                        compounds_data.append(compound_info)

            if not compounds_data:
                logger.warning(f"No compound data found for AID {aid}")
                return pd.DataFrame()

            return pd.DataFrame(compounds_data)

        except Exception as e:
            logger.error(f"Error getting data for AID {aid}: {e}")
            return pd.DataFrame()

    def get_compounds_smiles(self, cids: List[int], batch_size: int = 100) -> Dict[int, str]:
        """
        Get SMILES for a list of compound IDs

        Args:
            cids: List of PubChem compound IDs
            batch_size: Number of compounds per request

        Returns:
            Dictionary mapping CID to SMILES
        """
        smiles_dict = {}

        for i in tqdm(range(0, len(cids), batch_size), desc="Fetching SMILES"):
            batch = cids[i:i + batch_size]
            cid_str = ','.join(map(str, batch))

            url = f"{self.base_url}/compound/cid/{cid_str}/property/CanonicalSMILES/JSON"

            try:
                response = requests.get(url)
                time.sleep(self.rate_limit_delay)

                if response.status_code == 200:
                    data = response.json()
                    if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                        for prop in data['PropertyTable']['Properties']:
                            if 'CID' in prop and 'CanonicalSMILES' in prop:
                                smiles_dict[prop['CID']] = prop['CanonicalSMILES']

            except Exception as e:
                logger.warning(f"Error fetching SMILES for batch: {e}")
                continue

        return smiles_dict

    def process_virus_assays(self, virus_key: str) -> pd.DataFrame:
        """
        Process all assays for a specific virus

        Args:
            virus_key: Virus identifier

        Returns:
            Combined DataFrame with all assay data
        """
        if virus_key not in self.targets:
            raise ValueError(f"Unknown virus: {virus_key}")

        target_info = self.targets[virus_key]
        aids = target_info.get('pubchem_aid', [])

        if not aids:
            logger.warning(f"No PubChem assays defined for {virus_key}")
            return pd.DataFrame()

        logger.info(f"Processing {len(aids)} assays for {virus_key}")

        all_data = []

        for aid in aids:
            logger.info(f"Fetching data for AID {aid}")
            df = self.get_assay_data(aid)

            if not df.empty:
                all_data.append(df)
                logger.info(f"  Found {len(df)} compounds")

        if not all_data:
            logger.warning(f"No data found for {virus_key}")
            return pd.DataFrame()

        # Combine all assay data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates, keeping the active result if there's a conflict
        combined_df = combined_df.sort_values('is_active', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['cid'], keep='first')

        # Get SMILES for all compounds
        logger.info(f"Fetching SMILES for {len(combined_df)} compounds")
        unique_cids = combined_df['cid'].unique().tolist()
        smiles_dict = self.get_compounds_smiles(unique_cids)

        # Add SMILES to dataframe
        combined_df['canonical_smiles'] = combined_df['cid'].map(smiles_dict)

        # Remove compounds without SMILES
        combined_df = combined_df.dropna(subset=['canonical_smiles'])

        # Convert IC50 from μM to nM if present
        if 'ic50_um' in combined_df.columns:
            combined_df['activity_nm'] = combined_df['ic50_um'] * 1000
        else:
            # If no IC50, use activity classification with pseudo-values
            threshold = target_info['activity_threshold_nm']
            combined_df['activity_nm'] = combined_df['is_active'].apply(
                lambda x: threshold / 10 if x == 1 else threshold * 10
            )

        # Add metadata
        combined_df['source'] = 'PubChem'
        combined_df['virus'] = virus_key
        combined_df['activity_type'] = 'IC50' if 'ic50_um' in combined_df.columns else 'Binary'

        # Select final columns
        final_columns = [
            'canonical_smiles', 'activity_nm', 'is_active',
            'activity_type', 'source', 'virus', 'cid', 'aid'
        ]

        # Add optional columns if they exist
        for col in ['percent_inhibition', 'ic50_um']:
            if col in combined_df.columns:
                final_columns.append(col)

        combined_df = combined_df[final_columns]

        logger.info(f"Total compounds for {virus_key}: {len(combined_df)}")
        logger.info(
            f"Active: {combined_df['is_active'].sum()}, Inactive: {(~combined_df['is_active'].astype(bool)).sum()}")

        return combined_df

    def download_all_targets(self, output_dir: str = "data/activity"):
        """Download data for all viral targets"""

        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Process assays for this target
                df = self.process_virus_assays(virus_key)

                if not df.empty:
                    # Save to file
                    output_path = Path(output_dir) / virus_key / "raw" / "pubchem_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved {len(df)} compounds to {output_path}")

                    # Store results
                    results[virus_key] = {
                        'total_compounds': len(df),
                        'active_compounds': df['is_active'].sum(),
                        'inactive_compounds': (~df['is_active'].astype(bool)).sum(),
                        'unique_assays': df['aid'].nunique() if 'aid' in df.columns else 0,
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
    print("PubChem Data Download for Viral Proteases")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize downloader
    downloader = PubChemDownloader()

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
            print(f"  Active: {stats['active_compounds']:,}")
            print(f"  Inactive: {stats['inactive_compounds']:,}")
            if stats.get('unique_assays'):
                print(f"  Unique assays: {stats['unique_assays']}")

    # Save summary
    summary_path = "data/activity/pubchem_download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    # Calculate totals
    total_compounds = sum(r.get('total_compounds', 0) for r in results.values() if 'error' not in r)
    total_active = sum(r.get('active_compounds', 0) for r in results.values() if 'error' not in r)

    print(f"\n{'=' * 60}")
    print(f"Total compounds downloaded: {total_compounds:,}")
    print(f"Total active compounds: {total_active:,}")
    print(f"{'=' * 60}")

    print("\n✓ PubChem download complete!")
    print("\nNext step: Run 09_data_integration.py to integrate all data sources")
    print("(Note: Steps 07-08 for ZINC and COVID Moonshot are optional)")


if __name__ == "__main__":
    main()
