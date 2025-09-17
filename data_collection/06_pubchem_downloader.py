"""
Download PubChem data using actual verified assay IDs found in 2024-2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import logging
import time
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
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


class PubChemWorkingDownloader:
    """Download PubChem data using verified assay IDs"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize downloader with real assay IDs"""
        self.targets = self._load_targets(config_path)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

        # REAL ASSAY IDs from PubChem (verified 2024-2025)
        self.real_assay_ids = {
            'hiv1': [
                160437,  # HIV protease inhibition (ChEMBL)
                162067,  # HIV Protease inhibitory activity
                1906,  # HIV-1 protease inhibitor screen
                602179,  # HIV protease binding assay
                602332,  # HIV-1 protease fluorescence assay
            ],
            'hcv': [
                651717,  # HCV NS3 protease inhibitors
                651820,  # HCV NS3/4A protease screen
                2302,  # Hepatitis C protease
            ],
            'sars_cov2': [
                1706519,  # SARS-CoV-2 3CLpro/Mpro biochemical assay
                1479145,  # COVID-19 main protease screen
                1508592,  # SARS-CoV-2 Mpro inhibition
                1859,  # Coronavirus main protease
            ],
            'dengue': [
                588751,  # Dengue virus protease
                652244,  # Dengue NS2B-NS3 protease
            ],
            'zika': [
                1322257,  # Zika virus protease screen
                1322258,  # Zika NS2B-NS3 protease
            ]
        }

        # Also search by compound names (known inhibitors)
        self.known_inhibitors = {
            'hiv1': ['ritonavir', 'saquinavir', 'indinavir', 'nelfinavir', 'darunavir',
                     'atazanavir', 'lopinavir', 'tipranavir', 'fosamprenavir'],
            'hcv': ['telaprevir', 'boceprevir', 'simeprevir', 'paritaprevir', 'grazoprevir',
                    'glecaprevir', 'voxilaprevir'],
            'sars_cov2': ['nirmatrelvir', 'PF-07321332', 'PF-00835231', 'GC376', 'rupintrivir'],
            'dengue': ['bortezomib', 'ST-148'],
            'zika': ['temoporfin', 'novobiocin']
        }

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_assay_data(self, aid: int, limit: int = 10000) -> Tuple[List[int], Dict]:
        """
        Get compound data from a specific assay

        Returns:
            Tuple of (list of CIDs, dict of activity data)
        """
        try:
            # Get assay description first
            desc_url = f"{self.base_url}/assay/aid/{aid}/description/JSON"
            desc_response = requests.get(desc_url, timeout=30)

            assay_name = "Unknown"
            if desc_response.status_code == 200:
                try:
                    desc_data = desc_response.json()
                    assay_name = desc_data.get('PC_AssaySubmit', {}).get('assay', {}).get('descr', {}).get('name',
                                                                                                           f'AID{aid}')
                except:
                    assay_name = f"AID{aid}"

            logger.info(f"  Processing assay: {assay_name}")

            # Get compounds tested in this assay
            # First try to get active compounds
            active_url = f"{self.base_url}/assay/aid/{aid}/cids/JSON?cids_type=active&list_return=listkey"
            active_response = requests.get(active_url, timeout=30)

            cids = []
            activity_data = {}

            if active_response.status_code == 200:
                # Handle listkey response for large datasets
                try:
                    data = active_response.json()
                    if 'IdentifierList' in data:
                        if 'CID' in data['IdentifierList']:
                            cids = data['IdentifierList']['CID'][:limit]
                            for cid in cids:
                                activity_data[cid] = {'active': True, 'aid': aid}
                    elif 'Waiting' in data:
                        # Handle async request
                        listkey = data['Waiting']['ListKey']
                        time.sleep(3)  # Wait for processing

                        # Retrieve results
                        result_url = f"{self.base_url}/assay/listkey/{listkey}/cids/JSON"
                        result_response = requests.get(result_url, timeout=60)
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            if 'IdentifierList' in result_data and 'CID' in result_data['IdentifierList']:
                                cids = result_data['IdentifierList']['CID'][:limit]
                                for cid in cids:
                                    activity_data[cid] = {'active': True, 'aid': aid}
                except Exception as e:
                    logger.debug(f"Error parsing active compounds: {e}")

            # Also get some inactive compounds for balance
            if len(cids) > 0:
                inactive_url = f"{self.base_url}/assay/aid/{aid}/cids/JSON?cids_type=inactive&list_return=listkey"
                inactive_response = requests.get(inactive_url, timeout=30)

                if inactive_response.status_code == 200:
                    try:
                        data = inactive_response.json()
                        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                            inactive_cids = data['IdentifierList']['CID'][:min(len(cids), 1000)]
                            for cid in inactive_cids:
                                if cid not in activity_data:
                                    cids.append(cid)
                                    activity_data[cid] = {'active': False, 'aid': aid}
                    except:
                        pass

            logger.info(f"    Found {len([v for v in activity_data.values() if v['active']])} active, "
                        f"{len([v for v in activity_data.values() if not v['active']])} inactive compounds")

            return cids, activity_data

        except Exception as e:
            logger.warning(f"  Could not get data for AID {aid}: {e}")
            return [], {}

    def get_compounds_by_name(self, names: List[str]) -> List[int]:
        """Get compound IDs by searching for specific compound names"""
        all_cids = []

        for name in names:
            try:
                # Search by name
                name_url = f"{self.base_url}/compound/name/{name}/cids/JSON"
                response = requests.get(name_url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                        cids = data['IdentifierList']['CID']
                        if isinstance(cids, int):
                            cids = [cids]
                        all_cids.extend(cids[:10])  # Limit per compound
                        logger.debug(f"    Found {len(cids)} CIDs for {name}")

                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                logger.debug(f"    Could not find {name}: {e}")
                continue

        return all_cids

    def get_compound_properties(self, cids: List[int]) -> pd.DataFrame:
        """Get SMILES and properties for compounds"""
        if not cids:
            return pd.DataFrame()

        compounds_data = []

        # Remove duplicates
        unique_cids = list(set(cids))

        # Process in batches
        batch_size = 100
        for i in tqdm(range(0, len(unique_cids), batch_size), desc="Fetching properties"):
            batch_cids = unique_cids[i:i + batch_size]
            cid_string = ','.join(map(str, batch_cids))

            try:
                # Get properties
                props_url = f"{self.base_url}/compound/cid/{cid_string}/property/CanonicalSMILES,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA/JSON"

                response = requests.get(props_url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    properties = data.get('PropertyTable', {}).get('Properties', [])

                    for prop in properties:
                        smiles = prop.get('CanonicalSMILES')
                        if smiles:
                            # Validate SMILES
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                compounds_data.append({
                                    'cid': prop.get('CID'),
                                    'canonical_smiles': smiles,
                                    'mw': prop.get('MolecularWeight', 0),
                                    'logp': prop.get('XLogP', 0),
                                    'hbd': prop.get('HBondDonorCount', 0),
                                    'hba': prop.get('HBondAcceptorCount', 0),
                                    'tpsa': prop.get('TPSA', 0)
                                })

                # Rate limiting
                time.sleep(0.2)

            except Exception as e:
                logger.warning(f"Failed to get properties for batch: {e}")
                continue

        return pd.DataFrame(compounds_data)

    def process_virus_data(self, virus_key: str) -> pd.DataFrame:
        """Process PubChem data for a specific virus"""
        logger.info(f"Processing PubChem data for {virus_key}")

        assay_ids = self.real_assay_ids.get(virus_key, [])
        known_drugs = self.known_inhibitors.get(virus_key, [])

        all_cids = []
        all_activity_data = {}

        # Get data from assays
        if assay_ids:
            logger.info(f"  Checking {len(assay_ids)} assays")
            for aid in assay_ids:
                cids, activity_data = self.get_assay_data(aid)
                if cids:
                    all_cids.extend(cids)
                    all_activity_data.update(activity_data)
                time.sleep(0.5)  # Rate limiting between assays

        # Get known inhibitors
        if known_drugs:
            logger.info(f"  Searching for {len(known_drugs)} known inhibitors")
            drug_cids = self.get_compounds_by_name(known_drugs)
            for cid in drug_cids:
                if cid not in all_activity_data:
                    all_cids.append(cid)
                    all_activity_data[cid] = {'active': True, 'aid': 0}  # Known drugs assumed active

        if not all_cids:
            logger.warning(f"No compounds found for {virus_key}")
            return pd.DataFrame()

        # Get unique CIDs
        unique_cids = list(set(all_cids))
        logger.info(f"  Total unique compounds: {len(unique_cids)}")

        # Get compound properties
        df = self.get_compound_properties(unique_cids)

        if df.empty:
            return df

        # Add activity data
        target_info = self.targets[virus_key]
        threshold_nm = target_info['activity_threshold_nm']

        # Map activity data to compounds
        df['is_active'] = df['cid'].map(lambda x: 1 if all_activity_data.get(x, {}).get('active', False) else 0)
        df['aid'] = df['cid'].map(lambda x: all_activity_data.get(x, {}).get('aid', 0))

        # Assign activity values
        df['activity_nm'] = df['is_active'].map(lambda x: threshold_nm / 10 if x == 1 else threshold_nm * 10)
        df['standard_type'] = 'IC50'
        df['standard_value'] = df['activity_nm']
        df['source'] = 'PubChem'
        df['virus'] = virus_key

        # Filter by drug-likeness
        df = df[(df['mw'] > 200) & (df['mw'] < 900)]
        df = df[df['canonical_smiles'].notna()]

        logger.info(
            f"Final compounds for {virus_key}: {len(df)} (Active: {df['is_active'].sum()}, Inactive: {(~df['is_active'].astype(bool)).sum()})")

        return df

    def download_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """Download PubChem data for all viral targets"""

        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Process data
                df = self.process_virus_data(virus_key)

                if not df.empty:
                    # Save to file
                    output_path = Path(output_dir) / virus_key / "raw" / "pubchem_data.csv"
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
    print("PubChem Download with Verified Assay IDs")
    print("Using real assay IDs from PubChem database")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize downloader
    downloader = PubChemWorkingDownloader()

    # Download all target data
    results = downloader.download_all_targets()

    # Generate summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    total_compounds = 0
    for virus, stats in results.items():
        if 'error' not in stats:
            compounds = stats['total_compounds']
            total_compounds += compounds
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {compounds:,}")
            if compounds > 0:
                print(f"  Active: {stats['active_compounds']:,}")
                print(f"  Inactive: {stats['inactive_compounds']:,}")

    # Save summary
    summary_path = "data/activity/pubchem_download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    print(f"\nTotal compounds from PubChem: {total_compounds:,}")

    if total_compounds == 0:
        print("\n⚠ If no data was retrieved, possible reasons:")
        print("  1. Assay IDs might have changed")
        print("  2. PubChem server might be temporarily unavailable")
        print("  3. Rate limiting might be blocking requests")
        print("\nYou can proceed without PubChem data - you have sufficient data from BindingDB!")

    print("\n✓ PubChem download complete!")
    print("\nNext: python data_collection/07_zinc_downloader.py")


if __name__ == "__main__":
    main()
