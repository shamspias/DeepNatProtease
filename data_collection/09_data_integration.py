"""
Integrate data from multiple sources (ChEMBL, BindingDB, PubChem, ZINC, COVID Moonshot)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
# Correct imports for RDKit 2024.03.5
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataIntegrator:
    """Integrate and standardize data from multiple sources"""

    def __init__(self, config_path: str = "configs/viral_targets.json"):
        """Initialize data integrator"""
        self.targets = self._load_targets(config_path)

        # Initialize RDKit tools - Correct way for RDKit 2024.03.5
        self.normalizer = rdMolStandardize.Normalizer()
        self.lfc = rdMolStandardize.LargestFragmentChooser()
        self.uc = rdMolStandardize.Uncharger()
        self.salt_remover = SaltRemover()

        # Data sources to check
        self.sources = [
            'chembl_data.csv',
            'bindingdb_data.csv',
            'pubchem_data.csv',
            'zinc_data.csv',
            'covid_moonshot_data.csv'
        ]

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def standardize_molecule(self, smiles: str) -> Optional[str]:
        """
        Standardize a molecule SMILES string

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized canonical SMILES or None if invalid
        """
        try:
            # Parse molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Remove salts
            mol = self.salt_remover.StripMol(mol)

            # Get largest fragment
            mol = self.lfc.choose(mol)

            # Normalize
            mol = self.normalizer.normalize(mol)

            # Uncharge
            mol = self.uc.uncharge(mol)

            # Get canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            return canonical_smiles

        except Exception as e:
            logger.debug(f"Could not standardize {smiles}: {e}")
            return None

    def calculate_properties(self, smiles: str) -> Dict:
        """
        Calculate molecular properties

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular properties
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            properties = {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'qed': QED.qed(mol),
            }

            # Calculate Lipinski violations
            properties['lipinski_violations'] = sum([
                properties['mw'] > 500,
                properties['logp'] > 5,
                properties['hbd'] > 5,
                properties['hba'] > 10
            ])

            return properties

        except Exception as e:
            logger.debug(f"Could not calculate properties for {smiles}: {e}")
            return {}

    def load_source_data(self, virus_key: str, source_file: str) -> pd.DataFrame:
        """
        Load data from a specific source file

        Args:
            virus_key: Virus identifier
            source_file: Source filename

        Returns:
            DataFrame with source data
        """
        file_path = Path(f"data/activity/{virus_key}/raw/{source_file}")

        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            logger.info(f"  Loaded {len(df)} compounds from {source_file}")

            # Ensure required columns exist
            if 'canonical_smiles' not in df.columns:
                # Try to find SMILES column
                smiles_cols = [col for col in df.columns if 'smile' in col.lower()]
                if smiles_cols:
                    df['canonical_smiles'] = df[smiles_cols[0]]
                else:
                    logger.warning(f"  No SMILES column found in {source_file}")
                    return pd.DataFrame()

            # Add source column if not present
            if 'source' not in df.columns:
                source_name = source_file.replace('_data.csv', '').upper()
                df['source'] = source_name

            return df

        except Exception as e:
            logger.error(f"  Error loading {file_path}: {e}")
            return pd.DataFrame()

    def integrate_virus_data(self, virus_key: str) -> pd.DataFrame:
        """
        Integrate all data sources for a specific virus

        Args:
            virus_key: Virus identifier

        Returns:
            Integrated and standardized DataFrame
        """
        logger.info(f"\nProcessing {virus_key.upper()}")
        logger.info("-" * 40)

        all_data = []

        # Load data from each source
        for source_file in self.sources:
            df = self.load_source_data(virus_key, source_file)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning(f"No data found for {virus_key}")
            return pd.DataFrame()

        # Combine all sources
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"  Combined: {len(combined_df)} total compounds")

        # Standardize SMILES
        logger.info("  Standardizing molecules...")
        standardized_smiles = []

        for smiles in tqdm(combined_df['canonical_smiles'], desc="Standardizing"):
            std_smiles = self.standardize_molecule(smiles)
            standardized_smiles.append(std_smiles)

        combined_df['standardized_smiles'] = standardized_smiles

        # Remove compounds that couldn't be standardized
        valid_mask = combined_df['standardized_smiles'].notna()
        combined_df = combined_df[valid_mask]
        logger.info(f"  Valid molecules: {len(combined_df)}")

        # Remove duplicates based on standardized SMILES
        # Keep the most potent (lowest activity value) for each compound
        if 'activity_nm' in combined_df.columns:
            combined_df = combined_df.sort_values('activity_nm')
        combined_df = combined_df.drop_duplicates(subset=['standardized_smiles'], keep='first')
        logger.info(f"  Unique molecules: {len(combined_df)}")

        # Calculate molecular properties
        logger.info("  Calculating molecular properties...")
        properties_list = []

        for smiles in tqdm(combined_df['standardized_smiles'], desc="Properties"):
            props = self.calculate_properties(smiles)
            properties_list.append(props)

        properties_df = pd.DataFrame(properties_list)

        # Combine with original data
        combined_df = pd.concat([combined_df.reset_index(drop=True),
                                 properties_df.reset_index(drop=True)], axis=1)

        # Ensure activity labels
        target_info = self.targets[virus_key]
        threshold_nm = target_info['activity_threshold_nm']

        if 'is_active' not in combined_df.columns:
            if 'activity_nm' in combined_df.columns:
                combined_df['is_active'] = (combined_df['activity_nm'] <= threshold_nm).astype(int)
            else:
                # If no activity data, assume inactive (conservative)
                combined_df['is_active'] = 0

        # Add virus identifier
        combined_df['virus'] = virus_key

        # Select final columns
        final_columns = [
            'standardized_smiles', 'is_active', 'activity_nm', 'standard_type',
            'source', 'virus', 'mw', 'logp', 'hbd', 'hba', 'tpsa',
            'rotatable_bonds', 'aromatic_rings', 'heavy_atoms', 'qed', 'lipinski_violations'
        ]

        # Keep only columns that exist
        final_columns = [col for col in final_columns if col in combined_df.columns]
        combined_df = combined_df[final_columns]

        # Report class balance
        if 'is_active' in combined_df.columns:
            active_count = combined_df['is_active'].sum()
            inactive_count = (~combined_df['is_active'].astype(bool)).sum()
            logger.info(f"  Active: {active_count}, Inactive: {inactive_count}")

            # Balance classes if very imbalanced
            if active_count > 0 and inactive_count > 0:
                ratio = max(active_count, inactive_count) / min(active_count, inactive_count)
                if ratio > 10:
                    logger.warning(f"  Class imbalance ratio: {ratio:.1f}:1")

        return combined_df

    def integrate_all_targets(self, output_dir: str = "data/activity") -> Dict:
        """
        Integrate data for all viral targets

        Args:
            output_dir: Output directory

        Returns:
            Summary statistics
        """
        results = {}

        for virus_key in self.targets.keys():
            # Integrate data
            integrated_df = self.integrate_virus_data(virus_key)

            if not integrated_df.empty:
                # Save processed data
                output_path = Path(output_dir) / virus_key / "processed" / "integrated_data.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                integrated_df.to_csv(output_path, index=False)
                logger.info(f"  ✓ Saved to {output_path}")

                # Store results
                results[virus_key] = {
                    'total_compounds': len(integrated_df),
                    'active_compounds': int(
                        integrated_df['is_active'].sum()) if 'is_active' in integrated_df.columns else 0,
                    'inactive_compounds': int((~integrated_df['is_active'].astype(
                        bool)).sum()) if 'is_active' in integrated_df.columns else 0,
                    'sources': integrated_df[
                        'source'].value_counts().to_dict() if 'source' in integrated_df.columns else {},
                    'file_path': str(output_path)
                }
            else:
                results[virus_key] = {
                    'total_compounds': 0,
                    'active_compounds': 0,
                    'inactive_compounds': 0,
                    'sources': {},
                    'file_path': None
                }

        return results


def main():
    """Main execution function"""

    print("=" * 60)
    print("Data Integration for Viral Proteases")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize integrator
    integrator = DataIntegrator()

    # Integrate all data
    results = integrator.integrate_all_targets()

    # Generate summary
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)

    total_compounds = 0
    for virus, stats in results.items():
        if stats['total_compounds'] > 0:
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {stats['total_compounds']:,}")
            print(f"  Active: {stats['active_compounds']:,}")
            print(f"  Inactive: {stats['inactive_compounds']:,}")

            if stats['sources']:
                print("  Sources:")
                for source, count in stats['sources'].items():
                    print(f"    - {source}: {count}")

            total_compounds += stats['total_compounds']

    # Save summary
    summary_path = "data/activity/integration_summary.json"

    # Convert numpy types to Python types for JSON serialization
    json_safe_results = {}
    for virus, stats in results.items():
        json_safe_results[virus] = {
            'total_compounds': int(stats['total_compounds']),
            'active_compounds': int(stats['active_compounds']),
            'inactive_compounds': int(stats['inactive_compounds']),
            'sources': {k: int(v) for k, v in stats['sources'].items()},
            'file_path': stats['file_path']
        }

    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    print(f"\n{'=' * 60}")
    print(f"Total integrated compounds: {total_compounds:,}")
    print(f"{'=' * 60}")

    print("\n✓ Data integration complete!")
    print("\nNext steps:")
    print("  1. Run data_processing/12_create_splits.py to create train/val/test splits")
    print("  2. Run model_training scripts to train models")


if __name__ == "__main__":
    main()