"""
Integrate activity data from all sources (ChEMBL, BindingDB, PubChem) for each virus
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
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
        self.standardizer = rdMolStandardize.Standardizer()
        self.uncharger = rdMolStandardize.Uncharger()

    def _load_targets(self, config_path: str) -> Dict:
        """Load viral target configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_source_data(self, virus_key: str, source: str) -> pd.DataFrame:
        """
        Load data from a specific source for a virus

        Args:
            virus_key: Virus identifier
            source: Data source ('chembl', 'bindingdb', 'pubchem')

        Returns:
            DataFrame with source data
        """
        file_path = Path(f"data/activity/{virus_key}/raw/{source}_data.csv")

        if not file_path.exists():
            logger.warning(f"No {source} data found for {virus_key}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} compounds from {source} for {virus_key}")
            return df
        except Exception as e:
            logger.error(f"Error loading {source} data for {virus_key}: {e}")
            return pd.DataFrame()

    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize SMILES string using RDKit

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES or None if invalid
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Standardize molecule
            mol = self.standardizer.standardize(mol)

            # Remove charges
            mol = self.uncharger.uncharge(mol)

            # Remove salts (keep largest fragment)
            mol = rdMolStandardize.FragmentParent(mol)

            # Convert back to canonical SMILES
            return Chem.MolToSmiles(mol, canonical=True)

        except Exception:
            return None

    def calculate_basic_descriptors(self, smiles: str) -> Dict:
        """
        Calculate basic molecular descriptors

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of descriptors
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'qed': Descriptors.qed(mol),
                'lipinski_violations': self._count_lipinski_violations(mol)
            }
        except Exception:
            return {}

    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule of five violations"""
        violations = 0

        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Crippen.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1

        return violations

    def integrate_virus_data(self, virus_key: str) -> pd.DataFrame:
        """
        Integrate data from all sources for a specific virus

        Args:
            virus_key: Virus identifier

        Returns:
            Integrated DataFrame
        """
        logger.info(f"\nIntegrating data for {virus_key}")

        # Load data from each source
        sources = ['chembl', 'bindingdb', 'pubchem']
        all_data = []

        for source in sources:
            df = self.load_source_data(virus_key, source)
            if not df.empty:
                # Ensure source column exists
                if 'source' not in df.columns:
                    df['source'] = source
                all_data.append(df)

        if not all_data:
            logger.warning(f"No data found for {virus_key}")
            return pd.DataFrame()

        # Combine all sources
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total entries")

        # Standardize SMILES
        logger.info("Standardizing SMILES...")
        tqdm.pandas(desc="Standardizing")
        combined_df['standardized_smiles'] = combined_df['canonical_smiles'].progress_apply(
            self.standardize_smiles
        )

        # Remove invalid molecules
        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['standardized_smiles'])
        logger.info(f"Removed {initial_count - len(combined_df)} invalid molecules")

        # Handle duplicates
        combined_df = self._handle_duplicates(combined_df, virus_key)

        # Calculate descriptors
        logger.info("Calculating molecular descriptors...")
        descriptor_df = pd.DataFrame(
            combined_df['standardized_smiles'].progress_apply(
                self.calculate_basic_descriptors
            ).tolist()
        )

        # Combine with original data
        combined_df = pd.concat([combined_df.reset_index(drop=True),
                                 descriptor_df], axis=1)

        # Filter by drug-likeness criteria
        combined_df = self._filter_drug_like(combined_df)

        # Ensure balanced dataset
        combined_df = self._balance_dataset(combined_df)

        # Add virus identifier
        combined_df['virus'] = virus_key

        logger.info(f"Final dataset for {virus_key}: {len(combined_df)} compounds")
        logger.info(f"  Active: {combined_df['is_active'].sum()}")
        logger.info(f"  Inactive: {(~combined_df['is_active'].astype(bool)).sum()}")

        return combined_df

    def _handle_duplicates(self, df: pd.DataFrame, virus_key: str) -> pd.DataFrame:
        """
        Handle duplicate SMILES by aggregating activity data

        Args:
            df: DataFrame with potential duplicates
            virus_key: Virus identifier

        Returns:
            DataFrame with duplicates resolved
        """
        # Group by standardized SMILES
        grouped = df.groupby('standardized_smiles')

        aggregated_data = []

        for smiles, group in grouped:
            # If all sources agree on activity, use that
            if group['is_active'].nunique() == 1:
                # Take the row with lowest activity value (most potent)
                best_row = group.loc[group['activity_nm'].idxmin()].copy()
                best_row['data_sources'] = ','.join(group['source'].unique())
                best_row['n_sources'] = len(group['source'].unique())
                aggregated_data.append(best_row)
            else:
                # If sources disagree, use majority vote or most potent value
                activity_values = group['activity_nm'].values
                is_active_votes = group['is_active'].values

                # Use median activity value
                median_activity = np.median(activity_values)

                # Determine activity based on threshold
                threshold = self.targets[virus_key]['activity_threshold_nm']
                is_active = int(median_activity <= threshold)

                # Create aggregated row
                agg_row = group.iloc[0].copy()
                agg_row['activity_nm'] = median_activity
                agg_row['is_active'] = is_active
                agg_row['data_sources'] = ','.join(group['source'].unique())
                agg_row['n_sources'] = len(group['source'].unique())
                agg_row['activity_agreement'] = group['is_active'].mean()

                aggregated_data.append(agg_row)

        result_df = pd.DataFrame(aggregated_data)

        logger.info(f"Resolved {len(df)} entries to {len(result_df)} unique compounds")

        return result_df

    def _filter_drug_like(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter compounds by drug-likeness criteria

        Args:
            df: DataFrame with molecular descriptors

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        # Basic filters
        df = df[(df['mw'] >= 150) & (df['mw'] <= 900)]
        df = df[(df['logp'] >= -3) & (df['logp'] <= 7)]
        df = df[df['hbd'] <= 8]
        df = df[df['hba'] <= 15]
        df = df[df['rotatable_bonds'] <= 15]
        df = df[df['heavy_atoms'] >= 10]

        # Remove compounds with too many Lipinski violations
        df = df[df['lipinski_violations'] <= 2]

        logger.info(f"Filtered {initial_count - len(df)} non-drug-like compounds")

        return df

    def _balance_dataset(self, df: pd.DataFrame, max_ratio: float = 3.0) -> pd.DataFrame:
        """
        Balance active/inactive ratio

        Args:
            df: DataFrame to balance
            max_ratio: Maximum ratio of inactive to active

        Returns:
            Balanced DataFrame
        """
        n_active = df['is_active'].sum()
        n_inactive = (~df['is_active'].astype(bool)).sum()

        if n_inactive > n_active * max_ratio:
            # Downsample inactive compounds
            inactive_df = df[df['is_active'] == 0]
            active_df = df[df['is_active'] == 1]

            n_keep = int(n_active * max_ratio)
            inactive_df = inactive_df.sample(n=n_keep, random_state=42)

            df = pd.concat([active_df, inactive_df], ignore_index=True)
            logger.info(f"Balanced dataset to {len(active_df)} active and {len(inactive_df)} inactive")

        return df

    def integrate_all_viruses(self, output_dir: str = "data/activity"):
        """Integrate data for all viruses"""

        results = {}

        for virus_key in self.targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Integrate data
                integrated_df = self.integrate_virus_data(virus_key)

                if not integrated_df.empty:
                    # Save integrated data
                    output_path = Path(output_dir) / virus_key / "processed" / "integrated_data.csv"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    integrated_df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved integrated data to {output_path}")

                    # Save summary statistics
                    results[virus_key] = {
                        'total_compounds': len(integrated_df),
                        'active_compounds': integrated_df['is_active'].sum(),
                        'inactive_compounds': (~integrated_df['is_active'].astype(bool)).sum(),
                        'unique_sources': integrated_df['data_sources'].str.split(',').apply(len).mean(),
                        'avg_mw': integrated_df['mw'].mean(),
                        'avg_logp': integrated_df['logp'].mean(),
                        'file_path': str(output_path)
                    }
                else:
                    results[virus_key] = {
                        'total_compounds': 0,
                        'error': 'No data after integration'
                    }

            except Exception as e:
                logger.error(f"Error integrating data for {virus_key}: {str(e)}")
                results[virus_key] = {'error': str(e)}

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
    results = integrator.integrate_all_viruses()

    # Generate summary
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)

    for virus, stats in results.items():
        if 'error' not in stats:
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {stats['total_compounds']:,}")
            print(f"  Active: {stats['active_compounds']:,}")
            print(f"  Inactive: {stats['inactive_compounds']:,}")
            print(f"  Avg sources per compound: {stats['unique_sources']:.1f}")
            print(f"  Avg MW: {stats['avg_mw']:.1f}")
            print(f"  Avg LogP: {stats['avg_logp']:.2f}")
        else:
            print(f"\n{virus.upper()}: ERROR - {stats['error']}")

    # Save summary
    summary_path = "data/activity/integration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    print("\n✓ Data integration complete!")
    print("\nNext step: Run 12_create_splits.py to create train/validation/test splits")


if __name__ == "__main__":
    main()
