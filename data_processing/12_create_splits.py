"""
Create scaffold-based train/validation/test splits for each virus dataset
Fixed: Added proper tqdm pandas integration
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Fix: Properly import and initialize tqdm pandas integration
tqdm.pandas()
import logging
from typing import Dict, List, Tuple, Set, Optional
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/create_splits.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScaffoldSplitter:
    """Create scaffold-based data splits to ensure proper model evaluation"""

    def __init__(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15):
        """
        Initialize scaffold splitter

        Args:
            train_size: Fraction for training set
            val_size: Fraction for validation set
            test_size: Fraction for test set
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Verify splits sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 0.001

    def get_scaffold(self, smiles: str, include_chirality: bool = False) -> Optional[str]:
        """
        Get Murcko scaffold from SMILES

        Args:
            smiles: Input SMILES string
            include_chirality: Whether to include stereochemistry

        Returns:
            Scaffold SMILES or None if failed
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
        except:
            return None

    def scaffold_split(self, df: pd.DataFrame, smiles_col: str = 'standardized_smiles',
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset based on scaffolds

        Args:
            df: Input DataFrame
            smiles_col: Column containing SMILES
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Creating scaffold-based splits for {len(df)} compounds")

        # Extract scaffolds - Fixed: using progress_apply with tqdm.pandas()
        logger.info("Extracting molecular scaffolds...")
        df['scaffold'] = df[smiles_col].progress_apply(self.get_scaffold)

        # Remove compounds without valid scaffolds
        initial_count = len(df)
        df = df.dropna(subset=['scaffold'])
        logger.info(f"Removed {initial_count - len(df)} compounds without valid scaffolds")

        # Group compounds by scaffold
        scaffold_groups = df.groupby('scaffold').apply(lambda x: x.index.tolist()).to_dict()

        # Sort scaffolds by size (number of compounds)
        scaffold_sizes = {scaffold: len(indices) for scaffold, indices in scaffold_groups.items()}
        sorted_scaffolds = sorted(scaffold_sizes.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(sorted_scaffolds)} unique scaffolds")
        logger.info(f"Largest scaffold has {sorted_scaffolds[0][1]} compounds")
        logger.info(f"Smallest scaffold has {sorted_scaffolds[-1][1]} compounds")

        # Assign scaffolds to splits
        train_indices = []
        val_indices = []
        test_indices = []

        train_cutoff = int(len(df) * self.train_size)
        val_cutoff = int(len(df) * (self.train_size + self.val_size))

        current_train_size = 0
        current_val_size = 0
        current_test_size = 0

        # Use deterministic assignment based on scaffold order
        np.random.seed(random_state)
        shuffled_scaffolds = sorted_scaffolds.copy()
        np.random.shuffle(shuffled_scaffolds)

        for scaffold, size in shuffled_scaffolds:
            indices = scaffold_groups[scaffold]

            if current_train_size < train_cutoff:
                train_indices.extend(indices)
                current_train_size += size
            elif current_val_size < (val_cutoff - train_cutoff):
                val_indices.extend(indices)
                current_val_size += size
            else:
                test_indices.extend(indices)
                current_test_size += size

        # Create split DataFrames
        train_df = df.loc[train_indices].copy()
        val_df = df.loc[val_indices].copy()
        test_df = df.loc[test_indices].copy()

        # Verify no scaffold overlap
        train_scaffolds = set(train_df['scaffold'].unique())
        val_scaffolds = set(val_df['scaffold'].unique())
        test_scaffolds = set(test_df['scaffold'].unique())

        assert len(train_scaffolds & val_scaffolds) == 0, "Scaffold overlap between train and val"
        assert len(train_scaffolds & test_scaffolds) == 0, "Scaffold overlap between train and test"
        assert len(val_scaffolds & test_scaffolds) == 0, "Scaffold overlap between val and test"

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Split percentages - Train: {len(train_df) / len(df) * 100:.1f}%, "
                    f"Val: {len(val_df) / len(df) * 100:.1f}%, Test: {len(test_df) / len(df) * 100:.1f}%")

        # Check class balance in each split
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'is_active' in split_df.columns:
                active_ratio = split_df['is_active'].mean()
                logger.info(f"{split_name} - Active ratio: {active_ratio:.3f}")

        return train_df, val_df, test_df

    def random_split(self, df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split (for comparison with scaffold split)

        Args:
            df: Input DataFrame
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split off test set
        train_val_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=random_state,
            stratify=df['is_active'] if 'is_active' in df.columns else None
        )

        # Then split train_val into train and val
        relative_val_size = self.val_size / (self.train_size + self.val_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, random_state=random_state,
            stratify=train_val_df['is_active'] if 'is_active' in train_val_df.columns else None
        )

        logger.info(f"Random split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def cluster_based_split(self, df: pd.DataFrame, smiles_col: str = 'standardized_smiles',
                            n_clusters: int = 100, random_state: int = 42) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Cluster-based split using molecular fingerprints

        Args:
            df: Input DataFrame
            smiles_col: Column containing SMILES
            n_clusters: Number of clusters to create
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Creating cluster-based splits with {n_clusters} clusters")

        # Calculate fingerprints
        logger.info("Calculating molecular fingerprints...")
        fingerprints = []
        valid_indices = []

        for idx, smiles in enumerate(tqdm(df[smiles_col], desc="Fingerprints")):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
                valid_indices.append(idx)

        # Filter to valid molecules
        df_valid = df.iloc[valid_indices].copy()
        X = np.array(fingerprints)

        # Perform clustering
        logger.info("Clustering molecules...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1000)
        cluster_labels = kmeans.fit_predict(X)

        df_valid['cluster'] = cluster_labels

        # Assign clusters to splits
        unique_clusters = np.unique(cluster_labels)
        np.random.seed(random_state)
        np.random.shuffle(unique_clusters)

        n_train_clusters = int(len(unique_clusters) * self.train_size)
        n_val_clusters = int(len(unique_clusters) * self.val_size)

        train_clusters = unique_clusters[:n_train_clusters]
        val_clusters = unique_clusters[n_train_clusters:n_train_clusters + n_val_clusters]
        test_clusters = unique_clusters[n_train_clusters + n_val_clusters:]

        train_df = df_valid[df_valid['cluster'].isin(train_clusters)].copy()
        val_df = df_valid[df_valid['cluster'].isin(val_clusters)].copy()
        test_df = df_valid[df_valid['cluster'].isin(test_clusters)].copy()

        logger.info(f"Cluster split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df


class DataSplitCreator:
    """Create and save data splits for all viruses"""

    def __init__(self, split_method: str = 'scaffold'):
        """
        Initialize data split creator

        Args:
            split_method: Method to use ('scaffold', 'random', 'cluster')
        """
        self.split_method = split_method
        self.splitter = ScaffoldSplitter()

    def create_splits_for_virus(self, virus_key: str, data_dir: str = "data/activity") -> Dict:
        """
        Create splits for a specific virus

        Args:
            virus_key: Virus identifier
            data_dir: Base data directory

        Returns:
            Dictionary with split statistics
        """
        # Load integrated data
        data_path = Path(data_dir) / virus_key / "processed" / "integrated_data.csv"

        if not data_path.exists():
            logger.error(f"No integrated data found for {virus_key}")
            return {'error': 'No integrated data found'}

        logger.info(f"Loading data for {virus_key}")
        df = pd.read_csv(data_path)

        # Create splits based on method
        if self.split_method == 'scaffold':
            train_df, val_df, test_df = self.splitter.scaffold_split(df)
        elif self.split_method == 'random':
            train_df, val_df, test_df = self.splitter.random_split(df)
        elif self.split_method == 'cluster':
            train_df, val_df, test_df = self.splitter.cluster_based_split(df)
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        # Save splits
        splits_dir = Path(data_dir) / virus_key / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)

        logger.info(f"✓ Saved splits to {splits_dir}")

        # Calculate statistics - ensure all are native Python types
        stats = {
            'total_compounds': int(len(df)),
            'train_size': int(len(train_df)),
            'val_size': int(len(val_df)),
            'test_size': int(len(test_df)),
            'train_active': int(train_df['is_active'].sum()) if 'is_active' in train_df.columns else 0,
            'train_inactive': int(
                (~train_df['is_active'].astype(bool)).sum()) if 'is_active' in train_df.columns else 0,
            'val_active': int(val_df['is_active'].sum()) if 'is_active' in val_df.columns else 0,
            'val_inactive': int((~val_df['is_active'].astype(bool)).sum()) if 'is_active' in val_df.columns else 0,
            'test_active': int(test_df['is_active'].sum()) if 'is_active' in test_df.columns else 0,
            'test_inactive': int((~test_df['is_active'].astype(bool)).sum()) if 'is_active' in test_df.columns else 0,
            'train_active_ratio': float(train_df['is_active'].mean()) if 'is_active' in train_df.columns else 0.0,
            'val_active_ratio': float(val_df['is_active'].mean()) if 'is_active' in val_df.columns else 0.0,
            'test_active_ratio': float(test_df['is_active'].mean()) if 'is_active' in test_df.columns else 0.0
        }

        if 'scaffold' in train_df.columns:
            stats['train_scaffolds'] = int(train_df['scaffold'].nunique())
            stats['val_scaffolds'] = int(val_df['scaffold'].nunique())
            stats['test_scaffolds'] = int(test_df['scaffold'].nunique())

        return stats

    def create_all_splits(self, data_dir: str = "data/activity") -> Dict:
        """Create splits for all viruses"""

        # Load viral targets
        with open("configs/viral_targets.json", 'r') as f:
            targets = json.load(f)

        results = {}

        for virus_key in targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Creating splits for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                stats = self.create_splits_for_virus(virus_key, data_dir)
                results[virus_key] = stats
            except Exception as e:
                logger.error(f"Error creating splits for {virus_key}: {str(e)}")
                results[virus_key] = {'error': str(e)}

        return results


def main():
    """Main execution function"""

    print("=" * 60)
    print("Creating Train/Validation/Test Splits")
    print("=" * 60)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Create splits using scaffold-based method
    creator = DataSplitCreator(split_method='scaffold')
    results = creator.create_all_splits()

    # Generate summary
    print("\n" + "=" * 60)
    print("SPLIT CREATION SUMMARY")
    print("=" * 60)

    for virus, stats in results.items():
        if 'error' not in stats:
            print(f"\n{virus.upper()}:")
            print(f"  Total compounds: {stats['total_compounds']:,}")
            print(
                f"  Train: {stats['train_size']:,} (Active: {stats['train_active']:,}, Inactive: {stats['train_inactive']:,})")
            print(
                f"  Val: {stats['val_size']:,} (Active: {stats['val_active']:,}, Inactive: {stats['val_inactive']:,})")
            print(
                f"  Test: {stats['test_size']:,} (Active: {stats['test_active']:,}, Inactive: {stats['test_inactive']:,})")
            print(
                f"  Active ratios - Train: {stats['train_active_ratio']:.3f}, Val: {stats['val_active_ratio']:.3f}, Test: {stats['test_active_ratio']:.3f}")

            if 'train_scaffolds' in stats:
                print(
                    f"  Unique scaffolds - Train: {stats['train_scaffolds']}, Val: {stats['val_scaffolds']}, Test: {stats['test_scaffolds']}")

    # Save summary
    summary_path = "data/activity/split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")

    print("\n✓ Split creation complete!")
    print("\nNext step: Run training scripts (14_train_chemprop.py, etc.) to train models")


if __name__ == "__main__":
    main()
