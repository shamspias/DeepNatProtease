"""
Train Random Forest models for each viral protease
Handles class imbalance with class weights and feature importance analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import yaml
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_random_forest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Train Random Forest models for viral protease inhibitors"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize Random Forest trainer"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['random_forest']

        # Model hyperparameters
        self.default_params = {
            'n_estimators': self.config.get('n_estimators', 1000),
            'max_depth': self.config.get('max_depth', None),
            'min_samples_split': self.config.get('min_samples_split', 2),
            'min_samples_leaf': self.config.get('min_samples_leaf', 1),
            'max_features': self.config.get('max_features', 'sqrt'),
            'bootstrap': self.config.get('bootstrap', True),
            'n_jobs': -1,  # Use all cores
            'random_state': 42,
            'class_weight': 'balanced'  # Handle imbalanced data
        }

        # Virus-specific parameter adjustments
        self.virus_params = {
            'hiv1': {
                'n_estimators': 1500,  # More trees for larger dataset
            },
            'hcv': {},  # Use defaults
            'sars_cov2': {
                'n_estimators': 1500,
                'min_samples_leaf': 2  # Prevent overfitting with imbalanced data
            },
            'dengue': {},
            'zika': {
                'n_estimators': 500,  # Fewer trees for small dataset
                'max_depth': 10,  # Limit depth to prevent overfitting
                'min_samples_leaf': 5  # More conservative
            }
        }

        # Feature settings
        self.fingerprint_bits = 2048
        self.fingerprint_radius = 2

    def load_data(self, virus_key: str, data_dir: str = "data/activity") -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test splits for a virus"""

        splits_dir = Path(data_dir) / virus_key / "splits"

        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        # Log class distribution
        logger.info(f"Train: {len(train_df)} samples, {train_df['is_active'].mean():.2%} active")
        logger.info(f"Val: {len(val_df)} samples, {val_df['is_active'].mean():.2%} active")
        logger.info(f"Test: {len(test_df)} samples, {test_df['is_active'].mean():.2%} active")

        return train_df, val_df, test_df

    def calculate_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors for a SMILES string"""

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'QED': Descriptors.qed(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'MolMR': Descriptors.MolMR(mol),
                'FractionCsp3': Descriptors.FractionCsp3(mol)
            }

            return descriptors
        except:
            return {}

    def generate_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """Generate Morgan fingerprints for a list of SMILES"""

        fingerprints = []

        for smiles in tqdm(smiles_list, desc="Generating fingerprints"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.fingerprint_radius,
                    nBits=self.fingerprint_bits
                )
                arr = np.zeros((self.fingerprint_bits,))
                fp.ConvertToNumpyArray(arr)
                fingerprints.append(arr)
            else:
                # Use zero vector for invalid molecules
                fingerprints.append(np.zeros(self.fingerprint_bits))

        return np.array(fingerprints)

    def prepare_features(self, df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                         fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[StandardScaler]]:
        """Prepare features combining fingerprints and molecular descriptors"""

        logger.info("Preparing features...")

        # Generate fingerprints
        fingerprints = self.generate_fingerprints(df['standardized_smiles'].tolist())

        # Calculate molecular descriptors if not present
        if 'mw' not in df.columns:
            logger.info("Calculating molecular descriptors...")
            descriptor_list = []

            for smiles in tqdm(df['standardized_smiles'], desc="Calculating descriptors"):
                desc = self.calculate_molecular_descriptors(smiles)
                descriptor_list.append(desc)

            descriptor_df = pd.DataFrame(descriptor_list).fillna(0)
        else:
            # Use existing descriptors
            descriptor_cols = ['mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds',
                               'aromatic_rings', 'heavy_atoms', 'qed']
            available_cols = [col for col in descriptor_cols if col in df.columns]
            descriptor_df = df[available_cols].fillna(0)

        # Convert to numpy array
        descriptors = descriptor_df.values

        # Scale descriptors
        if fit_scaler:
            scaler = StandardScaler()
            descriptors_scaled = scaler.fit_transform(descriptors)
        elif scaler is not None:
            descriptors_scaled = scaler.transform(descriptors)
        else:
            descriptors_scaled = descriptors

        # Combine fingerprints and descriptors
        features = np.hstack([fingerprints, descriptors_scaled])

        logger.info(f"Feature matrix shape: {features.shape}")

        return features, scaler

    def train_model(self, virus_key: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    params: Dict) -> Tuple[RandomForestClassifier, StandardScaler, Dict]:
        """Train Random Forest model for a specific virus"""

        logger.info(f"Training Random Forest for {virus_key}")

        # Prepare features
        X_train, scaler = self.prepare_features(train_df, fit_scaler=True)
        X_val, _ = self.prepare_features(val_df, scaler=scaler)

        # Get targets
        y_train = train_df['is_active'].values
        y_val = val_df['is_active'].values

        # Calculate class weights manually if needed
        if params.get('class_weight') == 'balanced':
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([class_weights[int(y)] for y in y_train])
            logger.info(f"Class weights: {dict(zip(classes, class_weights))}")
        else:
            sample_weights = None

        # Train model
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(**params)

        # Fit with progress tracking
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Validate model
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate validation metrics
        val_metrics = {
            'train_auc': roc_auc_score(y_train, train_pred_proba),
            'val_auc': roc_auc_score(y_val, val_pred_proba),
            'train_auprc': average_precision_score(y_train, train_pred_proba),
            'val_auprc': average_precision_score(y_val, val_pred_proba)
        }

        logger.info(f"Training AUC: {val_metrics['train_auc']:.4f}")
        logger.info(f"Validation AUC: {val_metrics['val_auc']:.4f}")

        # Feature importance analysis
        feature_importance = model.feature_importances_

        # Get top features
        n_fingerprints = self.fingerprint_bits
        n_descriptors = X_train.shape[1] - n_fingerprints

        fp_importance = feature_importance[:n_fingerprints].mean()
        desc_importance = feature_importance[n_fingerprints:].mean() if n_descriptors > 0 else 0

        val_metrics['feature_importance'] = {
            'fingerprint_avg': float(fp_importance),
            'descriptor_avg': float(desc_importance),
            'top_features': np.argsort(feature_importance)[-20:].tolist()
        }

        return model, scaler, val_metrics

    def evaluate_model(self, model: RandomForestClassifier, scaler: StandardScaler,
                       test_df: pd.DataFrame) -> Dict:
        """Evaluate model on test set"""

        # Prepare features
        X_test, _ = self.prepare_features(test_df, scaler=scaler)
        y_test = test_df['is_active'].values

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report

        # ROC curve points
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist()[:100],  # Limit points for storage
            'tpr': tpr.tolist()[:100]
        }

        # PR curve points
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist()[:100],
            'recall': recall.tolist()[:100]
        }

        return metrics

    def save_model(self, model: RandomForestClassifier, scaler: StandardScaler,
                   virus_key: str, params: Dict, metrics: Dict,
                   output_dir: str = "data/models"):
        """Save trained model and metadata"""

        # Create output directory
        model_dir = Path(output_dir) / virus_key / "random_forest"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, model_dir / "model.pkl")

        # Save scaler
        joblib.dump(scaler, model_dir / "scaler.pkl")

        # Save metadata
        metadata = {
            'params': params,
            'metrics': metrics,
            'feature_info': {
                'n_fingerprint_bits': self.fingerprint_bits,
                'fingerprint_radius': self.fingerprint_radius,
                'total_features': model.n_features_in_
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved model to {model_dir}")

        return str(model_dir)

    def plot_feature_importance(self, model: RandomForestClassifier, virus_key: str,
                                output_dir: str = "data/models"):
        """Plot and save feature importance"""

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.title(f'Top 20 Feature Importances - {virus_key.upper()}')
        plt.barh(range(len(indices)), importances[indices])
        plt.xlabel('Importance')
        plt.tight_layout()

        # Save plot
        plot_dir = Path(output_dir) / virus_key / "random_forest"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / "feature_importance.png", dpi=300)
        plt.close()

    def train_all_viruses(self, data_dir: str = "data/activity") -> Dict:
        """Train Random Forest models for all viruses"""

        # Load viral targets
        with open("configs/viral_targets.json", 'r') as f:
            targets = json.load(f)

        results = {}

        for virus_key in targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training Random Forest for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Load data
                train_df, val_df, test_df = self.load_data(virus_key, data_dir)

                # Get parameters for this virus
                params = self.default_params.copy()
                params.update(self.virus_params.get(virus_key, {}))

                # Train model
                model, scaler, val_metrics = self.train_model(virus_key, train_df, val_df, params)

                # Evaluate on test set
                test_metrics = self.evaluate_model(model, scaler, test_df)

                # Combine metrics
                all_metrics = {**val_metrics, **test_metrics}

                # Save model
                model_path = self.save_model(model, scaler, virus_key, params, all_metrics)

                # Plot feature importance
                self.plot_feature_importance(model, virus_key)

                # Store results
                results[virus_key] = {
                    'model_path': model_path,
                    'params': params,
                    'metrics': all_metrics,
                    'n_samples_train': len(train_df),
                    'n_samples_test': len(test_df)
                }

                logger.info(f"✓ {virus_key} - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

            except Exception as e:
                logger.error(f"Error training {virus_key}: {str(e)}")
                results[virus_key] = {'error': str(e)}

        return results


def main():
    """Main execution function"""

    print("=" * 60)
    print("Random Forest Training for Viral Proteases")
    print("=" * 60)

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/models").mkdir(exist_ok=True)

    # Initialize trainer
    trainer = RandomForestTrainer()

    # Train models for all viruses
    results = trainer.train_all_viruses()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for virus, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{virus.upper()}:")
            print(f"  Samples: Train={result['n_samples_train']}, Test={result['n_samples_test']}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  MCC: {metrics['mcc']:.4f}")

            if 'feature_importance' in metrics:
                print(f"  Feature Importance - FP: {metrics['feature_importance']['fingerprint_avg']:.4f}, "
                      f"Desc: {metrics['feature_importance']['descriptor_avg']:.4f}")
        else:
            print(f"\n{virus.upper()}: {result['error']}")

    # Save summary
    summary_path = "data/models/random_forest_training_summary.json"

    # Convert to JSON-safe format
    json_safe_results = {}
    for virus, result in results.items():
        if 'error' not in result:
            json_safe_results[virus] = {
                'model_path': result['model_path'],
                'n_samples_train': result['n_samples_train'],
                'n_samples_test': result['n_samples_test'],
                'test_metrics': {
                    'roc_auc': float(result['metrics']['roc_auc']),
                    'pr_auc': float(result['metrics']['pr_auc']),
                    'balanced_accuracy': float(result['metrics']['balanced_accuracy']),
                    'f1_score': float(result['metrics']['f1_score']),
                    'mcc': float(result['metrics']['mcc'])
                }
            }
        else:
            json_safe_results[virus] = result

    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    print("\n✓ Random Forest training complete!")
    print("\nNext: Run 16_train_xgboost.py")


if __name__ == "__main__":
    main()
