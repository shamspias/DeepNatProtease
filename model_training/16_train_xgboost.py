"""
Train XGBoost models for each viral protease
Optimized for handling class imbalance with scale_pos_weight
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import yaml
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from scipy import stats

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_xgboost.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Train XGBoost models for viral protease inhibitors"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize XGBoost trainer"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['xgboost']

        # Default hyperparameters
        self.default_params = {
            'n_estimators': self.config.get('n_estimators', 500),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.01),
            'objective': self.config.get('objective', 'binary:logistic'),
            'eval_metric': self.config.get('eval_metric', 'auc'),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'device': 'cpu'  # Change to 'cuda' if GPU available
        }

        # Virus-specific adjustments
        self.virus_params = {
            'hiv1': {
                'n_estimators': 1000,
                'max_depth': 8
            },
            'hcv': {
                'n_estimators': 800
            },
            'sars_cov2': {
                'n_estimators': 1000,
                'max_depth': 8,
                'min_child_weight': 5  # More conservative for imbalanced data
            },
            'dengue': {
                'n_estimators': 800
            },
            'zika': {
                'n_estimators': 300,  # Fewer trees for small dataset
                'max_depth': 4,  # Shallower trees
                'learning_rate': 0.05,  # Higher learning rate
                'min_child_weight': 10,  # More conservative
                'subsample': 0.6  # More aggressive subsampling
            }
        }

        # Feature settings
        self.fingerprint_bits = 2048
        self.fingerprint_radius = 2

        # Hyperparameter tuning settings
        self.tune_hyperparameters = True
        self.n_trials = 20  # Number of Optuna trials

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
                'FractionCsp3': Descriptors.FractionCsp3(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
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

    def calculate_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """Calculate scale_pos_weight for imbalanced data"""

        n_negative = np.sum(y_train == 0)
        n_positive = np.sum(y_train == 1)

        if n_positive > 0:
            scale_pos_weight = n_negative / n_positive
        else:
            scale_pos_weight = 1.0

        logger.info(f"Class distribution - Negative: {n_negative}, Positive: {n_positive}")
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        return scale_pos_weight

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 scale_pos_weight: float) -> Dict:
        """Optimize hyperparameters using Optuna"""

        logger.info("Optimizing hyperparameters with Optuna...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'scale_pos_weight': scale_pos_weight,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'device': 'cpu'
            }

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            return auc

        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cpu'

        logger.info(f"Best validation AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params

    def train_model(self, virus_key: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    params: Dict) -> Tuple[xgb.XGBClassifier, StandardScaler, Dict]:
        """Train XGBoost model for a specific virus"""

        logger.info(f"Training XGBoost for {virus_key}")

        # Prepare features
        X_train, scaler = self.prepare_features(train_df, fit_scaler=True)
        X_val, _ = self.prepare_features(val_df, scaler=scaler)

        # Get targets
        y_train = train_df['is_active'].values
        y_val = val_df['is_active'].values

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = self.calculate_scale_pos_weight(y_train)
        params['scale_pos_weight'] = scale_pos_weight

        # Optimize hyperparameters if enabled
        if self.tune_hyperparameters and len(train_df) >= 100:  # Only tune for sufficient data
            optimized_params = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, scale_pos_weight
            )
            params.update(optimized_params)

        # Train final model
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(**params)

        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=['auc', 'aucpr'],
            verbose=100,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)]
        )

        # Get predictions
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred_proba),
            'val_auc': roc_auc_score(y_val, val_pred_proba),
            'train_auprc': average_precision_score(y_train, train_pred_proba),
            'val_auprc': average_precision_score(y_val, val_pred_proba),
            'best_iteration': model.best_iteration,
            'best_score': model.best_score
        }

        logger.info(f"Training AUC: {metrics['train_auc']:.4f}")
        logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")
        logger.info(f"Best iteration: {metrics['best_iteration']}")

        # Feature importance
        importance = model.feature_importances_
        n_fingerprints = self.fingerprint_bits
        n_descriptors = X_train.shape[1] - n_fingerprints

        fp_importance = importance[:n_fingerprints].mean()
        desc_importance = importance[n_fingerprints:].mean() if n_descriptors > 0 else 0

        metrics['feature_importance'] = {
            'fingerprint_avg': float(fp_importance),
            'descriptor_avg': float(desc_importance),
            'top_features': np.argsort(importance)[-20:].tolist(),
            'gain_importance': model.get_booster().get_score(importance_type='gain')
        }

        return model, scaler, metrics

    def evaluate_model(self, model: xgb.XGBClassifier, scaler: StandardScaler,
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
            'fpr': fpr.tolist()[:100],
            'tpr': tpr.tolist()[:100]
        }

        # PR curve points
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist()[:100],
            'recall': recall.tolist()[:100]
        }

        # Calculate prediction statistics
        metrics['prediction_stats'] = {
            'mean_pred_proba': float(y_pred_proba.mean()),
            'std_pred_proba': float(y_pred_proba.std()),
            'min_pred_proba': float(y_pred_proba.min()),
            'max_pred_proba': float(y_pred_proba.max()),
            'median_pred_proba': float(np.median(y_pred_proba))
        }

        return metrics

    def save_model(self, model: xgb.XGBClassifier, scaler: StandardScaler,
                   virus_key: str, params: Dict, metrics: Dict,
                   output_dir: str = "data/models"):
        """Save trained model and metadata"""

        # Create output directory
        model_dir = Path(output_dir) / virus_key / "xgboost"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model in multiple formats
        model.save_model(model_dir / "model.json")  # JSON format
        joblib.dump(model, model_dir / "model.pkl")  # Pickle format

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
            'xgboost_version': xgb.__version__,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance plot
        self.plot_feature_importance(model, virus_key, model_dir)

        logger.info(f"✓ Saved model to {model_dir}")

        return str(model_dir)

    def plot_feature_importance(self, model: xgb.XGBClassifier, virus_key: str, output_dir: Path):
        """Plot and save feature importance"""

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Weight importance
        ax1.barh(range(len(indices)), importances[indices])
        ax1.set_xlabel('Weight Importance')
        ax1.set_title(f'Top 20 Features by Weight - {virus_key.upper()}')

        # Gain importance
        try:
            xgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax2)
            ax2.set_title(f'Top 20 Features by Gain - {virus_key.upper()}')
        except:
            pass

        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300)
        plt.close()

    def train_all_viruses(self, data_dir: str = "data/activity") -> Dict:
        """Train XGBoost models for all viruses"""

        # Load viral targets
        with open("configs/viral_targets.json", 'r') as f:
            targets = json.load(f)

        results = {}

        for virus_key in targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training XGBoost for {virus_key.upper()}")
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
    print("XGBoost Training for Viral Proteases")
    print("With hyperparameter optimization using Optuna")
    print("=" * 60)

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/models").mkdir(exist_ok=True)

    # Initialize trainer
    trainer = XGBoostTrainer()

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

            if 'best_iteration' in metrics:
                print(f"  Best iteration: {metrics['best_iteration']}")

            if 'feature_importance' in metrics:
                print(f"  Feature Importance - FP: {metrics['feature_importance']['fingerprint_avg']:.4f}, "
                      f"Desc: {metrics['feature_importance']['descriptor_avg']:.4f}")
        else:
            print(f"\n{virus.upper()}: {result['error']}")

    # Save summary
    summary_path = "data/models/xgboost_training_summary.json"

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
                },
                'best_iteration': result['metrics'].get('best_iteration', 0)
            }
        else:
            json_safe_results[virus] = result

    with open(summary_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    print("\n✓ XGBoost training complete!")
    print("\nNext: Run 17_train_lightgbm.py or 20_evaluate_models.py to compare models")


if __name__ == "__main__":
    main()
