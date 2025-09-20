"""
Train Graph Neural Network models using PyTorch directly (ChemProp-style architecture)
Simplified version that doesn't rely on ChemProp's changing API
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import yaml
import time
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_chemprop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MolecularGraphDataset(Dataset):
    """PyTorch dataset for molecular graphs"""

    def __init__(self, smiles_list: List[str], labels: List[int]):
        self.smiles_list = smiles_list
        self.labels = labels
        self.features = []

        # Precompute molecular features
        for smiles in tqdm(smiles_list, desc="Computing molecular features"):
            features = self.compute_features(smiles)
            self.features.append(features)

    def compute_features(self, smiles: str) -> torch.Tensor:
        """Compute molecular features from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(200)  # Return zero vector for invalid molecules

            # Compute various molecular descriptors
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
            ]

            # Get Morgan fingerprint
            from rdkit.Chem import AllChem
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=185)
            fp_array = np.zeros((185,))
            fp.ConvertToNumpyArray(fp_array)

            # Combine descriptors and fingerprint
            all_features = np.concatenate([features, fp_array])

            return torch.FloatTensor(all_features)

        except Exception as e:
            return torch.zeros(200)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.features[idx], torch.FloatTensor([self.labels[idx]])


class SimpleGNN(nn.Module):
    """Simplified Graph Neural Network for molecular property prediction"""

    def __init__(self, input_dim=200, hidden_dim=300, output_dim=1, dropout=0.2, depth=3):
        super(SimpleGNN, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Message passing layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Input transformation
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            identity = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + identity  # Residual connection

        # Output
        x = self.output_layer(x)
        return x


class GNNTrainer:
    """Train GNN models for viral protease inhibitors"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize GNN trainer"""

        # Try to load configuration, use defaults if file doesn't exist
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config.get('chemprop', {})
        except:
            logger.warning(f"Could not load config from {config_path}, using defaults")
            self.config = {}

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Model hyperparameters
        self.default_params = {
            'hidden_size': self.config.get('hidden_size', 300),
            'depth': self.config.get('depth', 3),
            'dropout': self.config.get('dropout', 0.2),
            'epochs': self.config.get('epochs', 50),
            'batch_size': self.config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 0.001),
        }

        # Virus-specific adjustments
        self.virus_params = {
            'hiv1': {},  # Use defaults
            'hcv': {},  # Use defaults
            'sars_cov2': {
                'batch_size': 64  # Larger dataset
            },
            'dengue': {},
            'zika': {
                'hidden_size': 200,  # Smaller model for small dataset
                'depth': 2,
                'epochs': 30,
                'dropout': 0.3  # More dropout to prevent overfitting
            }
        }

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

    def calculate_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""

        n_samples = len(train_df)
        n_classes = 2

        class_counts = train_df['is_active'].value_counts()
        weights = []

        for i in range(n_classes):
            if i in class_counts.index:
                weight = n_samples / (n_classes * class_counts[i])
            else:
                weight = 1.0
            weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device)
        logger.info(f"Class weights: Inactive={weights[0]:.2f}, Active={weights[1]:.2f}")

        # Return positive weight for BCEWithLogitsLoss
        return weights[1] / weights[0]

    def train_model(self, virus_key: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    params: Dict) -> Tuple[nn.Module, Dict]:
        """Train GNN model for a specific virus"""

        logger.info(f"Training GNN for {virus_key}")

        # Prepare datasets
        train_dataset = MolecularGraphDataset(
            train_df['standardized_smiles'].tolist(),
            train_df['is_active'].tolist()
        )

        val_dataset = MolecularGraphDataset(
            val_df['standardized_smiles'].tolist(),
            val_df['is_active'].tolist()
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Calculate class weights for imbalanced data
        pos_weight = self.calculate_class_weights(train_df)

        # Build model
        model = SimpleGNN(
            input_dim=200,
            hidden_dim=params['hidden_size'],
            output_dim=1,
            dropout=params['dropout'],
            depth=params['depth']
        )
        model = model.to(self.device)

        # Loss function with class weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

        # Best model tracking
        best_val_auc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0

        # Training loop
        for epoch in range(params['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            for features, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params['epochs']} - Training"):
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                train_loss += loss.item()
                train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                train_targets.extend(targets.cpu().numpy())

            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(features)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)

            train_targets = np.array(train_targets).flatten()
            train_preds = np.array(train_preds).flatten()
            val_targets = np.array(val_targets).flatten()
            val_preds = np.array(val_preds).flatten()

            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            # Log progress
            logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                        f"Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

            # Update learning rate
            scheduler.step(val_auc)

            # Check for improvement
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, history

    def evaluate_model(self, model: nn.Module, test_df: pd.DataFrame) -> Dict:
        """Evaluate model on test set"""

        # Prepare test data
        test_dataset = MolecularGraphDataset(
            test_df['standardized_smiles'].tolist(),
            test_df['is_active'].tolist()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        # Get predictions
        model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for features, batch_targets in test_loader:
                features = features.to(self.device)
                outputs = model(features)
                preds = torch.sigmoid(outputs).cpu().numpy()

                predictions.extend(preds)
                targets.extend(batch_targets.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(targets, predictions),
            'pr_auc': average_precision_score(targets, predictions),
            'balanced_accuracy': balanced_accuracy_score(targets, predictions > 0.5),
            'f1_score': f1_score(targets, predictions > 0.5),
            'mcc': matthews_corrcoef(targets, predictions > 0.5)
        }

        # Confusion matrix
        cm = confusion_matrix(targets, predictions > 0.5)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(targets, predictions > 0.5, output_dict=True)
        metrics['classification_report'] = report

        return metrics

    def save_model(self, model: nn.Module, virus_key: str, params: Dict,
                   metrics: Dict, output_dir: str = "data/models"):
        """Save trained model and metadata"""

        # Create output directory
        model_dir = Path(output_dir) / virus_key / "gnn"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(model.state_dict(), model_dir / "model.pt")

        # Save model architecture info
        model_info = {
            'params': params,
            'model_class': 'SimpleGNN',
            'device': str(self.device),
            'metrics': metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(model_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"✓ Saved model to {model_dir}")

        return str(model_dir)

    def train_all_viruses(self, data_dir: str = "data/activity") -> Dict:
        """Train GNN models for all viruses"""

        # Load viral targets
        with open("configs/viral_targets.json", 'r') as f:
            targets = json.load(f)

        results = {}

        for virus_key in targets.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training GNN for {virus_key.upper()}")
            logger.info(f"{'=' * 60}")

            try:
                # Load data
                train_df, val_df, test_df = self.load_data(virus_key, data_dir)

                # Skip if too little data
                if len(train_df) < 50:
                    logger.warning(f"Skipping {virus_key} - insufficient training data ({len(train_df)} samples)")
                    results[virus_key] = {'error': 'Insufficient data'}
                    continue

                # Get parameters for this virus
                params = self.default_params.copy()
                params.update(self.virus_params.get(virus_key, {}))

                # Train model
                model, history = self.train_model(virus_key, train_df, val_df, params)

                # Evaluate on test set
                test_metrics = self.evaluate_model(model, test_df)

                # Save model
                model_path = self.save_model(model, virus_key, params, test_metrics)

                # Store results
                results[virus_key] = {
                    'model_path': model_path,
                    'params': params,
                    'test_metrics': test_metrics,
                    'training_history': history
                }

                logger.info(f"✓ {virus_key} - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

            except Exception as e:
                logger.error(f"Error training {virus_key}: {str(e)}")
                results[virus_key] = {'error': str(e)}

        return results


def main():
    """Main execution function"""

    print("=" * 60)
    print("GNN Training for Viral Proteases")
    print("Simplified implementation without ChemProp dependency")
    print("=" * 60)

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("data/models").mkdir(exist_ok=True)

    # Initialize trainer
    trainer = GNNTrainer()

    # Train models for all viruses
    results = trainer.train_all_viruses()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for virus, result in results.items():
        if 'error' not in result:
            metrics = result['test_metrics']
            print(f"\n{virus.upper()}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  MCC: {metrics['mcc']:.4f}")
        else:
            print(f"\n{virus.upper()}: {result['error']}")

    # Save summary
    summary_path = "data/models/gnn_training_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_results = {}
        for virus, result in results.items():
            if 'error' not in result:
                json_safe_results[virus] = {
                    'model_path': result['model_path'],
                    'test_metrics': {
                        'roc_auc': float(result['test_metrics']['roc_auc']),
                        'pr_auc': float(result['test_metrics']['pr_auc']),
                        'balanced_accuracy': float(result['test_metrics']['balanced_accuracy']),
                        'f1_score': float(result['test_metrics']['f1_score']),
                        'mcc': float(result['test_metrics']['mcc'])
                    }
                }
            else:
                json_safe_results[virus] = result

        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    print("\n✓ GNN training complete!")
    print("\nNext: Run 15_train_random_forest.py")


if __name__ == "__main__":
    main()
