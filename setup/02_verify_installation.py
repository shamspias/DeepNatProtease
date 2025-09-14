"""
Verify that all required packages are properly installed and create directory structure
"""

import os
import sys
import importlib
from pathlib import Path


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def create_directory_structure():
    """Create the required directory structure"""
    base_dirs = [
        "setup",
        "data_collection",
        "data_processing",
        "model_training",
        "model_evaluation",
        "screening",
        "docking",
        "analysis",
        "configs",
        "utils",
        "logs",
        "notebooks"
    ]

    data_dirs = [
        "data/activity/hiv1/raw",
        "data/activity/hiv1/processed",
        "data/activity/hiv1/splits",
        "data/activity/hcv/raw",
        "data/activity/hcv/processed",
        "data/activity/hcv/splits",
        "data/activity/sars_cov2/raw",
        "data/activity/sars_cov2/processed",
        "data/activity/sars_cov2/splits",
        "data/activity/dengue/raw",
        "data/activity/dengue/processed",
        "data/activity/dengue/splits",
        "data/activity/zika/raw",
        "data/activity/zika/processed",
        "data/activity/zika/splits",
        "data/natural/coconut",
        "data/models/hiv1",
        "data/models/hcv",
        "data/models/sars_cov2",
        "data/models/dengue",
        "data/models/zika",
        "data/results/screening",
        "data/results/docking"
    ]

    all_dirs = base_dirs + data_dirs

    for dir_path in all_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return all_dirs


def main():
    print("=" * 60)
    print("Verifying Installation and Setting Up Project Structure")
    print("=" * 60)

    # Define packages to check
    packages_to_check = [
        ("numpy", None),
        ("pandas", None),
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
        ("rdkit", "RDKit"),
        ("chembl_webresource_client", "ChEMBL Client"),
        ("pubchempy", "PubChemPy"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("chemprop", "ChemProp"),
        ("Bio", "BioPython"),
        ("requests", None),
        ("yaml", "PyYAML"),
        ("tqdm", None),
        ("mordred", "Mordred"),
        ("optuna", "Optuna")
    ]

    print("\nChecking package installations...")
    print("-" * 40)

    success_count = 0
    failed_packages = []

    for module_name, display_name in packages_to_check:
        if display_name is None:
            display_name = module_name

        success, version = check_import(module_name)

        if success:
            print(f"✓ {display_name:<20} Version: {version}")
            success_count += 1
        else:
            print(f"✗ {display_name:<20} Not installed")
            failed_packages.append(display_name)

    # Check for special requirements
    print("\n" + "-" * 40)
    print("Checking special requirements...")

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("ℹ CUDA not available - will use CPU for deep learning")
    except:
        print("✗ PyTorch not properly installed")

    # Check RDKit functionality
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        if mol:
            print("✓ RDKit molecular operations working")
    except:
        print("✗ RDKit not functioning properly")

    # Create directory structure
    print("\n" + "=" * 60)
    print("Creating directory structure...")
    print("-" * 40)

    created_dirs = create_directory_structure()
    print(f"✓ Created {len(created_dirs)} directories")

    # Create initial configuration files
    print("\nCreating configuration files...")

    # Database configuration
    db_config = """# Database Configuration
chembl:
  version: 34
  batch_size: 1000
  timeout: 30

bindingdb:
  url: "https://www.bindingdb.org/bind/downloads/"
  version: "2025_09"

pubchem:
  batch_size: 100
  rate_limit: 5  # requests per second
  timeout: 30

zinc:
  subset: "fda-approved"
  format: "sdf"

covid_moonshot:
  url: "https://covid.postera.ai/covid/activity_data"
"""

    with open("configs/database_config.yaml", "w") as f:
        f.write(db_config)
    print("✓ Created configs/database_config.yaml")

    # Model configuration
    model_config = """# Model Configuration
chemprop:
  hidden_size: 300
  depth: 3
  dropout: 0.2
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

random_forest:
  n_estimators: 1000
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  class_weight: balanced

xgboost:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.01
  objective: binary:logistic
  eval_metric: auc

lightgbm:
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.01
  n_estimators: 500
  objective: binary
  metric: auc

deep_neural_network:
  layers: [512, 256, 128, 64]
  dropout: 0.3
  activation: relu
  optimizer: adam
  learning_rate: 0.001
  epochs: 100
  batch_size: 64
"""

    with open("configs/model_config.yaml", "w") as f:
        f.write(model_config)
    print("✓ Created configs/model_config.yaml")

    # Docking configuration
    docking_config = """# Docking Configuration
proteins:
  hiv1:
    pdb_id: "7BWJ"
    center: [0.0, 0.0, 0.0]
    size: [20, 20, 20]
  hcv:
    pdb_id: "3SV6"
    center: [0.0, 0.0, 0.0]
    size: [20, 20, 20]
  sars_cov2:
    pdb_id: "7BQY"
    center: [-10.7, 12.4, 68.8]
    size: [20, 20, 20]
  dengue:
    pdb_id: "2FOM"
    center: [0.0, 0.0, 0.0]
    size: [20, 20, 20]
  zika:
    pdb_id: "5LC0"
    center: [0.0, 0.0, 0.0]
    size: [20, 20, 20]

vina:
  exhaustiveness: 32
  num_modes: 20
  energy_range: 3
  cpu: 0  # use all available CPUs
"""

    with open("configs/docking_config.yaml", "w") as f:
        f.write(docking_config)
    print("✓ Created configs/docking_config.yaml")

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"✓ Packages verified: {success_count}/{len(packages_to_check)}")
    print(f"✓ Directories created: {len(created_dirs)}")
    print("✓ Configuration files created: 3")

    if failed_packages:
        print(f"\n⚠ Missing packages ({len(failed_packages)}):")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages before proceeding.")
        sys.exit(1)
    else:
        print("\n✓ All systems ready!")
        print("\nNext step: Run data_collection/03_define_targets.py to define viral targets")


if __name__ == "__main__":
    main()
