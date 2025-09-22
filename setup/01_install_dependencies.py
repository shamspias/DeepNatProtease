"""
Install all required dependencies for viral protease discovery pipeline
Updated: September 2025 - Latest stable versions
"""

import subprocess
import sys
from pathlib import Path


def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("=" * 60)
    print("Installing Dependencies for Viral Protease Discovery")
    print("Updated: September 22, 2025")
    print("=" * 60)

    # First, upgrade pip itself
    print("\nUpgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Core scientific packages (latest stable)
    core_packages = [
        "numpy==2.3.3",  # Latest NumPy 2.x
        "pandas==2.3.2",  # Latest Pandas
        "scipy==1.16.2",  # Latest SciPy
        "scikit-learn==1.7.2",  # Latest sklearn
        "matplotlib==3.10.6",  # Latest matplotlib
        "seaborn==0.13.2",  # Latest seaborn
        "tqdm==4.67.1",  # Progress bars
        "joblib==1.5.2"  # Parallel processing
    ]

    # Chemistry packages (2025 versions)
    chemistry_packages = [
        "rdkit==2025.3.6",
        "chembl-webresource-client==0.10.9",  # ChEMBL client
        "pubchempy==1.0.5",  # PubChem API
        "mordred==1.2.0",  # Molecular descriptors
        "molvs==0.1.1"  # Molecule standardization
    ]

    # Machine Learning packages (September 2025 latest)
    ml_packages = [
        "torch==2.8.0",  # PyTorch latest stable
        "torch-geometric==2.6.1",  # Graph neural networks
        "xgboost==3.0.5",  # Latest XGBoost
        "lightgbm==4.6.0",  # Latest LightGBM
        "catboost==1.2.8",  # Latest CatBoost
        "optuna==4.5.0",  # Hyperparameter optimization
        "shap==0.48.0"  # Model interpretability
    ]

    # Data handling
    data_packages = [
        "requests==2.32.5",
        "beautifulsoup4==4.13.5",
        "lxml==6.0.1",
        "pyyaml==6.0.2",
        "h5py==3.14.0",
        "pyarrow==21.0.0",
        "openpyxl==3.1.5"  # Excel file support
    ]

    # Additional utilities
    utility_packages = [
        "jupyterlab==4.4.7",
        "notebook==7.4.5",
        "ipywidgets==8.1.7",
        "python-dotenv==1.1.1",
        "click==8.3.0",
        "rich==14.1.0"  # Beautiful terminal output
    ]

    # Combine all packages
    all_packages = (
            core_packages +
            chemistry_packages +
            ml_packages +
            data_packages +
            utility_packages
    )

    print(f"\nInstalling {len(all_packages)} packages...")

    failed_packages = []
    installed_packages = []

    for i, package in enumerate(all_packages, 1):
        try:
            print(f"[{i}/{len(all_packages)}] Installing {package}...")
            install_package(package)
            installed_packages.append(package)
            print(f"✓ {package}")
        except Exception as e:
            print(f"✗ Failed: {package}")
            failed_packages.append(package)

    # Create requirements.txt
    print("\nCreating requirements.txt...")
    with open("requirements.txt", "w") as f:
        for package in all_packages:
            f.write(f"{package}\n")

    # Create project structure
    print("\nCreating project directories...")
    dirs_to_create = [
        "data/raw",
        "data/activity",
        "data/natural/coconut",
        "data/models",
        "data/results",
        "logs",
        "configs"
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    print(f"✓ Successfully installed: {len(installed_packages)}")

    if failed_packages:
        print(f"✗ Failed: {len(failed_packages)}")
        for pkg in failed_packages:
            print(f"  - {pkg}")
    else:
        print("\n✓ All packages installed successfully!")

    print("\nNext: Run 03_define_targets.py")


if __name__ == "__main__":
    main()
