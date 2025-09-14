"""
Install all required dependencies for the viral protease inhibitor discovery pipeline
Updated for September 2025 - Using latest stable versions
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("=" * 60)
    print("Installing Dependencies for Viral Protease Discovery Pipeline")
    print("Date: September 2025")
    print("=" * 60)

    # Core scientific packages
    core_packages = [
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "tqdm==4.66.5"
    ]

    # Chemistry packages
    chemistry_packages = [
        "rdkit==2024.03.5",  # Latest RDKit version as of 2025
        "chembl-webresource-client==0.10.9",
        "pubchempy==1.0.4",
        "openbabel-wheel==3.1.1.2",
        "mordred==1.2.0",  # Molecular descriptor calculator
        "molvs==0.1.1"  # Molecule validation and standardization
    ]

    # Machine Learning packages (latest versions for 2025)
    ml_packages = [
        "torch==2.4.0",  # PyTorch latest stable
        "torch-geometric==2.5.3",  # For graph neural networks
        "torchvision==0.19.0",
        "chemprop==2.0.3",  # Latest ChemProp for molecular property prediction
        "xgboost==2.1.0",
        "lightgbm==4.5.0",
        "catboost==1.2.5",
        "optuna==3.6.1",  # Hyperparameter optimization
        "shap==0.45.1"  # Model interpretability
    ]

    # Data handling and APIs
    data_packages = [
        "requests==2.32.3",
        "beautifulsoup4==4.12.3",
        "lxml==5.2.2",
        "pyyaml==6.0.2",
        "jsonschema==4.23.0",
        "h5py==3.11.0",
        "pyarrow==17.0.0",  # For efficient data storage
        "fastparquet==2024.5.0"
    ]

    # Molecular docking and structural biology
    docking_packages = [
        "biopython==1.84",
        "prody==2.4.1",
        "meeko==0.5.0",  # AutoDock Vina tools
        "plip==2.3.2",  # Protein-ligand interaction profiler
        "pymol-open-source==3.0.0"  # For visualization
    ]

    # Additional utilities
    utility_packages = [
        "jupyterlab==4.2.4",
        "notebook==7.2.1",
        "ipywidgets==8.1.3",
        "py3Dmol==2.3.0",  # 3D molecular visualization
        "wandb==0.17.7",  # Experiment tracking
        "rich==13.7.1",  # Beautiful terminal output
        "click==8.1.7",  # CLI creation
        "python-dotenv==1.0.1"
    ]

    # Combine all packages
    all_packages = (
            core_packages +
            chemistry_packages +
            ml_packages +
            data_packages +
            docking_packages +
            utility_packages
    )

    print("\nInstalling packages...")
    print(f"Total packages to install: {len(all_packages)}\n")

    failed_packages = []

    for i, package in enumerate(all_packages, 1):
        try:
            print(f"[{i}/{len(all_packages)}] Installing {package}...")
            install_package(package)
            print(f"✓ Successfully installed {package}")
        except Exception as e:
            print(f"✗ Failed to install {package}: {str(e)}")
            failed_packages.append(package)

    # Install packages that require special handling
    print("\n" + "=" * 60)
    print("Installing special packages...")

    # Install PyTorch with CUDA support if available
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA is available. PyTorch with CUDA support already installed.")
        else:
            print("CUDA not available. Using CPU version of PyTorch.")
    except ImportError:
        print("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])

    # Create requirements.txt for reproducibility
    print("\n" + "=" * 60)
    print("Creating requirements.txt file...")

    with open("requirements.txt", "w") as f:
        for package in all_packages:
            f.write(f"{package}\n")

    print("✓ requirements.txt created successfully")

    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    print(f"✓ Successfully installed: {len(all_packages) - len(failed_packages)} packages")

    if failed_packages:
        print(f"✗ Failed installations: {len(failed_packages)} packages")
        print("\nFailed packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install these packages manually or check for compatibility issues.")
    else:
        print("\n✓ All packages installed successfully!")

    print("\nNext step: Run 02_verify_installation.py to verify the installation")


if __name__ == "__main__":
    main()
