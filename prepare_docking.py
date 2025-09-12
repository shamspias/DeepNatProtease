import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def prepare_for_docking(virus_name, n_compounds=100):
    """Prepare top hits for molecular docking"""
    
    print(f"\n🔗 Preparing {virus_name.upper()} hits for docking...")
    
    # Load top hits
    hits_file = Path(f'results/{virus_name}/top_1000_hits.csv')
    if not hits_file.exists():
        print(f"   ⚠️ No hits found for {virus_name}")
        return
    
    hits_df = pd.read_csv(hits_file).head(n_compounds)
    
    # Create docking directory
    docking_dir = Path(f'docking/{virus_name}')
    docking_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert SMILES to 3D structures
    for idx, row in hits_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                # Add hydrogens
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Save as PDB
                pdb_file = docking_dir / f"{row['coconut_id']}.pdb"
                Chem.MolToPDBFile(mol, str(pdb_file))
                
        except Exception as e:
            print(f"   Error with {row['coconut_id']}: {e}")
    
    # Create AutoDock Vina config (template; update receptor/box as needed)
    vina_config = f"""
# AutoDock Vina configuration for {virus_name}
receptor = receptors/{virus_name}_protease.pdbqt
ligand = ligands/EXAMPLE_LIGAND.pdbqt

# Search space (adjust based on protease active site)
center_x = 0.0
center_y = 0.0
center_z = 0.0
size_x = 20.0
size_y = 20.0
size_z = 20.0

exhaustiveness = 8
num_modes = 9
energy_range = 3
"""
    
    with open(docking_dir / 'vina_config.txt', 'w') as f:
        f.write(vina_config)
    
    print(f"   ✅ Prepared {len(hits_df)} compounds for docking")
    print(f"   📁 Files saved to docking/{virus_name}/")
    print(f"   📝 Next: Run AutoDock Vina with the prepared files")

# Prepare docking for each virus
viruses = ['dengue', 'zika', 'hiv_1', 'sars_cov_2', 'hcv']
for virus in viruses:
    prepare_for_docking(virus, n_compounds=100)
