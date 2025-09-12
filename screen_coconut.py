import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess

def screen_coconut_for_virus(virus_name, coconut_path='COCONUT_DB.sdf', top_n=1000):
    """Screen COCONUT database for specific virus"""
    
    print(f"\n🔬 Screening COCONUT for {virus_name.upper()} inhibitors...")
    
    # Check if model exists
    model_dir = Path(f'models/{virus_name}')
    if not model_dir.exists():
        print(f"   ⚠️ No model found for {virus_name}")
        return None
    
    # Prepare COCONUT for screening (convert SDF to CSV with SMILES)
    if not Path('coconut_smiles.csv').exists():
        print("   Converting COCONUT to SMILES format...")
        from rdkit import Chem
        
        supplier = Chem.SDMolSupplier(str(coconut_path))
        coconut_data = []
        
        for i, mol in enumerate(tqdm(supplier, desc="Processing COCONUT")):
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                coconut_data.append({
                    'smiles': smiles,
                    'coconut_id': f'COCONUT_{i:07d}'
                })
            
            if len(coconut_data) >= 100000:  # Process first 100k for demo
                break
        
        coconut_df = pd.DataFrame(coconut_data)
        coconut_df.to_csv('coconut_smiles.csv', index=False)
        print(f"   ✅ Prepared {len(coconut_df)} COCONUT compounds")
    
    # Run ChemProp prediction
    print(f"   🚀 Running {virus_name} model on COCONUT...")
    Path(f'results/{virus_name}').mkdir(exist_ok=True, parents=True)
    
    cmd = [
        'chemprop_predict',
        '--test_path', 'coconut_smiles.csv',
        '--checkpoint_dir', f'models/{virus_name}',
        '--preds_path', f'results/{virus_name}/coconut_predictions.csv',
        '--gpu', '0'  # Use GPU if available
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✅ Predictions saved to results/{virus_name}/")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # Load and rank predictions
    preds_df = pd.read_csv(f'results/{virus_name}/coconut_predictions.csv')
    coconut_df = pd.read_csv('coconut_smiles.csv')
    
    # Combine and sort by prediction score
    results_df = pd.concat([coconut_df, preds_df[['active']]], axis=1)
    results_df = results_df.rename(columns={'active': f'{virus_name}_score'})
    results_df = results_df.sort_values(f'{virus_name}_score', ascending=False)
    
    # Save top hits
    top_hits = results_df.head(top_n)
    top_hits.to_csv(f'results/{virus_name}/top_{top_n}_hits.csv', index=False)
    
    print(f"   📊 Top 10 {virus_name.upper()} hits:")
    for idx, row in top_hits.head(10).iterrows():
        print(f"      {row['coconut_id']}: Score = {row[f'{virus_name}_score']:.3f}")
    
    return top_hits

# Screen for each virus
viruses = ['dengue', 'zika', 'hiv_1', 'sars_cov_2', 'hcv']

all_hits = {}
for virus in viruses:
    Path(f'results/{virus}').mkdir(exist_ok=True, parents=True)
    hits = screen_coconut_for_virus(virus, top_n=1000)
    if hits is not None:
        all_hits[virus] = hits

print("\n✅ COCONUT screening complete!")
print(f"   Found hits for {len(all_hits)} viruses")
print("   Top hits saved to results/[virus_name]/top_1000_hits.csv")
