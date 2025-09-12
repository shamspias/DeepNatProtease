import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_cross_viral_activity():
    """Analyze compounds active against multiple viruses"""
    
    print("\n🔄 Analyzing cross-viral activity...")
    
    # Collect all predictions
    all_predictions = {}
    viruses = []
    
    for virus_dir in Path('results').iterdir():
        if virus_dir.is_dir():
            virus = virus_dir.name
            pred_file = virus_dir / 'coconut_predictions.csv'
            
            if pred_file.exists():
                viruses.append(virus)
                preds = pd.read_csv(pred_file)
                if 'active' not in preds.columns:
                    print(f"   ⚠️ results/{virus}/coconut_predictions.csv missing 'active' column")
                    continue
                all_predictions[virus] = preds['active'].values
    
    if not viruses:
        print("   ⚠️ No predictions found")
        return
    
    # Create cross-activity matrix
    n_compounds = len(next(iter(all_predictions.values())))
    activity_matrix = np.zeros((n_compounds, len(viruses)))
    
    for i, virus in enumerate(viruses):
        activity_matrix[:, i] = all_predictions[virus]
    
    # Find multi-viral hits
    activity_threshold = 0.7  # Score > 0.7 considered active
    active_matrix = (activity_matrix > activity_threshold).astype(int)
    n_active_viruses = active_matrix.sum(axis=1)
    
    # Statistics
    print(f"\n📊 Cross-Viral Activity Statistics:")
    for n in range(1, len(viruses) + 1):
        count = int((n_active_viruses == n).sum())
        if count > 0:
            print(f"   Active against {n} virus(es): {count} compounds")
    
    # Find pan-viral inhibitors
    pan_viral_idx = np.where(n_active_viruses == len(viruses))[0]
    if len(pan_viral_idx) > 0:
        print(f"\n🌟 Found {len(pan_viral_idx)} potential pan-viral inhibitors!")
        
        # Save pan-viral hits
        coconut_df = pd.read_csv('coconut_smiles.csv')
        pan_viral_hits = coconut_df.iloc[pan_viral_idx].copy()
        
        for i, virus in enumerate(viruses):
            pan_viral_hits[f'{virus}_score'] = activity_matrix[pan_viral_idx, i]
        
        Path('results').mkdir(exist_ok=True, parents=True)
        pan_viral_hits.to_csv('results/pan_viral_hits.csv', index=False)
        print("   💾 Saved to results/pan_viral_hits.csv")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(activity_matrix.T)
    sns.heatmap(correlation_matrix, 
                xticklabels=viruses,
                yticklabels=viruses,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0)
    plt.title('Cross-Viral Activity Correlation')
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True, parents=True)
    plt.savefig('results/cross_viral_correlation.png', dpi=300)
    print("   📊 Correlation plot saved to results/cross_viral_correlation.png")

if __name__ == '__main__':
    analyze_cross_viral_activity()
