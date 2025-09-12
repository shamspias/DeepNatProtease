import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import json

def evaluate_virus_model(virus_name):
    """Evaluate a trained virus-specific model"""
    
    # Load test predictions
    test_preds_path = Path(f'models/{virus_name}/test_preds.csv')
    if not test_preds_path.exists():
        print(f"No predictions found for {virus_name}")
        return None
    
    # Load predictions
    preds_df = pd.read_csv(test_preds_path)
    if 'active' not in preds_df.columns or 'active_pred' not in preds_df.columns:
        print(f"Predictions file for {virus_name} must contain columns 'active' and 'active_pred'.")
        return None

    y_true = preds_df['active']
    y_pred_proba = preds_df['active_pred']
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'virus': virus_name,
        'test_size': int(len(y_true)),
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'accuracy': float((y_pred == y_true).mean()),
        'sensitivity': float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else None,
        'specificity': float((y_pred[y_true == 0] == 0).mean()) if (y_true == 0).any() else None
    }
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['pr_auc'] = float(auc(recall, precision))
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {
        'tn': int(tn), 'fp': int(fp), 
        'fn': int(fn), 'tp': int(tp)
    }
    
    return metrics

# Evaluate all models
results = {}
models_dir = Path('models')
models_dir.mkdir(exist_ok=True, parents=True)

for virus_dir in models_dir.iterdir():
    if virus_dir.is_dir():
        virus_name = virus_dir.name
        metrics = evaluate_virus_model(virus_name)
        if metrics:
            results[virus_name] = metrics
            print(f"\n{virus_name.upper()} Performance:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.3f}" if metrics['sensitivity'] is not None else "  Sensitivity: N/A")
            print(f"  Specificity: {metrics['specificity']:.3f}" if metrics['specificity'] is not None else "  Specificity: N/A")

# Save results
Path('results').mkdir(exist_ok=True, parents=True)
with open('results/model_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Evaluation complete! Results saved to results/model_evaluation.json")
