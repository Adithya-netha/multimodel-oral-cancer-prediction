import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(labels, preds, probs):
    return {
        'accuracy':  accuracy_score(labels, preds),
        'f1':        f1_score(labels, preds, zero_division=0),
        'auc':       roc_auc_score(labels, probs),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'confusion': confusion_matrix(labels, preds),
    }

def print_report(metrics):
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        if k == 'confusion':
            print(f"\nConfusion Matrix:\n{v}")
        else:
            print(f"  {k.capitalize():12s}: {v:.4f}")
    print("="*50 + "\n")

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Cancer'],
                yticklabels=['Normal', 'Cancer'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Ensemble')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
