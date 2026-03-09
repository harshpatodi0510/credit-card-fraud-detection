"""
evaluate.py
Metrics, confusion matrix, and PR / ROC curve utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score
)


def print_metrics(y_true, y_pred, model_name: str = "Model"):
    """Print precision, recall, F1, PR-AUC, ROC-AUC."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    if hasattr(y_pred, '__len__'):
        pr_auc  = average_precision_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f"PR-AUC  : {pr_auc:.4f}")
        print(f"ROC-AUC : {roc_auc:.4f}")


def plot_confusion_matrix(y_true, y_pred_binary, model_name: str = "Model", save_path: str = None):
    """Plot a clean confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight='bold')
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pr_curve(y_true, y_scores, model_name: str = "Model", save_path: str = None):
    """Plot Precision-Recall curve with optimal threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Find threshold that maximises F1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx  = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color='royalblue', lw=2, label=f"PR curve (AP={ap:.3f})")
    ax.scatter(recall[best_idx], precision[best_idx], color='red', zorder=5,
               label=f"Best threshold = {best_thresh:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Recommended threshold: {best_thresh:.4f}")
    return best_thresh
