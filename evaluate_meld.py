"""
MELD Dataset - Model Evaluation Script
Evaluates baseline and multimodal CNN models for emotion recognition
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings

# Configuration
RESULTS_DIR = 'results_meld'
FIGURES_DIR = 'figures_meld'
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
DATASET_NAME = 'MELD'

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data_and_models():
    """Load test data and trained models"""
    print(f"Loading {DATASET_NAME} test data and models...")
    
    # Load the preprocessed test data saved by train_meld.py
    test_data = np.load(f'{RESULTS_DIR}/test_data.npz')
    X_test_audio = test_data['X_audio_test']
    X_test_text = test_data['X_text_test']
    y_test = test_data['y_test']
    
    # Audio is already in the correct shape from training
    
    baseline_model = tf.keras.models.load_model(f'{RESULTS_DIR}/baseline_cnn.h5')
    multimodal_model = tf.keras.models.load_model(f'{RESULTS_DIR}/multimodal_cnn.h5')
    
    print(f"Test samples: {len(y_test)}")
    print(f"Audio shape: {X_test_audio.shape}, Text shape: {X_test_text.shape}")
    
    return X_test_audio, X_test_text, y_test, baseline_model, multimodal_model


def generate_predictions(X_audio, X_text, baseline_model, multimodal_model):
    """Generate predictions from both models"""
    print("\nGenerating predictions")
    
    y_pred_baseline_proba = baseline_model.predict(X_audio, verbose=0)
    y_pred_baseline = np.argmax(y_pred_baseline_proba, axis=1)
    
    y_pred_multimodal_proba = multimodal_model.predict([X_audio, X_text], verbose=0)
    y_pred_multimodal = np.argmax(y_pred_multimodal_proba, axis=1)
    
    return y_pred_baseline, y_pred_baseline_proba, y_pred_multimodal, y_pred_multimodal_proba


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate evaluation metrics"""
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }


def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_multimodal, baseline_acc, multimodal_acc):
    """Plot confusion matrices for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[0])
    axes[0].set_title(f'Baseline CNN\nAccuracy: {baseline_acc:.3f}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    cm_multimodal = confusion_matrix(y_test, y_pred_multimodal)
    sns.heatmap(cm_multimodal, annot=True, fmt='d', cmap='Greens',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[1])
    axes[1].set_title(f'Multimodal CNN\nAccuracy: {multimodal_acc:.3f}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    save_path = f'{FIGURES_DIR}/confusion_matrices_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_roc_curves(y_test, y_pred_baseline_proba, y_pred_multimodal_proba):
    """Plot ROC curves for both models"""
    y_test_bin = label_binarize(y_test, classes=range(len(EMOTIONS)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    def plot_roc(y_true_bin, y_pred_proba, ax, title):
        fpr, tpr, roc_auc = {}, {}, {}
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
        
        for i, color in zip(range(len(EMOTIONS)), colors):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=color, lw=2, 
                   label=f'{EMOTIONS[i]} (AUC={roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.50)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        return np.mean(list(roc_auc.values()))
    
    mean_auc_baseline = plot_roc(y_test_bin, y_pred_baseline_proba, axes[0], 'Baseline CNN - ROC Curves')
    mean_auc_multimodal = plot_roc(y_test_bin, y_pred_multimodal_proba, axes[1], 'Multimodal CNN - ROC Curves')
    
    plt.tight_layout()
    save_path = f'{FIGURES_DIR}/roc_curves_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    print(f"\nROC Analysis:")
    print(f"  Baseline Mean AUC: {mean_auc_baseline:.3f}")
    print(f"  Multimodal Mean AUC: {mean_auc_multimodal:.3f}")
    print(f"  Improvement: {((mean_auc_multimodal - mean_auc_baseline) / mean_auc_baseline * 100):+.2f}%")


def plot_precision_recall(y_test, y_pred_baseline_proba, y_pred_multimodal_proba):
    """Plot precision-recall curves"""
    y_test_bin = label_binarize(y_test, classes=range(len(EMOTIONS)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    def plot_pr(y_true_bin, y_pred_proba, ax, title):
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
        for i, color in zip(range(len(EMOTIONS)), colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, color=color, lw=2, label=EMOTIONS[i])
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    plot_pr(y_test_bin, y_pred_baseline_proba, axes[0], 'Baseline CNN')
    plot_pr(y_test_bin, y_pred_multimodal_proba, axes[1], 'Multimodal CNN')
    
    plt.tight_layout()
    save_path = f'{FIGURES_DIR}/precision_recall_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_f1_comparison(y_test, y_pred_baseline, y_pred_multimodal):
    """Plot per-class F1-score comparison"""
    f1_baseline = f1_score(y_test, y_pred_baseline, average=None)
    f1_multimodal = f1_score(y_test, y_pred_multimodal, average=None)
    
    f1_df = pd.DataFrame({
        'Emotion': EMOTIONS,
        'Baseline': f1_baseline,
        'Multimodal': f1_multimodal,
        'Improvement': f1_multimodal - f1_baseline
    })
    
    print("\nPer-Class F1-Score Comparison:")
    print(f1_df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(EMOTIONS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, f1_baseline, width, label='Baseline CNN', 
                   color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_multimodal, width, label='Multimodal CNN', 
                   color='lightgreen', edgecolor='black')
    
    ax.set_xlabel('Emotions', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = f'{FIGURES_DIR}/f1_score_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline"""
    print(f"{DATASET_NAME} - Model Evaluation")
 
    X_test_audio, X_test_text, y_test, baseline_model, multimodal_model = load_data_and_models()
    
    y_pred_baseline, y_pred_baseline_proba, y_pred_multimodal, y_pred_multimodal_proba = \
        generate_predictions(X_test_audio, X_test_text, baseline_model, multimodal_model)
    
    baseline_metrics = calculate_metrics(y_test, y_pred_baseline, 'Baseline CNN')
    multimodal_metrics = calculate_metrics(y_test, y_pred_multimodal, 'Multimodal CNN')
    
    print("\nOverall Metrics:")
    metrics_df = pd.DataFrame([baseline_metrics, multimodal_metrics])
    print(metrics_df.to_string(index=False))
    
    print("\nGenerating visualizations")
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_multimodal, 
                           baseline_metrics['Accuracy'], multimodal_metrics['Accuracy'])
    plot_roc_curves(y_test, y_pred_baseline_proba, y_pred_multimodal_proba)
    plot_precision_recall(y_test, y_pred_baseline_proba, y_pred_multimodal_proba)
    plot_f1_comparison(y_test, y_pred_baseline, y_pred_multimodal)
    
    print(" All figures saved to:", FIGURES_DIR)

if __name__ == "__main__":
    main()
