"""
HSI Evaluation Metrics
Multi-class evaluation metrics and visualization for CCARS results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from pathlib import Path


def compute_multiclass_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive multi-class evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(12, 10)):
    """
    Plot 16x16 confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    return cm


def plot_wavelength_frequency(coefficients_df, wavelengths, save_path=None, figsize=(14, 6)):
    """
    Plot wavelength selection frequency across CARS runs
    
    Args:
        coefficients_df: DataFrame with columns ['Run', 'Iteration', 'Wavelength', 'Coefficient']
        wavelengths: Array of all wavelength values
        save_path: Path to save figure
        figsize: Figure size
    """
    # Ensure wavelengths is numpy array
    wavelengths = np.array(wavelengths)
    # Get final iterations from each run
    final_iters = coefficients_df.groupby('Run')['Iteration'].max()
    
    # Collect wavelengths from final iterations
    final_wls = []
    for run_idx, max_iter in final_iters.items():
        run_final = coefficients_df[
            (coefficients_df['Run'] == run_idx) &
            (coefficients_df['Iteration'] == max_iter)
        ]
        final_wls.extend(run_final['Wavelength'].values)
    
    # Count frequencies
    wl_counts = pd.Series(final_wls).value_counts().sort_index()
    
    # Plot
    plt.figure(figsize=figsize)
    plt.bar(wl_counts.index, wl_counts.values, width=5, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Selection Frequency', fontsize=12)
    plt.title('Wavelength Selection Frequency Across CARS Runs', fontsize=14, pad=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Wavelength frequency plot saved to: {save_path}")
    
    plt.close()
    
    return wl_counts


def plot_cars_convergence(statistics_df, save_path=None, figsize=(14, 10)):
    """
    Plot CARS convergence: variables selected and accuracy over iterations
    
    Args:
        statistics_df: DataFrame with CARS statistics
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Selected Variables vs Iteration (average across runs)
    avg_selected = statistics_df.groupby('Iteration')['Selected_Variables'].mean()
    std_selected = statistics_df.groupby('Iteration')['Selected_Variables'].std()
    
    ax1.plot(avg_selected.index, avg_selected.values, color='steelblue', linewidth=2, label='Mean')
    ax1.fill_between(avg_selected.index,
                      avg_selected.values - std_selected.values,
                      avg_selected.values + std_selected.values,
                      alpha=0.3, color='steelblue', label='±1 std')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Selected Variables', fontsize=12)
    ax1.set_title('CARS Variable Selection Convergence', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot 2: Accuracy vs Iteration
    avg_acc = statistics_df.groupby('Iteration')['Accuracy'].mean()
    std_acc = statistics_df.groupby('Iteration')['Accuracy'].std()
    
    ax2.plot(avg_acc.index, avg_acc.values, color='forestgreen', linewidth=2, label='Mean Accuracy')
    ax2.fill_between(avg_acc.index,
                      avg_acc.values - std_acc.values,
                      avg_acc.values + std_acc.values,
                      alpha=0.3, color='forestgreen', label='±1 std')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy During Variable Selection', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Convergence plot saved to: {save_path}")
    
    plt.close()


def generate_evaluation_report(y_true, y_pred, class_names, selected_wavelengths, 
                                full_wavelengths, save_path=None):
    """
    Generate comprehensive text evaluation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        selected_wavelengths: Array of selected wavelength values
        full_wavelengths: Array of all wavelength values
        save_path: Path to save report
    
    Returns:
        Report string
    """
    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)
    
    report = []
    report.append("=" * 70)
    report.append("CCARS+PLS-DA Evaluation Report")
    report.append("=" * 70)
    report.append("")
    
    # Overall metrics
    report.append("Overall Metrics:")
    report.append(f"  Accuracy:           {metrics['accuracy']:.4f}")
    report.append(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    report.append(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    report.append(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
    report.append("")
    report.append(f"  Precision (macro):    {metrics['precision_macro']:.4f}")
    report.append(f"  Recall (macro):       {metrics['recall_macro']:.4f}")
    report.append(f"  F1-Score (macro):     {metrics['f1_macro']:.4f}")
    report.append("")
    
    # Wavelength reduction
    n_original = len(full_wavelengths)
    n_selected = len(selected_wavelengths)
    reduction = (1 - n_selected / n_original) * 100
    
    report.append("Dimensionality Reduction:")
    report.append(f"  Original wavelengths: {n_original}")
    report.append(f"  Selected wavelengths: {n_selected}")
    report.append(f"  Reduction:            {reduction:.1f}%")
    report.append("")
    
    # Selected wavelengths
    report.append("Selected Wavelengths:")
    if n_selected <= 20:
        wl_str = ", ".join([f"{wl:.2f}" for wl in selected_wavelengths])
        report.append(f"  {wl_str} nm")
    else:
        wl_str = ", ".join([f"{wl:.2f}" for wl in selected_wavelengths[:10]])
        report.append(f"  First 10: {wl_str} nm")
        report.append(f"  ... and {n_selected - 10} more")
    report.append("")
    
    # Per-class metrics
    report.append("Per-Class Metrics:")
    report.append(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    report.append("-" * 70)
    for i, class_name in enumerate(class_names):
        if i < len(metrics['precision_per_class']):
            report.append(f"{class_name:<30} {metrics['precision_per_class'][i]:>10.4f} "
                         f"{metrics['recall_per_class'][i]:>10.4f} {metrics['f1_per_class'][i]:>10.4f}")
    
    report.append("")
    report.append("=" * 70)
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        print(f"✓ Evaluation report saved to: {save_path}")
    
    return report_str


if __name__ == '__main__':
    """Test evaluation functions"""
    print("=" * 70)
    print("Testing Evaluation Functions")
    print("=" * 70)
    
    # Generate synthetic results
    np.random.seed(42)
    n_samples = 500
    n_classes = 16
    
    y_true = np.random.randint(0, n_classes, n_samples)
    # Make predictions correlated with true labels (80% accuracy)
    y_pred = y_true.copy()
    wrong_mask = np.random.random(n_samples) > 0.8
    y_pred[wrong_mask] = np.random.randint(0, n_classes, wrong_mask.sum())
    
    class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Test metrics
    print("\n1. Computing metrics...")
    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    
    # Test confusion matrix
    print("\n2. Plotting confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    print(f"   Confusion matrix shape: {cm.shape}")
    
    # Test report
    print("\n3. Generating evaluation report...")
    selected_wl = np.linspace(450, 900, 30)
    full_wl = np.linspace(400, 2500, 200)
    report = generate_evaluation_report(y_true, y_pred, class_names, selected_wl, full_wl)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation functions test complete!")
    print("=" * 70)
