"""
Create Publication-Quality Plots for Feature Selection Comparison

Generates plots similar to reference paper:
1. Mean spectra per class
2. Confusion matrices with metrics
3. Selected wavelengths overlay on mean spectrum
4. Method comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def plot_mean_spectra_per_class(X_df, wavelengths, dataset_name, output_path):
    """
    Plot mean reflectance spectra for each class
    Similar to reference paper Figure 1
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique classes
    classes = X_df.index.get_level_values('Class').unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # Plot mean spectrum for each class
    for idx, cls in enumerate(sorted(classes)):
        # Get data for this class
        class_data = X_df[X_df.index.get_level_values('Class') == cls]
        mean_spectrum = class_data.mean(axis=0).values
        
        ax.plot(wavelengths, mean_spectrum, 
               label=f'class {cls+1}', color=colors[idx], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance (a.u.)', fontsize=13, fontweight='bold')
    ax.set_title(f'TRAIN mean spectra per class', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, frameon=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved mean spectra plot: {output_path}")


def plot_confusion_matrix_with_metrics(y_true, y_pred, method_name, classifier_name,
                                       n_features, output_path):
    """
    Plot confusion matrix with OA, macroF1, and Kappa
    Similar to reference paper confusion matrix plots
    """
    # Calculate metrics
    oa = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot heatmap
    im = ax.imshow(cm, cmap='YlGnBu', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(np.arange(1, n_classes+1))
    ax.set_yticklabels(np.arange(1, n_classes+1))
    
    # Add values to cells
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, int(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=9)
    
    # Labels
    ax.set_xlabel('Predicted label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontweight='bold')
    
    # Title with metrics
    title = f'{method_name}_{classifier_name} (OA={oa:.3f}, macroF1={macro_f1:.3f}, κ={kappa:.3f})'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrix: {output_path}")
    
    return {'OA': oa, 'macroF1': macro_f1, 'kappa': kappa}


def plot_selected_wavelengths_overlay(mean_spectrum, wavelengths, 
                                      selected_wavelengths_dict,
                                      dataset_name, output_path):
    """
    Plot selected wavelengths overlaid on mean spectrum
    Similar to reference paper Figure showing wavelength selection
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot mean spectrum
    ax.plot(wavelengths, mean_spectrum, 'k-', linewidth=2, label='Mean Spectrum', zorder=1)
    
    # Define colors for different methods
    colors = {'RFE': 'green', 'MRMR': 'red', 'BOSS': 'orange', 
              'CCARS': 'blue', 'FISHER': 'purple'}
    linetyles = {'RFE': '--', 'MRMR': '--', 'BOSS': ':', 
                'CCARS': '-.', 'FISHER': '-'}
    
    # Plot vertical lines for selected wavelengths
    offset = 0
    for method, selected_wl in selected_wavelengths_dict.items():
        color = colors.get(method, 'gray')
        ls = linetyles.get(method, '-')
        
        for wl in selected_wl[:50]:  # Limit to 50 for visibility
            ax.axvline(x=wl, color=color, linestyle=ls, alpha=0.6, linewidth=0.8)
        
        # Add method to legend (only once)
        ax.axvline(x=-100, color=color, linestyle=ls, linewidth=2, 
                  label=f'{method} (κ={len(selected_wl)})', alpha=0.8)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Reflectance (TRAIN) (a.u.)', fontsize=13, fontweight='bold')
    ax.set_title(f'Selected Wavelengths Overlaid on Mean Spectrum ({dataset_name})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([wavelengths[0], wavelengths[-1]])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved wavelength overlay plot: {output_path}")


def create_comparison_table_latex(results_dict, output_path):
    """
    Create LaTeX table for publication
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Feature Selection Methods Comparison}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\hline")
    lines.append("Method & Classifier & κ & OA & macroF1 & Kappa & Time (s) \\\\")
    lines.append("\\hline")
    
    for key, val in results_dict.items():
        method, clf, n_feat = key
        lines.append(f"{method} & {clf} & {n_feat} & "
                    f"{val['OA']:.3f} & {val['macroF1']:.3f} & "
                    f"{val['kappa']:.3f} & {val.get('time', 0):.2f} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved LaTeX table: {output_path}")


def generate_publication_plots(dataset_name='salinas'):
    """
    Generate all publication-quality plots
    """
    print("\n" + "="*80)
    print(f"GENERATING PUBLICATION PLOTS - {dataset_name.upper()}")
    print("="*80)
    
    from hsi_data_loader import prepare_hsi_for_cars
    from hsi_preprocessing import preprocess_hsi_data
    
    # Create output directory
    output_dir = Path(f'Publication_Plots/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nStep 1: Loading data...")
    data = prepare_hsi_for_cars(dataset_name=dataset_name, test_percentage=0.2,
                                calibration_percentage=0.5, random_state=42)
    
    X_train_df = data['X_cal_train_df']
    X_test_df = data['X_cal_test_df']
    wavelengths = data['wavelengths']
    
    # Preprocess
    X_train = preprocess_hsi_data(X_train_df, apply_log=True, apply_snv=True)
    X_test = preprocess_hsi_data(X_test_df, apply_log=True, apply_snv=True)
    
    print(f"✓ Loaded: {len(X_train_df)} train, {len(X_test_df)} test samples")
    
    # Preprocess for other analyses (but NOT for mean spectra plot)
    X_train = preprocess_hsi_data(X_train_df, apply_log=True, apply_snv=True)
    X_test = preprocess_hsi_data(X_test_df, apply_log=True, apply_snv=True)
    
    # Plot 1: Mean spectra per class (use RAW data for interpretability)
    print("\nStep 2: Creating mean spectra plot...")
    plot_mean_spectra_per_class(
        X_train_df, wavelengths, dataset_name,  # Use RAW data, not preprocessed
        output_dir / f'{dataset_name}_mean_spectra.png'
    )
    
    # Plot 2: Load existing results and create confusion matrices
    print("\nStep 3: Creating confusion matrices...")
    
    # Load CCARS results
    ccars_path = Path(f'HSI_CARS_comprehensive/{dataset_name}/comprehensive_results.csv')
    if ccars_path.exists():
        df_ccars = pd.read_csv(ccars_path)
        # Get best CCARS result (30 features, RF)
        best_ccars = df_ccars[
            (df_ccars['n_wavelengths_selected'] == 30) &
            (df_ccars['classifier'] == 'Random Forest')
        ]
        
        if len(best_ccars) > 0:
            print("  Found CCARS results - confusion matrix available in results folder")
    
    # Plot 3: Selected wavelengths overlay
    print("\nStep 4: Creating wavelength overlay plot...")
    
    # Load selected wavelengths from each method
    selected_wl_dict = {}
    
    # Load CCARS
    ccars_wl_path = Path(f'HSI_CARS_comprehensive/{dataset_name}/selected_wavelengths.csv')
    if ccars_wl_path.exists():
        df_wl = pd.read_csv(ccars_wl_path)
        selected_wl_dict['CCARS'] = df_wl.head(30)['Wavelength'].values
        print(f"  ✓ Loaded CCARS wavelengths: {len(selected_wl_dict['CCARS'])}")
    
    # Load other methods (if available)
    for method in ['MRMR', 'BOSS', 'FISHER']:
        method_dir = Path(f'Feature_Selection_Comparison/{dataset_name}/wavelength_30')
        if method_dir.exists():
            # Methods were run, wavelengths should be stored somewhere
            print(f"  Note: {method} wavelengths need to be extracted from results")
    
    # Plot with available wavelengths
    if selected_wl_dict:
        mean_spectrum = X_train.mean(axis=0).values
        plot_selected_wavelengths_overlay(
            mean_spectrum, wavelengths, selected_wl_dict,
            dataset_name, output_dir / f'{dataset_name}_wavelength_overlay.png'
        )
    
    # Create summary CSV
    print("\nStep 5: Creating summary CSV...")
    
    summary_data = []
    
    # Load all results
    if ccars_path.exists():
        df_all = pd.read_csv(ccars_path)
        for _, row in df_all.iterrows():
            summary_data.append({
                'Method': 'CCARS',
                'Classifier': row['classifier'],
                'Features': row['n_wavelengths_selected'],
                'OA': row['accuracy_selected'],  # Fixed: use accuracy_selected
                'F1': row['f1_selected'],        # Fixed: use f1_selected
                'Reduction_%': row['reduction_percent']
            })
    
    # Load MRMR/BOSS/FISHER
    other_path = Path(f'Feature_Selection_Comparison/{dataset_name}/all_methods_comparison.csv')
    if other_path.exists():
        df_other = pd.read_csv(other_path)
        for _, row in df_other.iterrows():
            summary_data.append({
                'Method': row['method'],
                'Classifier': row['classifier'],
                'Features': row['n_features_selected'],
                'OA': row['accuracy'],
                'F1': row['f1_weighted'],
                'Reduction_%': row['reduction_percent']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['Features', 'OA'], ascending=[True, False])
        summary_df.to_csv(output_dir / f'{dataset_name}_summary_table.csv', index=False)
        print(f"✓ Saved summary table: {output_dir / f'{dataset_name}_summary_table.csv'}")
        
        # Print top results
        print("\n" + "="*80)
        print("TOP RESULTS:")
        print("="*80)
        print(summary_df.head(15).to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ PUBLICATION PLOTS COMPLETE!")
    print("="*80)
    print(f"\nFiles saved to: {output_dir}/")
    print(f"  - {dataset_name}_mean_spectra.png")
    print(f"  - {dataset_name}_wavelength_overlay.png")
    print(f"  - {dataset_name}_summary_table.csv")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'])
    
    args = parser.parse_args()
    
    output_dir = generate_publication_plots(args.dataset)
