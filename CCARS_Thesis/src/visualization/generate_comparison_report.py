"""
Comprehensive Feature Selection Comparison with Visualization

Creates publication-quality comparison tables and plots similar to Nicola's work:
- Unified CSV comparison table
- Accuracy vs feature count plots
- Method comparison bar charts
- Selection frequency plots
- Performance heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


def load_all_results(dataset_name='salinas'):
    """
    Load all results from different experiments
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
    
    Returns:
        Combined DataFrame with all results
    """
    results = []
    
    # Load CCARS comprehensive results
    ccars_path = Path(f'HSI_CARS_comprehensive/{dataset_name}/comprehensive_results.csv')
    if ccars_path.exists():
        df_ccars = pd.read_csv(ccars_path)
        df_ccars['method'] = 'CCARS'
        df_ccars['n_features_selected'] = df_ccars['n_wavelengths_selected'].astype(int)
        results.append(df_ccars)
        print(f"✓ Loaded CCARS results: {len(df_ccars)} rows")
    
    # Load other methods comparison
    other_path = Path(f'Feature_Selection_Comparison/{dataset_name}/all_methods_comparison.csv')
    if other_path.exists():
        df_other = pd.read_csv(other_path)
        results.append(df_other)
        print(f"✓ Loaded MRMR/BOSS/FISHER results: {len(df_other)} rows")
    
    if not results:
        raise FileNotFoundError(f"No results found for {dataset_name}")
    
    # Combine all results
    df_combined = pd.concat(results, ignore_index=True)
    
    # Ensure proper data types
    df_combined['n_features_selected'] = df_combined['n_features_selected'].astype(int)
    df_combined['accuracy'] = pd.to_numeric(df_combined['accuracy'], errors='coerce')
    df_combined['f1_weighted'] = pd.to_numeric(df_combined['f1_weighted'], errors='coerce')
    df_combined['train_time'] = pd.to_numeric(df_combined['train_time'], errors='coerce')
    df_combined['selection_time'] = pd.to_numeric(df_combined['selection_time'], errors='coerce')
    
    return df_combined


def create_unified_comparison_table(df, output_path):
    """
    Create unified comparison table (CSV)
    
    Args:
        df: Combined results DataFrame
        output_path: Output file path
    """
    # Create pivot table
    summary = df.groupby(['method', 'n_features_selected', 'classifier']).agg({
        'accuracy': 'mean',
        'f1_weighted': 'mean',
        'train_time': 'mean',
        'selection_time': 'first',
        'reduction_percent': 'first'
    }).reset_index()
    
    # Sort by accuracy descending
    summary = summary.sort_values(['n_features_selected', 'accuracy'], ascending=[True, False])
    
    # Round numerical columns
    summary['accuracy'] = summary['accuracy'].round(4)
    summary['f1_weighted'] = summary['f1_weighted'].round(4)
    summary['train_time'] = summary['train_time'].round(2)
    summary['selection_time'] = summary['selection_time'].round(2)
    summary['reduction_percent'] = summary['reduction_percent'].round(2)
    
    # Save to CSV
    summary.to_csv(output_path, index=False)
    print(f"✓ Saved unified comparison table: {output_path}")
    
    return summary


def plot_accuracy_vs_features(df, dataset_name, output_dir):
    """
    Plot accuracy vs number of features for each method
    Similar to Nicola's convergence plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df['method'].unique()
    classifiers = ['SVM-RBF', 'Random Forest']  # Focus on main classifiers
    
    for ax, classifier in zip([ax1, ax2], classifiers):
        for method in methods:
            subset = df[(df['method'] == method) & (df['classifier'] == classifier)]
            if len(subset) > 0:
                # Group by feature count and average
                grouped = subset.groupby('n_features_selected')['accuracy'].mean()
                
                # Convert to lists for plotting
                x_vals = list(grouped.index.astype(int))
                y_vals= list(grouped.values.astype(float) * 100)
                
                if len(x_vals) > 0:
                    ax.plot(x_vals, y_vals, 
                           marker='o', linewidth=2, markersize=8, label=method)
        
        ax.set_xlabel('Number of Selected Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{classifier}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.set_ylim([0, 100])
    
    plt.suptitle(f'Accuracy vs Number of Features - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_accuracy_vs_features.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved plot: {save_path}")


def plot_method_comparison_bars(df, dataset_name, output_dir):
    """
    Bar chart comparing all methods at specific feature counts
    """
    # Filter to main classifiers only
    df_filtered = df[df['classifier'].isin(['SVM-RBF', 'Random Forest'])]
    feature_counts = sorted(df_filtered['n_features_selected'].unique())
    
    fig, axes = plt.subplots(1, len(feature_counts), figsize=(5 * len(feature_counts), 6))
    if len(feature_counts) == 1:
        axes = [axes]
    
    for ax, n_feat in zip(axes, feature_counts):
        subset = df_filtered[df_filtered['n_features_selected'] == n_feat]
        
        # Pivot for grouped bar chart
        pivot = subset.pivot_table(
            values='accuracy', 
            index='method', 
            columns='classifier', 
            aggfunc='mean'
        )
        
        pivot_pct = pivot * 100
        pivot_pct.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Feature Selection Method', fontsize=11, fontweight='bold')
        ax.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{n_feat} Features', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(title='Classifier', frameon=True, shadow=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
    
    plt.suptitle(f'Method Comparison - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_method_comparison_bars.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved plot: {save_path}")


def plot_performance_heatmap(df, dataset_name, output_dir):
    """
    Heatmap showing performance across methods × feature counts
    """
    # Filter to main classifiers
    classifiers = ['SVM-RBF', 'Random Forest']
    
    fig, axes = plt.subplots(1, len(classifiers), figsize=(8 * len(classifiers), 6))
    if len(classifiers) == 1:
        axes = [axes]
    
    for ax, classifier in zip(axes, classifiers):
        subset = df[df['classifier'] == classifier]
        
        # Pivot table
        pivot = subset.pivot_table(
            values='accuracy',
            index='method',
            columns='n_features_selected',
            aggfunc='mean'
        )
        
        # Convert to percentage
        pivot_pct = pivot * 100
        
        # Create heatmap
        sns.heatmap(pivot_pct, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature Selection Method', fontsize=11, fontweight='bold')
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Performance Heatmap - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_performance_heatmap.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved plot: {save_path}")


def plot_efficiency_comparison(df, dataset_name, output_dir):
    """
    Scatter plot: Accuracy vs Selection Time (efficiency)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    classifiers = ['SVM-RBF', 'Random Forest']
    
    for ax, classifier in zip(axes, classifiers):
        subset = df[df['classifier'] == classifier]
        
        methods = subset['method'].unique()
        for method in methods:
            method_data = subset[subset['method'] == method]
            
            # Average by feature count
            grouped = method_data.groupby('n_features_selected').agg({
                'accuracy': 'mean',
                'selection_time': 'first'
            })
            
            ax.scatter(grouped['selection_time'], grouped['accuracy'] * 100, 
                      s=200, alpha=0.7, label=method, edgecolors='black', linewidth=1.5)
            
            # Add labels for each point
            for idx, row in grouped.iterrows():
                ax.annotate(f'{idx}', 
                          (row['selection_time'], row['accuracy'] * 100),
                          fontsize=8, ha='center', va='center')
        
        ax.set_xlabel('Selection Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{classifier}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(frameon=True, shadow=True)
        ax.set_xscale('log')
    
    plt.suptitle(f'Efficiency: Accuracy vs Selection Time - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = output_dir / f'{dataset_name}_efficiency_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved plot: {save_path}")


def create_latex_table(df, dataset_name, output_path):
    """
    Create LaTeX-formatted table for publications
    """
    # Best results per configuration
    summary = df.groupby(['method', 'n_features_selected', 'classifier']).agg({
        'accuracy': 'mean',
        'f1_weighted': 'mean',
    }).reset_index()
    
    # Format for LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Feature Selection Methods Comparison on " + dataset_name.upper() + "}")
    latex_lines.append("\\begin{tabular}{llrrr}")
    latex_lines.append("\\hline")
    latex_lines.append("Method & Classifier & Features & Accuracy (\\%) & F1-Score (\\%) \\\\")
    latex_lines.append("\\hline")
    
    for _, row in summary.sort_values(['n_features_selected', 'accuracy'], ascending=[True, False]).iterrows():
        latex_lines.append(
            f"{row['method']} & {row['classifier']} & {int(row['n_features_selected'])} & "
            f"{row['accuracy']*100:.2f} & {row['f1_weighted']*100:.2f} \\\\"
        )
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ Saved LaTeX table: {output_path}")


def generate_comprehensive_report(dataset_name='salinas'):
    """
    Generate comprehensive comparison report with all visualizations
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
    """
    print("\n" + "="*80)
    print(f"GENERATING COMPREHENSIVE COMPARISON REPORT - {dataset_name.upper()}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(f'Comprehensive_Comparison/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\nStep 1: Loading Results...")
    df = load_all_results(dataset_name)
    print(f"Total configurations: {len(df)}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Classifiers: {df['classifier'].unique()}")
    print(f"Feature counts: {sorted(df['n_features_selected'].unique())}")
    
    # Create unified comparison table
    print("\nStep 2: Creating Unified Comparison Table...")
    table_path = output_dir / f'{dataset_name}_unified_comparison.csv'
    summary = create_unified_comparison_table(df, table_path)
    
    # Generate plots
    print("\nStep 3: Generating Plots...")
    
    print("  Creating accuracy vs features plot...")
    plot_accuracy_vs_features(df, dataset_name, output_dir)
    
    print("  Creating method comparison bars...")
    plot_method_comparison_bars(df, dataset_name, output_dir)
    
    print("  Creating performance heatmap...")
    plot_performance_heatmap(df, dataset_name, output_dir)
    
    print("  Creating efficiency comparison...")
    plot_efficiency_comparison(df, dataset_name, output_dir)
    
    # Create LaTeX table
    print("\nStep 4: Creating LaTeX Table...")
    latex_path = output_dir / f'{dataset_name}_table.tex'
    create_latex_table(df, dataset_name, latex_path)
    
    # Create summary report
    print("\nStep 5: Creating Summary Report...")
    report_path = output_dir / f'{dataset_name}_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"COMPREHENSIVE FEATURE SELECTION COMPARISON - {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("BEST RESULTS:\n")
        f.write("-" * 80 + "\n")
        
        # Best overall
        best_overall = summary.loc[summary['accuracy'].idxmax()]
        f.write(f"\nBest Overall Accuracy:\n")
        f.write(f"  Method: {best_overall['method']}\n")
        f.write(f"  Classifier: {best_overall['classifier']}\n")
        f.write(f"  Features: {int(best_overall['n_features_selected'])}\n")
        f.write(f"  Accuracy: {best_overall['accuracy']*100:.2f}%\n")
        f.write(f"  F1-Score: {best_overall['f1_weighted']*100:.2f}%\n")
        
        # Fastest
        fastest = summary.loc[summary['selection_time'].idxmin()]
        f.write(f"\nFastest Selection:\n")
        f.write(f"  Method: {fastest['method']}\n")
        f.write(f"  Time: {fastest['selection_time']:.2f}s\n")
        
        # Best per method
        f.write(f"\n\nBEST RESULT PER METHOD:\n")
        f.write("-" * 80 + "\n")
        for method in summary['method'].unique():
            method_best = summary[summary['method'] == method].nlargest(1, 'accuracy').iloc[0]
            f.write(f"\n{method}:\n")
            f.write(f"  Best: {method_best['accuracy']*100:.2f}% ")
            f.write(f"({method_best['classifier']}, {int(method_best['n_features_selected'])} features)\n")
    
    print(f"✓ Saved summary report: {report_path}")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE REPORT COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {output_dir}/")
    print("  - {}_unified_comparison.csv".format(dataset_name))
    print("  - {}_accuracy_vs_features.png".format(dataset_name))
    print("  - {}_method_comparison_bars.png".format(dataset_name))
    print("  - {}_performance_heatmap.png".format(dataset_name))
    print("  - {}_efficiency_comparison.png".format(dataset_name))
    print("  - {}_table.tex".format(dataset_name))
    print("  - {}_summary_report.txt".format(dataset_name))
    
    return summary, output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Comprehensive Comparison Report')
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'])
    
    args = parser.parse_args()
    
    summary, output_dir = generate_comprehensive_report(args.dataset)
    
    print(f"\n📊 Summary Statistics:")
    print(summary.to_string(index=False))
