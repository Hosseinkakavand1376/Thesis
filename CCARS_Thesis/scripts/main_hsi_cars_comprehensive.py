"""
Comprehensive CCARS Evaluation with Multiple Classifiers

Extends main pipeline to evaluate multiple classifiers and wavelength counts:
- Tests: 10, 20, 30, 50 wavelengths
- Classifiers: PLS-DA, SVM-Linear, SVM-RBF, Random Forest, k-NN
- Baselines: Full spectrum for each classifier
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json

from hsi_data_loader import prepare_hsi_for_cars
from hsi_preprocessing import preprocess_hsi_data
from multiclass_cars import MultiClassCARS
from multiclass_classifiers import MultiClassifierFramework
from hsi_evaluation import (
    compute_multiclass_metrics,
    plot_confusion_matrix
)


def run_comprehensive_evaluation(
    dataset_name='salinas',
    wavelength_counts=[10, 20, 30, 50],
    classifiers=['PLS-DA', 'SVM-RBF', 'Random Forest'],
    cars_runs=500,
    cars_iterations=10,
    pls_components=3,
    output_dir=None,
    random_state=42
):
    """
    Comprehensive CCARS evaluation with multiple configurations
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        wavelength_counts: List of wavelength counts to test
        classifiers: List of classifier names to evaluate
        cars_runs: Number of CCARS Monte Carlo runs
        cars_iterations: Iterations per run
        pls_components: PLS components
        output_dir: Output directory
        random_state: Random seed
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*80)
    print(f"COMPREHENSIVE CCARS EVALUATION - {dataset_name.upper()}")
    print("="*80)
    print(f"Wavelength counts to test: {wavelength_counts}")
    print(f"Classifiers: {classifiers}")
    print(f"CCARS: {cars_runs} runs × {cars_iterations} iterations")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(f"HSI_CARS_comprehensive/{dataset_name}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    print("\n" + "="*80)
    print("Step 1: Data Preparation")
    print("="*80)
    
    data = prepare_hsi_for_cars(
        dataset_name=dataset_name,
        test_percentage=0.2,
        calibration_percentage=0.5,
        random_state=random_state,
        auto_download=True
    )
    
    X_train_df = data['X_cal_train_df']
    X_test_df = data['X_cal_test_df']
    wavelengths = data['wavelengths']
    
    # Preprocess
    X_train_preprocessed = preprocess_hsi_data(X_train_df, apply_log=True, apply_snv=True)
    X_test_preprocessed = preprocess_hsi_data(X_test_df, apply_log=True, apply_snv=True)
    
    X_train_full = X_train_preprocessed.values
    y_train = X_train_preprocessed.index.get_level_values('Class').values
    X_test_full = X_test_preprocessed.values
    y_test = X_test_preprocessed.index.get_level_values('Class').values
    
    print(f"\n✓ Data prepared: {X_train_full.shape[0]} train, {X_test_full.shape[0]} test")
    
    # =========================================================================
    # Step 2: Run CCARS (if not already done)
    # =========================================================================
    print("\n" + "="*80)
    print("Step 2: CCARS Wavelength Selection")
    print("="*80)
    
    cars_results_path = output_dir / 'cars_results'
    
    # Check if CCARS already run
    if (cars_results_path / 'coefficients_all.csv').exists():
        print("Loading existing CCARS results...")
        cars = MultiClassCARS(
            output_path=cars_results_path,
            n_components=pls_components,
            random_state=random_state
        )
        cars.coefficients_df = pd.read_csv(cars_results_path / 'coefficients_all.csv')
        cars.statistics_df = pd.read_csv(cars_results_path / 'statistics_all.csv')
        print(f"✓ Loaded CCARS results from {cars_results_path}")
    else:
        print(f"Running CCARS ({cars_runs} runs)...")
        cars = MultiClassCARS(
            output_path=cars_results_path,
            n_components=pls_components,
            test_percentage=0.2,
            calibration=True,
            random_state=random_state
        )
        
        cars.fit(X_train_full, y_train, X_test_full, y_test, wavelengths)
        cars.run_cars(
            n_runs=cars_runs,
            n_iterations=cars_iterations,
            mc_samples=0.8,
            use_ars=True
        )
        print("✓ CCARS complete")
    
    # =========================================================================
   # Step 3: Evaluate Multiple Wavelength Counts
    # =========================================================================
    print("\n" + "="*80)
    print("Step 3: Multi-Wavelength-Count Evaluation")
    print("="*80)
    
    all_results = []
    framework = MultiClassifierFramework(n_components=pls_components, random_state=random_state)
    
    for n_wl in wavelength_counts:
        print(f"\n{'='*80}")
        print(f"Testing with {n_wl} wavelengths")
        print(f"{'='*80}")
        
        # Select wavelengths
        selected_wl, wl_freq_df = cars.get_selected_wavelengths(top_n=n_wl)
        
        if len(selected_wl) == 0:
            print(f"Warning: No wavelengths selected for n={n_wl}, skipping...")
            continue
        
        selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
        
        print(f"Selected {len(selected_wl)} wavelengths")
        
        # Extract selected wavelength data
        X_train_sel = X_train_full[:, selected_wl_indices]
        X_test_sel = X_test_full[:, selected_wl_indices]
        
        # Evaluate all classifiers
        comparison = framework.compare_selected_vs_full(
            X_train_sel, X_train_full,
            y_train, X_test_sel, X_test_full, y_test,
            n_selected=len(selected_wl)
        )
        
        # Save results for this wavelength count
        wl_dir = output_dir / f'wavelength_{len(selected_wl)}'
        wl_dir.mkdir(exist_ok=True)
        
        # Save wavelength list
        wl_df = pd.DataFrame({
            'wavelength_nm': selected_wl,
            'index': selected_wl_indices
        })
        wl_df.to_csv(wl_dir / 'selected_wavelengths.csv', index=False)
        
        # Save results for each classifier
        for clf_name in comparison['selected'].keys():
            if comparison['selected'][clf_name] is None:
                continue
                
            clf_dir = wl_dir / clf_name.lower().replace('-', '_').replace(' ', '_')
            clf_dir.mkdir(exist_ok=True)
            
            # Save metrics
            metrics = {
                'selected': comparison['selected'][clf_name],
                'full': comparison['full'][clf_name]
            }
            
            with open(clf_dir / 'metrics.json', 'w') as f:
                # Convert numpy types to Python types
                metrics_serializable = {}
                for key, val in metrics.items():
                    if val is not None:
                        metrics_serializable[key] = {
                            k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                            for k, v in val.items() if k != 'predictions'
                        }
                json.dump(metrics_serializable, f, indent=2)
        
        # Collect summary
        all_results.extend(comparison['summary'])
        
        # Print comparison
        framework.print_comparison_table(comparison['summary'])
    
    # =========================================================================
    # Step 4: Create Comprehensive Summary
    # =========================================================================
    print("\n" + "="*80)
    print("Step 4: Creating Comprehensive Summary")
    print("="*80)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'comprehensive_results.csv', index=False)
    
    # Create summary report
    with open(output_dir / 'comprehensive_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"COMPREHENSIVE CCARS EVALUATION REPORT - {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"CCARS Runs: {cars_runs}\n")
        f.write(f"Wavelength Counts Tested: {wavelength_counts}\n")
        f.write(f"Classifiers Evaluated: {classifiers}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Group by wavelength count
        for n_wl in wavelength_counts:
            subset = results_df[results_df['n_wavelengths_selected'] == n_wl]
            if len(subset) == 0:
                continue
                
            f.write(f"\n{n_wl} Wavelengths:\n")
            f.write("-" * 80 + "\n")
            
            for _, row in subset.iterrows():
                acc_sel = row['accuracy_selected'] * 100
                acc_full = row['accuracy_full'] * 100
                retention = row['accuracy_retention'] * 100
                
                f.write(f"  {row['classifier']:<20}: ")
                f.write(f"Selected={acc_sel:5.2f}%  Full={acc_full:5.2f}%  ")
                f.write(f"Retention={retention:5.1f}%\n")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - comprehensive_results.csv")
    print(f"  - comprehensive_report.txt")
    print(f"  - wavelength_*/")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*80)
    
    return {
        'results_df': results_df,
        'output_dir': output_dir
    }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive CCARS Evaluation with Multiple Classifiers'
    )
    
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'],
                       help='Dataset name')
    parser.add_argument('--wavelengths', type=int, nargs='+', default=[10, 20, 30, 50],
                       help='Wavelength counts to test')
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['PLS-DA', 'SVM-RBF', 'Random Forest'],
                       help='Classifiers to evaluate')
    parser.add_argument('--cars_runs', type=int, default=500,
                       help='CCARS Monte Carlo runs')
    parser.add_argument('--cars_iterations', type=int, default=10,
                       help='CCARS iterations per run')
    parser.add_argument('--components', type=int, default=3,
                       help='PLS components')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    results = run_comprehensive_evaluation(
        dataset_name=args.dataset,
        wavelength_counts=args.wavelengths,
        classifiers=args.classifiers,
        cars_runs=args.cars_runs,
        cars_iterations=args.cars_iterations,
        pls_components=args.components,
        output_dir=args.output,
        random_state=args.random_state
    )
    
    return results


if __name__ == '__main__':
    results = main()
