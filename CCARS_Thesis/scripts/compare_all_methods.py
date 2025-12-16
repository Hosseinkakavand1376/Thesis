"""
Comprehensive Feature Selection Comparison

Compares CCARS, MRMR, BOSS, and FISHER methods with multiple classifiers
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from hsi_data_loader import prepare_hsi_for_cars
from hsi_preprocessing import preprocess_hsi_data
from multiclass_cars import MultiClassCARS
from feature_selection_methods import UnifiedFeatureSelector
from multiclass_classifiers import MultiClassifierFramework


def run_all_methods_comparison(
    dataset_name='salinas',
    feature_counts=[10, 20, 30, 50],
    methods=['CCARS', 'MRMR', 'BOSS', 'FISHER'],
    classifiers=['SVM-RBF', 'Random Forest'],
    cars_runs=500,
    cars_iterations=10,
    pls_components=3,
    output_dir=None,
    random_state=42
):
    """
    Compare all feature selection methods
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        feature_counts: List of feature counts to test
        methods: List of methods to compare
        classifiers: List of classifiers to evaluate
        cars_runs: CCARS Monte Carlo runs
        cars_iterations: CCARS iterations
        pls_components: PLS components
        output_dir: Output directory
        random_state: Random seed
    
    Returns:
        DataFrame with comprehensive results
    """
    print("\n" + "="*80)
    print(f"COMPREHENSIVE FEATURE SELECTION COMPARISON - {dataset_name.upper()}")
    print("="*80)
    print(f"Methods: {methods}")
    print(f"Feature counts: {feature_counts}")
    print(f"Classifiers: {classifiers}")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(f"Feature_Selection_Comparison/{dataset_name}")
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
        random_state=random_state
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
    print(f"  Features: {X_train_full.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # =========================================================================
    # Step 2: Run All Feature Selection Methods
    # =========================================================================
    all_results = []
    framework = MultiClassifierFramework(n_components=pls_components, random_state=random_state)
    
    for method in methods:
        print("\n" + "="*80)
        print(f"METHOD: {method}")
        print("="*80)
        
        for n_features in feature_counts:
            print(f"\n{'-'*80}")
            print(f"{method} - Selecting {n_features} features")
            print(f"{'-'*80}")
            
            start_time = time.time()
            
            # Feature selection
            if method == 'CCARS':
                # Run CCARS
                cars = MultiClassCARS(
                    output_path=output_dir / f'{method}_results',
                    n_components=pls_components,
                    random_state=random_state
                )
                
                # Check if already run
                cars_cache = output_dir / f'ccars_{n_features}features.npy'
                if cars_cache.exists():
                    print(f"  Loading cached CCARS results...")
                    selected_indices = np.load(cars_cache)
                else:
                    print(f"  Running CCARS ({cars_runs} runs)...")
                    cars.fit(X_train_full, y_train, X_test_full, y_test, wavelengths)
                    cars.run_cars(
                        n_runs=cars_runs,
                        n_iterations=cars_iterations,
                        mc_samples=0.8,
                        use_ars=True
                    )
                    
                    selected_wl, _ = cars.get_selected_wavelengths(top_n=n_features)
                    selected_indices = np.array([
                        np.where(np.array(wavelengths) == wl)[0][0] 
                        for wl in selected_wl
                    ])
                    np.save(cars_cache, selected_indices)
                
            else:
                # Run other methods (MRMR, BOSS, FISHER)
                selector = UnifiedFeatureSelector(
                    method=method,
                    n_features=n_features,
                    n_bootstrap=100,  # For BOSS
                    random_state=random_state
                )
                
                print(f"  Running {method}...")
                selector.fit(X_train_full, y_train)
                selected_indices = selector.selected_features_
            
            selection_time = time.time() - start_time
            
            print(f"  ✓ Selected {len(selected_indices)} features in {selection_time:.1f}s")
            
            # Extract selected features
            X_train_sel = X_train_full[:, selected_indices]
            X_test_sel = X_test_full[:, selected_indices]
            
            # Evaluate with each classifier
            for clf_name in classifiers:
                print(f"    Evaluating {clf_name}...", end=' ')
                
                # Train and evaluate
                result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    clf_name, f'{method}_{n_features}'
                )
                
                # Add metadata
                result['method'] = method
                result['n_features_selected'] = n_features
                result['selection_time'] = selection_time
                result['n_features_original'] = X_train_full.shape[1]
                result['reduction_percent'] = (1 - n_features / X_train_full.shape[1]) * 100
                
                all_results.append(result)
                
                print(f"Accuracy: {result['accuracy']:.4f}")
    
    # =========================================================================
    # Step 3: Create Comprehensive Summary
    # =========================================================================
    print("\n" + "="*80)
    print("Step 3: Creating Summary")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'all_methods_comparison.csv', index=False)
    
    # Create summary report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"FEATURE SELECTION METHODS COMPARISON - {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Methods Tested: {', '.join(methods)}\n")
        f.write(f"Feature Counts: {feature_counts}\n")
        f.write(f"Classifiers: {', '.join(classifiers)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY METHOD AND FEATURE COUNT\n")
        f.write("="*80 + "\n\n")
        
        for method in methods:
            f.write(f"\n{method}:\n")
            f.write("-" * 80 + "\n")
            
            method_results = results_df[results_df['method'] == method]
            
            for n_feat in feature_counts:
                subset = method_results[method_results['n_features_selected'] == n_feat]
                if len(subset) == 0:
                    continue
                
                f.write(f"\n  {n_feat} Features:\n")
                for _, row in subset.iterrows():
                    acc = row['accuracy'] * 100
                    f1 = row['f1_weighted'] * 100
                    f.write(f"    {row['classifier']:<20}: ")
                    f.write(f"Acc={acc:5.2f}%  F1={f1:5.2f}%\n")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - all_methods_comparison.csv")
    print(f"  - comparison_report.txt")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*80)
    
    return results_df


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Feature Selection Comparison'
    )
    
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'])
    parser.add_argument('--features', type=int, nargs='+', default=[10, 20, 30, 50])
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['CCARS', 'MRMR', 'BOSS', 'FISHER'])
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['SVM-RBF', 'Random Forest'])
    parser.add_argument('--cars_runs', type=int, default=500)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    results_df = run_all_methods_comparison(
        dataset_name=args.dataset,
        feature_counts=args.features,
        methods=args.methods,
        classifiers=args.classifiers,
        cars_runs=args.cars_runs,
        output_dir=args.output
    )
    
    return results_df


if __name__ == '__main__':
    results = main()
