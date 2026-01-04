"""
Comprehensive CCARS Evaluation with Multiple Classifiers

Extends main pipeline to evaluate multiple classifiers and wavelength counts:
- Tests: 10, 20, 30, 50 wavelengths
- Classifiers: PLS-DA, SVM-Linear, SVM-RBF, Random Forest, k-NN
- Baselines: Full spectrum for each classifier
"""

# Set matplotlib backend to non-GUI to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

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
from permutation_test import PermutationTest
from learning_curve import LearningCurve
import matplotlib.pyplot as plt


def find_best_wavelength_count(results_dict):
    """
    Find wavelength count with highest accuracy
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
    
    Returns:
        best_count: Wavelength count with highest accuracy
        best_accuracy: Best accuracy achieved
    """
    if not results_dict:
        return None, None
    
    best_count = max(results_dict, key=results_dict.get)
    best_accuracy = results_dict[best_count]
    
    return best_count, best_accuracy


def find_minimum_acceptable(results_dict, threshold=0.85):
    """
    Find minimum wavelength count achieving threshold of best accuracy
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
        threshold: Fraction of best accuracy (default 0.85 = 85%)
    
    Returns:
        min_count: Minimum wavelength count meeting threshold
        min_accuracy: Accuracy at that count
    """
    if not results_dict:
        return None, None
    
    # Find best accuracy
    best_count, best_accuracy = find_best_wavelength_count(results_dict)
    target_accuracy = best_accuracy * threshold
    
    # Find minimum count achieving target
    acceptable = {k: v for k, v in results_dict.items() if v >= target_accuracy}
    
    if not acceptable:
        # If no count meets threshold, return the one closest to target
        closest = min(results_dict, key=lambda k: abs(results_dict[k] - target_accuracy))
        return closest, results_dict[closest]
    
    min_count = min(acceptable.keys())
    min_accuracy = acceptable[min_count]
    
    return min_count, min_accuracy


def detect_knee_point(results_dict):
    """
    Detect knee point (elbow) in accuracy vs wavelength count curve
    Uses derivative-based approach to find point of diminishing returns
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
    
    Returns:
        knee_count: Wavelength count at knee point
        knee_accuracy: Accuracy at knee point
    """
    if not results_dict or len(results_dict) < 3:
        return None, None
    
    # Sort by wavelength count
    sorted_counts = sorted(results_dict.keys())
    accuracies = [results_dict[c] for c in sorted_counts]
    
    # Compute first derivative (rate of change)
    derivatives = []
    for i in range(len(accuracies) - 1):
        dx = sorted_counts[i + 1] - sorted_counts[i]
        dy = accuracies[i + 1] - accuracies[i]
        derivatives.append(dy / dx if dx != 0 else 0)
    
    # Compute second derivative (change in rate of change)
    second_derivatives = []
    for i in range(len(derivatives) - 1):
        second_derivatives.append(derivatives[i + 1] - derivatives[i])
    
    if not second_derivatives:
        # Fallback: use point where derivative drops significantly
        threshold = max(derivatives) * 0.5
        for i, deriv in enumerate(derivatives):
            if deriv < threshold:
                knee_count = sorted_counts[i]
                return knee_count, results_dict[knee_count]
        return sorted_counts[len(sorted_counts)//2], results_dict[sorted_counts[len(sorted_counts)//2]]
    
    # Find point with largest negative second derivative (sharpest deceleration)
    knee_idx = second_derivatives.index(min(second_derivatives))
    knee_count = sorted_counts[knee_idx + 1]  # +1 because second deriv is offset
    knee_accuracy = results_dict[knee_count]
    
    return knee_count, knee_accuracy


def plot_optimization_curve(results_dict, best_count, min_count, knee_count, 
                           classifier_name, dataset_name, output_dir):
    """
    Plot accuracy vs wavelength count with optimal points marked
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
        best_count, min_count, knee_count: Optimal wavelength counts
        classifier_name: Name of classifier
        dataset_name: Dataset name
        output_dir: Output directory for plot
    """
    if not results_dict:
        return
    
    # Sort data
    sorted_counts = sorted(results_dict.keys())
    accuracies = [results_dict[c] * 100 for c in sorted_counts]  # Convert to percentage
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot curve
    plt.plot(sorted_counts, accuracies, 'o-', linewidth=2, markersize=6, 
             label='Accuracy', color='steelblue')
    
    # Mark optimal points
    if best_count is not None:
        plt.axvline(best_count, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best ({best_count} WL, {results_dict[best_count]*100:.2f}%)')
        plt.scatter([best_count], [results_dict[best_count]*100], 
                   color='green', s=200, zorder=5, marker='*')
    
    if min_count is not None:
        plt.axvline(min_count, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Minimum (≥85%, {min_count} WL, {results_dict[min_count]*100:.2f}%)')
        plt.scatter([min_count], [results_dict[min_count]*100], 
                   color='orange', s=150, zorder=5, marker='D')
    
    if knee_count is not None and knee_count != min_count:
        plt.axvline(knee_count, color='red', linestyle='--', alpha=0.7, 
                   label=f'Knee Point ({knee_count} WL, {results_dict[knee_count]*100:.2f}%)')
        plt.scatter([knee_count], [results_dict[knee_count]*100], 
                   color='red', s=150, zorder=5, marker='^')
    
    plt.xlabel('Number of Wavelengths', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Wavelength Count Optimization - {dataset_name.upper()}\n{classifier_name}', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'optimization_curve_{classifier_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved optimization plot: {output_path.name}")


def run_comprehensive_evaluation(
    dataset_name='salinas',
    wavelength_counts=[10, 20, 30, 50],
    classifiers=['PLS-DA', 'SVM-RBF', 'Random Forest'],
    cars_runs=500,
    cars_iterations=10,
    pls_components=3,
    output_dir=None,
    random_state=42,
    optimize_wavelengths=False,
    use_holdout_validation=False,
    use_calibration=True,
    calibration_fraction=0.5,
    n_permutations=1000,
    skip_permutation=False,
    compute_learning_curves=False,
    lc_train_sizes=None,
    preprocessing_method='snv_only',
    optimize_roc=False,
    adaptive_permutations=False
):
    """
    Comprehensive CCARS evaluation with multiple configurations
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        wavelength_counts: List of wavelength counts to test (if optimize_wavelengths=False)
                          or None to use default range [5-50] (if optimize_wavelengths=True)
        classifiers: List of classifiers to evaluate
        cars_runs: Number of CCARS Monte Carlo runs
        cars_iterations: Iterations per CCARS run
        pls_components: Number of PLS components
        output_dir: Output directory path
        random_state: Random seed for reproducibility
        optimize_wavelengths: If True, enables wavelength optimization mode
        use_holdout_validation: If True, performs final independent validation on hold-out set
        classifiers: List of classifier names to evaluate
        cars_runs: Number of CCARS Monte Carlo runs
        cars_iterations: Iterations per run
        pls_components: PLS components
        output_dir: Output directory
        random_state: Random seed
        optimize_wavelengths: If True, tests range [5-50] and auto-selects best/min/knee.
                             If False, uses Nicola's approach (single CARS run, natural convergence)
    
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
        output_dir = Path(f"HSI_CARS_comprehensive/{dataset_name}/components_{pls_components}")
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
    X_train_preprocessed = preprocess_hsi_data(X_train_df, method=preprocessing_method)
    X_test_preprocessed = preprocess_hsi_data(X_test_df, method=preprocessing_method)
    
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
            use_calibration=use_calibration,
            random_state=random_state
        )
        cars.coefficients_df = pd.read_csv(cars_results_path / 'coefficients_all.csv')
        cars.statistics_df = pd.read_csv(cars_results_path / 'statistics_all.csv')
        print(f"✓ Loaded CCARS results from {cars_results_path}")
        
        # Set calibration mode flags if needed
        if use_calibration:
            cars.is_calibration_mode = True
            cars.calibration_note = f"CCARS Mode: Loaded from saved results"
    else:
        print(f"Running CCARS ({cars_runs} runs)...")
        cars = MultiClassCARS(
            output_path=cars_results_path,
            n_components=pls_components,
            test_percentage=0.2,
            use_calibration=use_calibration,
            random_state=random_state
        )
        
        # CCARS MODE: Use calibration/final split
        if use_calibration:
            print(f"\n⚠️  CCARS MODE ENABLED: Calibration/Final Split ({calibration_fraction*100:.0f}%/{(1-calibration_fraction)*100:.0f}%)")
            print(f"   Wavelengths will be selected from CALIBRATION set only")
            print(f"   Final evaluation uses SEPARATE final set\n")
            
           # Prepare calibration split
            split_dict = cars.prepare_calibration_split(
                X_train_full, y_train, wavelengths,
                calibration_fraction=calibration_fraction
            )
            
            # Fit CARS with CALIBRATION data only
            cars.fit(
                split_dict['X_cal_train'],
                split_dict['y_cal_train'],
                split_dict['X_cal_test'],
                split_dict['y_cal_test'],
                wavelengths
            )
            
            # Run CARS (uses ONLY calibration data)
            cars.run_cars(
                n_runs=cars_runs,
                n_iterations=cars_iterations,
                mc_samples=0.8,
                use_ars=True
            )
            
        else:
            # ORIGINAL CARS MODE: Use all data
            print(f"\n   Original CARS MODE: Using all data\n")
            
            cars.fit(X_train_full, y_train, X_test_full, y_test, wavelengths)
            cars.run_cars(
                n_runs=cars_runs,
                n_iterations=cars_iterations,
                mc_samples=0.8,
                use_ars=True
            )
        
        print("✓ CCARS complete")
    
    # =========================================================================
    # Step 3: Wavelength Selection Strategy
    # =========================================================================
    print("\n" + "="*80)
    if optimize_wavelengths:
        print("Step 3: Wavelength Range Optimization (Enhanced Mode)")
    else:
        print("Step 3: Multi-Wavelength-Count Evaluation (Nicola's Approach)")
    print("="*80)
    
    all_results = []
    framework = MultiClassifierFramework(n_components=pls_components, random_state=random_state)
    
    # =========================================================================
    # OPTIMIZATION: Evaluate full spectrum ONCE (reused for all comparisons)
    # =========================================================================
    print("\nEvaluating full spectrum baseline (204 wavelengths)...")
    full_spectrum_results = {}
    for clf_name in classifiers:
        print(f"  Training {clf_name} on full spectrum...", end=" ")
        result = framework.train_and_evaluate(
            X_train_full, y_train, X_test_full, y_test,
            classifier_name=clf_name,
            wavelength_type='full_spectrum'
        )
        full_spectrum_results[clf_name] = result
        print(f"✓ Accuracy: {result['accuracy']:.4f}")
    print("✓ Full spectrum baseline complete\n")
    
    # Determine testing strategy
    if optimize_wavelengths:
        # ENHANCED MODE: Test range [5, 10, 15, ..., 50] and find optimal
        if isinstance(wavelength_counts, list) and len(wavelength_counts) > 0:
            # Use provided range if specified
            test_range = wavelength_counts
            print(f"Testing specific wavelength counts: {test_range}")
        else:
            # Default: test range [5, 10, 15, ..., 50] (10 counts)
            test_range = list(range(5, 55, 5))
            print(f"Testing wavelength range: {test_range[0]} to {test_range[-1]} (step 5)")
        
        # Data structure to collect results for each classifier
        classifier_results = {clf: {} for clf in classifiers}
        
        # Test all wavelength counts
        for n_wl in test_range:
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
            
            # Evaluate selected wavelengths and compare with pre-computed full spectrum
            selected_results = {}
            for clf_name in classifiers:
                result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    classifier_name=clf_name,
                    wavelength_type=f'{len(selected_wl)}_selected'
                )
                selected_results[clf_name] = result
                
                # Collect accuracy for optimization
                classifier_results[clf_name][len(selected_wl)] = result['accuracy']
                print(f"  {clf_name}: {result['accuracy']:.4f}")
            
            # Create comparison summary using pre-computed full spectrum results
            for clf_name in classifiers:
                if selected_results[clf_name] is not None and full_spectrum_results[clf_name] is not None:
                    acc_sel = selected_results[clf_name]['accuracy']
                    acc_full = full_spectrum_results[clf_name]['accuracy']
                    retention = (acc_sel / acc_full) * 100
                    
                    summary_row = {
                        'classifier': clf_name,
                        'n_wavelengths_selected': len(selected_wl),
                        'n_wavelengths_full': len(wavelengths),
                        'accuracy_selected': acc_sel,
                        'accuracy_full': acc_full,
                        'accuracy_diff': acc_sel - acc_full,
                        'accuracy_retention': retention / 100,
                        'reduction_percent': (1 - len(selected_wl) / len(wavelengths)) * 100
                    }
                    all_results.append(summary_row)
    
    else:
        # NICOLA'S APPROACH: Test specified wavelength counts only
        print(f"Testing wavelength counts: {wavelength_counts}")
        test_range = wavelength_counts
        classifier_results = None  # No optimization needed
        
        for n_wl in test_range:
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
            
            # Evaluate selected wavelengths and compare with pre-computed full spectrum
            selected_results = {}
            for clf_name in classifiers:
                result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    classifier_name=clf_name,
                    wavelength_type=f'{len(selected_wl)}_selected'
                )
                selected_results[clf_name] = result
                print(f"  {clf_name}: {result['accuracy']:.4f}")
            
            # Create comparison summary using pre-computed full spectrum results
            for clf_name in classifiers:
                if selected_results[clf_name] is not None and full_spectrum_results[clf_name] is not None:
                    acc_sel = selected_results[clf_name]['accuracy']
                    acc_full = full_spectrum_results[clf_name]['accuracy']
                    retention = (acc_sel / acc_full) * 100
                    
                    summary_row = {
                        'classifier': clf_name,
                        'n_wavelengths_selected': len(selected_wl),
                        'n_wavelengths_full': len(wavelengths),
                        'accuracy_selected': acc_sel,
                        'accuracy_full': acc_full,
                        'accuracy_diff': acc_sel - acc_full,
                        'accuracy_retention': retention / 100,
                        'reduction_percent': (1 - len(selected_wl) / len(wavelengths)) * 100
                    }
                    all_results.append(summary_row)
            
            # Print comparison table for this wavelength count
            framework.print_comparison_table(all_results[-len(classifiers):])
    
    # =========================================================================
    # Step 4: Identify Optimal Wavelength Counts (Enhanced Mode Only)
    # =========================================================================
    optimal_configs = {}
    
    if optimize_wavelengths and classifier_results:
        print("\n" + "="*80)
        print("Step 4: Identifying Optimal Wavelength Counts")
        print("="*80)
        
        # For primary classifier (Random Forest or first in list), find optimal counts
        primary_classifier = 'Random Forest' if 'Random Forest' in classifiers else classifiers[0]
        
        if primary_classifier in classifier_results and len(classifier_results[primary_classifier]) >= 3:
            results_dict = classifier_results[primary_classifier]
            
            # Find optimal points
            best_count, best_acc = find_best_wavelength_count(results_dict)
            min_count, min_acc = find_minimum_acceptable(results_dict, threshold=0.85)
            knee_count, knee_acc = detect_knee_point(results_dict)
            
            optimal_configs = {
                'best': (best_count, best_acc),
                'minimum': (min_count, min_acc),
                'knee': (knee_count, knee_acc)
            }
            
            print(f"\nOptimal configurations for {primary_classifier}:")
            print(f"  Best: {best_count} wavelengths → {best_acc:.4f} accuracy")
            print(f"  Minimum (≥85%): {min_count} wavelengths → {min_acc:.4f} accuracy")
            print(f"  Knee point: {knee_count} wavelengths → {knee_acc:.4f} accuracy")
            
            # Create optimization plot
            print(f"\nCreating optimization curve...")
            plot_optimization_curve(
                results_dict, best_count, min_count, knee_count,
                primary_classifier, dataset_name, output_dir
            )
        else:
            print(f"Warning: Insufficient data for optimization analysis")
    else:
        print("\n" + "="*80)
        print("Step 4: Skipping Optimization (Using Nicola's Approach)")
        print("="*80)
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("\n" + "="*80)
    if optimize_wavelengths:
        print("Step 5: Saving Optimal Configuration Results")
    else:
        print("Step 5: Saving All Tested Configurations")
    print("="*80)
    
    # Determine which wavelength counts to save
    if optimize_wavelengths and 'best' in optimal_configs:
        # Enhanced mode: save only optimal configs
        counts_to_save = [
            optimal_configs['best'][0],
            optimal_configs['minimum'][0],
            optimal_configs['knee'][0]
        ]
        # Remove duplicates
        counts_to_save = sorted(list(set([c for c in counts_to_save if c is not None])))
        print(f"Saving 3 optimal configurations: {counts_to_save}")
    else:
        # Nicola's mode: save all tested counts
        counts_to_save = test_range
        print(f"Saving all {len(counts_to_save)} tested configurations: {counts_to_save}")
    
    # Save detailed results for optimal configurations
    for n_wl in counts_to_save:
        # Select wavelengths
        selected_wl, wl_freq_df = cars.get_selected_wavelengths(top_n=n_wl)
        
        if len(selected_wl) == 0:
            continue
        
        selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
        
        # Determine configuration type and directory name
        if optimize_wavelengths and 'best' in optimal_configs:
            # Enhanced mode: use descriptive labels
            config_type = []
            if n_wl == optimal_configs['best'][0]:
                config_type.append('best')
            if n_wl == optimal_configs['minimum'][0]:
                config_type.append('minimum')
            if n_wl == optimal_configs['knee'][0]:
                config_type.append('knee')
            config_label = '_'.join(config_type) if config_type else 'wavelength'
            wl_dir = output_dir / f'{config_label}_{n_wl}'
        else:
            # Nicola's mode: simple wavelength count naming
            wl_dir = output_dir / f'wavelength_{n_wl}'
        wl_dir.mkdir(exist_ok=True)
        
        # Save wavelength list
        wl_df = pd.DataFrame({
            'wavelength_nm': selected_wl,
            'index': selected_wl_indices
        })
        wl_df.to_csv(wl_dir / 'selected_wavelengths.csv', index=False)
        
        print(f"\n  Saved: {config_label if optimize_wavelengths and 'best' in optimal_configs else 'wavelength'}_{n_wl}/ ({len(selected_wl)} wavelengths)")
        
        # Evaluate ONLY selected wavelengths (reuse pre-computed full spectrum)
        X_train_sel = X_train_full[:, selected_wl_indices]
        X_test_sel = X_test_full[:, selected_wl_indices]
        
        # Save metrics for each classifier
        for clf_name in classifiers:
            # Evaluate selected wavelengths
            selected_result = framework.train_and_evaluate(
                X_train_sel, y_train, X_test_sel, y_test,
                classifier_name=clf_name,
                wavelength_type=f'{len(selected_wl)}_selected'
            )
            
            # Get pre-computed full spectrum result
            full_result = full_spectrum_results.get(clf_name)
            
            if selected_result is None or full_result is None:
                continue
            
            clf_dir = wl_dir / clf_name.lower().replace('-', '_').replace(' ', '_')
            clf_dir.mkdir(exist_ok=True)
            
            # Create metrics dict with both selected and full results
            metrics = {
                'selected': selected_result,
                'full': full_result
            }
            
            with open(clf_dir / 'metrics.json', 'w') as f:
                # Convert numpy types to Python types
                metrics_serializable = {}
                for key, val in metrics.items():
                    if val is not None:
                        metrics_serializable[key] = {
                            k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                            for k, v in val.items() if k not in ['predictions', 'model']  # Exclude non-serializable objects
                        }
                json.dump(metrics_serializable, f, indent=2)
    
    # =========================================================================
    # Step 5.5: Permutation Testing (Statistical Validation)
    # =========================================================================
    if not skip_permutation:
        print("\n" + "="*80)
        print("Step 5.5: Permutation Testing (Statistical Validation)")
        print("="*80)
        print(f"Testing statistical significance of optimal configurations...")
        print(f"Permutations per test: {n_permutations}")
        print()
        
        # Create permutation directory
        perm_dir = output_dir / 'permutation_tests'
        perm_dir.mkdir(exist_ok=True)
        
        # Store all permutation results
        all_perm_results = []
        
        # Test each wavelength configuration
        for n_wl in counts_to_save:
            # Get selected wavelengths
            selected_wl, _ = cars.get_selected_wavelengths(top_n=n_wl)
            
            if len(selected_wl) == 0:
                continue
            
            selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
            
            # Get configuration label
            if optimize_wavelengths and 'best' in optimal_configs:
                config_type = []
                if n_wl == optimal_configs['best'][0]:
                    config_type.append('best')
                if n_wl == optimal_configs['minimum'][0]:
                    config_type.append('minimum')
                if n_wl == optimal_configs['knee'][0]:
                    config_type.append('knee')
                config_label = '_'.join(config_type) if config_type else 'wavelength'
                config_name = f'{config_label}_{n_wl}'
            else:
                config_name = f'wavelength_{n_wl}'
            
            print(f"\n{'='*80}")
            print(f"Testing {config_name} ({len(selected_wl)} wavelengths)")
            print(f"{'='*80}")
            
            # Extract data with selected wavelengths
            X_train_sel = X_train_full[:, selected_wl_indices]
            X_test_sel = X_test_full[:, selected_wl_indices]
            
            # Test each classifier
            for clf_name in classifiers:
                print(f"\nClassifier: {clf_name}")
                
                # Train model
                clf_result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    classifier_name=clf_name,
                    wavelength_type=f'{len(selected_wl)}_selected'
                )
                
                if clf_result is None or 'model' not in clf_result:
                    print(f"  ⚠️  Skipping {clf_name} - no model available")
                    continue
                
                # Get trained model
                model = clf_result['model']
                
                # Determine number of permutations (adaptive for slow classifiers)
                if adaptive_permutations:
                    # Adaptive permutation counts based on classifier speed
                    if clf_name == 'PLS-DA':
                        n_perms = n_permutations  # Fast classifier, keep full (e.g., 1000)
                    elif 'SVM' in clf_name:
                        n_perms = min(100, n_permutations)  # Slow classifier, reduce to 100
                    elif 'Random Forest' in clf_name or 'RF' in clf_name:
                        n_perms = min(200, n_permutations)  # Medium speed, reduce to 200
                    elif 'k-NN' in clf_name or 'kNN' in clf_name:
                        n_perms = min(200, n_permutations)  # Medium speed, reduce to 200
                    else:
                        n_perms = n_permutations  # Unknown classifier, use full
                    
                    if n_perms != n_permutations:
                        print(f"  ⚡ Adaptive permutations: Using {n_perms} permutations for {clf_name} (vs {n_permutations} default)")
                else:
                    n_perms = n_permutations  # Use default without adaptation
                
                # Run permutation test with adaptive count
                perm_test = PermutationTest(n_permutations=n_perms, random_state=random_state)
                p_values = perm_test.run_test(
                    model, X_train_sel, y_train, X_test_sel, y_test,
                    metrics=['accuracy', 'precision', 'recall', 'f1']
                )
                
                # Save results
                perm_csv_path = perm_dir / f'permutation_{config_name}_{clf_name.lower().replace(" ", "_").replace("-", "_")}.csv'
                perm_test.save_results(perm_csv_path)
                
                # Save plot
                plot_path = perm_dir / f'permutation_dist_{config_name}_{clf_name.lower().replace(" ", "_").replace("-", "_")}.png'
                perm_test.plot_permutation_distribution(plot_path, metric='accuracy')
                
                # Store results for summary
                for metric, p_val in p_values.items():
                    all_perm_results.append({
                        'configuration': config_name,
                        'n_wavelengths': len(selected_wl),
                        'classifier': clf_name,
                        'metric': metric,
                        'p_value': p_val,
                        'significant': 'Yes' if p_val < 0.05 else 'No',
                        'baseline_score': perm_test.results['baseline_scores'][metric]
                    })
        
        # Save comprehensive permutation results
        if len(all_perm_results) > 0:
            perm_df = pd.DataFrame(all_perm_results)
            perm_summary_path = perm_dir / 'all_permutation_results.csv'
            perm_df.to_csv(perm_summary_path, index=False)
            
            print(f"\n{'='*80}")
            print(f"✓ Permutation tests complete!")
            print(f"  Results saved to: {perm_dir}")
            print(f"  Total tests run: {len(all_perm_results)}")
            
            # Count significant results
            significant_count = sum(1 for r in all_perm_results if r['significant'] == 'Yes')
            total_count = len(all_perm_results)
            print(f"  Significant results (p < 0.05): {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
            print(f"{'='*80}")
    else:
        print("\n" + "="*80)
        print("Step 5.5: Permutation Testing SKIPPED (--skip_permutation flag)")
        print("="*80)
    
    # =========================================================================
    # Step 5.6: Learning Curves (Overfitting Detection)
    # =========================================================================
    if compute_learning_curves:
        print("\n" + "="*80)
        print("Step 5.6: Learning Curves (Overfitting Detection)")
        print("="*80)
        print(f"Analyzing model performance vs training set size...")
        print()
        
        # Create learning curves directory
        lc_dir = output_dir / 'learning_curves'
        lc_dir.mkdir(exist_ok=True)
        
        # Determine train sizes
        if lc_train_sizes is not None:
            train_sizes = np.array(lc_train_sizes)
            print(f"Using specified train sizes: {train_sizes}")
        else:
            # Default: percentage-based
            train_sizes = np.linspace(0.1, 1.0, 10)
            print(f"Using default train sizes: 10% to 100% in 10% steps")
        
        # Test each wavelength configuration
        for n_wl in counts_to_save:
            # Get selected wavelengths
            selected_wl, _ = cars.get_selected_wavelengths(top_n=n_wl)
            
            if len(selected_wl) == 0:
                continue
            
            selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
            
            # Get configuration label
            if optimize_wavelengths and 'best' in optimal_configs:
                config_type = []
                if n_wl == optimal_configs['best'][0]:
                    config_type.append('best')
                if n_wl == optimal_configs['minimum'][0]:
                    config_type.append('minimum')
                if n_wl == optimal_configs['knee'][0]:
                    config_type.append('knee')
                config_label = '_'.join(config_type) if config_type else 'wavelength'
                config_name = f'{config_label}_{n_wl}'
            else:
                config_name = f'wavelength_{n_wl}'
            
            print(f"\n{'='*80}")
            print(f"Learning Curves: {config_name} ({len(selected_wl)} wavelengths)")
            print(f"{'='*80}")
            
            # Extract data with selected wavelengths
            X_train_sel = X_train_full[:, selected_wl_indices]
            X_test_sel = X_test_full[:, selected_wl_indices]
            
            # Combine train and test for learning curve analysis
            X_combined = np.vstack([X_train_sel, X_test_sel])
            y_combined = np.concatenate([y_train, y_test])
            
            # Test each classifier
            for clf_name in classifiers:
                print(f"\nClassifier: {clf_name}")
                
                # Get classifier instance
                clf_result = framework.train_and_evaluate(
                    X_train_sel, y_train, X_test_sel, y_test,
                    classifier_name=clf_name,
                    wavelength_type=f'{len(selected_wl)}_selected'
                )
                
                if clf_result is None or 'model' not in clf_result:
                    print(f"  ⚠️  Skipping {clf_name} - no model available")
                    continue
                
                # Get fresh model instance (not trained)
                from sklearn.base import clone
                model = clone(clf_result['model'])
                
                # Compute learning curve
                lc = LearningCurve(cv_splits=5, random_state=random_state)
                results = lc.compute_learning_curve(
                    model, X_combined, y_combined,
                    train_sizes=train_sizes,
                    scoring='accuracy'
                )
                
                # Analyze overfitting
                analysis = lc.analyze_overfitting(results, threshold=0.05)
                
                # Print analysis
                print(f"\n  Overfitting Analysis:")
                print(f"    Final gap: {analysis['final_gap']:.4f} ({analysis['final_gap']*100:.2f}%)")
                print(f"    Severity: {analysis['overfitting_severity']}")
                print(f"    Status: {'⚠️  Overfitting detected' if analysis['is_overfitting'] else '✓  Good generalization'}")
                
                # Save plot
                plot_path = lc_dir / f'learning_curve_{config_name}_{clf_name.lower().replace(" ", "_").replace("-", "_")}.png'
                lc.plot_learning_curve(results, f'{clf_name} - {config_name}', plot_path)
                
                # Save results
                csv_path = lc_dir / f'learning_curve_{config_name}_{clf_name.lower().replace(" ", "_").replace("-", "_")}.csv'
                lc.save_results(results, analysis, csv_path)
        
        print(f"\n{'='*80}")
        print(f"✓ Learning curves complete!")
        print(f"  Results saved to: {lc_dir}")
        print(f"{'='*80}")
    else:
        print("\n" + "="*80)
        print("Step 5.6: Learning Curves SKIPPED (--compute_learning_curves not set)")
        print("="*80)
    
    # =========================================================================
    # Step 6: Create Comprehensive Summary
    # =========================================================================
    print("\n" + "="*80)
    print("Step 6: Creating Comprehensive Summary")
    print("="*80)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'comprehensive_results.csv', index=False)
    
    # Save optimal configurations summary
    if 'best' in optimal_configs:
        optimal_summary = pd.DataFrame([
            {
                'configuration': 'best',
                'wavelength_count': optimal_configs['best'][0],
                'accuracy': optimal_configs['best'][1],
                'reduction_percent': (1 - optimal_configs['best'][0] / len(wavelengths)) * 100
            },
            {
                'configuration': 'minimum_acceptable',
                'wavelength_count': optimal_configs['minimum'][0],
                'accuracy': optimal_configs['minimum'][1],
                'reduction_percent': (1 - optimal_configs['minimum'][0] / len(wavelengths)) * 100
            },
            {
                'configuration': 'knee_point',
                'wavelength_count': optimal_configs['knee'][0],
                'accuracy': optimal_configs['knee'][1],
                'reduction_percent': (1 - optimal_configs['knee'][0] / len(wavelengths)) * 100
            }
        ])
        optimal_summary.to_csv(output_dir / 'optimal_configurations.csv', index=False)
        print(f"✓ Saved optimal configurations summary")
    
    # Create summary report
    with open(output_dir / 'comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"COMPREHENSIVE CCARS EVALUATION REPORT - {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total wavelengths: {len(wavelengths)}\n")
        f.write(f"CCARS Runs: {cars_runs}\n")
        f.write(f"PLS Components: {pls_components}\n")
        f.write(f"Wavelength Range Tested: {test_range[0]} to {test_range[-1]}\n")
        f.write(f"Classifiers Evaluated: {classifiers}\n\n")
        
        # Optimal configurations
        if 'best' in optimal_configs:
            f.write("="*80 + "\n")
            f.write("OPTIMAL WAVELENGTH CONFIGURATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Primary Classifier: {primary_classifier}\n\n")
            
            best_count, best_acc = optimal_configs['best']
            min_count, min_acc = optimal_configs['minimum']
            knee_count, knee_acc = optimal_configs['knee']
            
            f.write(f"Best Performance:\n")
            f.write(f"  Wavelengths: {best_count}\n")
            f.write(f"  Accuracy: {best_acc:.4f}\n")
            f.write(f"  Reduction: {(1 - best_count/len(wavelengths))*100:.1f}%\n\n")
            
            f.write(f"Minimum Acceptable (>=85% of best):\n")
            f.write(f"  Wavelengths: {min_count}\n")
            f.write(f"  Accuracy: {min_acc:.4f}\n")
            f.write(f"  Reduction: {(1 - min_count/len(wavelengths))*100:.1f}%\n\n")
            
            f.write(f"Knee Point (optimal trade-off):\n")
            f.write(f"  Wavelengths: {knee_count}\n")
            f.write(f"  Accuracy: {knee_acc:.4f}\n")
            f.write(f"  Reduction: {(1 - knee_count/len(wavelengths))*100:.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS BY WAVELENGTH COUNT\n")
        f.write("="*80 + "\n\n")
        
        # Group by wavelength count
        for n_wl in counts_to_save:
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
    print(f"  - optimal_configurations.csv")
    print(f"  - comprehensive_report.txt")
    print(f"  - optimization_curve_*.png")
    print(f"  - {len(counts_to_save)} optimal configuration directories")
    
    # =========================================================================
    # Step 6: Hold-Out Validation (Optional - Independent Evaluation)
    # =========================================================================
    if use_holdout_validation:
        print("\n" + "="*80)
        print("Step 6: Hold-Out Validation (Independent Evaluation)")
        print("="*80)
        
        # Extract and preprocess hold-out data
        X_holdout_train_df = data['X_holdout_train_df']
        X_holdout_test_df = data['X_holdout_test_df']
        
        X_holdout_train_preprocessed = preprocess_hsi_data(X_holdout_train_df, apply_log=True, apply_snv=True)
        X_holdout_test_preprocessed = preprocess_hsi_data(X_holdout_test_df, apply_log=True, apply_snv=True)
        
        X_train_2 = X_holdout_train_preprocessed.values
        y_train_2 = X_holdout_train_preprocessed.index.get_level_values('Class').values
        X_test_2 = X_holdout_test_preprocessed.values
        y_test_2 = X_holdout_test_preprocessed.index.get_level_values('Class').values
        
        # Determine which configurations to validate
        if optimize_wavelengths and 'best' in optimal_configs:
            validation_counts = [optimal_configs[key][0] for key in ['best', 'minimum', 'knee']]
            config_labels = {optimal_configs[key][0]: key for key in ['best', 'minimum', 'knee']}
        else:
            validation_counts = counts_to_save
            config_labels = {n: f'wavelength_{n}' for n in validation_counts}
        
        print(f"\nValidating {len(validation_counts)} configurations on independent hold-out set...")
        print(f"Hold-out size: {len(X_test_2)} samples (completely unseen during wavelength selection)\n")
        
        # Preprocess hold-out data
        framework_holdout = MultiClassifierFramework(n_components=pls_components, random_state=random_state)
        
        # Store hold-out validation results
        holdout_results = []
        
        for n_wl in validation_counts:
            # Get selected wavelengths for this configuration
            selected_wl, _ = cars.get_selected_wavelengths(top_n=n_wl)
            if len(selected_wl) == 0:
                continue
            
            selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
            
            # Extract hold-out data with selected wavelengths
            X_train_holdout_sel = X_train_2[:, selected_wl_indices]
            X_test_holdout_sel = X_test_2[:, selected_wl_indices]
            
            config_label = config_labels.get(n_wl, f'wavelength_{n_wl}')
            print(f"\n{config_label} ({len(selected_wl)} wavelengths):")
            
            # Evaluate on hold-out set
            for clf_name in classifiers:
                result = framework_holdout.train_and_evaluate(
                    X_train_holdout_sel, y_train_2, X_test_holdout_sel, y_test_2,
                    classifier_name=clf_name,
                    wavelength_type=f'{len(selected_wl)}_selected_holdout'
                )
                
                # Find corresponding calibration result
                cal_result = results_df[
                    (results_df['classifier'] == clf_name) &
                    (results_df['n_wavelengths_selected'] == len(selected_wl))
                ]
                
                if len(cal_result) > 0:
                    cal_acc = cal_result.iloc[0]['accuracy_selected']
                    holdout_acc = result['accuracy']
                    diff = holdout_acc - cal_acc
                    
                    holdout_results.append({
                        'configuration': config_label,
                        'n_wavelengths': len(selected_wl),
                        'classifier': clf_name,
                        'calibration_accuracy': cal_acc,
                        'holdout_accuracy': holdout_acc,
                        'difference': diff,
                        'generalization_score': holdout_acc / cal_acc if cal_acc > 0 else 0
                    })
                    
                    print(f"  {clf_name:20s}: Calibration={cal_acc:.4f}, Hold-out={holdout_acc:.4f}, "
                          f"Diff={diff:+.4f}")
        
        # Save hold-out validation results
        holdout_df = pd.DataFrame(holdout_results)
        holdout_df.to_csv(output_dir / 'holdout_validation_results.csv', index=False)
        
        # Create comparison report
        with open(output_dir / 'holdout_validation_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HOLD-OUT VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Hold-out samples: {len(X_test_2)}\n")
            f.write(f"Configurations tested: {len(validation_counts)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("CALIBRATION vs HOLD-OUT PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            for config in validation_counts:
                subset = holdout_df[holdout_df['n_wavelengths'] == config]
                if len(subset) == 0:
                    continue
                
                config_label = config_labels.get(config, f'wavelength_{config}')
                f.write(f"\n{config_label} ({config} wavelengths):\n")
                f.write("-" * 80 + "\n")
                
                for _, row in subset.iterrows():
                    f.write(f"  {row['classifier']:20s}: ")
                    f.write(f"Cal={row['calibration_accuracy']:.4f}  ")
                    f.write(f"Hold-out={row['holdout_accuracy']:.4f}  ")
                    f.write(f"Diff={row['difference']:+.4f}  ")
                    f.write(f"Gen={row['generalization_score']:.3f}\n")
        
        print(f"\n✓ Hold-out validation complete!")
        print(f"  - holdout_validation_results.csv")
        print(f"  - holdout_validation_report.txt")
    
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
        description='Comprehensive CCARS Evaluation with Optional Wavelength Optimization'
    )
    
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'],
                       help='Dataset name')
    parser.add_argument('--wavelengths', type=int, nargs='*', default=None,
                       help='Specific wavelength counts to test (default: [10,20,30,50] in Nicola mode, [5-50] in optimize mode)')
    parser.add_argument('--optimize_wavelengths', action='store_true',
                       help='Enable wavelength optimization mode (tests range [5-50] and auto-selects best/min/knee). Default: False (Nicola\'s approach)')
    parser.add_argument('--classifiers', type=str, nargs='+',
                       default=['PLS-DA', 'SVM-RBF', 'Random Forest'],
                       help='Classifiers to evaluate')
    parser.add_argument('--cars_runs', type=int, default=500,
                       help='CCARS Monte Carlo runs')
    parser.add_argument('--cars_iterations', type=int, default=10,
                       help='CCARS iterations per run')
    parser.add_argument('--components', type=int, nargs='+', default=[2, 3, 4],
                       help='PLS components to test (can specify multiple, e.g., --components 2 3 4)')
    parser.add_argument('--validation', action='store_true',
                       help='Enable hold-out validation (final independent evaluation)')
    parser.add_argument('--use_calibration', action='store_true', default=True,
                       help='Use calibration/final split (CCARS mode). Use --no_use_calibration to disable.')
    parser.add_argument('--no_use_calibration', dest='use_calibration', action='store_false',
                       help='Disable calibration split (original CARS mode)')
    parser.add_argument('--calibration_fraction', type=float, default=0.5,
                       help='Fraction of data for calibration set (default: 0.5)')
    parser.add_argument('--n_permutations', type=int, default=1000,
                       help='Number of permutations for statistical testing (default: 1000)')
    parser.add_argument('--skip_permutation', action='store_true',
                       help='Skip permutation testing (faster for quick tests)')
    parser.add_argument('--compute_learning_curves', action='store_true',
                       help='Compute learning curves for overfitting detection')
    parser.add_argument('--lc_train_sizes', type=int, nargs='+', default=None,
                       help='Training sizes for learning curves (e.g., 100 200 500 1000)')
    parser.add_argument('--preprocessing', type=str, default='snv_only',
                       choices=['snv_only', 'log10_snv', 'none'],
                       help="Preprocessing method: 'snv_only' (Nicola's exact, default), 'log10_snv' (HSI-adapted), 'none'")
    parser.add_argument('--optimize_roc', action='store_true',
                       help='Enable ROC threshold optimization for improved classification')
    parser.add_argument('--adaptive_permutations', action='store_true',
                       help='Use adaptive permutation counts (faster for Kaggle: PLS-DA=1000, SVM=100, RF=200)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Determine wavelength counts to test
    if args.wavelengths is not None and len(args.wavelengths) > 0:
        # Use explicitly provided counts
        wavelength_counts = args.wavelengths
    elif args.optimize_wavelengths:
        # Optimization mode: will test range [5, 10, ..., 50]
        wavelength_counts = None  # Will be set to range(5, 55, 5) in function
    else:
        # Nicola's mode: default fixed counts
        wavelength_counts = [10, 20, 30, 50]
    
    
    # Normalize components to always be a list
    components_list = args.components if isinstance(args.components, list) else [args.components]
    
    all_results = {}
    
    # Loop through each PLS component value
    for pls_comp in components_list:
        print(f"\n{'='*80}")
        print(f"Testing with PLS Components: {pls_comp}")
        print(f"{'='*80}\n")
        
        # Create component-specific output directory
        if args.output:
            component_output_dir = str(Path(args.output) / f"component_{pls_comp}")
        else:
            component_output_dir = f"{args.dataset}_results/component_{pls_comp}"
        
        results = run_comprehensive_evaluation(
            dataset_name=args.dataset,
           wavelength_counts=wavelength_counts,
            classifiers=args.classifiers,
            cars_runs=args.cars_runs,
            cars_iterations=args.cars_iterations,
            pls_components=pls_comp,  # Pass single component value
            output_dir=component_output_dir,
            random_state=args.random_state,
            optimize_wavelengths=args.optimize_wavelengths,
            use_holdout_validation=args.validation,
            use_calibration=args.use_calibration,
            calibration_fraction=args.calibration_fraction,
            n_permutations=args.n_permutations,
            skip_permutation=args.skip_permutation,
            compute_learning_curves=args.compute_learning_curves,
            lc_train_sizes=args.lc_train_sizes,
            preprocessing_method=args.preprocessing,
            optimize_roc=args.optimize_roc,
            adaptive_permutations=args.adaptive_permutations
        )
        
        all_results[f'components_{pls_comp}'] = results
    
    return all_results


if __name__ == '__main__':
    results = main()
