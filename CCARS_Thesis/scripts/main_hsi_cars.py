"""
Main Pipeline: CCARS+PLS-DA for Multi-Class HSI Datasets

This script runs the complete pipeline:
1. Load HSI dataset (Salinas or Indian Pines)
2. Preprocess (Log10 + SNV)
3. Run CCARS wavelength selection
4. Train PLS-DA on selected wavelengths
5. Evaluate and generate reports

Usage:
    python main_hsi_cars.py --dataset salinas --runs 500 --components 3
    python main_hsi_cars.py --dataset indian_pines --runs 500 --components 5
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import our modules
from hsi_config import get_dataset_config
from hsi_data_loader import prepare_hsi_for_cars
from hsi_preprocessing import preprocess_hsi_data
from multiclass_cars import MultiClassCARS
from multiclass_plsda import MultiClassPLSDA
from hsi_evaluation import (
    compute_multiclass_metrics,
    plot_confusion_matrix,
    plot_wavelength_frequency,
    plot_cars_convergence,
    generate_evaluation_report
)


def run_ccars_pipeline(dataset_name='salinas',
                       n_runs=500,
                       n_iterations=100,
                       n_components=3,
                       mc_samples=0.8,
                       use_ars=True,
                       test_percentage=0.2,
                       calibration_percentage=0.5,
                       output_dir=None,
                       random_state=42):
    """
    Complete CCARS+PLS-DA pipeline for HSI dataset
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        n_runs: Number of Monte Carlo runs
        n_iterations: Number of iterations per run
        n_components: Number of PLS components
        mc_samples: Fraction of samples per iteration
        use_ars: Use Adaptive Reweighted Sampling
        test_percentage: Test set fraction
        calibration_percentage: Calibration set fraction
        output_dir: Output directory (auto-generated if None)
        random_state: Random seed
    
    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 80)
    print(f"CCARS+PLS-DA Pipeline for {dataset_name.upper()}")
    print("=" * 80)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(f"HSI_CARS_results/{dataset_name}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    class_names = config['class_names']
    
    # =========================================================================
    # Step 1: Load and Prepare Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading and Preparing Data")
    print("=" * 80)
    
    data = prepare_hsi_for_cars(
        dataset_name=dataset_name,
        test_percentage=test_percentage,
        calibration_percentage=calibration_percentage,
        random_state=random_state,
        auto_download=True
    )
    
    X_train_df = data['X_cal_train_df']
    X_test_df = data['X_cal_test_df']
    wavelengths = data['wavelengths']
    
    print(f"\n✓ Data loaded:")
    print(f"  Training samples: {len(X_train_df)}")
    print(f"  Test samples: {len(X_test_df)}")
    print(f"  Wavelengths: {len(wavelengths)}")
    print(f"  Classes: {len(class_names)}")
    
    # =========================================================================
    # Step 2: Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Preprocessing (Log10 + SNV)")
    print("=" * 80)
    
    X_train_preprocessed = preprocess_hsi_data(X_train_df, apply_log=True, apply_snv=True)
    X_test_preprocessed = preprocess_hsi_data(X_test_df, apply_log=True, apply_snv=True)
    
    # Convert to numpy arrays
    X_train = X_train_preprocessed.values
    y_train = X_train_preprocessed.index.get_level_values('Class').values
    X_test = X_test_preprocessed.values
    y_test = X_test_preprocessed.index.get_level_values('Class').values
    
    print(f"\n✓ Preprocessing complete")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # =========================================================================
    # Step 3: Run CCARS Wavelength Selection
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Running CCARS Wavelength Selection")
    print("=" * 80)
    
    cars = MultiClassCARS(
        output_path=output_dir / 'cars_results',
        n_components=n_components,
        test_percentage=test_percentage,
        calibration=True,
        random_state=random_state
    )
    
    cars.fit(X_train, y_train, X_test, y_test, wavelengths)
    
    cars.run_cars(
        n_runs=n_runs,
        n_iterations=n_iterations,
        mc_samples=mc_samples,
        use_ars=use_ars
    )
    
    # Get selected wavelengths
    # Use adaptive strategy: try frequency > 30%, fallback to top 10 if none found
    selected_wl, wl_freq_df = cars.get_selected_wavelengths(frequency_threshold=0.3)
    
    # If no wavelengths selected with 30% threshold, use top 10 most frequent
    if len(selected_wl) == 0:
        print("  Warning: No wavelengths met 30% frequency threshold. Using top 10 most frequent.")
        selected_wl, wl_freq_df = cars.get_selected_wavelengths(top_n=10)
    
    # If still empty (shouldn't happen), use top 5
    if len(selected_wl) == 0:
        print("  Warning: Using top 5 most frequent wavelengths.")
        selected_wl, wl_freq_df = cars.get_selected_wavelengths(top_n=5)
    
    selected_wl_indices = [np.where(np.array(wavelengths) == wl)[0][0] for wl in selected_wl]
    
    print(f"\n✓ CCARS complete:")
    print(f"  Selected wavelengths: {len(selected_wl)}")
    print(f"  Reduction: {(1 - len(selected_wl) / len(wavelengths)) * 100:.1f}%")
    
    # Save selected wavelengths
    wl_freq_df.to_csv(output_dir / 'selected_wavelengths.csv', index=False)
    
    # =========================================================================
    # Step 4: Train Final PLS-DA with Selected Wavelengths
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Training PLS-DA with Selected Wavelengths")
    print("=" * 80)
    
    # Train on selected wavelengths
    pls_selected = MultiClassPLSDA(n_components=n_components)
    pls_selected.fit(X_train[:, selected_wl_indices], y_train)
    y_pred_selected = pls_selected.predict(X_test[:, selected_wl_indices])
    
    # Train on full spectrum for comparison
    pls_full = MultiClassPLSDA(n_components=n_components)
    pls_full.fit(X_train, y_train)
    y_pred_full = pls_full.predict(X_test)
    
    print(f"\n✓ PLS-DA trained:")
    print(f"  Model with selected wavelengths: {len(selected_wl)} features")
    print(f"  Model with full spectrum: {len(wavelengths)} features")
    
    # =========================================================================
    # Step 5: Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Evaluation")
    print("=" * 80)
    
    # Compute metrics
    metrics_selected = compute_multiclass_metrics(y_test, y_pred_selected, class_names)
    metrics_full = compute_multiclass_metrics(y_test, y_pred_full, class_names)
    
    print(f"\n Performance Comparison:")
    print(f"  {'Metric':<25} {'Selected WL':>15} {'Full Spectrum':>15} {'Difference':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Accuracy':<25} {metrics_selected['accuracy']:>15.4f} "
          f"{metrics_full['accuracy']:>15.4f} "
          f"{(metrics_selected['accuracy'] - metrics_full['accuracy']):>12.4f}")
    print(f"  {'F1 (weighted)':<25} {metrics_selected['f1_weighted']:>15.4f} "
          f"{metrics_full['f1_weighted']:>15.4f} "
          f"{(metrics_selected['f1_weighted'] - metrics_full['f1_weighted']):>12.4f}")
    print(f"  {'Precision (weighted)':<25} {metrics_selected['precision_weighted']:>15.4f} "
          f"{metrics_full['precision_weighted']:>15.4f} "
          f"{(metrics_selected['precision_weighted'] - metrics_full['precision_weighted']):>12.4f}")
    print(f"  {'Recall (weighted)':<25} {metrics_selected['recall_weighted']:>15.4f} "
          f"{metrics_full['recall_weighted']:>15.4f} "
          f"{(metrics_selected['recall_weighted'] - metrics_full['recall_weighted']):>12.4f}")
    
    # =========================================================================
    # Step 6: Generate Visualizations and Reports
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Generating Visualizations and Reports")
    print("=" * 80)
    
    # Confusion matrix
    cm_selected = plot_confusion_matrix(
        y_test, y_pred_selected, class_names,
        save_path=output_dir / 'confusion_matrix_selected.png'
    )
    
    cm_full = plot_confusion_matrix(
        y_test, y_pred_full, class_names,
        save_path=output_dir / 'confusion_matrix_full.png'
    )
    
    # Wavelength frequency
    wl_freq = plot_wavelength_frequency(
        cars.coefficients_df, wavelengths,
        save_path=output_dir / 'wavelength_frequency.png'
    )
    
    # CARS convergence
    plot_cars_convergence(
        cars.statistics_df,
        save_path=output_dir / 'cars_convergence.png'
    )
    
    # Evaluation report
    report = generate_evaluation_report(
        y_test, y_pred_selected, class_names, selected_wl, wavelengths,
        save_path=output_dir / 'evaluation_report.txt'
    )
    
    print("\n" + report)
    
    # =========================================================================
    # Save Summary
    # =========================================================================
    summary = {
        'dataset': dataset_name,
        'n_runs': n_runs,
        'n_iterations': n_iterations,
        'n_components': n_components,
        'original_wavelengths': len(wavelengths),
        'selected_wavelengths': len(selected_wl),
        'reduction_percent': (1 - len(selected_wl) / len(wavelengths)) * 100,
        'accuracy_selected': metrics_selected['accuracy'],
        'accuracy_full': metrics_full['accuracy'],
        'f1_selected': metrics_selected['f1_weighted'],
        'f1_full': metrics_full['f1_weighted'],
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")
    
    return {
        'cars': cars,
        'selected_wavelengths': selected_wl,
        'metrics_selected': metrics_selected,
        'metrics_full': metrics_full,
        'summary': summary
    }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='CCARS+PLS-DA for Multi-Class HSI Wavelength Selection'
    )
    
    parser.add_argument('--dataset', type=str, default='salinas',
                       choices=['salinas', 'indian_pines'],
                       help='Dataset name (default: salinas)')
    parser.add_argument('--runs', type=int, default=500,
                       help='Number of Monte Carlo runs (default: 500)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per run (default: 100)')
    parser.add_argument('--components', type=int, default=3,
                       help='Number of PLS components (default: 3)')
    parser.add_argument('--mc_samples', type=float, default=0.8,
                       help='MC sampling fraction (default: 0.8)')
    parser.add_argument('--ars', action='store_true', default=True,
                       help='Use Adaptive Reweighted Sampling (default: True)')
    parser.add_argument('--test_pct', type=float, default=0.2,
                       help='Test set percentage (default: 0.2)')
    parser.add_argument('--calibration_pct', type=float, default=0.5,
                       help='Calibration set percentage (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: HSI_CARS_results/{dataset})')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_ccars_pipeline(
        dataset_name=args.dataset,
        n_runs=args.runs,
        n_iterations=args.iterations,
        n_components=args.components,
        mc_samples=args.mc_samples,
        use_ars=args.ars,
        test_percentage=args.test_pct,
        calibration_percentage=args.calibration_pct,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    return results


if __name__ == '__main__':
    results = main()
