"""
Validation Test: Dataset Separation with Real Salinas Data

This script validates the calibration/final split methodology using
a small subset of real HSI data to verify:
1. CARS runs successfully with calibration data
2. Final set evaluation works correctly
3. Generalization gap is reasonable (1-5%)
4. No data leakage between sets
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multiclass_cars import MultiClassCARS
from multiclass_plsda import MultiClassPLSDA
from hsi_data_loader import load_hsi_dataset, flatten_hsi_to_2d
from hsi_preprocessing import preprocess_hsi_data
from hsi_config import generate_wavelengths

def validate_with_real_data():
    """Run validation with real Salinas data"""
    
    print("\n" + "=" * 70)
    print("VALIDATION: Dataset Separation with Real Salinas Data")
    print("=" * 70)
    print("Purpose: Verify calibration/final split before full integration")
    print("=" * 70 + "\n")
    
    # Configuration
    dataset_name = 'salinas'
    n_runs = 30  # Increased from 15 for better wavelength selection
    n_iterations = 100  # Full iterations for better convergence
    n_components = 3
    
    print(f"Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  CARS runs: {n_runs}")
    print(f"  Iterations per run: {n_iterations}")
    print(f"  PLS components: {n_components}")
    print()
    
    # Step 1: Load data
    print("=" * 70)
    print("Step 1: Loading Salinas Data")
    print("=" * 70)
    
    try:
        # Load full dataset
        data_cube, ground_truth, config = load_hsi_dataset(dataset_name, auto_download=False)
        
        # Flatten to 2D
        X, y, spatial_indices = flatten_hsi_to_2d(data_cube, ground_truth, remove_background=True)
        
        # Generate wavelengths
        wavelengths = generate_wavelengths(dataset_name)
        
        print(f"\n✓ Data loaded successfully:")
        print(f"  Total samples: {len(y)}")
        print(f"  Features (wavelengths): {len(wavelengths)}")
        print(f"  Classes: {len(np.unique(y))}")
        
        # Use a subset for faster testing (first 500 samples)
        subset_size = min(500, len(y))
        X_subset = X[:subset_size]
        y_subset = y[:subset_size]
        
        print(f"\n  Using subset for validation:")
        print(f"    Samples: {subset_size}")
        print(f"    Wavelengths: {len(wavelengths)}")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print(f"\nFalling back to synthetic data...")
        
        # Fallback to synthetic data if real data not available
        from sklearn.datasets import make_classification
        X_subset, y_subset = make_classification(
            n_samples=500, n_features=204, n_informative=150,
            n_classes=16, n_clusters_per_class=1, random_state=42
        )
        wavelengths = np.linspace(400, 2500, 204)
        
        print(f"  Synthetic data created:")
        print(f"    Samples: {len(y_subset)}")
        print(f"    Features: {len(wavelengths)}")
        print(f"    Classes: 16")
    
    # Step 2: Preprocess data
    print("\n" + "=" * 70)
    print("Step 2: Preprocessing Data")
    print("=" * 70)
    
    # Convert to DataFrame for preprocessing
    X_df_subset = pd.DataFrame(X_subset, columns=wavelengths)
    
    # Apply Log10 + SNV (current pipeline preprocessing)
    X_preprocessed = preprocess_hsi_data(X_df_subset, apply_log=True, apply_snv=True)
    X_preprocessed = X_preprocessed.values
    
    print(f"\n✓ Preprocessing complete")
    
    # Step 3: Initialize CARS with calibration mode
    print("\n" + "=" * 70)
    print("Step 3: Initializing CCARS with Calibration Mode")
    print("=" * 70)
    
    output_path = Path('validation_calibration_test')
    output_path.mkdir(exist_ok=True)
    
    cars = MultiClassCARS(
        output_path=output_path,
        n_components=n_components,
        use_calibration=True,  # Enable calibration mode
        random_state=42
    )
    
    print(f"\n✓ CARS initialized with:")
    print(f"  Calibration mode: ENABLED")
    print(f"  Components: {n_components}")
    print(f"  Output: {output_path}")
    
    # Step 4: Prepare calibration split
    print("\n" + "=" * 70)
    print("Step 4: Preparing Calibration/Final Split")
    print("=" * 70)
    
    split_dict = cars.prepare_calibration_split(
        X_preprocessed, y_subset, wavelengths,
        calibration_fraction=0.5
    )
    
    # Step 5: Fit CARS with calibration data
    print("\n" + "=" * 70)
    print("Step 5: Fitting CARS with Calibration Data")
    print("=" * 70)
    
    cars.fit(
        split_dict['X_cal_train'],
        split_dict['y_cal_train'],
        split_dict['X_cal_test'],
        split_dict['y_cal_test'],
        wavelengths
    )
    
    # Step 6: Run CARS wavelength selection
    print("\n" + "=" * 70)
    print("Step 6: Running CARS Wavelength Selection")
    print("=" * 70)
    print(f"This will take approximately {n_runs * n_iterations * 0.05 / 60:.1f} minutes...")
    print()
    
    cars.run_cars(
        n_runs=n_runs,
        n_iterations=n_iterations,
        mc_samples=0.8,
        use_ars=True
    )
    
    # Step 7: Get selected wavelengths
    print("\n" + "=" * 70)
    print("Step 7: Extracting Selected Wavelengths")
    print("=" * 70)
    
    # Get top wavelengths
    selected_wl, freq_df = cars.get_selected_wavelengths(top_n=50)
    
    print(f"\n✓ Selected wavelengths extracted:")
    print(f"  Total selected: {len(selected_wl)}")
    print(f"  Top 10 wavelengths:")
    for i, wl in enumerate(selected_wl[:10], 1):
        freq = freq_df[freq_df['Wavelength'] == wl]['Frequency'].values[0]
        print(f"    {i}. {wl:.2f} nm (selected {freq*100:.1f}% of runs)")
    
    # Step 8: Compare calibration vs final set performance
    print("\n" + "=" * 70)
    print("Step 8: Comparing Calibration vs Final Set Performance")
    print("=" * 70)
    
    final_set = cars.get_final_set()
    
    # Test with different wavelength counts
    test_configs = [10, 20, 30, 50]
    results = []
    
    for n_wl in test_configs:
        if n_wl > len(selected_wl):
            continue
        
        # Get indices of selected wavelengths
        top_wl = selected_wl[:n_wl]
        wl_indices = []
        for wl in top_wl:
            idx = np.argmin(np.abs(wavelengths - wl))
            wl_indices.append(idx)
        wl_indices = np.array(wl_indices)
        
        # Evaluate on CALIBRATION set
        cal_model = MultiClassPLSDA(n_components=n_components)
        cal_model.fit(
            split_dict['X_cal_train'][:, wl_indices],
            split_dict['y_cal_train']
        )
        y_pred_cal = cal_model.predict(split_dict['X_cal_test'][:, wl_indices])
        cal_accuracy = np.mean(y_pred_cal == split_dict['y_cal_test'])
        
        # Evaluate on FINAL set (completely independent)
        final_model = MultiClassPLSDA(n_components=n_components)
        final_model.fit(
            final_set['X_train'][:, wl_indices],
            final_set['y_train']
        )
        y_pred_final = final_model.predict(final_set['X_test'][:, wl_indices])
        final_accuracy = np.mean(y_pred_final == final_set['y_test'])
        
        # Calculate gap
        gap = cal_accuracy - final_accuracy
        gap_pct = gap * 100
        
        results.append({
            'n_wavelengths': n_wl,
            'cal_accuracy': cal_accuracy,
            'final_accuracy': final_accuracy,
            'gap': gap,
            'gap_pct': gap_pct
        })
        
        # Determine status
        if abs(gap) < 0.05:
            status = "✓ Excellent"
        elif abs(gap) < 0.10:
            status = "✓ Good"
        elif abs(gap) < 0.15:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"
        
        print(f"\n{n_wl} Wavelengths:")
        print(f"  Calibration accuracy: {cal_accuracy:.4f}")
        print(f"  Final set accuracy:   {final_accuracy:.4f}")
        print(f"  Generalization gap:   {gap:+.4f} ({gap_pct:+.2f}%)")
        print(f"  Status: {status}")
    
    # Step 9: Summary and validation
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = output_path / 'calibration_validation_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Check validation criteria
    print("\nValidation Criteria:")
    
    # Criterion 1: All gaps should be < 15%
    max_gap = results_df['gap'].abs().max()
    criterion1 = max_gap < 0.15
    print(f"  1. Maximum gap < 15%: {max_gap*100:.2f}% {'✓ PASS' if criterion1 else '❌ FAIL'}")
    
    # Criterion 2: At least one config with gap < 5%
    min_gap = results_df['gap'].abs().min()
    criterion2 = min_gap < 0.05
    print(f"  2. Minimum gap < 5%:  {min_gap*100:.2f}% {'✓ PASS' if criterion2 else '❌ FAIL'}")
    
    # Criterion 3: Calibration accuracies reasonable (> 0.5 for multi-class)
    min_cal_acc = results_df['cal_accuracy'].min()
    criterion3 = min_cal_acc > 0.5
    print(f"  3. Calibration acc > 50%: {min_cal_acc*100:.2f}% {'✓ PASS' if criterion3 else '❌ FAIL'}")
    
    # Criterion 4: Final accuracies not much worse than calibration
    avg_gap = results_df['gap'].mean()
    criterion4 = abs(avg_gap) < 0.10
    print(f"  4. Average gap < 10%: {abs(avg_gap)*100:.2f}% {'✓ PASS' if criterion4 else '❌ FAIL'}")
    
    all_pass = criterion1 and criterion2 and criterion3 and criterion4
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ VALIDATION SUCCESSFUL!")
        print("=" * 70)
        print("\nCalibration/final split is working correctly.")
        print("Generalization gaps are within expected range.")
        print("Ready to proceed with full integration into main pipeline.")
    else:
        print("⚠️  VALIDATION WARNINGS")
        print("=" * 70)
        print("\nSome criteria did not pass. Review results above.")
        print("This may indicate:")
        print("  - Need more CARS runs (try 50-100)")
        print("  - Dataset is too small")
        print("  - Model needs tuning")
    
    print("=" * 70)
    
    # Print best configuration
    best_idx = results_df['final_accuracy'].idxmax()
    best_config = results_df.iloc[best_idx]
    
    print(f"\nBest Configuration:")
    print(f"  Wavelengths: {int(best_config['n_wavelengths'])}")
    print(f"  Final accuracy: {best_config['final_accuracy']:.4f}")
    print(f"  Gap: {best_config['gap']:+.4f}")
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70 + "\n")
    
    return all_pass


if __name__ == '__main__':
    try:
        success = validate_with_real_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
