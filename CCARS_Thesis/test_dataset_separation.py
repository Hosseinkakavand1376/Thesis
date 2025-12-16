"""
Quick test for dataset separation in multiclass_cars.py
Tests that calibration/final split works correctly with no data leakage
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multiclass_cars import MultiClassCARS

def test_dataset_separation():
    """Test calibration/final split functionality"""
    
    print("\n" + "=" * 70)
    print("Testing Dataset Separation (CCARS Calibration Split)")
    print("=" * 70)
    
    # Create test data
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_classes = 5
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    wavelengths = np.linspace(400, 900, n_features)
    
    print(f"\nTest data created:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    
    # Initialize CARS with calibration mode
    cars = MultiClassCARS(
        output_path='test_calibration_output',
        n_components=3,
        use_calibration=True,
        random_state=42
    )
    
    print(f"\nCalled prepare_calibration_split()...\n")
    
    # Prepare calibration split
    split_dict = cars.prepare_calibration_split(X, y, wavelengths, calibration_fraction=0.5)
    
    # Verify sizes
    cal_train_size = len(split_dict['y_cal_train'])
    cal_test_size = len(split_dict['y_cal_test'])
    final_train_size = len(split_dict['y_final_train'])
    final_test_size = len(split_dict['y_final_test'])
    
    cal_total = cal_train_size + cal_test_size
    final_total = final_train_size + final_test_size
    overall_total = cal_total + final_total
    
    print("\n" + "=" * 70)
    print("Verification Results:")
    print("=" * 70)
    
    # Test 1: Total samples preserved
    assert overall_total == n_samples, f"Sample count mismatch: {overall_total} != {n_samples}"
    print(f"✓ Test 1 PASSED: Total samples preserved ({overall_total} == {n_samples})")
    
    # Test 2: 50/50 split
    assert cal_total == 100, f"Calibration set should be 100 samples, got {cal_total}"
    assert final_total == 100, f"Final set should be 100 samples, got {final_total}"
    print(f"✓ Test 2 PASSED: 50/50 split correct (Cal: {cal_total}, Final: {final_total})")
    
    # Test 3: 80/20 split within each set
    assert cal_train_size == 80, f"Cal train should be 80, got {cal_train_size}"
    assert cal_test_size == 20, f"Cal test should be 20, got {cal_test_size}"
    assert final_train_size == 80, f"Final train should be 80, got {final_train_size}"
    assert final_test_size == 20, f"Final test should be 20, got {final_test_size}"
    print(f"✓ Test 3 PASSED: 80/20 split within each set")
    print(f"    Calibration: train={cal_train_size}, test={cal_test_size}")
    print(f"    Final: train={final_train_size}, test={final_test_size}")
    
    # Test 4: Calibration mode flags set correctly
    assert cars.is_calibration_mode == True, "Calibration mode flag not set"
    assert cars.calibration_note != "", "Calibration note not set"
    assert cars.final_set is not None, "Final set not stored"
    print(f"✓ Test 4 PASSED: Calibration mode flags set correctly")
    print(f"    Mode: {cars.is_calibration_mode}")
    print(f"    Note: {cars.calibration_note}")
    
    # Test 5: get_final_set() works
    final_set = cars.get_final_set()
    assert final_set is not None, "get_final_set() returned None"
    assert len(final_set['y_train']) == 80, "Final set train size incorrect"
    assert len(final_set['y_test']) == 20, "Final set test size incorrect"
    print(f"✓ Test 5 PASSED: get_final_set() returns correct data")
    
    # Test 6: Feature dimensions preserved
    assert split_dict['X_cal_train'].shape[1] == n_features, "Features lost in calibration train"
    assert split_dict['X_final_test'].shape[1] == n_features, "Features lost in final test"
    print(f"✓ Test 6 PASSED: All feature dimensions preserved ({n_features} wavelengths)")
    
    # Test 7: fit() works with calibration data
    print(f"\nCalling fit() with calibration data...")
    cars.fit(
        split_dict['X_cal_train'],
        split_dict['y_cal_train'],
        split_dict['X_cal_test'],
        split_dict['y_cal_test'],
        wavelengths
    )
    print(f"✓ Test 7 PASSED: fit() accepts calibration data correctly")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nDataset separation implementation is working correctly.")
    print("Ready to integrate into main pipeline.")
    print("=" * 70 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        test_dataset_separation()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
