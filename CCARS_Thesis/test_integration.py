"""
Quick integration test for calibration mode in main_hsi_cars_comprehensive.py

Tests that the full pipeline works with --use_calibration flag
"""

import subprocess
import sys
from pathlib import Path

def test_integration():
    """Test calibration mode integration"""
    
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Calibration Mode in Main Pipeline")
    print("=" * 70)
    print()
    
    # Test configuration
    test_cmd = [
        'python', 'main_hsi_cars_comprehensive.py',
        '--dataset', 'salinas',
        '--cars_runs', '5',  # Very quick test
        '--cars_iterations', '20',  # Reduced iterations
        '--components', '3',
        '--wavelengths', '10', '20',  # Only test 2 counts
        '--classifiers', 'PLS-DA',  # Only one classifier
        '--use_calibration',  # Enable calibration mode
        '--calibration_fraction', '0.5',
        '--output', 'test_integration_output'
    ]
    
    print("Running command:")
    print(" ".join(test_cmd))
    print()
    print("This will test:")
    print("  ✓ Calibration/final split (50/50)")
    print("  ✓ CARS wavelength selection from calibration set")
    print("  ✓ Evaluation with PLS-DA")
    print("  ✓ 5 CARS runs (quick test)")
    print()
    
    try:
        # Run the command
        result = subprocess.run(
            test_cmd,
            cwd=Path(__file__).parent,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ INTEGRATION TEST PASSED!")
            print("=" * 70)
            print()
            print("Calibration mode is successfully integrated into the pipeline.")
            print("You can now run full experiments with:")
            print()
            print("  python main_hsi_cars_comprehensive.py \\")
            print("    --dataset salinas \\")
            print("    --cars_runs 500 \\")
            print("    --use_calibration")
            print()
            return True
        else:
            print("\n" + "=" * 70)
            print("❌ INTEGRATION TEST FAILED")
            print("=" * 70)
            print(f"Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)
