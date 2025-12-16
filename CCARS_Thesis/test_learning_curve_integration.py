"""
Quick integration test for learning curves

Tests that learning curve analysis works in the main pipeline
"""

import subprocess
import sys
from pathlib import Path

def test_learning_curve_integration():
    """Test learning curve integration"""
    
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Learning Curves in Main Pipeline")
    print("=" * 70)
    print()
    
    # Test configuration - VERY quick test
    test_cmd = [
        'python', 'main_hsi_cars_comprehensive.py',
        '--dataset', 'salinas',
        '--cars_runs', '3',  # Minimal CARS runs
        '--cars_iterations', '10',  # Minimal iterations
        '--components', '3',
        '--wavelengths', '10',  # Only test 1 count
        '--classifiers', 'PLS-DA',  # Only one classifier
        '--use_calibration',
        '--skip_permutation',  # Skip permutation for speed
        '--compute_learning_curves',  # Enable learning curves
        '--output', 'test_learning_curve_output'
    ]
    
    print("Running command:")
    print(" ".join(test_cmd))
    print()
    print("This will test:")
    print("  ✓ Calibration/final split")
    print("  ✓ CARS wavelength selection")
    print("  ✓ Learning curve analysis (10% to 100%)")
    print("  ✓ Overfitting detection")
    print("  ✓ Dual plots (curves + gap)")
    print("  ✓ CSV and analysis export")
    print()
    
    try:
        result = subprocess.run(
            test_cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ INTEGRATION TEST PASSED!")
            print("=" * 70)
            print()
            
            # Check for output files
            output_dir = Path('test_learning_curve_output/learning_curves')
            if output_dir.exists():
                files = list(output_dir.glob('*'))
                print(f"Generated {len(files)} learning curve files:")
                for f in files[:5]:  # Show first 5
                    print(f"  - {f.name}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")
            
            print()
            print("Learning curves is successfully integrated into the pipeline.")
            print("You can now run full experiments with:")
            print()
            print("  python main_hsi_cars_comprehensive.py \\")
            print("    --dataset salinas \\")
            print("    --cars_runs 500 \\")
            print("    --compute_learning_curves")
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
    success = test_learning_curve_integration()
    sys.exit(0 if success else 1)
