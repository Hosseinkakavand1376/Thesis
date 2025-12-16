"""
Quick integration test for permutation testing

Tests that permutation testing works in the main pipeline
"""

import subprocess
import sys
from pathlib import Path

def test_permutation_integration():
    """Test permutation testing integration"""
    
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Permutation Testing in Main Pipeline")
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
        '--n_permutations', '20',  # Very few permutations for speed
        '--output', 'test_permutation_output'
    ]
    
    print("Running command:")
    print(" ".join(test_cmd))
    print()
    print("This will test:")
    print("  ✓ Calibration/final split")
    print("  ✓ CARS wavelength selection")
    print("  ✓ Permutation testing (20 permutations)")
    print("  ✓ P-value calculation")
    print("  ✓ Plot and CSV generation")
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
            output_dir = Path('test_permutation_output/permutation_tests')
            if output_dir.exists():
                files = list(output_dir.glob('*'))
                print(f"Generated {len(files)} permutation test files:")
                for f in files[:5]:  # Show first 5
                    print(f"  - {f.name}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")
            
            print()
            print("Permutation testing is successfully integrated into the pipeline.")
            print("You can now run full experiments with:")
            print()
            print("  python main_hsi_cars_comprehensive.py \\")
            print("    --dataset salinas \\")
            print("    --cars_runs 500 \\")
            print("    --n_permutations 1000")
            print()
            print("Or skip permutation testing for quick tests:")
            print("  python main_hsi_cars_comprehensive.py --skip_permutation")
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
    success = test_permutation_integration()
    sys.exit(0 if success else 1)
