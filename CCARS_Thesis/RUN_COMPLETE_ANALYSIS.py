"""
RUN_COMPLETE_ANALYSIS.py

Main entry point for complete HSI feature selection analysis

This runs the master pipeline from the scripts directory.
"""

import subprocess
import sys

if __name__ == '__main__':
    print("\n" + "="*80)
    print("HSI FEATURE SELECTION - COMPLETE ANALYSIS")
    print("="*80)
    print("\nRunning master pipeline from scripts/RUN_ALL_MASTER.py")
    print("="*80 + "\n")
    
    # Run the master script from scripts directory
    result = subprocess.run(
        [sys.executable, 'scripts/RUN_ALL_MASTER.py'] + sys.argv[1:],
        cwd='.'
    )
    
    sys.exit(result.returncode)
