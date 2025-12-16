"""
Cleanup script - Deletes all results and plot directories

This removes:
- HSI_CARS_comprehensive/
- Feature_Selection_Comparison/
- Publication_Plots/
- Advanced_Visualizations/
- pipeline_summary.json
"""

import shutil
from pathlib import Path


def cleanup_all_results():
    """Delete all result directories and files"""
    
    print("\n" + "="*80)
    print("CLEANING UP ALL PREVIOUS RESULTS")
    print("="*80 + "\n")
    
    # Directories to remove
    dirs_to_remove = [
        'HSI_CARS_comprehensive',
        'HSI_CARS_results',
        'Feature_Selection_Comparison',
        'Publication_Plots',
        'Advanced_Visualizations',
        'Comprehensive_Comparison'
    ]
    
    # Files to remove
    files_to_remove = [
        'pipeline_summary.json'
    ]
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Removing directory: {dir_name}...")
            shutil.rmtree(dir_path)
            print(f"  ✅ Deleted {dir_name}/")
        else:
            print(f"  ⏭️  {dir_name}/ does not exist, skipping")
    
    # Remove files
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"Removing file: {file_name}...")
            file_path.unlink()
            print(f"  ✅ Deleted {file_name}")
        else:
            print(f"  ⏭️  {file_name} does not exist, skipping")
    
    print("\n" + "="*80)
    print("✅ CLEANUP COMPLETE!")
    print("="*80)
    print("\nAll previous results have been removed.")
    print("Ready for fresh master pipeline run.")


if __name__ == '__main__':
    cleanup_all_results()
