"""
Utility to load actual wavelengths from reference CSV files
"""
import pandas as pd
from pathlib import Path


def load_wavelengths_from_csv(dataset_name):
    """
    Load actual wavelengths from reference CSV file
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
    
    Returns:
        List of wavelength values in nanometers
    """
    # Define CSV file paths and column names
    csv_files = {
        'salinas': {
            'file': 'wavelengths_salinas_corrected_204.csv',
            'column': 'wavelength_nm'
        },
        'indian_pines': {
            'file': 'indianpines_wavelengths_200.csv',
            'column': 'Value_1'
        }
    }
    
    if dataset_name not in csv_files:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = csv_files[dataset_name]
    
    # Get the CSV file path (in same directory as this script)
    current_dir = Path(__file__).parent
    csv_path = current_dir / config['file']
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Wavelength reference file not found: {csv_path}\n"
            f"Please ensure {config['file']} is in the CCARS_Thesis directory"
        )
    
    # Load wavelengths from CSV
    df = pd.read_csv(csv_path)
    
    if config['column'] not in df.columns:
        raise ValueError(
            f"Column '{config['column']}' not found in {config['file']}.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    wavelengths = df[config['column']].tolist()
    
    print(f"✓ Loaded {len(wavelengths)} wavelengths from {config['file']}")
    
    return wavelengths


if __name__ == '__main__':
    """Test wavelength loading"""
    print("Testing wavelength loading...\n")
    
    for dataset in ['salinas', 'indian_pines']:
        print(f"{dataset.upper()}:")
        wl = load_wavelengths_from_csv(dataset)
        print(f"  Total wavelengths: {len(wl)}")
        print(f"  Range: {min(wl):.2f} - {max(wl):.2f} nm")
        print(f"  First 5: {[f'{w:.2f}' for w in wl[:5]]}")
        print(f"  Last 5: {[f'{w:.2f}' for w in wl[-5:]]}")
        print()
