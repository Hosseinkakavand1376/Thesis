"""
HSI Dataset Configuration and Download Utilities
Provides metadata and download functions for Salinas and Indian Pines datasets
"""

import os
import urllib.request
from pathlib import Path
import scipy.io as sio

# Base directory for datasets
DATASET_DIR = Path(__file__).parent / "dataset"

# Dataset configurations
HSI_DATASETS = {
    'salinas': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'gt_url': 'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
        'data_file': 'Salinas_corrected.mat',
        'gt_file': 'Salinas_gt.mat',
        'data_key': 'salinas_corrected',  # Key in .mat file
        'gt_key': 'salinas_gt',
        'n_classes': 16,
        'n_bands': 204,
        'spatial_size': (512, 217),
        'wavelength_range': (360, 2500),  # nm (approximate)
        'class_names': [
            'Brocoli_green_weeds_1',
            'Brocoli_green_weeds_2',
            'Fallow',
            'Fallow_rough_plow',
            'Fallow_smooth',
            'Stubble',
            'Celery',
            'Grapes_untrained',
            'Soil_vinyard_develop',
            'Corn_senesced_green_weeds',
            'Lettuce_romaine_4wk',
            'Lettuce_romaine_5wk',
            'Lettuce_romaine_6wk',
            'Lettuce_romaine_7wk',
            'Vinyard_untrained',
            'Vinyard_vertical_trellis'
        ]
    },
    'indian_pines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url': 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_file': 'Indian_pines_corrected.mat',
        'gt_file': 'Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key': 'indian_pines_gt',
        'n_classes': 16,
        'n_bands': 200,
        'spatial_size': (145, 145),
        'wavelength_range': (400, 2500),  # nm (approximate)
        'class_names': [
            'Alfalfa',
            'Corn-notill',
            'Corn-mintill',
            'Corn',
            'Grass-pasture',
            'Grass-trees',
            'Grass-pasture-mowed',
            'Hay-windrowed',
            'Oats',
            'Soybean-notill',
            'Soybean-mintill',
            'Soybean-clean',
            'Wheat',
            'Woods',
            'Buildings-Grass-Trees-Drives',
            'Stone-Steel-Towers'
        ]
    }
}


def download_file(url, dest_path, force=False, max_retries=3):
    """
    Download a file from URL to destination path with retry logic
    
    Args:
        url: Source URL
        dest_path: Destination file path
        force: If True, re-download even if file exists
        max_retries: Maximum download attempts
    
    Returns:
        Path to downloaded file
    """
    import urllib.request
    import socket
    
    dest_path = Path(dest_path)
    
    if dest_path.exists() and not force:
        print(f"✓ File already exists: {dest_path.name}")
        return dest_path
    
    print(f"Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set timeout for socket operations
    socket.setdefaulttimeout(120)  # 2 minutes
    
    for attempt in range(max_retries):
        try:
            # Use urlretrieve with a custom reporthook for progress
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 / total_size)
                    if block_num % 100 == 0:  # Print every 100 blocks
                        print(f"  Progress: {percent:.1f}%", end='\r')
            
            urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
            print(f"\n✓ Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return dest_path
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠ Download attempt {attempt + 1} failed: {e}")
                print(f"  Retrying ({attempt + 2}/{max_retries})...")
                if dest_path.exists():
                    dest_path.unlink()  # Remove partial download
            else:
                print(f"\n✗ Error downloading {url} after {max_retries} attempts: {e}")
                print(f"\n💡 Manual download instructions:")
                print(f"   1. Visit: {url}")
                print(f"   2. Download manually to: {dest_path}")
                print(f"   3. Run the pipeline again")
                raise


def download_dataset(dataset_name, force=False):
    """
    Download both data and ground truth files for a dataset
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        force: If True, re-download even if files exist
    
    Returns:
        Tuple of (data_path, gt_path)
    """
    if dataset_name not in HSI_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(HSI_DATASETS.keys())}")
    
    config = HSI_DATASETS[dataset_name]
    
    # Download data file
    data_path = DATASET_DIR / config['data_file']
    download_file(config['data_url'], data_path, force=force)
    
    # Download ground truth file
    gt_path = DATASET_DIR / config['gt_file']
    download_file(config['gt_url'], gt_path, force=force)
    
    return data_path, gt_path


def load_mat_file(file_path, key):
    """
    Load .mat file and extract data using the specified key
    
    Args:
        file_path: Path to .mat file
        key: Key to extract from the .mat file
    
    Returns:
        Numpy array
    """
    try:
        mat_data = sio.loadmat(file_path)
        if key in mat_data:
            return mat_data[key]
        else:
            # Try to find the key automatically
            possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(possible_keys) == 1:
                print(f"Warning: Key '{key}' not found. Using '{possible_keys[0]}' instead.")
                return mat_data[possible_keys[0]]
            else:
                raise KeyError(f"Key '{key}' not found in .mat file. Available keys: {possible_keys}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def get_dataset_config(dataset_name):
    """
    Get configuration dictionary for a dataset
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
    
    Returns:
        Configuration dictionary
    """
    if dataset_name not in HSI_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(HSI_DATASETS.keys())}")
    return HSI_DATASETS[dataset_name]


def generate_wavelengths(n_bands, wavelength_range):
    """
    Generate wavelength values for spectral bands
    
    Args:
        n_bands: Number of spectral bands
        wavelength_range: Tuple of (min_wavelength, max_wavelength) in nm
    
    Returns:
        List of wavelength values
    """
    import numpy as np
    min_wl, max_wl = wavelength_range
    wavelengths = np.linspace(min_wl, max_wl, n_bands)
    return wavelengths.tolist()


if __name__ == '__main__':
    """Test dataset download"""
    print("=" * 60)
    print("HSI Dataset Downloader")
    print("=" * 60)
    
    for dataset_name in ['salinas', 'indian_pines']:
        print(f"\n📦 Downloading {dataset_name.upper()} dataset...")
        try:
            data_path, gt_path = download_dataset(dataset_name)
            
            # Verify files
            config = get_dataset_config(dataset_name)
            data = load_mat_file(data_path, config['data_key'])
            gt = load_mat_file(gt_path, config['gt_key'])
            
            print(f"  Data shape: {data.shape}")
            print(f"  GT shape: {gt.shape}")
            print(f"  Expected: {config['spatial_size']} × {config['n_bands']} bands")
            print(f"  Classes: {config['n_classes']}")
            print(f"  ✅ {dataset_name.upper()} ready!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
