"""
HSI Data Loader - Transform 3D HSI cube to 2D format for CCARS

This module bridges the gap between hyperspectral image format and 
Nicola's CARS expected input format.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from hsi_config import (
    download_dataset, 
    load_mat_file, 
    get_dataset_config,
    generate_wavelengths
)


def load_hsi_dataset(dataset_name='salinas', auto_download=True):
    """
    Load HSI dataset from .mat files
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        auto_download: If True, download files if not present
    
    Returns:
        data_cube: (H, W, Bands) numpy array
        ground_truth: (H, W) numpy array with class labels
        config: Dataset configuration dictionary
    """
    config = get_dataset_config(dataset_name)
    
    # Download if needed
    if auto_download:
        data_path, gt_path = download_dataset(dataset_name)
    else:
        from hsi_config import DATASET_DIR
        data_path = DATASET_DIR / config['data_file']
        gt_path = DATASET_DIR / config['gt_file']
    
    # Load .mat files
    data_cube = load_mat_file(data_path, config['data_key'])
    ground_truth = load_mat_file(gt_path, config['gt_key'])
    
    print(f"✓ Loaded {dataset_name.upper()}")
    print(f"  Data shape: {data_cube.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  Number of classes: {config['n_classes']}")
    
    return data_cube, ground_truth, config


def flatten_hsi_to_2d(data_cube, ground_truth, remove_background=True):
    """
    Convert 3D HSI cube to 2D matrix format
    
    Args:
        data_cube: (H, W, Bands) array
        ground_truth: (H, W) array with class labels
        remove_background: If True, remove pixels with label == 0
    
    Returns:
        X: (n_samples, n_bands) array - spectral signatures
        y: (n_samples,) array - class labels
        valid_indices: (n_samples, 2) array - spatial coordinates (h, w)
    """
    H, W, Bands = data_cube.shape
    
    # Reshape to 2D: (H*W, Bands)
    X_flat = data_cube.reshape(-1, Bands)
    y_flat = ground_truth.reshape(-1)
    
    # Create spatial indices
    h_indices, w_indices = np.meshgrid(range(H), range(W), indexing='ij')
    spatial_indices = np.column_stack([h_indices.ravel(), w_indices.ravel()])
    
    # Remove background (label 0) if requested
    if remove_background:
        valid_mask = y_flat > 0
        X = X_flat[valid_mask]
        y = y_flat[valid_mask]
        valid_indices = spatial_indices[valid_mask]
        
        print(f"✓ Removed background pixels")
        print(f"  Total pixels: {H * W}")
        print(f"  Valid pixels: {len(y)} ({100 * len(y) / (H * W):.1f}%)")
        print(f"  Background pixels removed: {np.sum(~valid_mask)}")
    else:
        X = X_flat
        y = y_flat
        valid_indices = spatial_indices
    
    # Adjust labels to be 0-indexed (1-16 → 0-15)
    y = y - 1
    
    return X, y, valid_indices


def create_dataframe_format(X, y, wavelengths, spatial_indices=None):
    """
    Create pandas DataFrame format matching Nicola's structure
    
    Args:
        X: (n_samples, n_bands) array
        y: (n_samples,) array - class labels (0-indexed)
        wavelengths: List of wavelength values
        spatial_indices: Optional (n_samples, 2) array of (h, w) coordinates
    
    Returns:
        X_df: DataFrame with MultiIndex=['Class', 'Sample_ID'], columns=wavelengths
    """
    n_samples, n_bands = X.shape
    
    # Create sample IDs
    sample_ids = np.arange(n_samples)
    
    # Create multi-index
    index_tuples = list(zip(y, sample_ids))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Class', 'Sample_ID'])
    
    # Create DataFrame
    X_df = pd.DataFrame(X, index=multi_index, columns=wavelengths)
    
    print(f"✓ Created DataFrame")
    print(f"  Shape: {X_df.shape}")
    print(f"  Index levels: {X_df.index.names}")
    print(f"  Class distribution:")
    class_counts = X_df.index.get_level_values('Class').value_counts().sort_index()
    for class_id, count in class_counts.items():
        print(f"    Class {class_id}: {count} samples")
    
    return X_df


def stratified_split(X_df, test_size=0.2, random_state=42):
    """
    Perform stratified train/test split maintaining class distribution
    
    Args:
        X_df: DataFrame with MultiIndex containing 'Class' level
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        X_train_df: Training DataFrame
        X_test_df: Test DataFrame
    """
    # Extract class labels from multi-index
    classes = X_df.index.get_level_values('Class').values
    
    # Stratified split by class
    train_indices, test_indices = train_test_split(
        np.arange(len(X_df)),
        test_size=test_size,
        stratify=classes,
        random_state=random_state
    )
    
    X_train_df = X_df.iloc[train_indices].copy()
    X_test_df = X_df.iloc[test_indices].copy()
    
    print(f"✓ Stratified split: {len(X_train_df)} train, {len(X_test_df)} test")
    
    return X_train_df, X_test_df


def prepare_hsi_for_cars(dataset_name='salinas', 
                         test_percentage=0.2, 
                         calibration_percentage=0.5,
                         random_state=42,
                         auto_download=True):
    """
    Complete pipeline to prepare HSI data for CARS algorithm
    
    Following Nicola's exact data splitting strategy:
    - If calibration=True: Split 50/50 into calibration and hold-out sets
    - Each set is then split into train/test with test_percentage
    
    Args:
        dataset_name: 'salinas' or 'indian_pines'
        test_percentage: Fraction for test split (default 0.2 = 20%)
        calibration_percentage: Fraction for calibration split (default 0.5 = 50%)
        random_state: Random seed for reproducibility
        auto_download: Auto-download dataset if not present
    
    Returns:
        Dictionary with:
            - X_cal_train_df: Calibration training set
            - X_cal_test_df: Calibration test set
            - X_holdout_train_df: Hold-out training set
            - X_holdout_test_df: Hold-out test set
            - config: Dataset configuration
            - wavelengths: List of wavelength values
    """
    print("=" * 70)
    print(f"Preparing {dataset_name.upper()} for CCARS")
    print("=" * 70)
    
    # Step 1: Load dataset
    data_cube, ground_truth, config = load_hsi_dataset(dataset_name, auto_download)
    
    # Step 2: Flatten to 2D
    X, y, spatial_indices = flatten_hsi_to_2d(data_cube, ground_truth, remove_background=True)
    
    # Step 3: Load actual wavelength values from CSV reference files
    from wavelength_loader import load_wavelengths_from_csv
    wavelengths = load_wavelengths_from_csv(dataset_name)
    
    # Step 4: Create DataFrame format
    X_df = create_dataframe_format(X, y, wavelengths, spatial_indices)
    
    # Step 5: Calibration split (50/50)
    print(f"\n📊 Performing calibration split ({calibration_percentage:.0%} calibration)...")
    X_cal_df, X_holdout_df = stratified_split(X_df, test_size=1 - calibration_percentage, 
                                                random_state=random_state)
    
    # Step 6: Train/test split for calibration set
    print(f"\n📊 Splitting calibration set ({test_percentage:.0%} test)...")
    X_cal_train_df, X_cal_test_df = stratified_split(X_cal_df, test_size=test_percentage, 
                                                       random_state=random_state + 1)
    
    # Step 7: Train/test split for hold-out set
    print(f"\n📊 Splitting hold-out set ({test_percentage:.0%} test)...")
    X_holdout_train_df, X_holdout_test_df = stratified_split(X_holdout_df, test_size=test_percentage, 
                                                               random_state=random_state + 2)
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    print(f"Calibration train: {len(X_cal_train_df)} samples")
    print(f"Calibration test:  {len(X_cal_test_df)} samples")
    print(f"Hold-out train:    {len(X_holdout_train_df)} samples")
    print(f"Hold-out test:     {len(X_holdout_test_df)} samples")
    print(f"Total:             {len(X_df)} samples")
    print("=" * 70)
    
    return {
        'X_cal_train_df': X_cal_train_df,
        'X_cal_test_df': X_cal_test_df,
        'X_holdout_train_df': X_holdout_train_df,
        'X_holdout_test_df': X_holdout_test_df,
        'X_full_df': X_df,
        'config': config,
        'wavelengths': wavelengths,
        'spatial_indices': spatial_indices
    }


if __name__ == '__main__':
    """Test data loading and preparation"""
    print("\n" + "=" * 70)
    print("Testing HSI Data Loader")
    print("=" * 70)
    
    # Test both datasets
    for dataset_name in ['salinas', 'indian_pines']:
        print(f"\n{'=' * 70}")
        print(f"Testing {dataset_name.upper()}")
        print('=' * 70)
        
        try:
            data = prepare_hsi_for_cars(
                dataset_name=dataset_name,
                test_percentage=0.2,
                calibration_percentage=0.5,
                random_state=42
            )
            
            # Verify data integrity
            print(f"\n✓ Verification:")
            print(f"  Wavelengths: {len(data['wavelengths'])} bands")
            print(f"  First wavelength: {data['wavelengths'][0]:.2f} nm")
            print(f"  Last wavelength: {data['wavelengths'][-1]:.2f} nm")
            print(f"  Config classes: {data['config']['n_classes']}")
            
            # Check class distribution in calibration train set
            cal_train_classes = data['X_cal_train_df'].index.get_level_values('Class').unique()
            print(f"  Classes in calibration train: {len(cal_train_classes)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Data loader test complete!")
    print("=" * 70)
