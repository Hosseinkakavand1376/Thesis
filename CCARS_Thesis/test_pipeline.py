# Test script to isolate the error
from hsi_data_loader import prepare_hsi_for_cars
from hsi_preprocessing import preprocess_hsi_data
import numpy as np

print("Step 1: Load data")
data = prepare_hsi_for_cars(
    dataset_name='salinas',
    test_percentage=0.2,
    calibration_percentage=0.5,
    random_state=42,
    auto_download=True
)

print("\nStep 2: Extract data")
X_train_df = data['X_cal_train_df']
X_test_df = data['X_cal_test_df']
wavelengths = data['wavelengths']

print(f"X_train shape: {X_train_df.shape}")
print(f"Wavelengths shape: {len(wavelengths) if hasattr(wavelengths, '__len__') else 'scalar'}")
print(f"Wavelengths type: {type(wavelengths)}")

print("\nStep 3: Preprocess train")
X_train_preprocessed = preprocess_hsi_data(X_train_df, apply_log=True, apply_snv=True)

print(f"\nPreprocessed train shape: {X_train_preprocessed.shape}")
print(f"Has NaN: {X_train_preprocessed.isna().any().any()}")
print(f"Has Inf: {np.isinf(X_train_preprocessed.values).any()}")

print("\nStep 4: Convert to numpy")
X_train = X_train_preprocessed.values
y_train = X_train_preprocessed.index.get_level_values('Class').values

print(f"X_train type: {X_train.dtype}")
print(f"y_train type: {y_train.dtype}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train has NaN: {np.isnan(X_train).any()}")
print(f"X_train has Inf: {np.isinf(X_train).any()}")

print("\n✓ All steps successful!")
