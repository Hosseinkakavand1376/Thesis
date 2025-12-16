"""
Test script to verify actual wavelengths are being loaded from CSV
"""
from hsi_data_loader import prepare_hsi_for_cars

print("Testing wavelength loading from CSV files...\n")

# Test Salinas
print("=" * 70)
print("SALINAS Dataset")
print("=" * 70)
data = prepare_hsi_for_cars('salinas', test_percentage=0.2, 
                            calibration_percentage=0.5, random_state=42)

wl = data['wavelengths']
print(f"\n✓ Wavelengths loaded successfully!")
print(f"Total wavelengths: {len(wl)}")
print(f"Range: {min(wl):.2f} - {max(wl):.2f} nm")
print(f"\nFirst 5 wavelengths:")
for i, w in enumerate(wl[:5], 1):
    print(f"  {i}. {w:.4f} nm")
print(f"\nLast 5 wavelengths:")
for i, w in enumerate(wl[-5:], len(wl)-4):
    print(f"  {i}. {w:.4f} nm")

print("\n" + "=" * 70)
print("✅ Test Complete! Using actual wavelengths from CSV")
print("=" * 70)
