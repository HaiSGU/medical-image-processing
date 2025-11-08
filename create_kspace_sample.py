"""Create a valid K-space sample for MRI reconstruction testing"""

import numpy as np
import matplotlib.pyplot as plt

# Create a simple phantom image (64x64)
size = 64
phantom = np.zeros((size, size), dtype=np.complex128)

# Add some features (circles and rectangles)
y, x = np.ogrid[:size, :size]
center_y, center_x = size // 2, size // 2

# Large circle
mask1 = (x - center_x) ** 2 + (y - center_y) ** 2 <= (size // 3) ** 2
phantom[mask1] = 1.0

# Small bright circle
mask2 = (x - center_x + 10) ** 2 + (y - center_y - 10) ** 2 <= (size // 8) ** 2
phantom[mask2] = 1.5

# Small dark circle
mask3 = (x - center_x - 10) ** 2 + (y - center_y + 10) ** 2 <= (size // 8) ** 2
phantom[mask3] = 0.5

# Generate K-space by 2D FFT
kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phantom)))

print(f"✅ Created K-space data")
print(f"   Shape: {kspace.shape}")
print(f"   Dtype: {kspace.dtype}")
print(f"   Size: {kspace.nbytes:,} bytes")

# Save the K-space
output_path = r"d:\Documents\Medical Image Processing\data\medical\slice_kspace.npy"
np.save(output_path, kspace)
print(f"\n💾 Saved to: {output_path}")

# Verify it can be loaded
kspace_loaded = np.load(output_path)
print(f"\n✅ Verification: Successfully loaded")
print(f"   Shape: {kspace_loaded.shape}")
print(f"   Dtype: {kspace_loaded.dtype}")

# Test reconstruction
reconstructed = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_loaded)))
magnitude = np.abs(reconstructed)
print(f"\n✅ Test reconstruction successful")
print(f"   Magnitude shape: {magnitude.shape}")
print(f"   Value range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
