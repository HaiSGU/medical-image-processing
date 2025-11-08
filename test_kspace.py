"""Test script to check slice_kspace.npy file"""

import numpy as np
import os

file_path = r"d:\Documents\Medical Image Processing\data\medical\slice_kspace.npy"

print(f"📁 File: {os.path.basename(file_path)}")
print(f"💾 Size: {os.path.getsize(file_path):,} bytes")

# Test reading without pickle
print("\n🔬 Testing without pickle...")
try:
    data = np.load(file_path, allow_pickle=False)
    print(f"✅ SUCCESS without pickle!")
    print(f"   Shape: {data.shape}")
    print(f"   Dtype: {data.dtype}")
except Exception as e:
    print(f"❌ FAILED: {e}")

    # Test with pickle
    print("\n🔬 Testing WITH pickle...")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"✅ SUCCESS with pickle!")
        print(f"   Type: {type(data)}")
        print(f"   Is ndarray: {isinstance(data, np.ndarray)}")
        if isinstance(data, np.ndarray):
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
        else:
            print(f"   Content type: {type(data).__name__}")
    except Exception as e2:
        print(f"❌ FAILED even with pickle: {e2}")
