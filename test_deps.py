"""Quick Test - Verify dependencies"""

import sys

print(f"Python: {sys.version}")

try:
    import numpy as np

    print("✅ numpy")
except:
    print("❌ numpy missing")

try:
    import pydicom

    print("✅ pydicom")
except:
    print("❌ pydicom missing")

try:
    import nibabel as nib

    print("✅ nibabel")
except:
    print("❌ nibabel missing")

try:
    import SimpleITK as sitk

    print("✅ SimpleITK")
except:
    print("❌ SimpleITK missing")

try:
    from skimage.transform import radon

    print("✅ scikit-image")
except:
    print("❌ scikit-image missing")

print("\nAll checks complete!")
