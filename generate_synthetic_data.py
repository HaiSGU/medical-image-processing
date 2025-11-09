"""
Script tạo synthetic data cho testing - KHÔNG CẦN Kaggle API
Tạo data cần thiết cho tất cả 5 trang của project
"""

import numpy as np
from pathlib import Path
from skimage.data import shepp_logan_phantom
from skimage.transform import radon
from scipy.fft import fft2, fftshift

print("=" * 70)
print("🎨 SYNTHETIC DATA GENERATOR - Medical Image Processing")
print("=" * 70)
print("\nTạo synthetic data để test project mà KHÔNG CẦN download từ Kaggle")
print("Bạn có thể test project ngay với data này!\n")

# Create folders
base_path = Path("data")
folders = {
    "ct": base_path / "synthetic_ct",
    "mri": base_path / "synthetic_mri",
    "preprocessing": base_path / "synthetic_preprocessing",
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)

# ============================================
# 1. ANONYMIZATION - Sử dụng data có sẵn
# ============================================
print("=" * 70)
print("1️⃣ ANONYMIZATION")
print("=" * 70)
existing_dicom = base_path / "anonym" / "dicom_dir"
if existing_dicom.exists():
    count = len(list(existing_dicom.glob("*.dcm")))
    print(f"✅ READY: Bạn đã có {count} DICOM files tại:")
    print(f"   {existing_dicom}")
else:
    print("⚠️  Không tìm thấy DICOM files")
    print("   Cần download từ Kaggle hoặc thêm file DICOM vào data/anonym/")

# ============================================
# 2. SEGMENTATION - Sử dụng data có sẵn
# ============================================
print("\n" + "=" * 70)
print("2️⃣ SEGMENTATION")
print("=" * 70)
existing_mri = base_path / "sitk"
if existing_mri.exists():
    mri_files = list(existing_mri.glob("*.nrrd")) + list(existing_mri.glob("*.mha"))
    print(f"✅ READY: Bạn đã có {len(mri_files)} MRI files tại:")
    print(f"   {existing_mri}")
    for f in mri_files[:3]:
        print(f"   - {f.name}")
else:
    print("⚠️  Không tìm thấy MRI files")

# ============================================
# 3. CT RECONSTRUCTION - Generate Phantom
# ============================================
print("\n" + "=" * 70)
print("3️⃣ CT RECONSTRUCTION - Generating Synthetic Data")
print("=" * 70)

try:
    ct_path = folders["ct"]

    # Generate Shepp-Logan phantom
    print("\n[1/5] Generating Shepp-Logan phantom...")
    phantom = shepp_logan_phantom()
    print(f"   ✅ Phantom shape: {phantom.shape}")

    # Generate sinogram with different angles
    print("\n[2/5] Generating full sinogram (180 angles)...")
    theta_180 = np.linspace(0.0, 180.0, 180, endpoint=False)
    sinogram_180 = radon(phantom, theta=theta_180, circle=True)
    print(f"   ✅ Sinogram shape: {sinogram_180.shape}")

    print("\n[3/5] Generating sparse sinogram (90 angles)...")
    theta_90 = np.linspace(0.0, 180.0, 90, endpoint=False)
    sinogram_90 = radon(phantom, theta=theta_90, circle=True)
    print(f"   ✅ Sinogram shape: {sinogram_90.shape}")

    print("\n[4/5] Generating very sparse sinogram (45 angles)...")
    theta_45 = np.linspace(0.0, 180.0, 45, endpoint=False)
    sinogram_45 = radon(phantom, theta=theta_45, circle=True)
    print(f"   ✅ Sinogram shape: {sinogram_45.shape}")

    print("\n[5/5] Generating limited angle sinogram (120 degrees)...")
    theta_120 = np.linspace(0.0, 120.0, 120, endpoint=False)
    sinogram_120 = radon(phantom, theta=theta_120, circle=True)
    print(f"   ✅ Sinogram shape: {sinogram_120.shape}")

    # Save all files
    print("\n💾 Saving files...")
    np.save(ct_path / "phantom_ground_truth.npy", phantom)
    np.save(ct_path / "sinogram_full_180angles.npy", sinogram_180)
    np.save(ct_path / "sinogram_sparse_90angles.npy", sinogram_90)
    np.save(ct_path / "sinogram_verysparse_45angles.npy", sinogram_45)
    np.save(ct_path / "sinogram_limited_120deg.npy", sinogram_120)

    # Save angle arrays
    np.save(ct_path / "angles_180.npy", theta_180)
    np.save(ct_path / "angles_90.npy", theta_90)
    np.save(ct_path / "angles_45.npy", theta_45)
    np.save(ct_path / "angles_120.npy", theta_120)

    print("✅ Saved 9 files to:", ct_path)

    # Create README
    readme = ct_path / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("CT RECONSTRUCTION SYNTHETIC DATA\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generated using Shepp-Logan phantom\n\n")
        f.write("Files:\n")
        f.write("------\n")
        f.write("1. phantom_ground_truth.npy\n")
        f.write("   - Ground truth image (400x400)\n")
        f.write("   - Use for comparison with reconstructed images\n\n")
        f.write("2. sinogram_full_180angles.npy\n")
        f.write("   - Full sampling: 180 projection angles\n")
        f.write("   - Best reconstruction quality\n\n")
        f.write("3. sinogram_sparse_90angles.npy\n")
        f.write("   - Sparse view: 90 projection angles\n")
        f.write("   - Test reconstruction with 50% data\n\n")
        f.write("4. sinogram_verysparse_45angles.npy\n")
        f.write("   - Very sparse: 45 projection angles\n")
        f.write("   - Test reconstruction with 25% data\n\n")
        f.write("5. sinogram_limited_120deg.npy\n")
        f.write("   - Limited angle: 0-120 degrees only\n")
        f.write("   - Test reconstruction with incomplete coverage\n\n")
        f.write("6. angles_*.npy\n")
        f.write("   - Corresponding projection angles for each sinogram\n\n")
        f.write("Usage:\n")
        f.write("------\n")
        f.write("Load sinogram and angles, then apply reconstruction:\n")
        f.write("- Filtered Back Projection (FBP)\n")
        f.write("- Algebraic Reconstruction Technique (ART)\n")
        f.write("- Simultaneous ART (SART)\n")

    print(f"📄 Created README: {readme}")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 4. MRI RECONSTRUCTION - Generate K-space
# ============================================
print("\n" + "=" * 70)
print("4️⃣ MRI RECONSTRUCTION - Generating Synthetic K-space")
print("=" * 70)

try:
    mri_path = folders["mri"]

    # Create synthetic brain-like images
    print("\n[1/3] Generating synthetic MRI images...")

    # Generate 5 different synthetic "MRI" images
    num_samples = 5
    image_size = 256

    for i in range(1, num_samples + 1):
        # Create synthetic brain-like pattern
        y, x = np.ogrid[-1 : 1 : image_size * 1j, -1 : 1 : image_size * 1j]

        # Create elliptical shape (brain-like)
        brain = ((x / 0.7) ** 2 + (y / 0.9) ** 2 < 1).astype(float)

        # Add some internal structures
        structure1 = ((x - 0.2) ** 2 + (y - 0.1) ** 2 < 0.1).astype(float) * 0.6
        structure2 = ((x + 0.2) ** 2 + (y + 0.1) ** 2 < 0.1).astype(float) * 0.6
        structure3 = ((x) ** 2 + (y - 0.3) ** 2 < 0.15).astype(float) * 0.8

        # Combine
        synthetic_mri = brain * 0.5 + structure1 + structure2 + structure3

        # Add some noise
        np.random.seed(i * 42)
        noise = np.random.normal(0, 0.01, (image_size, image_size))
        synthetic_mri = synthetic_mri + noise
        synthetic_mri = np.clip(synthetic_mri, 0, 1)

        print(f"   ✅ Sample {i}: {synthetic_mri.shape}")

        # Generate k-space (2D Fourier transform)
        kspace_full = fftshift(fft2(synthetic_mri))

        # Create undersampling masks
        center = image_size // 2

        # Mask 1: 50% sampling (center + random lines)
        mask_50 = np.zeros((image_size, image_size), dtype=bool)
        mask_50[center - 20 : center + 20, :] = True  # Center 40 lines
        random_lines = np.random.choice(image_size, image_size // 2, replace=False)
        mask_50[random_lines, :] = True
        kspace_50 = kspace_full * mask_50

        # Mask 2: 25% sampling (center + very sparse random)
        mask_25 = np.zeros((image_size, image_size), dtype=bool)
        mask_25[center - 10 : center + 10, :] = True  # Center 20 lines
        random_lines_25 = np.random.choice(image_size, image_size // 4, replace=False)
        mask_25[random_lines_25, :] = True
        kspace_25 = kspace_full * mask_25

        # Save files
        prefix = mri_path / f"sample_{i:02d}"
        np.save(f"{prefix}_original_image.npy", synthetic_mri)
        np.save(f"{prefix}_kspace_full.npy", kspace_full)
        np.save(f"{prefix}_kspace_50percent.npy", kspace_50)
        np.save(f"{prefix}_kspace_25percent.npy", kspace_25)
        np.save(f"{prefix}_mask_50percent.npy", mask_50)
        np.save(f"{prefix}_mask_25percent.npy", mask_25)

    print(f"\n✅ Generated {num_samples} synthetic MRI samples")
    print(f"💾 Saved to: {mri_path}")

    # Create README
    readme = mri_path / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("MRI K-SPACE RECONSTRUCTION SYNTHETIC DATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated {num_samples} synthetic MRI samples with k-space data\n\n")
        f.write("Files per sample (sample_XX_):\n")
        f.write("------\n")
        f.write("1. original_image.npy\n")
        f.write("   - Synthetic MRI image (256x256)\n")
        f.write("   - Ground truth for comparison\n\n")
        f.write("2. kspace_full.npy\n")
        f.write("   - Full k-space (100% sampling)\n")
        f.write("   - Complex-valued frequency domain data\n\n")
        f.write("3. kspace_50percent.npy\n")
        f.write("   - Undersampled k-space (50% data)\n")
        f.write("   - Center + random lines\n\n")
        f.write("4. kspace_25percent.npy\n")
        f.write("   - Heavily undersampled (25% data)\n")
        f.write("   - Center + sparse random lines\n\n")
        f.write("5. mask_50percent.npy & mask_25percent.npy\n")
        f.write("   - Binary sampling masks\n\n")
        f.write("Usage:\n")
        f.write("------\n")
        f.write("1. Load k-space data\n")
        f.write("2. Apply inverse FFT: ifft2(ifftshift(kspace))\n")
        f.write("3. Compare with original image\n")
        f.write("4. Test different reconstruction methods:\n")
        f.write("   - Zero-filling\n")
        f.write("   - Compressed sensing\n")
        f.write("   - Deep learning methods\n")

    print(f"📄 Created README: {readme}")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 5. PREPROCESSING - Generate test images
# ============================================
print("\n" + "=" * 70)
print("5️⃣ PREPROCESSING - Generating Test Images")
print("=" * 70)

try:
    prep_path = folders["preprocessing"]

    print("\n[1/1] Generating synthetic medical images...")

    # Generate different types of medical-like images
    image_size = 512
    num_images = 10

    for i in range(1, num_images + 1):
        y, x = np.ogrid[-1 : 1 : image_size * 1j, -1 : 1 : image_size * 1j]

        # Different patterns for each image
        if i <= 3:
            # Circular pattern (like X-ray)
            img = np.exp(-((x) ** 2 + (y) ** 2) / 0.5) * 255
        elif i <= 6:
            # Elliptical pattern (like CT)
            img = np.exp(-((x / 0.8) ** 2 + (y / 1.2) ** 2) / 0.3) * 255
        else:
            # Complex pattern (like MRI)
            img = (np.sin(x * 5) * np.cos(y * 5) + 1) * 127.5

        # Add noise
        np.random.seed(i * 10)
        noise = np.random.normal(0, 10, (image_size, image_size))
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Save
        np.save(prep_path / f"test_image_{i:02d}.npy", img)

    print(f"   ✅ Generated {num_images} test images ({image_size}x{image_size})")
    print(f"💾 Saved to: {prep_path}")

    # Create README
    readme = prep_path / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING TEST IMAGES\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated {num_images} synthetic medical-like images\n\n")
        f.write("Image types:\n")
        f.write("------\n")
        f.write("- test_image_01-03: Circular patterns (X-ray-like)\n")
        f.write("- test_image_04-06: Elliptical patterns (CT-like)\n")
        f.write("- test_image_07-10: Complex patterns (MRI-like)\n\n")
        f.write("All images: 512x512, grayscale (0-255)\n\n")
        f.write("Usage:\n")
        f.write("------\n")
        f.write("Test preprocessing operations:\n")
        f.write("- Normalization\n")
        f.write("- Denoising (Gaussian, Median filters)\n")
        f.write("- Resizing\n")
        f.write("- Contrast enhancement (CLAHE)\n")
        f.write("- Histogram equalization\n")

    print(f"📄 Created README: {readme}")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 70)

summary = f"""
📁 Data Structure:

{base_path}/
├── anonym/dicom_dir/          # ✅ EXISTING: DICOM files (Anonymization)
│
├── sitk/                      # ✅ EXISTING: Brain MRI (Segmentation)
│   ├── A1_grayT1.nrrd
│   ├── A1_grayT2.nrrd
│   └── training_001_mr_T1.mha
│
├── medical/                   # ✅ EXISTING: Original sinogram & k-space
│   ├── Schepp_Logan_sinogram 1.npy
│   └── slice_kspace.npy
│
├── synthetic_ct/              # 🆕 NEW: CT Reconstruction
│   ├── phantom_ground_truth.npy
│   ├── sinogram_full_180angles.npy
│   ├── sinogram_sparse_90angles.npy
│   ├── sinogram_verysparse_45angles.npy
│   ├── sinogram_limited_120deg.npy
│   └── angles_*.npy (4 files)
│
├── synthetic_mri/             # 🆕 NEW: MRI Reconstruction
│   ├── sample_01_*.npy (6 files per sample)
│   ├── sample_02_*.npy
│   ├── ... (5 samples total)
│   └── README.txt
│
└── synthetic_preprocessing/   # 🆕 NEW: Preprocessing tests
    ├── test_image_01.npy
    ├── test_image_02.npy
    ├── ... (10 images)
    └── README.txt

🎯 Ready to Use:
"""

print(summary)
print("   1️⃣  Anonymization:     data/anonym/dicom_dir/*.dcm")
print("   2️⃣  Segmentation:      data/sitk/*.nrrd, *.mha")
print("   3️⃣  CT Reconstruction: data/synthetic_ct/sinogram_*.npy")
print("   4️⃣  MRI Reconstruction: data/synthetic_mri/sample_*_kspace_*.npy")
print("   5️⃣  Preprocessing:     data/synthetic_preprocessing/test_image_*.npy")

print("\n💡 Next Steps:")
print("   1. Test mỗi trang với synthetic data")
print("   2. Chạy app: streamlit run app.py")
print("   3. (Optional) Setup Kaggle API để download real data:")
print("      - Xem hướng dẫn: KAGGLE_SETUP.md")
print("      - Chạy: python download_kaggle_data.py")

print("\n" + "=" * 70)
print("🎉 Done! Bạn có thể test project ngay bây giờ!")
print("=" * 70)
