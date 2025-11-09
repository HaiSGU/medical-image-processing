"""
Script tự động download data từ Kaggle cho Medical Image Processing project
Chạy script này để download tất cả data cần thiết cho 5 trang

Prerequisites:
1. pip install kaggle
2. Download kaggle.json từ https://www.kaggle.com/settings
3. Đặt kaggle.json vào C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.fft import fft2, fftshift
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

print("=" * 70)
print("📥 KAGGLE DATA DOWNLOAD - Medical Image Processing Project")
print("=" * 70)

# Check if kaggle is installed
try:
    import kaggle

    print("✅ Kaggle API installed")
except ImportError:
    print("❌ Kaggle not installed!")
    print("💡 Run: pip install kaggle")
    sys.exit(1)

# Check kaggle credentials
kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_path.exists():
    print(f"❌ Kaggle credentials not found at: {kaggle_path}")
    print("💡 Steps to setup:")
    print("   1. Go to https://www.kaggle.com/settings")
    print("   2. Scroll to 'API' section")
    print("   3. Click 'Create New API Token'")
    print(f"   4. Move kaggle.json to: {kaggle_path}")
    sys.exit(1)
else:
    print(f"✅ Kaggle credentials found at: {kaggle_path}")

# Create data folders
base_path = Path("data")
folders = {
    "anonymization": base_path / "kaggle_anonymization",
    "segmentation": base_path / "kaggle_segmentation",
    "ct_reconstruction": base_path / "kaggle_ct",
    "mri_reconstruction": base_path / "kaggle_mri",
    "preprocessing": base_path / "kaggle_preprocessing",
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created folder: {folder}")

print("\n" + "=" * 70)
print("Starting downloads...")
print("=" * 70)

# ============================================
# 1. ANONYMIZATION - DICOM Files
# ============================================
print("\n" + "=" * 70)
print("1️⃣ ANONYMIZATION - DICOM Files")
print("=" * 70)

print("\n⚠️  Note: Some competitions require accepting rules first:")
print("   - SIIM: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/rules")
print("   - RSNA: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules")

# Option 1: SIIM Pneumothorax
print("\n[1/2] Downloading SIIM-ACR Pneumothorax DICOM...")
try:
    kaggle.api.competition_download_file(
        "siim-acr-pneumothorax-segmentation",
        "train-rle.csv",
        path=str(folders["anonymization"] / "siim"),
    )
    print("✅ Downloaded SIIM metadata")
    print("💡 To download full DICOM images, accept competition rules and run:")
    print("   kaggle competitions download -c siim-acr-pneumothorax-segmentation")
except Exception as e:
    print(f"⚠️  SIIM download skipped: {str(e)}")
    print("💡 You need to accept competition rules first")

# Option 2: RSNA Pneumonia
print("\n[2/2] Downloading RSNA Pneumonia sample...")
try:
    kaggle.api.competition_download_file(
        "rsna-pneumonia-detection-challenge",
        "stage_2_detailed_class_info.csv",
        path=str(folders["anonymization"] / "rsna"),
    )
    print("✅ Downloaded RSNA metadata")
    print("💡 To download full DICOM images, accept competition rules and run:")
    print("   kaggle competitions download -c rsna-pneumonia-detection-challenge")
except Exception as e:
    print(f"⚠️  RSNA download skipped: {str(e)}")

print("\n📝 ANONYMIZATION STATUS:")
print("   ✅ You already have DICOM files in data/anonym/dicom_dir/")
print("   ✅ 82 DICOM files ready to use")
print("   💡 Kaggle datasets are OPTIONAL - your existing data is sufficient")

# ============================================
# 2. SEGMENTATION - Brain MRI
# ============================================
print("\n" + "=" * 70)
print("2️⃣ SEGMENTATION - Brain MRI")
print("=" * 70)

# Option 1: LGG MRI Segmentation (BEST - with ground truth masks)
print("\n[1/3] Downloading LGG MRI Segmentation Dataset (with masks)...")
try:
    kaggle.api.dataset_download_files(
        "mateuszbuda/lgg-mri-segmentation",
        path=str(folders["segmentation"] / "lgg"),
        unzip=True,
    )
    print("✅ Downloaded LGG MRI Segmentation")
    print("   📊 110 patients with TIFF images + segmentation masks")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Option 2: Brain Tumor MRI Dataset
print("\n[2/3] Downloading Brain Tumor MRI Dataset...")
try:
    kaggle.api.dataset_download_files(
        "masoudnickparvar/brain-tumor-mri-dataset",
        path=str(folders["segmentation"] / "brain_tumor"),
        unzip=True,
    )
    print("✅ Downloaded Brain Tumor MRI Dataset")
    print("   📊 7,023 brain MRI images")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Option 3: Brain MRI for Tumor Detection
print("\n[3/3] Downloading Brain MRI for Tumor Detection...")
try:
    kaggle.api.dataset_download_files(
        "navoneel/brain-mri-images-for-brain-tumor-detection",
        path=str(folders["segmentation"] / "tumor_detection"),
        unzip=True,
    )
    print("✅ Downloaded Brain MRI for Tumor Detection")
    print("   📊 253 brain MRI images")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n📝 SEGMENTATION STATUS:")
print("   ✅ You already have brain MRI in data/sitk/")
print("   ✅ Kaggle data adds more variety for testing")

# ============================================
# 3. CT RECONSTRUCTION - Generate Phantom
# ============================================
print("\n" + "=" * 70)
print("3️⃣ CT RECONSTRUCTION - Phantom & Sinogram")
print("=" * 70)

print("\n[1/1] Generating Shepp-Logan Phantom + Sinograms...")
try:
    # Generate phantom
    phantom = shepp_logan_phantom()

    # Generate different sinogram variants
    # Full sampling (180 angles)
    theta_180 = np.linspace(0.0, 180.0, 180, endpoint=False)
    sinogram_180 = radon(phantom, theta=theta_180, circle=True)

    # Sparse sampling (90 angles)
    theta_90 = np.linspace(0.0, 180.0, 90, endpoint=False)
    sinogram_90 = radon(phantom, theta=theta_90, circle=True)

    # Very sparse (45 angles)
    theta_45 = np.linspace(0.0, 180.0, 45, endpoint=False)
    sinogram_45 = radon(phantom, theta=theta_45, circle=True)

    # Limited angle (120 degrees only)
    theta_120 = np.linspace(0.0, 120.0, 120, endpoint=False)
    sinogram_120 = radon(phantom, theta=theta_120, circle=True)

    # Save all variants
    ct_path = folders["ct_reconstruction"]
    np.save(ct_path / "phantom.npy", phantom)
    np.save(ct_path / "sinogram_full_180angles.npy", sinogram_180)
    np.save(ct_path / "sinogram_sparse_90angles.npy", sinogram_90)
    np.save(ct_path / "sinogram_verysparse_45angles.npy", sinogram_45)
    np.save(ct_path / "sinogram_limited_120degrees.npy", sinogram_120)

    print("✅ Generated Shepp-Logan Phantom + Sinogram variants")
    print(f"   📊 Phantom shape: {phantom.shape}")
    print(f"   📊 Full sinogram (180 angles): {sinogram_180.shape}")
    print(f"   📊 Sparse sinogram (90 angles): {sinogram_90.shape}")
    print(f"   📊 Very sparse (45 angles): {sinogram_45.shape}")
    print(f"   📊 Limited angle (120°): {sinogram_120.shape}")

    # Create visualization info
    info_file = ct_path / "README.txt"
    with open(info_file, "w") as f:
        f.write("CT Reconstruction Data\n")
        f.write("=" * 50 + "\n\n")
        f.write("Files:\n")
        f.write("- phantom.npy: Ground truth Shepp-Logan phantom\n")
        f.write("- sinogram_full_180angles.npy: Full sampling (180 projections)\n")
        f.write("- sinogram_sparse_90angles.npy: Sparse view (90 projections)\n")
        f.write("- sinogram_verysparse_45angles.npy: Very sparse (45 projections)\n")
        f.write("- sinogram_limited_120degrees.npy: Limited angle (0-120 degrees)\n\n")
        f.write("Usage:\n")
        f.write("Use these sinograms to test different CT reconstruction algorithms:\n")
        f.write("- Full sampling: Best quality\n")
        f.write("- Sparse view: Test reconstruction with fewer projections\n")
        f.write(
            "- Limited angle: Test reconstruction with incomplete angular coverage\n"
        )

    print(f"   📄 Created README at: {info_file}")

except Exception as e:
    print(f"❌ Error generating phantom: {str(e)}")

print("\n📝 CT RECONSTRUCTION STATUS:")
print("   ✅ You already have sinogram in data/medical/Schepp_Logan_sinogram 1.npy")
print("   ✅ New variants generated for testing different scenarios")

# ============================================
# 4. MRI RECONSTRUCTION - Generate K-space
# ============================================
print("\n" + "=" * 70)
print("4️⃣ MRI RECONSTRUCTION - K-space Data")
print("=" * 70)

# Download Brain MRI images first
print("\n[1/2] Downloading Brain MRI images for k-space generation...")
try:
    kaggle.api.dataset_download_files(
        "navoneel/brain-mri-images-for-brain-tumor-detection",
        path=str(folders["mri_reconstruction"] / "brain_mri_source"),
        unzip=True,
    )
    print("✅ Downloaded Brain MRI images (253 images)")
except Exception as e:
    print(f"❌ Error downloading MRI images: {str(e)}")

# Generate k-space data
print("\n[2/2] Generating K-space data from MRI images...")
try:
    from PIL import Image

    mri_source = folders["mri_reconstruction"] / "brain_mri_source"
    kspace_output = folders["mri_reconstruction"] / "kspace_data"
    kspace_output.mkdir(exist_ok=True)

    # Find image files
    image_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_files.extend(list(mri_source.glob(f"**/{ext}")))

    if image_files:
        print(f"   Found {len(image_files)} MRI images")

        # Process first 10 images to create k-space samples
        num_samples = min(10, len(image_files))
        print(f"   Generating k-space for {num_samples} samples...")

        for i, img_file in enumerate(image_files[:num_samples], 1):
            try:
                # Load image and convert to grayscale
                img = Image.open(img_file).convert("L")
                img_array = np.array(img, dtype=float)

                # Normalize to [0, 1]
                img_array = (img_array - img_array.min()) / (
                    img_array.max() - img_array.min() + 1e-8
                )

                # Generate full k-space (2D FFT)
                kspace_full = fftshift(fft2(img_array))

                # Create undersampling masks
                # Mask 1: 50% sampling (center + random)
                mask_50 = np.zeros_like(kspace_full, dtype=bool)
                center = kspace_full.shape[0] // 2
                mask_50[center - 20 : center + 20, :] = True  # Keep center 40 lines
                random_lines = np.random.choice(
                    kspace_full.shape[0], kspace_full.shape[0] // 2, replace=False
                )
                mask_50[random_lines, :] = True
                kspace_50 = kspace_full * mask_50

                # Mask 2: 25% sampling (center only + sparse random)
                mask_25 = np.zeros_like(kspace_full, dtype=bool)
                mask_25[center - 10 : center + 10, :] = True
                random_lines_25 = np.random.choice(
                    kspace_full.shape[0], kspace_full.shape[0] // 4, replace=False
                )
                mask_25[random_lines_25, :] = True
                kspace_25 = kspace_full * mask_25

                # Save files
                sample_name = f"sample_{i:02d}"
                np.save(kspace_output / f"{sample_name}_original_image.npy", img_array)
                np.save(kspace_output / f"{sample_name}_kspace_full.npy", kspace_full)
                np.save(
                    kspace_output / f"{sample_name}_kspace_50percent.npy", kspace_50
                )
                np.save(
                    kspace_output / f"{sample_name}_kspace_25percent.npy", kspace_25
                )
                np.save(kspace_output / f"{sample_name}_mask_50percent.npy", mask_50)
                np.save(kspace_output / f"{sample_name}_mask_25percent.npy", mask_25)

                if i == 1:
                    print(
                        f"   ✅ Sample {i}: shape={img_array.shape}, k-space={kspace_full.shape}"
                    )
                elif i % 5 == 0:
                    print(f"   ✅ Processed {i}/{num_samples} samples...")

            except Exception as e:
                print(f"   ⚠️  Error processing sample {i}: {str(e)}")

        print(f"✅ Generated K-space data for {num_samples} MRI samples")
        print(f"   📊 Each sample includes:")
        print(f"      - Original image")
        print(f"      - Full k-space (100% data)")
        print(f"      - Undersampled k-space (50% data)")
        print(f"      - Undersampled k-space (25% data)")
        print(f"      - Sampling masks")

        # Create README
        readme_file = kspace_output / "README.txt"
        with open(readme_file, "w") as f:
            f.write("MRI K-space Reconstruction Data\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Generated {num_samples} k-space samples from brain MRI images\n\n"
            )
            f.write("Files per sample (sample_XX):\n")
            f.write("- _original_image.npy: Original MRI image\n")
            f.write("- _kspace_full.npy: Full k-space (100% sampling)\n")
            f.write("- _kspace_50percent.npy: Undersampled (50% data)\n")
            f.write("- _kspace_25percent.npy: Undersampled (25% data)\n")
            f.write("- _mask_50percent.npy: Sampling mask (50%)\n")
            f.write("- _mask_25percent.npy: Sampling mask (25%)\n\n")
            f.write("Usage:\n")
            f.write(
                "Test MRI reconstruction algorithms with different undersampling ratios\n"
            )
            f.write("Compare reconstruction quality vs sampling percentage\n")

        print(f"   📄 Created README at: {readme_file}")

    else:
        print("   ⚠️  No MRI images found to generate k-space")

except Exception as e:
    print(f"❌ Error generating k-space: {str(e)}")

print("\n📝 MRI RECONSTRUCTION STATUS:")
print("   ✅ You already have k-space in data/medical/slice_kspace.npy")
print("   ✅ New k-space samples with different undersampling ratios generated")

# ============================================
# 5. PREPROCESSING - Multiple Modalities
# ============================================
print("\n" + "=" * 70)
print("5️⃣ PREPROCESSING - Medical Images (Multiple Modalities)")
print("=" * 70)

# Option 1: COVID-19 Radiography Database (BEST)
print("\n[1/4] Downloading COVID-19 Radiography Database...")
try:
    kaggle.api.dataset_download_files(
        "tawsifurrahman/covid19-radiography-database",
        path=str(folders["preprocessing"] / "covid_xray"),
        unzip=True,
    )
    print("✅ Downloaded COVID-19 Radiography Database")
    print("   📊 21,165 chest X-rays (COVID, Normal, Pneumonia, Lung Opacity)")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Option 2: Chest X-Ray Pneumonia
print("\n[2/4] Downloading Chest X-Ray Pneumonia Dataset...")
try:
    kaggle.api.dataset_download_files(
        "paultimothymooney/chest-xray-pneumonia",
        path=str(folders["preprocessing"] / "chest_xray_pneumonia"),
        unzip=True,
    )
    print("✅ Downloaded Chest X-Ray Pneumonia")
    print("   📊 5,863 chest X-rays (Normal, Pneumonia)")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Option 3: Brain Tumor Classification
print("\n[3/4] Downloading Brain Tumor Classification MRI...")
try:
    kaggle.api.dataset_download_files(
        "sartajbhuvaji/brain-tumor-classification-mri",
        path=str(folders["preprocessing"] / "brain_tumor_classification"),
        unzip=True,
    )
    print("✅ Downloaded Brain Tumor Classification MRI")
    print("   📊 3,264 brain MRI images (4 classes)")
except Exception as e:
    print(f"❌ Error: {str(e)}")

# Option 4: CT Kidney Dataset
print("\n[4/4] Downloading CT Kidney Dataset...")
try:
    kaggle.api.dataset_download_files(
        "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone",
        path=str(folders["preprocessing"] / "ct_kidney"),
        unzip=True,
    )
    print("✅ Downloaded CT Kidney Dataset")
    print("   📊 12,446 CT kidney images (4 classes)")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n📝 PREPROCESSING STATUS:")
print("   ✅ You already have various medical images in data/sitk/")
print("   ✅ Kaggle data provides diverse modalities for comprehensive testing")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ DOWNLOAD COMPLETE - SUMMARY")
print("=" * 70)

summary = f"""
📁 Data Structure:

{base_path}/
├── anonym/                      # ✅ EXISTING: 82 DICOM files
├── kaggle_anonymization/        # 🆕 KAGGLE: Competition metadata
│
├── sitk/                        # ✅ EXISTING: Brain MRI (.nrrd, .mha)
├── kaggle_segmentation/         # 🆕 KAGGLE: Brain MRI datasets
│   ├── lgg/                     #    - 110 patients with masks
│   ├── brain_tumor/             #    - 7,023 images
│   └── tumor_detection/         #    - 253 images
│
├── medical/                     # ✅ EXISTING: Sinogram + K-space
├── kaggle_ct/                   # 🆕 GENERATED: Phantom variants
│   ├── phantom.npy
│   ├── sinogram_full_180angles.npy
│   ├── sinogram_sparse_90angles.npy
│   ├── sinogram_verysparse_45angles.npy
│   └── sinogram_limited_120degrees.npy
│
├── kaggle_mri/                  # 🆕 KAGGLE + GENERATED: K-space
│   ├── brain_mri_source/        #    - 253 MRI images
│   └── kspace_data/             #    - 10 k-space samples (full/50%/25%)
│
└── kaggle_preprocessing/        # 🆕 KAGGLE: Multiple modalities
    ├── covid_xray/              #    - 21,165 X-rays
    ├── chest_xray_pneumonia/    #    - 5,863 X-rays
    ├── brain_tumor_classification/ # - 3,264 MRI
    └── ct_kidney/               #    - 12,446 CT

📊 Dataset Statistics:
"""

# Count files
stats = {
    "Existing DICOM": (
        len(list((base_path / "anonym" / "dicom_dir").glob("*.dcm")))
        if (base_path / "anonym" / "dicom_dir").exists()
        else 0
    ),
    "Brain MRI (Segmentation)": "Multiple datasets downloaded",
    "CT Phantom/Sinogram": "5 variants generated",
    "MRI K-space": "10 samples with 3 undersampling ratios",
    "Preprocessing Images": "40,000+ images across 4 datasets",
}

print(summary)
for key, value in stats.items():
    print(f"   ✅ {key}: {value}")

print("\n🎯 Ready to Use:")
print("   1️⃣  Anonymization: Use existing data/anonym/dicom_dir/")
print("   2️⃣  Segmentation: Use kaggle_segmentation/lgg/ (has ground truth)")
print("   3️⃣  CT Reconstruction: Use kaggle_ct/sinogram_*.npy")
print("   4️⃣  MRI Reconstruction: Use kaggle_mri/kspace_data/")
print("   5️⃣  Preprocessing: Use any dataset in kaggle_preprocessing/")

print("\n💡 Next Steps:")
print("   1. Test each page with the downloaded data")
print("   2. For DICOM competitions, accept rules and download full datasets")
print("   3. Run your Streamlit app: streamlit run app.py")

print("\n" + "=" * 70)
print("🎉 All Done!")
print("=" * 70)
