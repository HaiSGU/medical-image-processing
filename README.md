# Medical Image Processing System

A comprehensive Python toolkit and web application for medical image processing, featuring file I/O, anonymization, reconstruction, segmentation, and preprocessing capabilities.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Application](#web-application)
- [Python Library Usage](#python-library-usage)
- [Project Structure](#project-structure)
- [Supported Formats](#supported-formats)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides both a **Python library** and an **interactive web application** for medical image processing. Built for researchers, students, and healthcare professionals working with medical imaging data.

**Key Capabilities:**
- Multi-format medical image I/O (NIfTI, DICOM, NRRD, MetaImage, NumPy)
- DICOM anonymization (PHI removal)
- Brain segmentation (4 methods)
- CT reconstruction from sinograms (FBP, SART)
- MRI K-space reconstruction
- Image preprocessing pipeline
- Interactive web interface

## Features

### File I/O & Visualization
- Read/write NIfTI, DICOM, NRRD, MetaImage, NumPy formats
- 2D slice viewing with navigation
- 3D visualization with Plotly
- Metadata extraction and display

### DICOM Anonymization
- Remove Protected Health Information (PHI)
- Batch processing support
- Customizable patient ID mapping
- ZIP export for anonymized files

### Image Segmentation
- **Threshold-based** segmentation
- **Otsu's method** automatic thresholding
- **Region growing** with seed points
- **Automatic** brain extraction pipeline

### CT Reconstruction
- **Filtered Backprojection (FBP)** with 4 filters:
  - Ram-Lak (standard)
  - Shepp-Logan (smooth)
  - Cosine (smoother)
  - Hamming (smoothest)
- **SART (Simultaneous Algebraic Reconstruction Technique)**
- Shepp-Logan phantom generation
- Quality metrics (PSNR, SSIM)

### MRI Reconstruction
- K-space to image domain conversion
- Magnitude and phase extraction
- Partial Fourier reconstruction
- Interactive visualization

### Image Preprocessing
- **Intensity normalization**: Min-Max, Z-Score, Percentile clipping
- **Spatial transforms**: Resize, Crop, Pad
- **Denoising**: Gaussian blur, Median filter
- **Contrast enhancement**: Histogram Equalization, CLAHE, Gamma correction
- **Augmentation**: Flip, Rotate, Noise injection
- Pipeline builder with JSON export

## Demo

**Web Application:** [Coming Soon - Streamlit Cloud]

**Screenshots:**

```
[Main Page]          [Anonymization]       [Segmentation]
File Upload      →   DICOM Processing  →   Brain Extraction
   +                      +                      +
Preview              Before/After           3D Visualization
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Install from GitHub

```bash
# Clone repository
git clone https://github.com/HaiSGU/medical-image-processing.git
cd medical-image-processing

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `streamlit` - Web application framework
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `plotly` - Interactive 3D plots
- `SimpleITK` - Medical image processing
- `pydicom` - DICOM file handling
- `nibabel` - NIfTI file handling
- `scikit-image` - Image processing algorithms
- `scipy` - Scientific computing

## Quick Start

### Launch Web Application

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

### Use as Python Library

```python
from utils.file_io import MedicalImageIO
from src.segmentation.brain_segmentation import BrainSegmentation

# Load medical image
io_handler = MedicalImageIO()
image, metadata = io_handler.read_image("brain_mri.nii.gz")

# Segment brain
segmenter = BrainSegmentation()
mask = segmenter.segment_brain(image, method='auto')

# Save result
io_handler.save_image(mask, "brain_mask.nii.gz", metadata)
```

## Web Application

### Pages Overview

**1. File Upload & Preview**
- Upload medical images (all supported formats)
- View 2D slices with navigation
- Display metadata and statistics
- 3D visualization for volumetric data

**2. DICOM Anonymization**
- Batch upload DICOM files
- Remove patient information
- Preview before/after metadata
- Download anonymized files as ZIP

**3. Brain Segmentation**
- Upload brain MRI
- Choose segmentation method
- Adjust parameters interactively
- View segmentation overlay
- Export mask

**4. CT Reconstruction**
- Load sinogram data
- Select reconstruction algorithm
- Adjust parameters (filter type, iterations)
- Compare with ground truth
- View quality metrics

**5. MRI Reconstruction**
- Upload K-space data
- Perform inverse FFT
- Extract magnitude/phase
- Partial Fourier reconstruction
- Visualize results

**6. Image Preprocessing**
- Apply normalization
- Spatial transformations
- Denoising filters
- Contrast enhancement
- Build processing pipeline
- Export processed image + config

### Running the Web App

```bash
# Start app
streamlit run app.py

# Access at
http://localhost:8501

# Stop app
Ctrl + C
```

## Python Library Usage

### File I/O

```python
from utils.file_io import MedicalImageIO

io = MedicalImageIO()

# Read image
image_data, metadata = io.read_image("scan.nii.gz")
print(f"Shape: {metadata['shape']}")
print(f"Spacing: {metadata['spacing']}")

# Write image
io.save_image(image_data, "output.nrrd", metadata)
```

### DICOM Anonymization

```python
from src.anonymization.dicom_anonymizer import DICOMAnonymizer

anonymizer = DICOMAnonymizer(patient_prefix="ANON")

# Anonymize single file
mapping = anonymizer.anonymize_file("input.dcm", "output.dcm")

# Anonymize directory
mappings = anonymizer.anonymize_directory(
    input_dir="dicom_files/",
    output_dir="anonymized/"
)
```

### Brain Segmentation

```python
from src.segmentation.brain_segmentation import BrainSegmentation

seg = BrainSegmentation()

# Automatic segmentation
mask = seg.segment_brain(image, method='auto')

# Threshold method
mask = seg.threshold_segmentation(image, threshold=50)

# Otsu method
mask = seg.otsu_segmentation(image)

# Region growing
mask = seg.region_growing(image, seed_point=(128, 128, 64))
```

### CT Reconstruction

```python
from src.reconstruction.ct_reconstruction import CTReconstructor

recon = CTReconstructor()

# Filtered backprojection
image = recon.fbp_reconstruction(sinogram, filter_type='ram-lak')

# SART
image = recon.sart_reconstruction(
    sinogram, 
    iterations=10,
    relaxation=0.15
)

# Evaluate quality
psnr = recon.calculate_psnr(original, reconstructed)
ssim = recon.calculate_ssim(original, reconstructed)
```

### MRI Reconstruction

```python
from src.reconstruction.mri_reconstruction import MRIReconstructor

recon = MRIReconstructor()

# K-space to image
image = recon.kspace_to_image(kspace_data)

# Extract magnitude and phase
magnitude = recon.get_magnitude(image)
phase = recon.get_phase(image)

# Partial Fourier
image = recon.partial_fourier_reconstruction(kspace_data, factor=0.75)
```

### Image Preprocessing

```python
from src.preprocessing.image_transforms import ImageTransforms

transforms = ImageTransforms()

# Normalize
normalized = transforms.normalize_intensity(image, method='minmax')

# Resize
resized = transforms.resize(image, output_shape=(256, 256, 128))

# Denoise
denoised = transforms.gaussian_filter(image, sigma=1.0)

# Enhance contrast
enhanced = transforms.enhance_contrast(image, method='clahe')
```

## Project Structure

```
medical-image-processing/
│
├── app.py                      # Streamlit web application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_Anonymization.py      # DICOM anonymization page
│   ├── 2_Segmentation.py       # Brain segmentation page
│   ├── 3_CT_Reconstruction.py  # CT reconstruction page
│   ├── 4_MRI_Reconstruction.py # MRI reconstruction page
│   └── 5_Preprocessing.py      # Image preprocessing page
│
├── src/                        # Core library modules
│   ├── anonymization/
│   │   └── dicom_anonymizer.py # DICOM PHI removal
│   ├── preprocessing/
│   │   ├── image_transforms.py # Normalization, transforms, augmentation
│   │   └── registration.py     # Image registration
│   ├── reconstruction/
│   │   ├── ct_reconstruction.py  # FBP, SART algorithms
│   │   └── mri_reconstruction.py # K-space processing
│   ├── segmentation/
│   │   └── brain_segmentation.py # Brain extraction methods
│   └── visualization/
│       └── slice_viewer.py     # 2D/3D visualization
│
├── utils/                      # Utility functions
│   ├── file_io.py              # Multi-format I/O
│   └── image_utils.py          # Image processing utilities
│
├── data/                       # Sample data (not in git)
│   ├── anonym/                 # DICOM samples
│   ├── medical/                # Sinogram, K-space
│   ├── mri/                    # NIfTI files
│   └── sitk/                   # SimpleITK formats
│
├── notebooks/                  # Jupyter notebooks
│   ├── AnonymizingImg.ipynb
│   ├── MedImgModal.ipynb
│   ├── MRI.ipynb
│   ├── SITK.ipynb
│   └── ...
│
└── examples/                   # Example scripts
    ├── demo_file_io.py
    ├── demo_anonymization.py
    ├── demo_segmentation.py
    └── ...
```

## Supported Formats

| Format | Extension | Read | Write | Metadata |
|--------|-----------|------|-------|----------|
| **NIfTI** | `.nii`, `.nii.gz` | ✅ | ✅ | ✅ |
| **DICOM** | `.dcm` | ✅ | ✅ | ✅ |
| **NRRD** | `.nrrd` | ✅ | ✅ | ✅ |
| **MetaImage** | `.mha`, `.mhd` | ✅ | ✅ | ✅ |
| **NumPy** | `.npy` | ✅ | ✅ | ❌ |

**Metadata includes:** Spacing, origin, direction, data type, dimensions

## Documentation

### Jupyter Notebooks

Detailed tutorials available in `notebooks/`:

- **AnonymizingImg.ipynb** - DICOM anonymization techniques
- **MedImgModal.ipynb** - CT and MRI reconstruction
- **MRI.ipynb** - MRI processing workflow
- **SITK.ipynb** - SimpleITK registration and segmentation
- **ImgforML.ipynb** - Preparing images for machine learning

### Example Scripts

Ready-to-run examples in `examples/`:

```bash
python examples/demo_file_io.py
python examples/demo_anonymization.py
python examples/demo_segmentation.py
python examples/demo_ct_reconstruction.py
python examples/demo_mri_reconstruction.py
python examples/demo_preprocessing.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/medical-image-processing.git

# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Format code
black src/ utils/ pages/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SimpleITK** - Medical image processing library
- **Streamlit** - Web application framework
- **scikit-image** - Image processing algorithms
- Sample data from various open medical imaging datasets

## Contact

**Developer:** HaiSGU  
**Repository:** [https://github.com/HaiSGU/medical-image-processing](https://github.com/HaiSGU/medical-image-processing)  
**Issues:** [https://github.com/HaiSGU/medical-image-processing/issues](https://github.com/HaiSGU/medical-image-processing/issues)

---

**Made with ❤️ for the medical imaging community**