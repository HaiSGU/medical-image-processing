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

### 3. Phân đoạn Ảnh (Segmentation)
Tách vùng quan tâm ra khỏi ảnh (ví dụ: tìm vùng não)
- **Ngưỡng thủ công:** Tự chọn giá trị ngưỡng
- **Otsu:** Tự động tìm ngưỡng tốt nhất
- **Region Growing:** Phát triển vùng từ điểm chọn
- **Tự động:** Phân đoạn não hoàn toàn tự động

### 4. Tái tạo ảnh CT
Tái tạo ảnh CT từ dữ liệu sinogram (dữ liệu thô từ máy chụp)
- **FBP (Filtered Backprojection):** Thuật toán tái tạo nhanh
- **SART:** Thuật toán lặp, chính xác hơn
- Tạo phantom để test
- Đo lường chất lượng ảnh tái tạo

### 5. Tái tạo ảnh MRI
Tái tạo ảnh MRI từ K-space (dữ liệu tần số)
- Chuyển đổi từ K-space sang ảnh thực
- Hiển thị magnitude (độ lớn) và phase (pha)
- Partial Fourier: tái tạo từ dữ liệu thiếu
- Trực quan hóa tương tác

### 6. Tiền xử lý Ảnh
Cải thiện chất lượng ảnh trước khi phân tích
- **Chuẩn hóa:** Min-Max, Z-Score
- **Thay đổi kích thước:** Resize, Crop
- **Khử nhiễu:** Gaussian blur, Median filter  
- **Tăng độ tương phản:** Histogram Equalization, CLAHE
- **Augmentation:** Lật, xoay, thêm nhiễu

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

## Cài đặt

### Yêu cầu hệ thống
- Python 3.9 trở lên
- Windows/Linux/Mac

### Hướng dẫn cài đặt

**Bước 1: Tải code về**
```bash
git clone https://github.com/HaiSGU/medical-image-processing.git
cd medical-image-processing
```

**Bước 2: Tạo môi trường ảo (khuyến nghị)**
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate
```

**Bước 3: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

### Các thư viện chính

- `streamlit` - Framework tạo web app
- `numpy` - Tính toán số học
- `matplotlib` - Vẽ biểu đồ
- `SimpleITK` - Xử lý ảnh y tế
- `pydicom` - Đọc/ghi file DICOM
- `nibabel` - Đọc/ghi file NIfTI
- `scikit-image` - Thuật toán xử lý ảnh
- `scipy` - Tính toán khoa học

## Cách sử dụng

### Chạy ứng dụng Web

**Bước 1: Mở Terminal/Command Prompt**

**Bước 2: Chạy lệnh**
```bash
streamlit run app.py
```

**Bước 3: Mở trình duyệt**
- Tự động mở hoặc vào: `http://localhost:8501`

**Bước 4: Sử dụng**
1. Tải ảnh lên từ máy tính
2. Xem thông tin và thống kê
3. Chọn công cụ xử lý ở menu bên trái
4. Làm theo hướng dẫn trong từng công cụ

### Ví dụ sử dụng code Python

```python
# Đọc ảnh y tế
from utils.file_io import MedicalImageIO

io_handler = MedicalImageIO()
image, metadata = io_handler.read_image("path/to/image.nii")
print(f"Kích thước ảnh: {image.shape}")

# Phân đoạn não
from src.segmentation.brain_segmentation import BrainSegmentation

segmentor = BrainSegmentation(image)
mask = segmentor.segment(method="automatic")
print(f"Đã tìm thấy vùng não với {mask.sum()} pixels")

# Load medical image
io_handler = MedicalImageIO()
image, metadata = io_handler.read_image("brain_mri.nii.gz")

# Segment brain
segmenter = BrainSegmentation()
mask = segmenter.segment_brain(image, method='auto')

# Save result
io_handler.save_image(mask, "brain_mask.nii.gz", metadata)
```

## Cấu trúc Project

```
medical-image-processing/
│
├── app.py                      # Trang chủ - Tải và xem ảnh
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # File này
│
├── pages/                      # Các trang công cụ
│   ├── 1_Anonymization.py     # Ẩn danh hóa DICOM
│   ├── 2_Segmentation.py      # Phân đoạn ảnh
│   ├── 3_CT_Reconstruction.py # Tái tạo CT
│   ├── 4_MRI_Reconstruction.py# Tái tạo MRI
│   └── 5_Preprocessing.py     # Tiền xử lý
│
├── src/                        # Mã nguồn xử lý
│   ├── anonymization/         # Module ẩn danh
│   ├── segmentation/          # Module phân đoạn
│   ├── reconstruction/        # Module tái tạo
│   └── preprocessing/         # Module tiền xử lý
│
├── utils/                      # Công cụ hỗ trợ
│   ├── file_io.py             # Đọc/ghi file ảnh
│   └── image_utils.py         # Xử lý ảnh cơ bản
│
├── data/                       # Thư mục chứa ảnh mẫu
│   ├── test_output/           # Ảnh test có dữ liệu
│   └── medical/               # Ảnh y tế mẫu
│
└── examples/                   # Code ví dụ
    └── demo_*.py              # Các file demo
```

## Định dạng ảnh hỗ trợ

| Định dạng | Phần mở rộng | Mô tả | Dùng cho |
|-----------|--------------|-------|----------|
| **NIfTI** | .nii, .nii.gz | Phổ biến trong nghiên cứu não | MRI, fMRI |
| **DICOM** | .dcm | Tiêu chuẩn y tế quốc tế | CT, MRI, X-quang |
| **NRRD** | .nrrd | Nearly Raw Raster Data | Nghiên cứu |
| **MetaImage** | .mha, .mhd | ITK format | Xử lý ảnh y tế |
| **NumPy** | .npy | Mảng Python | Dữ liệu đã xử lý |

## Lưu ý

### Dành cho sinh viên

1. **Mục đích:** Đồ án này phục vụ học tập, nghiên cứu. KHÔNG dùng để chẩn đoán y khoa thực tế.

2. **File test:** Sử dụng file trong `data/test_output/` để test các chức năng:
   - `synthetic_dicom.dcm` - Test ẩn danh hóa
   - `test_volume.mha` - Test phân đoạn/tiền xử lý
   - `slice_kspace.npy` - Test tái tạo MRI

3. **Hiểu thuật toán:** Mỗi công cụ có giải thích ngắn gọn về thuật toán. Đọc kỹ để hiểu cách hoạt động.

4. **Tham khảo code:** Xem code trong `src/` và `examples/` để hiểu cách implement.

### Xử lý lỗi thường gặp

**Lỗi: "Module not found"**
```bash
# Cài lại requirements
pip install -r requirements.txt
```

**Lỗi: "Port already in use"**
```bash
# Đổi port
streamlit run app.py --server.port 8502
```

**Lỗi: "Cannot read file"**
- Kiểm tra định dạng file có đúng không
- Thử file khác trong `data/test_output/`
- File có thể bị hỏng hoặc rỗng

### Đóng góp và Phản hồi

Nếu gặp lỗi hoặc có đề xuất cải tiến:
1. Tạo Issue trên GitHub
2. Hoặc email: [Thêm email của bạn]

### Bản quyền

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

---

**Tác giả:** HaiSGU  
**Repository:** https://github.com/HaiSGU/medical-image-processing  
**Năm:** 2025

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