# 🏥 Medical Image Processing System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Hệ thống xử lý ảnh y tế toàn diện với giao diện web tương tác, tích hợp **CORE Processing** và **AI Analysis**.

---

## 🎯 Tổng quan

Dự án cung cấp cả **thư viện Python** và **ứng dụng web Streamlit** để xử lý ảnh y tế, phù hợp cho:
- 🎓 Sinh viên y khoa và kỹ thuật y sinh
- 🔬 Nhà nghiên cứu xử lý ảnh y tế
- 💻 Kỹ sư AI/Computer Vision
- 🏥 Chuyên gia y tế (mục đích nghiên cứu)

---

## ✨ Tính năng chính

### 🔧 **CORE Processing** (6 công cụ)

1. **Preprocessing** - Tiền xử lý ảnh
   - Normalization (Min-Max, Z-Score)
   - Denoising (Gaussian, Median, Bilateral)
   - Histogram Equalization, CLAHE
   - Edge Enhancement, Gamma Correction

2. **Anonymization** - Ẩn danh hóa DICOM
   - Xóa thông tin bệnh nhân (PHI)
   - Batch processing
   - Export ZIP files

3. **Segmentation** - Phân đoạn ảnh
   - Threshold-based (Manual, Otsu)
   - Region Growing
   - Watershed
   - Active Contour (Snake)

4. **CT Reconstruction** - Tái tạo ảnh CT
   - FBP (4 filters: Ramp, Shepp-Logan, Cosine, Hamming)
   - SART (iterative)
   - Shepp-Logan phantom generation
   - Quality metrics (PSNR, SSIM, MSE, SNR)

5. **MRI Reconstruction** - Tái tạo ảnh MRI
   - K-space ↔ Image domain conversion
   - Magnitude & Phase extraction
   - Partial Fourier reconstruction
   - Interactive visualization

6. **Registration** - Căn chỉnh ảnh
   - Rigid (6 DOF)
   - Affine (12 DOF)
   - Deformable (B-spline)
   - Metrics: MSE, NCC
   - Visualization: Side-by-side, Overlay, Checkerboard, Difference

### 🧠 **AI Analysis** (3 modules)

1. **Classification** - Phân loại khối u não
   - Deep Learning models: ResNet50, VGG16, EfficientNet
   - 4 classes: Glioma, Meningioma, Pituitary, Normal
   - Grad-CAM visualization
   - Confidence scores

2. **Detection** - Phát hiện tổn thương
   - Threshold-based detection
   - Blob detection
   - Bounding box visualization
   - Statistics (area, centroid, confidence)

3. **Feature Extraction** - Trích xuất đặc trưng
   - Texture: GLCM (contrast, homogeneity), LBP
   - Shape: Area, perimeter, circularity, eccentricity
   - Intensity: Mean, std, skewness, kurtosis
   - CSV export

---

## 🚀 Cài đặt nhanh

### Yêu cầu hệ thống

- Python 3.9 trở lên
- Windows/Linux/Mac
- 4GB RAM (khuyến nghị 8GB cho AI Analysis)

### Hướng dẫn cài đặt

**Bước 1: Clone repository**

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

# Optional: Cài TensorFlow cho Classification (AI Analysis)
pip install tensorflow>=2.13.0

# Optional: Cài OpenCV cho Computer Vision
pip install opencv-python
```

**Bước 4: Chạy ứng dụng**

```bash
streamlit run app.py
```

Mở trình duyệt tại: `http://localhost:8501`

---

## 📖 Cách sử dụng

### Giao diện Web

1. **Khởi động ứng dụng:** `streamlit run app.py`
2. **Chọn pipeline:**
   - **CORE Processing:** Xử lý cơ bản (Preprocessing, Anonymization, Segmentation, Reconstruction, Registration)
   - **AI Analysis:** Phân tích thông minh (Classification, Detection, Feature Extraction)
3. **Upload ảnh** từ máy tính
4. **Chọn công cụ** từ dropdown menu
5. **Điều chỉnh tham số** trong sidebar
6. **Xem kết quả** và **download** file đã xử lý

### Python Library

```python
# Đọc ảnh y tế
from utils.file_io import MedicalImageIO

io_handler = MedicalImageIO()
image, metadata = io_handler.read_image("brain_mri.nii.gz")
print(f"Shape: {image.shape}")

# Phân đoạn não
from src.segmentation.brain_segmentation import BrainSegmentation

segmenter = BrainSegmentation()
mask = segmenter.segment_brain(image, method='auto')

# Tái tạo CT
from src.reconstruction.ct_reconstruction import CTReconstructor

recon = CTReconstructor()
ct_image = recon.fbp_reconstruction(sinogram, filter_type='ramp')

# Tính metrics
psnr = recon.calculate_psnr(original, reconstructed)
ssim = recon.calculate_ssim(original, reconstructed)

# Tiền xử lý
from src.preprocessing.image_transforms import ImageTransforms

transforms = ImageTransforms()
normalized = transforms.normalize_intensity(image, method='minmax')
enhanced = transforms.enhance_contrast(image, method='clahe')
```

---

## 📁 Cấu trúc dự án

```
medical-image-processing/
│
├── app.py                          # Home page
│
├── pages/                          # Streamlit pages
│   ├── 1_Processing_Pipeline.py   # CORE Processing (6 tools)
│   ├── 2_AI_Analysis.py            # AI Analysis (3 tabs)
│   └── _archived/                  # Old standalone pages (backup)
│
├── src/                            # Core modules
│   ├── anonymization/
│   │   └── dicom_anonymizer.py
│   ├── preprocessing/
│   │   └── image_transforms.py
│   ├── segmentation/
│   │   └── brain_segmentation.py
│   ├── reconstruction/
│   │   ├── ct_reconstruction.py
│   │   └── mri_reconstruction.py
│   ├── registration/
│   │   └── image_registration.py
│   └── computer_vision/
│       ├── classification.py       # Requires TensorFlow
│       ├── detection.py
│       └── feature_extraction.py
│
├── utils/                          # Utilities
│   ├── file_io.py                  # Medical image I/O
│   ├── interpretation.py           # Result visualization
│   └── image_explainer.py          # Input image explanation
│
├── data/                           # Sample data
│   ├── anonym/dicom_dir/          # 100 DICOM files
│   ├── sitk/                      # Brain MRI (.nrrd, .mha)
│   ├── medical/                   # Sinogram & K-space
│   ├── synthetic_ct/              # CT test data
│   ├── synthetic_mri/             # MRI test data
│   └── synthetic_preprocessing/   # Preprocessing test
│
├── examples/                       # Example scripts
├── notebooks/                      # Jupyter notebooks
├── requirements.txt                # Dependencies
├── WORKFLOW.md                     # Detailed workflow documentation
└── README.md                       # This file
```

---

## 📊 Định dạng ảnh hỗ trợ

| Định dạng | Phần mở rộng | Mô tả | Dùng cho |
|-----------|--------------|-------|----------|
| **NIfTI** | `.nii`, `.nii.gz` | Phổ biến trong nghiên cứu não | MRI, fMRI |
| **DICOM** | `.dcm` | Tiêu chuẩn y tế quốc tế | CT, MRI, X-quang |
| **NRRD** | `.nrrd` | Nearly Raw Raster Data | Nghiên cứu |
| **MetaImage** | `.mha`, `.mhd` | ITK format | Xử lý ảnh y tế |
| **NumPy** | `.npy` | Mảng Python | Dữ liệu đã xử lý |
| **Images** | `.png`, `.jpg` | Standard images | Computer Vision |

---

## 🧪 Test nhanh (5 phút)

### CORE Processing

1. **Preprocessing:**
   - Upload: `data/sitk/A1_grayT1.nrrd`
   - Apply: CLAHE + Gaussian Denoising
   - Compare: Original vs Processed

2. **Segmentation:**
   - Upload: `data/sitk/A1_grayT1.nrrd`
   - Method: Automatic
   - View: Brain mask overlay

3. **CT Reconstruction:**
   - Data Source: Generate Phantom
   - Method: FBP (Ramp filter)
   - View: Sinogram + Reconstructed image + Metrics

4. **MRI Reconstruction:**
   - Upload: `data/sitk/A1_grayT1.nrrd`
   - Auto generate K-space
   - View: K-space + Magnitude + Phase

5. **Registration:**
   - Fixed: `data/sitk/A1_grayT1.nrrd`
   - Moving: Same file (for demo)
   - Type: Rigid
   - View: Comparison visualizations

### AI Analysis

1. **Classification:**
   - Upload brain MRI
   - Model: ResNet50
   - View: Predicted class + Grad-CAM

2. **Detection:**
   - Upload brain scan with lesions
   - Threshold: 0.7
   - View: Bounding boxes + Statistics

3. **Feature Extraction:**
   - Upload any medical image
   - Extract: Texture + Intensity features
   - Download: CSV file

---

## 🔧 Troubleshooting

**Module not found:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**File cannot be read:**
- Check file format (`.nii`, `.dcm`, `.nrrd`, `.mha`, `.npy`)
- Try other files in `data/` directory
- Ensure file is not corrupted

**TensorFlow not found (for Classification):**
```bash
pip install tensorflow>=2.13.0
```

**OpenCV not found (for Computer Vision):**
```bash
pip install opencv-python
```

**Access denied when installing:**
```bash
# Option 1: Run as Administrator
# Option 2: Install for user only
pip install --user opencv-python
```

---

## 📚 Tài liệu

### Jupyter Notebooks

Hướng dẫn chi tiết trong `notebooks/`:
- **AnonymizingImg.ipynb** - DICOM anonymization
- **MedImgModal.ipynb** - CT/MRI reconstruction
- **MRI.ipynb** - MRI processing workflow
- **SITK.ipynb** - SimpleITK usage
- **ImgforML.ipynb** - ML preprocessing

### Example Scripts

```bash
python examples/demo_file_io.py
python examples/demo_anonymization.py
python examples/demo_segmentation.py
python examples/demo_ct_reconstruction.py
python examples/demo_mri_reconstruction.py
python examples/demo_preprocessing.py
```

### Workflow Documentation

Xem file `WORKFLOW.md` để hiểu chi tiết về:
- Kiến trúc hệ thống
- Workflow từng công cụ
- Use cases thực tế
- Best practices

---

## ⚠️ Lưu ý quan trọng

- **Mục đích:** Học tập & nghiên cứu - **KHÔNG** dùng cho chẩn đoán lâm sàng
- **Yêu cầu:** Python 3.9+
- **Dữ liệu:** Tất cả file mẫu đã có sẵn trong `data/`
- **GPU:** Khuyến nghị cho Classification (AI Analysis), nhưng CPU vẫn chạy được (chậm hơn)
- **Bản quyền:** MIT License - Tự do sử dụng cho mục đích phi thương mại

---

## 🤝 Đóng góp

Contributions welcome! Vui lòng:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - Tự do sử dụng cho học tập và nghiên cứu.

---

## 👨‍💻 Tác giả

**HaiSGU**  
- Repository: https://github.com/HaiSGU/medical-image-processing  
- Issues: https://github.com/HaiSGU/medical-image-processing/issues

---

## 🌟 Acknowledgments

- **SimpleITK** - Medical image processing
- **Streamlit** - Web framework
- **scikit-image** - Image processing algorithms
- **TensorFlow** - Deep learning (optional)
- Medical imaging community

---

**Made with ❤️ for the medical imaging community**

**Last Updated:** 2025-11-24  
**Version:** 2.0.0