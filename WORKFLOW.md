# 🏥 Medical Image Processing - Complete Workflow

## 📋 Tổng quan Dự án

Dự án này là một hệ thống xử lý ảnh y tế toàn diện, tích hợp **CORE Processing** và **AI Analysis** với giao diện Streamlit.

---

## 🎯 Workflow Tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    🏠 HOME PAGE (app.py)                     │
│                                                              │
│  • Giới thiệu hệ thống                                       │
│  • Navigation đến 2 pipeline chính                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ 🔧 CORE       │    │ 🧠 AI          │
│ Processing    │    │ Analysis       │
└───────────────┘    └────────────────┘
```

---

## 🔧 Pipeline 1: CORE Processing (`1_Processing_Pipeline.py`)

### **Mục đích:** Xử lý cơ bản ảnh y tế, tái tạo, và căn chỉnh

### **6 Công cụ chính:**

#### 1️⃣ **Preprocessing** (Tiền xử lý)
```
Input: Ảnh y tế (.nii, .dcm, .nrrd, .mha)
  ↓
Operations:
  • Normalization (Chuẩn hóa)
  • Denoising (Giảm nhiễu): Gaussian, Median, Bilateral
  • Histogram Equalization (Cân bằng histogram)
  • Edge Enhancement (Tăng cường cạnh)
  • Gamma Correction (Điều chỉnh độ sáng)
  ↓
Output: Ảnh đã xử lý + Metrics + Download
```

**Use case:** Cải thiện chất lượng ảnh trước khi phân tích

---

#### 2️⃣ **Anonymization** (Ẩn danh hóa)
```
Input: DICOM file
  ↓
Operations:
  • Xóa Patient Name, ID, Birth Date
  • Xóa Study/Series metadata
  • Giữ nguyên pixel data
  ↓
Output: DICOM ẩn danh + Metadata table
```

**Use case:** Bảo vệ thông tin bệnh nhân khi chia sẻ dữ liệu

---

#### 3️⃣ **Segmentation** (Phân đoạn)
```
Input: Ảnh MRI não
  ↓
Methods:
  • Threshold-based (Ngưỡng)
  • Region Growing (Phát triển vùng)
  • Watershed (Phân thủy)
  • Active Contour (Snake)
  ↓
Output: 
  • Brain mask (Mặt nạ não)
  • Overlay visualization
  • Volume metrics
```

**Use case:** Tách não ra khỏi nền, phân đoạn khối u

---

#### 4️⃣ **CT Reconstruction** (Tái tạo CT)
```
Input Option 1: Tạo Phantom
  • Shepp-Logan Phantom
  • Số góc quét: 90-360
  ↓
Input Option 2: Upload Sinogram (.npy)
  ↓
Methods:
  • FBP (Filtered Back Projection) - Nhanh
  • SART (Algebraic) - Chất lượng cao
  ↓
Output:
  • Reconstructed image
  • Quality metrics (PSNR, SSIM, MSE)
  • Comparison plots
```

**Use case:** Hiểu cách CT scanner hoạt động, nghiên cứu thuật toán tái tạo

---

#### 5️⃣ **MRI Reconstruction** (Tái tạo MRI)
```
Input Option 1: Tạo từ Ảnh
  • Upload ảnh MRI
  • FFT → K-space
  • (Optional) Partial Fourier (50-100%)
  • IFFT → Reconstructed image
  ↓
Input Option 2: Upload K-space (.npy)
  • IFFT → Image
  ↓
Output:
  • K-space visualization (log magnitude)
  • Magnitude image (ảnh giải phẫu)
  • Phase image (artifacts, flow)
  • Quality metrics (MSE, PSNR)
```

**Use case:** Nghiên cứu K-space, mô phỏng quét MRI nhanh

---

#### 6️⃣ **Registration** (Căn chỉnh ảnh)
```
Input: 
  • Fixed Image (Ảnh tham chiếu)
  • Moving Image (Ảnh cần căn chỉnh)
  ↓
Methods:
  • Rigid (Translation + Rotation) - 6 DOF
  • Affine (+ Scaling + Shearing) - 12 DOF
  • Deformable (Local deformation) - Nhiều DOF
  ↓
Metrics:
  • Mean Squares
  • Mutual Information
  ↓
Output:
  • Registered image
  • Metrics (MSE, NCC improvement)
  • Visualization (Side-by-side, Overlay, Checkerboard, Difference)
  • Transform file (.tfm)
```

**Use case:** So sánh ảnh trước/sau điều trị, multi-modal alignment (CT-MRI)

---

## 🧠 Pipeline 2: AI Analysis (`2_AI_Analysis.py`)

### **Mục đích:** Phân tích ảnh bằng Computer Vision và Deep Learning

### **3 Tab chính:**

#### 1️⃣ **Classification** (Phân loại khối u)
```
Input: Ảnh MRI não
  ↓
Models:
  • ResNet50 (Khuyến nghị)
  • VGG16
  • EfficientNet
  ↓
Processing:
  • Preprocessing (resize, normalize)
  • CNN inference
  • Grad-CAM visualization
  ↓
Output:
  • Predicted class (Glioma, Meningioma, Pituitary, Normal)
  • Confidence score
  • Top-k predictions
  • Grad-CAM heatmap (vùng ảnh hưởng)
```

**Use case:** Chẩn đoán tự động loại khối u não

**Yêu cầu:** TensorFlow >= 2.13.0

---

#### 2️⃣ **Detection** (Phát hiện tổn thương)
```
Input: Ảnh chứa tổn thương
  ↓
Methods:
  • Threshold-based detection
  • Blob detection (SimpleBlobDetector)
  ↓
Parameters:
  • Detection threshold (0.0-1.0)
  • Minimum lesion area (pixels)
  • Enable/disable blob detection
  ↓
Output:
  • Bounding boxes
  • Number of lesions
  • Statistics (area, centroid, confidence)
  • Detailed table
```

**Use case:** Phát hiện và đếm số lượng tổn thương, khối u

---

#### 3️⃣ **Feature Extraction** (Trích xuất đặc trưng)
```
Input: Ảnh y tế
  ↓
Feature Types:
  ┌─────────────────────────────────┐
  │ 🎨 Texture Features             │
  │   • GLCM (contrast, homogeneity)│
  │   • LBP (Local Binary Pattern)  │
  ├─────────────────────────────────┤
  │ ⭕ Shape Features                │
  │   • Area, Perimeter             │
  │   • Circularity, Eccentricity   │
  │   • Solidity, Axis lengths      │
  ├─────────────────────────────────┤
  │ 💡 Intensity Features            │
  │   • Mean, Std, Min, Max         │
  │   • Skewness, Kurtosis, Energy  │
  └─────────────────────────────────┘
  ↓
Output:
  • Feature dictionary
  • CSV export
  • Categorized display
```

**Use case:** Phân tích định lượng, machine learning features

---

## 📂 Cấu trúc Dự án

```
Medical Image Processing/
│
├── app.py                          # 🏠 Home page
│
├── pages/
│   ├── 1_Processing_Pipeline.py    # 🔧 CORE Processing (6 tools)
│   └── 2_AI_Analysis.py            # 🧠 AI Analysis (3 tabs)
│
├── src/
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
│       ├── classification.py       # (Requires TensorFlow)
│       ├── detection.py
│       └── feature_extraction.py
│
├── utils/
│   ├── file_io.py                  # Medical image I/O
│   ├── interpretation.py           # Result visualization
│   └── image_explainer.py          # Input image explanation
│
└── data/                           # Sample data (optional)
```

---

## 🔄 Workflow Sử dụng Thực tế

### **Scenario 1: Phân tích khối u não**
```
1. CORE Processing → Preprocessing
   • Upload ảnh MRI
   • Denoising + Histogram Equalization
   • Download ảnh đã xử lý
   ↓
2. CORE Processing → Segmentation
   • Upload ảnh đã xử lý
   • Watershed segmentation
   • Lấy brain mask
   ↓
3. AI Analysis → Classification
   • Upload ảnh/mask
   • ResNet50 classification
   • Xem kết quả: Glioma (85% confidence)
   ↓
4. AI Analysis → Detection
   • Phát hiện vị trí khối u
   • Đếm số lượng tổn thương
   ↓
5. AI Analysis → Feature Extraction
   • Trích xuất texture + shape features
   • Export CSV cho machine learning
```

### **Scenario 2: So sánh ảnh trước/sau điều trị**
```
1. CORE Processing → Preprocessing
   • Xử lý cả 2 ảnh (before/after)
   ↓
2. CORE Processing → Registration
   • Fixed: Ảnh trước điều trị
   • Moving: Ảnh sau điều trị
   • Rigid registration
   • Xem Difference map
   ↓
3. AI Analysis → Detection
   • Phát hiện tổn thương trên cả 2 ảnh
   • So sánh số lượng và kích thước
```

### **Scenario 3: Nghiên cứu kỹ thuật MRI**
```
1. CORE Processing → MRI Reconstruction
   • Upload ảnh MRI
   • Tạo K-space
   • Thử Partial Fourier 50%, 75%, 100%
   • So sánh chất lượng (PSNR)
   • Hiểu trade-off: Tốc độ vs Chất lượng
```

---

## 🚀 Cách Chạy Dự án

### **1. Cài đặt Dependencies**
```bash
pip install streamlit numpy matplotlib scikit-image pydicom SimpleITK pandas opencv-python

# Optional (cho Classification)
pip install tensorflow>=2.13.0
```

### **2. Chạy Ứng dụng**
```bash
# Chạy Home page
streamlit run app.py

# Hoặc chạy trực tiếp từng page
streamlit run pages/1_Processing_Pipeline.py
streamlit run pages/2_AI_Analysis.py
```

### **3. Truy cập**
```
Browser tự động mở: http://localhost:8501
```

---

## 📊 Tính năng Nổi bật

✅ **All-in-One:** 9 công cụ trong 1 hệ thống  
✅ **User-Friendly:** Giao diện Streamlit trực quan  
✅ **Explanations:** Hướng dẫn chi tiết cho từng tool  
✅ **Visualization:** Plots, metrics, comparisons  
✅ **Export:** Download kết quả (.npy, .png, .csv, .dcm, .tfm)  
✅ **Professional:** Không emoji, thiết kế y tế chuẩn  
✅ **Modular:** Code tổ chức rõ ràng, dễ mở rộng  

---

## 🎓 Học tập & Nghiên cứu

Dự án này phù hợp cho:
- 🏥 **Sinh viên Y khoa:** Hiểu cách xử lý ảnh y tế
- 💻 **Kỹ sư AI:** Thực hành Computer Vision
- 🔬 **Nhà nghiên cứu:** Thử nghiệm thuật toán
- 🏢 **Bệnh viện:** Công cụ hỗ trợ chẩn đoán

---

## 📝 Notes

- **CORE Processing:** Không cần GPU, chạy nhanh
- **AI Analysis:** Classification cần GPU (khuyến nghị) hoặc CPU (chậm hơn)
- **Data Format:** Hỗ trợ NIfTI, DICOM, NRRD, MHA, NPY, PNG, JPG
- **3D Images:** Tự động lấy slice giữa để hiển thị 2D

---

## 🔮 Tương lai

Có thể mở rộng:
- [ ] Thêm 3D visualization
- [ ] Batch processing
- [ ] Model training interface
- [ ] PACS integration
- [ ] Report generation
- [ ] Multi-language support

---

**Created by:** Medical Image Processing Team  
**Last Updated:** 2025-11-24  
**Version:** 1.0.0
