# 🎨 UX IMPROVEMENTS - Hướng dẫn sử dụng

## 📋 **Tổng quan tính năng mới**

Đã thêm 4 tính năng UX chính:

### 1. **Progress Bars & Status** ✅
- Hiển thị tiến độ xử lý real-time
- Ước tính thời gian còn lại (ETA)
- Status messages chi tiết
- Thông báo hoàn thành/lỗi

### 2. **Image Comparison Slider** 🔄
- So sánh trực quan trước/sau xử lý
- 3 chế độ:
  - **Side by Side**: Xem song song
  - **Overlay**: Trộn 2 ảnh với slider
  - **Difference Map**: Bản đồ khác biệt

### 3. **Batch Processing** 📦
- Upload nhiều files cùng lúc (max 50)
- Xử lý tự động toàn bộ
- Progress tracking cho từng file
- Xử lý song song (optional)

### 4. **PDF/ZIP Export** 💾
- Export báo cáo PDF đầy đủ
- Tạo ZIP archive với tất cả kết quả
- Download từng ảnh riêng lẻ
- Bao gồm metrics và description

---

## 🚀 **Cách sử dụng**

### **A. Test Demo Components**

```bash
# Chạy demo để test features
streamlit run demo_ui_features.py
```

**Chức năng demo:**
- Tab 1: Test progress bars
- Tab 2: Test image comparison
- Tab 3: Test batch upload
- Tab 4: Test export functionality

---

### **B. Sử dụng trong Pages**

#### **1. Import Components**

```python
from utils.ui_components import (
    ProgressTracker,
    ImageComparer,
    BatchProcessor,
    ResultExporter,
    show_metrics_dashboard,
    show_preview_gallery,
    create_download_section
)
```

#### **2. Progress Bars**

```python
# Tạo tracker
tracker = ProgressTracker("Đang xử lý", total_steps=5)

# Update progress
for i in range(5):
    tracker.update(i+1, f"Step {i+1}...")
    # Do work...

# Hoàn thành
tracker.complete("✅ Xong!")
```

#### **3. Image Comparison**

```python
# So sánh 2 ảnh
comparer = ImageComparer()
comparer.show(
    original_image,
    processed_image,
    "Ảnh gốc",
    "Đã xử lý"
)
```

#### **4. Batch Processing**

```python
# Upload nhiều files
processor = BatchProcessor()
files = processor.upload_multiple(
    "Upload files",
    ["dcm", "nii", "npy"],
    max_files=20
)

# Xử lý batch
def process_func(file):
    # Load và xử lý file
    return result

results = processor.process_files(files, process_func)
```

#### **5. Export Results**

```python
# Chuẩn bị data
results = {
    'images': {
        'original': original_img,
        'processed': processed_img
    },
    'metrics': {
        'Size': '512x512',
        'Processing Time': '2.5s'
    },
    'description': 'Report description here'
}

# Tạo download section
create_download_section(results, "segmentation")
```

---

## 📄 **Enhanced Page Example: Preprocessing**

File `pages/5_Preprocessing_Enhanced.py` đã được tạo với:

✅ **Single & Batch modes**
```python
mode = st.radio("Chế độ:", ["Single Image", "Batch Processing"])
```

✅ **Progress tracking**
```python
tracker = ProgressTracker("Đang xử lý", total_steps)
for i, step in enumerate(steps):
    tracker.update(i+1, f"Processing: {step}")
```

✅ **Image comparison**
```python
comparer = ImageComparer()
comparer.show(original, processed)
```

✅ **Gallery view cho batch**
```python
show_preview_gallery(processed_images, columns=3)
```

✅ **Export options**
```python
create_download_section(results, "preprocessing")
```

---

## 🎯 **Apply to Other Pages**

### **Segmentation Page**

```python
# Thêm batch segmentation
uploaded_files = batch_processor.upload_multiple(
    "Upload DICOM files",
    ["dcm"],
    max_files=30
)

# Xử lý batch với progress
tracker = ProgressTracker("Segmenting brains", len(uploaded_files))

for i, file in enumerate(uploaded_files):
    tracker.update(i+1, f"Segmenting: {file.name}")
    
    # Load
    image = load_image(file)
    
    # Segment
    segmenter = BrainSegmentation(image)
    mask = segmenter.threshold_otsu()
    
    results.append((file.name, mask))

tracker.complete("✅ All brains segmented!")

# So sánh
comparer.show(image, mask, "Original", "Segmented")

# Export
create_download_section({
    'images': {'original': image, 'mask': mask},
    'metrics': {'Volume': volume, 'Coverage': coverage},
    'description': 'Brain segmentation report'
}, "segmentation")
```

### **CT Reconstruction Page**

```python
# Batch reconstruction với nhiều góc
angle_configs = [45, 90, 180]

tracker = ProgressTracker("Reconstructing", len(angle_configs))

for i, num_angles in enumerate(angle_configs):
    tracker.update(i+1, f"Reconstructing with {num_angles} angles")
    
    # Reconstruct
    reconstructor = CTReconstructor(sinogram, num_angles)
    recon = reconstructor.fbp_reconstruct()
    
    results[f"{num_angles}_angles"] = recon

# Gallery comparison
show_preview_gallery(results, columns=3, title="Reconstructions")

# Export comparison report
create_download_section({
    'images': results,
    'metrics': {
        'Method': 'FBP',
        'Angles': str(angle_configs)
    },
    'description': 'CT reconstruction comparison'
}, "ct_reconstruction")
```

### **MRI Reconstruction Page**

```python
# Batch với different undersampling
undersampling_rates = [1.0, 0.5, 0.25]

tracker = ProgressTracker("MRI Reconstruction", len(undersampling_rates))

for i, rate in enumerate(undersampling_rates):
    tracker.update(i+1, f"Undersampling: {rate*100}%")
    
    # Apply mask
    kspace_masked = apply_undersampling_mask(kspace, rate)
    
    # Reconstruct
    reconstructor = MRIReconstructor(kspace_masked)
    magnitude = np.abs(reconstructor.kspace_to_image())
    
    results[f"{int(rate*100)}percent"] = magnitude

# Compare với slider
comparer.show(
    results["100percent"],
    results["50percent"],
    "Full sampling",
    "50% undersampled"
)
```

---

## 📊 **Metrics Dashboard**

Hiển thị metrics đẹp:

```python
metrics = {
    "Total Images": 10,
    "Processing Time": "5.2s",
    "Success Rate": "100%",
    "Average Size": "512×512",
    "Memory Used": "120MB",
    "Method": "Otsu"
}

show_metrics_dashboard(metrics, title="📊 Processing Statistics")
```

---

## 💾 **Export Formats**

### **PDF Report**

Bao gồm:
- ✅ Title page với timestamp
- ✅ Metrics summary
- ✅ All images với captions
- ✅ Shape và dtype info

### **ZIP Archive**

Chứa:
- ✅ Tất cả ảnh (.png)
- ✅ Metrics file (.txt)
- ✅ Log file (optional)
- ✅ Metadata (optional)

### **Individual Images**

- ✅ Download từng ảnh
- ✅ PNG format (normalized)
- ✅ Timestamp trong filename

---

## 🔧 **Configuration**

### **Tùy chỉnh Progress Bar**

```python
tracker = ProgressTracker(
    title="Custom title",
    total_steps=100
)
```

### **Tùy chỉnh Image Comparer**

```python
comparer.show(
    img1, 
    img2,
    label1="Before",
    label2="After",
    slider_position=0.5  # Initial position
)
```

### **Tùy chỉnh Batch Processor**

```python
files = batch_processor.upload_multiple(
    label="Custom label",
    accepted_types=["dcm", "nii"],
    max_files=100  # Increase limit
)
```

---

## 🐛 **Troubleshooting**

### **Progress bar không hiển thị**

```python
# Đảm bảo gọi update() trong loop
for i in range(n):
    tracker.update(i+1, "Processing...")
    # Do work here
```

### **Image comparison lỗi shape**

```python
# Kiểm tra shape trước khi compare
if img1.shape != img2.shape:
    # Resize về cùng kích thước
    img2 = resize(img2, img1.shape)

comparer.show(img1, img2)
```

### **Export PDF quá lớn**

```python
# Giảm số ảnh hoặc resize
images_small = {
    name: resize(img, (256, 256)) 
    for name, img in images.items()
}

pdf_bytes = ResultExporter.create_pdf_report(images_small, ...)
```

---

## 📈 **Performance Tips**

### **1. Batch Processing**

```python
# Sử dụng multiprocessing cho batch lớn
from concurrent.futures import ProcessPoolExecutor

def process_batch_parallel(files, process_func, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, files))
    return results
```

### **2. Caching**

```python
@st.cache_data
def load_and_process_image(file_path):
    # Expensive operation
    return processed_image

# Sử dụng cache
image = load_and_process_image(path)
```

### **3. Progress Optimization**

```python
# Update progress mỗi N steps thay vì mọi step
UPDATE_INTERVAL = 10

for i in range(1000):
    # Process
    
    if i % UPDATE_INTERVAL == 0:
        tracker.update(i, f"Processing {i}/1000")
```

---

## ✅ **Checklist Implementation**

Apply vào pages:

- [ ] **Preprocessing** - ✅ Done (Enhanced version)
- [ ] **Segmentation** - 🔄 Đang cập nhật
- [ ] **CT Reconstruction** - 🔄 Đang cập nhật
- [ ] **MRI Reconstruction** - 🔄 Đang cập nhật
- [ ] **Anonymization** - 🔄 Đang cập nhật

---

## 🎉 **Next Steps**

1. **Test demo components:**
   ```bash
   streamlit run demo_ui_features.py
   ```

2. **Apply to Segmentation page** (easiest first)

3. **Apply to other pages** one by one

4. **Test full workflow** với batch processing

5. **User feedback** và iterations

---

## 📞 **Support**

Nếu gặp vấn đề:
1. Check `demo_ui_features.py` examples
2. Review `utils/ui_components.py` docstrings
3. Test với sample data trước
4. Ask for help!

Happy coding! 🚀
