# 🧠 Hướng dẫn tích hợp Interpretation vào các trang

## 📋 Tổng quan

File `utils/interpretation.py` cung cấp các components giải thích kết quả cho người không chuyên:

### ✅ Đã tạo:
- ✅ `utils/interpretation.py` - Core interpretation library
- ✅ `pages/6_Interpretation_Report.py` - Standalone interpretation page
- 🔄 Chờ tích hợp vào 5 trang còn lại

---

## 🎯 Components có sẵn

### 1. ResultVisualizer
```python
from utils.interpretation import ResultVisualizer

visualizer = ResultVisualizer()

# So sánh ảnh trước/sau
visualizer.compare_images(
    img_before=original,
    img_after=processed,
    title_before="Ảnh gốc",
    title_after="Ảnh đã xử lý",
    description="Giải thích cho người không chuyên"
)

# Overlay phân đoạn với chú thích
labels = {
    1: "Khối u nghi ngờ",
    2: "Mô bình thường"
}
visualizer.show_overlay_with_legend(
    image=mri_image,
    mask=segmentation_mask,
    labels=labels,
    title="Kết quả phân đoạn não bộ"
)

# Hiển thị nhiều slices 3D
visualizer.show_3d_slices(
    volume=volume_3d,
    axis=2,
    num_slices=9,
    title="Các lát cắt MRI"
)
```

### 2. MetricsExplainer
```python
from utils.interpretation import MetricsExplainer

metrics_explainer = MetricsExplainer()

# Tự động giải thích metrics
explanation = metrics_explainer.explain_metric('PSNR', 35.2)
# Returns: {
#     'name': 'Độ rõ nét (PSNR)',
#     'value': 35.2,
#     'unit': 'dB',
#     'assessment': 'good',
#     'description': '...',
#     'interpretation': 'Chất lượng tốt'
# }

# Dashboard metrics với màu sắc
metrics = {
    'PSNR': 35.2,
    'SSIM': 0.94,
    'Dice': 0.87
}
metrics_explainer.show_metrics_dashboard(
    metrics,
    title="Chỉ số chất lượng"
)
```

### 3. InterpretationGenerator
```python
from utils.interpretation import InterpretationGenerator

# Tạo đoạn giải thích tự động
interpretation = InterpretationGenerator.generate_interpretation(
    task_type='segmentation',  # hoặc 'preprocessing', 'reconstruction', 'anonymization'
    metrics={'Dice': 0.87, 'IoU': 0.76},
    image_info={'region_percentage': 15.3}
)

# Hoặc dùng helper function
from utils.interpretation import show_interpretation_section

show_interpretation_section(
    task_type='preprocessing',
    metrics={'PSNR': 35.2, 'SSIM': 0.94},
    image_info={'operations': ['normalize', 'denoise', 'enhance']}
)
```

### 4. ReportBuilder
```python
from utils.interpretation import ReportBuilder

# Tạo báo cáo PDF/HTML
report_bytes = ReportBuilder.create_interpretation_report(
    title="Báo cáo Phân đoạn MRI Não",
    task_type='segmentation',
    images={
        'Ảnh gốc': original_image,
        'Kết quả phân đoạn': overlay_image
    },
    metrics={'Dice': 0.87, 'IoU': 0.76},
    interpretation=interpretation_text,
    output_format='pdf'  # hoặc 'html'
)

# Download
st.download_button(
    "📥 Tải báo cáo",
    data=report_bytes,
    file_name="report.pdf",
    mime="application/pdf"
)
```

---

## 📝 Tích hợp vào từng trang

### 🎨 1. Preprocessing Page (`pages/5_Preprocessing.py`)

**Vị trí tích hợp:** Sau khi xử lý xong ảnh, trước phần download

```python
# Thêm imports
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Sau khi có original và processed
if processed_image is not None:
    
    # 1. So sánh trực quan
    st.markdown("---")
    st.subheader("🔍 So sánh kết quả")
    
    visualizer = ResultVisualizer()
    visualizer.compare_images(
        original_image,
        processed_image,
        title_before="Ảnh gốc",
        title_after="Ảnh sau tiền xử lý",
        description="Ảnh đã được chuẩn hóa, giảm nhiễu và tăng độ tương phản "
                   "để làm nổi bật các chi tiết mô và cấu trúc trong ảnh y tế."
    )
    
    # 2. Tính metrics
    psnr = peak_signal_noise_ratio(original_image, processed_image)
    ssim = structural_similarity(original_image, processed_image, data_range=1.0)
    mse = np.mean((original_image - processed_image) ** 2)
    
    metrics = {
        'PSNR': psnr,
        'SSIM': ssim,
        'MSE': mse
    }
    
    # 3. Dashboard metrics
    st.markdown("---")
    MetricsExplainer().show_metrics_dashboard(metrics)
    
    # 4. Giải thích
    show_interpretation_section(
        task_type='preprocessing',
        metrics=metrics,
        image_info={'operations': selected_operations}  # list các operations đã chọn
    )
```

---

### 🧠 2. Segmentation Page (`pages/2_Segmentation.py`)

**Vị trí tích hợp:** Sau khi segmentation xong

```python
# Thêm imports
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section
)

# Sau khi có mask
if segmentation_mask is not None:
    
    # 1. Overlay với chú thích
    st.markdown("---")
    st.subheader("🎨 Kết quả phân đoạn")
    
    visualizer = ResultVisualizer()
    
    labels = {
        0: "Nền (background)",
        1: "Mô não trắng (White Matter)",
        2: "Mô não xám (Gray Matter)",
        3: "Dịch não tủy (CSF)"
    }
    
    visualizer.show_overlay_with_legend(
        image=original_brain_image,
        mask=segmentation_mask,
        labels=labels,
        title="Phân đoạn não bộ tự động"
    )
    
    # 2. Tính metrics (nếu có ground truth)
    if ground_truth is not None:
        from sklearn.metrics import jaccard_score
        
        # Flatten arrays
        y_true = ground_truth.flatten()
        y_pred = segmentation_mask.flatten()
        
        # Dice coefficient
        intersection = np.sum(y_true * y_pred)
        dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
        
        # IoU
        iou = jaccard_score(y_true, y_pred, average='macro')
        
        metrics = {
            'Dice': dice,
            'IoU': iou
        }
        
        # 3. Dashboard
        st.markdown("---")
        MetricsExplainer().show_metrics_dashboard(metrics)
    
    # 4. Giải thích
    total_pixels = segmentation_mask.size
    region_pixels = np.sum(segmentation_mask > 0)
    region_pct = (region_pixels / total_pixels) * 100
    
    show_interpretation_section(
        task_type='segmentation',
        metrics=metrics if ground_truth else {},
        image_info={'region_percentage': region_pct}
    )
```

---

### 🔄 3. CT Reconstruction Page (`pages/3_CT_Reconstruction.py`)

**Vị trí tích hợp:** Sau reconstruction

```python
# Thêm imports
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Sau khi có reconstructed image
if reconstructed_image is not None:
    
    # 1. So sánh với ground truth (nếu có)
    st.markdown("---")
    st.subheader("🔍 Đánh giá chất lượng tái tạo")
    
    visualizer = ResultVisualizer()
    
    if ground_truth_image is not None:
        visualizer.compare_images(
            ground_truth_image,
            reconstructed_image,
            title_before="Ảnh ground truth",
            title_after="Ảnh tái tạo",
            description=f"Tái tạo từ {num_angles} góc quét. "
                       f"Ít góc quét hơn → nhanh hơn nhưng chất lượng thấp hơn."
        )
        
        # 2. Metrics
        psnr = peak_signal_noise_ratio(ground_truth_image, reconstructed_image)
        ssim = structural_similarity(ground_truth_image, reconstructed_image, data_range=1.0)
        mse = np.mean((ground_truth_image - reconstructed_image) ** 2)
        
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'SNR': psnr - 10  # approximation
        }
        
        # 3. Dashboard
        st.markdown("---")
        MetricsExplainer().show_metrics_dashboard(metrics)
        
    else:
        # Chỉ hiển thị reconstructed
        st.image(reconstructed_image, caption="Ảnh CT tái tạo", use_container_width=True)
        metrics = {}
    
    # 4. Giải thích
    show_interpretation_section(
        task_type='reconstruction',
        metrics=metrics,
        image_info={
            'method': reconstruction_method,  # 'FBP', 'SART', etc.
            'num_angles': num_angles
        }
    )
```

---

### 🧲 4. MRI Reconstruction Page (`pages/4_MRI_Reconstruction.py`)

**Vị trí tích hợp:** Tương tự CT Reconstruction

```python
# Thêm imports
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section
)

# Sau reconstruction
if reconstructed_mri is not None:
    
    # 1. So sánh
    st.markdown("---")
    visualizer = ResultVisualizer()
    
    if ground_truth_mri is not None:
        visualizer.compare_images(
            ground_truth_mri,
            reconstructed_mri,
            title_before="MRI đầy đủ",
            title_after="MRI tái tạo",
            description=f"Tái tạo từ k-space undersampling {sampling_rate}%. "
                       f"Undersampling cao → quét nhanh hơn nhưng mất thông tin."
        )
        
        # Metrics
        psnr = peak_signal_noise_ratio(ground_truth_mri, reconstructed_mri)
        ssim = structural_similarity(ground_truth_mri, reconstructed_mri, data_range=1.0)
        
        metrics = {
            'PSNR': psnr,
            'SSIM': ssim
        }
        
        MetricsExplainer().show_metrics_dashboard(metrics)
    
    # Giải thích
    show_interpretation_section(
        task_type='reconstruction',
        metrics=metrics,
        image_info={
            'method': 'Inverse FFT with undersampled k-space',
            'sampling_rate': sampling_rate
        }
    )
```

---

### 🔒 5. Anonymization Page (`pages/1_Anonymization.py`)

**Vị trí tích hợp:** Sau anonymization

```python
# Thêm imports
from utils.interpretation import show_interpretation_section

# Sau khi anonymize
if anonymized_success:
    
    st.success("✅ Đã ẩn danh hóa thành công!")
    
    # Hiển thị metadata before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metadata gốc")
        st.json({
            'PatientName': original_metadata.get('PatientName', 'N/A'),
            'PatientID': original_metadata.get('PatientID', 'N/A'),
            'PatientBirthDate': original_metadata.get('PatientBirthDate', 'N/A'),
            # ... other fields
        })
    
    with col2:
        st.subheader("Metadata sau ẩn danh")
        st.json({
            'PatientName': 'ANONYMIZED',
            'PatientID': 'ANONYMIZED',
            'PatientBirthDate': 'ANONYMIZED',
            # ...
        })
    
    # Giải thích
    st.markdown("---")
    show_interpretation_section(
        task_type='anonymization',
        metrics={},
        image_info={
            'fields_removed': [
                'PatientName', 'PatientID', 'PatientBirthDate',
                'InstitutionName', 'ReferringPhysicianName'
            ]
        }
    )
```

---

## 🎨 Thêm Export Báo cáo vào tất cả các trang

Thêm section này vào cuối mỗi trang:

```python
# Ở cuối trang, sau tất cả kết quả
if 'results' in st.session_state and st.session_state.results:
    
    st.markdown("---")
    st.subheader("📄 Xuất báo cáo giải thích")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_format = st.selectbox(
            "Định dạng",
            ['PDF', 'HTML']
        )
    
    with col2:
        report_title = st.text_input(
            "Tiêu đề báo cáo",
            value=f"Báo cáo {page_name} - {datetime.now().strftime('%Y%m%d')}"
        )
    
    with col3:
        st.write("")  # spacer
        st.write("")
        generate_report = st.button("🚀 Tạo báo cáo", type="primary")
    
    if generate_report:
        with st.spinner("Đang tạo báo cáo..."):
            try:
                from utils.interpretation import (
                    ReportBuilder,
                    InterpretationGenerator
                )
                
                # Tạo interpretation
                interpretation = InterpretationGenerator.generate_interpretation(
                    task_type=task_type,  # 'segmentation', 'preprocessing', etc.
                    metrics=metrics_dict,
                    image_info=additional_info
                )
                
                # Tạo báo cáo
                report_bytes = ReportBuilder.create_interpretation_report(
                    title=report_title,
                    task_type=task_type,
                    images=images_dict,  # {'name': numpy_array}
                    metrics=metrics_dict,
                    interpretation=interpretation,
                    output_format=report_format.lower()
                )
                
                # Download button
                file_ext = 'pdf' if report_format == 'PDF' else 'html'
                mime_type = 'application/pdf' if report_format == 'PDF' else 'text/html'
                
                st.download_button(
                    label=f"📥 Tải báo cáo {report_format}",
                    data=report_bytes,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                    mime=mime_type
                )
                
                st.success(f"✅ Báo cáo {report_format} đã được tạo!")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi tạo báo cáo: {e}")
```

---

## 🎯 Checklist tích hợp

### ✅ Phase 1: Core Components (DONE)
- [x] Tạo `utils/interpretation.py`
- [x] Tạo `pages/6_Interpretation_Report.py` (standalone demo)
- [x] Test các components hoạt động độc lập

### 🔄 Phase 2: Integration (TODO)
- [ ] **Preprocessing** (`pages/5_Preprocessing.py`)
  - [ ] Add comparison visualization
  - [ ] Add metrics dashboard
  - [ ] Add interpretation section
  - [ ] Add PDF/HTML export

- [ ] **Segmentation** (`pages/2_Segmentation.py`)
  - [ ] Add overlay with legend
  - [ ] Add Dice/IoU metrics
  - [ ] Add interpretation section
  - [ ] Add export

- [ ] **CT Reconstruction** (`pages/3_CT_Reconstruction.py`)
  - [ ] Add comparison with ground truth
  - [ ] Add PSNR/SSIM metrics
  - [ ] Add interpretation section
  - [ ] Add export

- [ ] **MRI Reconstruction** (`pages/4_MRI_Reconstruction.py`)
  - [ ] Add comparison
  - [ ] Add metrics
  - [ ] Add interpretation
  - [ ] Add export

- [ ] **Anonymization** (`pages/1_Anonymization.py`)
  - [ ] Add metadata comparison
  - [ ] Add interpretation
  - [ ] Add export

### 🧪 Phase 3: Testing
- [ ] Test tất cả visualizations
- [ ] Test metrics calculations
- [ ] Test PDF generation
- [ ] Test HTML generation
- [ ] Test với real medical images

---

## 💡 Tips

### 1. Normalize images trước khi visualize
```python
from utils.image_utils import normalize_image

# Normalize về 0-1
img_normalized = normalize_image(img_array)
```

### 2. Handle 3D volumes
```python
# Lấy middle slice
if len(img_array.shape) == 3:
    display_image = img_array[img_array.shape[0] // 2]
else:
    display_image = img_array
```

### 3. Catch exceptions
```python
try:
    # Your visualization code
    visualizer.show_overlay_with_legend(...)
except Exception as e:
    st.error(f"Lỗi khi hiển thị: {e}")
    st.image(image, caption="Ảnh gốc (fallback)")
```

### 4. Progress indicators cho report generation
```python
with st.spinner("Đang tạo báo cáo..."):
    report_bytes = ReportBuilder.create_interpretation_report(...)
```

---

## 📚 References

- `utils/interpretation.py` - Main library
- `pages/6_Interpretation_Report.py` - Example usage
- `UX_IMPROVEMENTS_GUIDE.md` - General UX guidelines

---

## 🎉 Kết quả mong đợi

Sau khi tích hợp xong:

✅ Người dùng thấy **giải thích rõ ràng** cho mọi kết quả
✅ **Metrics được diễn giải** bằng ngôn ngữ dễ hiểu
✅ **Báo cáo PDF/HTML** chuyên nghiệp, đầy đủ
✅ **Trực quan hóa** giúp "nhìn là hiểu"
✅ Phù hợp cho **người không chuyên y học**

---

**Next step:** Bắt đầu tích hợp vào Preprocessing page (đơn giản nhất) rồi lan sang các trang khác! 🚀
