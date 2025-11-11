"""
Trang Tiền xử lý Ảnh - ENHANCED VERSION

Áp dụng các phép tiền xử lý cho ảnh y tế với:
- Progress bars
- Image comparison slider
- Batch processing
- PDF/ZIP export

Tác giả: HaiSGU
Ngày: 2025-11-11
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.preprocessing.image_transforms import ImageTransforms
from utils.file_io import MedicalImageIO
from utils.ui_components import (
    ProgressTracker,
    ImageComparer,
    BatchProcessor,
    ResultExporter,
    show_metrics_dashboard,
    show_preview_gallery,
    create_download_section,
)

# Page config
st.set_page_config(
    page_title="🔧 Tiền xử lý Ảnh", layout="wide", initial_sidebar_state="expanded"
)

# Initialize session state
if "prep_images" not in st.session_state:
    st.session_state.prep_images = {}  # {filename: original_image}
if "prep_processed" not in st.session_state:
    st.session_state.prep_processed = {}  # {filename: processed_image}
if "prep_operations" not in st.session_state:
    st.session_state.prep_operations = []
if "prep_metrics" not in st.session_state:
    st.session_state.prep_metrics = {}

# Header
st.title("🔧 Tiền xử lý Ảnh")
st.markdown("Biến đổi và nâng cao chất lượng ảnh y tế để phân tích")

# Info
with st.expander("📖 Hướng dẫn sử dụng"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Tính năng mới:**
        
        - ✅ **Upload nhiều ảnh** cùng lúc
        - ✅ **Xử lý batch** tự động
        - ✅ **So sánh trước/sau** với slider
        - ✅ **Export PDF** hoặc ZIP
        - ✅ **Progress bar** hiển thị tiến độ
        
        **Quy trình:**
        1. Upload 1 hoặc nhiều ảnh
        2. Chọn các phép toán
        3. Click "Áp dụng"
        4. So sánh kết quả
        5. Download PDF/ZIP
        """
        )

    with col2:
        st.markdown(
            """
        **Thứ tự khuyến nghị:**
        
        1. **Normalize** - Chuẩn hóa cường độ
        2. **Denoise** - Khử nhiễu
        3. **Resize** - Thay đổi kích thước
        4. **Enhance** - Tăng độ tương phản
        
        **Mẹo:**
        - Áp dụng từng bước một
        - Kiểm tra preview trước khi lưu
        - Dùng batch cho nhiều ảnh cùng loại
        """
        )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")

    # Processing mode
    mode = st.radio(
        "Chế độ xử lý:",
        ["Single Image", "Batch Processing"],
        help="Single: 1 ảnh | Batch: nhiều ảnh",
    )

    st.markdown("---")
    st.markdown("### 🔧 Các phép toán")

    # Normalize
    normalize_enabled = st.checkbox("✓ Chuẩn hóa Cường độ", value=False)
    if normalize_enabled:
        norm_method = st.selectbox(
            "Phương pháp:",
            ["Min-Max (0-1)", "Z-Score", "Cắt phân vị"],
        )

        if norm_method == "Cắt phân vị":
            lower_p = st.slider("Phân vị dưới (%)", 0, 50, 2)
            upper_p = st.slider("Phân vị trên (%)", 50, 100, 98)

    # Denoise
    denoise_enabled = st.checkbox("✓ Khử nhiễu", value=False)
    if denoise_enabled:
        denoise_method = st.selectbox(
            "Phương pháp:",
            ["Gaussian", "Median"],
        )

        if denoise_method == "Gaussian":
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        else:  # Median
            kernel_size = st.slider("Kernel size", 3, 11, 5, 2)

    # Resize
    resize_enabled = st.checkbox("✓ Thay đổi kích thước", value=False)
    if resize_enabled:
        resize_method = st.selectbox(
            "Loại:",
            ["Scale Factor", "Target Size"],
        )

        if resize_method == "Scale Factor":
            scale = st.slider("Scale factor", 0.1, 3.0, 1.0, 0.1)
        else:
            target_width = st.number_input("Width", 64, 2048, 512, 64)
            target_height = st.number_input("Height", 64, 2048, 512, 64)

    # Enhance Contrast
    enhance_enabled = st.checkbox("✓ Tăng độ tương phản", value=False)
    if enhance_enabled:
        enhance_method = st.selectbox(
            "Phương pháp:",
            ["Histogram Equalization", "CLAHE", "Adaptive"],
        )

        if enhance_method == "CLAHE":
            clip_limit = st.slider("Clip limit", 0.01, 0.1, 0.03, 0.01)

st.markdown("---")

# Main content
if mode == "Single Image":
    st.subheader("📁 Upload Ảnh")

    uploaded_file = st.file_uploader(
        "Chọn file ảnh y tế",
        type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
        help="Hỗ trợ: NIfTI, DICOM, NRRD, MetaImage, NumPy",
    )

    if uploaded_file:
        # Load image with progress
        tracker = ProgressTracker("Đang tải ảnh", total_steps=1)
        tracker.update(0, f"Đọc file: {uploaded_file.name}")

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

            # Store in session state
            st.session_state.prep_images = {uploaded_file.name: image_data}

            tracker.complete(f"✅ Đã tải: {uploaded_file.name}")

            # Show image info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Shape", f"{image_data.shape}")
            col2.metric("Dtype", f"{image_data.dtype}")
            col3.metric("Min", f"{image_data.min():.2f}")
            col4.metric("Max", f"{image_data.max():.2f}")

        except Exception as e:
            tracker.error(f"❌ Lỗi: {str(e)}")
            st.stop()

else:  # Batch Processing
    st.subheader("📁 Upload Nhiều Ảnh (Batch)")

    batch_processor = BatchProcessor()
    uploaded_files = batch_processor.upload_multiple(
        "Chọn nhiều files", ["nii", "gz", "dcm", "nrrd", "mha", "npy"], max_files=20
    )

    if uploaded_files:
        # Load all images
        def load_image(file):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix
            ) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            io_handler = MedicalImageIO()
            image_data, _ = io_handler.read_image(tmp_path)
            return image_data

        results = batch_processor.process_files(uploaded_files, load_image)

        # Store in session state
        st.session_state.prep_images = {
            name: img for name, img in results if img is not None
        }

        if st.session_state.prep_images:
            st.success(f"✅ Đã tải {len(st.session_state.prep_images)} ảnh")

# Processing section
if st.session_state.prep_images:
    st.markdown("---")
    st.subheader("🎨 Xử lý Ảnh")

    if st.button("▶️ Áp dụng Các phép toán", type="primary", use_container_width=True):
        # Calculate total operations
        ops_enabled = sum(
            [normalize_enabled, denoise_enabled, resize_enabled, enhance_enabled]
        )

        if ops_enabled == 0:
            st.warning("⚠️ Vui lòng chọn ít nhất 1 phép toán!")
        else:
            total_steps = len(st.session_state.prep_images) * ops_enabled
            tracker = ProgressTracker("Đang xử lý", total_steps)

            step = 0
            processed_images = {}
            operations_applied = []

            for filename, original in st.session_state.prep_images.items():
                processed = original.copy()

                # Apply operations
                transformer = ImageTransforms(processed)

                if normalize_enabled:
                    step += 1
                    tracker.update(step, f"{filename}: Normalize...")

                    if norm_method == "Min-Max (0-1)":
                        processed = transformer.normalize_minmax(0.0, 1.0)
                        operations_applied.append("Normalize: Min-Max")
                    elif norm_method == "Z-Score":
                        processed = transformer.normalize_zscore()
                        operations_applied.append("Normalize: Z-Score")
                    else:
                        processed = transformer.normalize_percentile(lower_p, upper_p)
                        operations_applied.append(
                            f"Normalize: Percentile {lower_p}-{upper_p}"
                        )

                    transformer = ImageTransforms(processed)

                if denoise_enabled:
                    step += 1
                    tracker.update(step, f"{filename}: Denoise...")

                    if denoise_method == "Gaussian":
                        processed = transformer.denoise_gaussian(sigma=sigma)
                        operations_applied.append(f"Denoise: Gaussian (σ={sigma})")
                    else:
                        processed = transformer.denoise_median(size=kernel_size)
                        operations_applied.append(f"Denoise: Median (k={kernel_size})")

                    transformer = ImageTransforms(processed)

                if resize_enabled:
                    step += 1
                    tracker.update(step, f"{filename}: Resize...")

                    if resize_method == "Scale Factor":
                        processed = transformer.resize_by_factor(scale)
                        operations_applied.append(f"Resize: Scale {scale}x")
                    else:
                        processed = transformer.resize_to_shape(
                            (target_height, target_width)
                        )
                        operations_applied.append(
                            f"Resize: {target_width}×{target_height}"
                        )

                    transformer = ImageTransforms(processed)

                if enhance_enabled:
                    step += 1
                    tracker.update(step, f"{filename}: Enhance...")

                    if enhance_method == "Histogram Equalization":
                        processed = transformer.histogram_equalization()
                        operations_applied.append("Enhance: Histogram Eq")
                    elif enhance_method == "CLAHE":
                        processed = transformer.adaptive_histogram_equalization(
                            clip_limit
                        )
                        operations_applied.append(f"Enhance: CLAHE (clip={clip_limit})")
                    else:
                        processed = transformer.adaptive_histogram_equalization()
                        operations_applied.append("Enhance: Adaptive")

                processed_images[filename] = processed

            # Store results
            st.session_state.prep_processed = processed_images
            st.session_state.prep_operations = list(set(operations_applied))

            tracker.complete("✅ Hoàn thành!")

# Display results
if st.session_state.prep_processed:
    st.markdown("---")
    st.subheader("📊 Kết quả")

    # Metrics
    total_images = len(st.session_state.prep_processed)
    operations_count = len(st.session_state.prep_operations)

    metrics = {
        "Số ảnh": total_images,
        "Phép toán": operations_count,
        "Thời gian": f"{datetime.now():%H:%M:%S}",
    }

    # Add shape metrics for first image
    first_name = list(st.session_state.prep_processed.keys())[0]
    first_processed = st.session_state.prep_processed[first_name]

    metrics.update(
        {
            "Shape mới": str(first_processed.shape),
            "Min": f"{first_processed.min():.4f}",
            "Max": f"{first_processed.max():.4f}",
            "Mean": f"{first_processed.mean():.4f}",
        }
    )

    show_metrics_dashboard(metrics)

    st.markdown("---")

    # Image comparison
    if len(st.session_state.prep_processed) == 1:
        # Single image - detailed comparison
        filename = list(st.session_state.prep_processed.keys())[0]
        original = st.session_state.prep_images[filename]
        processed = st.session_state.prep_processed[filename]

        comparer = ImageComparer()
        comparer.show(original, processed, "Ảnh gốc", "Đã xử lý")

    else:
        # Multiple images - gallery view
        tab1, tab2 = st.tabs(["📸 Gallery Gốc", "✨ Gallery Đã xử lý"])

        with tab1:
            show_preview_gallery(
                st.session_state.prep_images, columns=3, title="Ảnh Gốc"
            )

        with tab2:
            show_preview_gallery(
                st.session_state.prep_processed, columns=3, title="Đã Xử lý"
            )

    # Export section
    results_to_export = {
        "images": {
            **{f"original_{k}": v for k, v in st.session_state.prep_images.items()},
            **{f"processed_{k}": v for k, v in st.session_state.prep_processed.items()},
        },
        "metrics": metrics,
        "description": f"Preprocessing Report\n\nOperations Applied:\n"
        + "\n".join(f"- {op}" for op in st.session_state.prep_operations),
    }

    create_download_section(results_to_export, "preprocessing")

else:
    st.info("👆 Upload ảnh và chọn phép toán để bắt đầu")

# Footer
st.markdown("---")
st.caption(
    "💡 **Mẹo:** Dùng batch processing cho nhiều ảnh cùng loại. Export PDF để lưu report đầy đủ."
)
