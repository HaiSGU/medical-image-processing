"""
Ứng dụng Xử lý Ảnh Y tế

Giao diện đơn giản để xem và xử lý ảnh y tế.

Author: HaiSGU
Date: 2025-10-28
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import sys

# Thêm src vào path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_io import MedicalImageIO

# Cấu hình trang
st.set_page_config(page_title="Xử lý Ảnh Y tế", layout="wide", page_icon="🏥")

# Khởi tạo session state
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "filename" not in st.session_state:
    st.session_state.filename = None

# Thanh bên
with st.sidebar:
    st.title("🏥 Xử lý Ảnh Y tế")
    st.markdown("---")

    st.info(
        """
    **Tính năng:**
    - Hỗ trợ nhiều định dạng
    - Hiển thị 2D/3D
    - Trích xuất thông tin
    - Phân tích thống kê
    
    **Dùng thanh bên → cho các công cụ khác**
    """
    )

    if st.session_state.image_data is not None:
        st.markdown("---")
        st.subheader("📁 File hiện tại")
        st.text(st.session_state.filename)
        meta = st.session_state.metadata
        st.text(f"Kích thước: {' × '.join(map(str, meta['shape']))}")
        st.text(f"Kiểu: {meta['dtype']}")

# Trang chính
st.title("📤 Tải lên & Xem trước")
st.markdown("Tải lên và xem ảnh y tế")

# Tải file lên
uploaded_file = st.file_uploader(
    "Chọn file ảnh y tế",
    type=["nii", "gz", "dcm", "nrrd", "mha", "mhd", "npy"],
    help="Hỗ trợ: NIfTI, DICOM, NRRD, MetaImage, NumPy",
)

if uploaded_file:
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Đọc file
    try:
        with st.spinner("Đang tải..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

        st.session_state.image_data = image_data
        st.session_state.metadata = metadata
        st.session_state.filename = uploaded_file.name

        st.success(f"✅ Đã tải: {uploaded_file.name}")

    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {str(e)}")
        st.stop()

    # Hiển thị thông tin
    st.markdown("---")
    st.subheader("📊 Thông tin ảnh")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kích thước", f"{' × '.join(map(str, metadata['shape']))}")
    col2.metric("Chiều", f"{metadata['ndim']}D")
    col3.metric("Kiểu dữ liệu", metadata["dtype"])
    col4.metric("Dung lượng (MB)", f"{image_data.nbytes / 1024 / 1024:.2f}")

    # Thống kê
    st.markdown("---")
    st.subheader("📈 Thống kê")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá trị nhỏ nhất", f"{image_data.min():.2f}")
    col2.metric("Giá trị lớn nhất", f"{image_data.max():.2f}")
    col3.metric("Trung bình", f"{image_data.mean():.2f}")
    col4.metric("Độ lệch chuẩn", f"{image_data.std():.2f}")

    # Xem trước
    st.markdown("---")
    st.subheader("🖼️ Xem trước ảnh")

    # Với ảnh 3D, hiển thị lát cắt
    if image_data.ndim == 3:
        slice_idx = st.slider(
            "Chọn lát cắt", 0, image_data.shape[2] - 1, image_data.shape[2] // 2
        )
        slice_data = image_data[:, :, slice_idx]
    else:
        slice_data = image_data

    # Hiển thị ảnh
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(slice_data, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

    # Biểu đồ phân bố
    st.markdown("---")
    st.subheader("📉 Phân bố cường độ")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(image_data.flatten(), bins=50, color="steelblue", alpha=0.7)
    ax.set_xlabel("Cường độ")
    ax.set_ylabel("Tần số")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

else:
    st.info("👆 Tải file lên để bắt đầu")

    with st.expander("📋 Các định dạng hỗ trợ"):
        st.markdown(
            """
        - **NIfTI** (.nii, .nii.gz) - Định dạng ảnh não
        - **DICOM** (.dcm) - Định dạng ảnh y tế chuẩn
        - **NRRD** (.nrrd) - Định dạng nghiên cứu
        - **MetaImage** (.mha, .mhd) - Định dạng ITK
        - **NumPy** (.npy) - Mảng Python
        """
        )

    st.markdown("---")
    st.markdown(
        """
        ### 💡 Hướng dẫn sử dụng
        
        1. **Tải ảnh lên**: Click nút "Browse files" ở trên
        2. **Xem thông tin**: Kiểm tra kích thước, kiểu dữ liệu
        3. **Xem ảnh**: Với ảnh 3D, dùng thanh trượt chọn lát cắt
        4. **Xử lý**: Dùng các công cụ ở thanh bên trái
        
        ### 🔧 Các công cụ khác
        
        - **Anonymization**: Ẩn danh hóa thông tin bệnh nhân
        - **Segmentation**: Phân đoạn vùng quan tâm
        - **CT Reconstruction**: Tái tạo ảnh CT
        - **MRI Reconstruction**: Tái tạo ảnh MRI
        - **Preprocessing**: Tiền xử lý ảnh
        """
    )
