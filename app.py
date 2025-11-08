"""
Ứng dụng Xử lý Ảnh Y tế - Đồ án Sinh viên

Ứng dụng đơn giản giúp hiểu và xử lý ảnh y tế.
Phù hợp cho người không chuyên ngành y.

Tác giả: HaiSGU
Ngày: 2025-10-28
"""

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Thêm src vào path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_io import MedicalImageIO

# Cấu hình trang
st.set_page_config(page_title="Xử lý Ảnh Y tế - Đồ án", layout="wide")

# Khởi tạo session state
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "filename" not in st.session_state:
    st.session_state.filename = None

# Thanh bên
with st.sidebar:
    st.title("Xử lý Ảnh Y tế")
    st.caption("Đồ án Sinh viên - Xử lý Ảnh")
    st.markdown("---")

    st.info(
        """
    **Các chức năng chính:**
    - Hỗ trợ nhiều định dạng ảnh y tế
    - Hiển thị ảnh 2D và 3D
    - Trích xuất thông tin từ ảnh
    - Phân tích và xử lý ảnh
    
    **Hướng dẫn:** Chọn công cụ ở menu bên trái
    """
    )

    if st.session_state.image_data is not None:
        st.markdown("---")
        st.subheader("File hiện tại")
        st.text(st.session_state.filename)
        meta = st.session_state.metadata
        size_str = " × ".join(map(str, meta["shape"]))
        st.text(f"Kích thước: {size_str}")
        st.text(f"Kiểu dữ liệu: {meta['dtype']}")

# Trang chính
st.title("Tải lên và Xem ảnh")
st.markdown("**Bước 1:** Tải ảnh y tế lên để bắt đầu phân tích và xử lý")

# Tải file lên
uploaded_file = st.file_uploader(
    "Chọn file ảnh y tế từ máy tính",
    type=["nii", "gz", "dcm", "nrrd", "mha", "mhd", "npy"],
    help="Các định dạng hỗ trợ: NIfTI, DICOM, NRRD, MetaImage, NumPy",
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

        st.success(f"Đã tải thành công: {uploaded_file.name}")

    except Exception as e:
        st.error(f"Lỗi: Không thể đọc file - {str(e)}")
        st.stop()

    # Hiển thị thông tin
    st.markdown("---")
    st.subheader("Thông tin Ảnh")

    col1, col2, col3, col4 = st.columns(4)
    size_str = " × ".join(map(str, metadata["shape"]))
    col1.metric("Kích thước", size_str)
    col2.metric("Số chiều", f"{metadata['ndim']}D")
    col3.metric("Kiểu dữ liệu", metadata["dtype"])
    size_mb = image_data.nbytes / 1024 / 1024
    col4.metric("Dung lượng (MB)", f"{size_mb:.2f}")

    # Thống kê
    st.markdown("---")
    st.subheader("Thống kê Giá trị Pixel")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá trị nhỏ nhất", f"{image_data.min():.2f}")
    col2.metric("Giá trị lớn nhất", f"{image_data.max():.2f}")
    col3.metric("Giá trị trung bình", f"{image_data.mean():.2f}")
    col4.metric("Độ lệch chuẩn", f"{image_data.std():.2f}")

    # Xem trước
    st.markdown("---")
    st.subheader("Hiển thị Ảnh")

    # Với ảnh 3D, hiển thị lát cắt
    if image_data.ndim == 3:
        max_slice = image_data.shape[2] - 1
        mid_slice = image_data.shape[2] // 2
        slice_idx = st.slider("Chọn lát cắt (slice) để xem", 0, max_slice, mid_slice)
        slice_data = image_data[:, :, slice_idx]
        st.caption(f"Đang xem lát cắt số {slice_idx} / {max_slice}")
    else:
        slice_data = image_data
        st.caption("Ảnh 2D - Hiển thị toàn bộ")

    # Hiển thị ảnh
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(slice_data, cmap="gray")
    ax.axis("off")
    ax.set_title("Ảnh Y tế (Grayscale)", fontsize=14, pad=10)
    st.pyplot(fig)
    plt.close()

    # Biểu đồ phân bố
    st.markdown("---")
    st.subheader("Biểu đồ Phân bố Giá trị Pixel")
    st.caption(
        "Biểu đồ này cho thấy tần suất xuất hiện của " "các giá trị độ sáng trong ảnh"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        image_data.flatten(), bins=50, color="steelblue", alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Giá trị Pixel (Độ sáng)")
    ax.set_ylabel("Tần số xuất hiện")
    ax.set_title("Histogram - Phân bố giá trị pixel")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

else:
    st.info(
        "Vui lòng tải file ảnh y tế lên để bắt đầu. " "Nhấn nút 'Browse files' ở trên."
    )

    with st.expander("Các định dạng file được hỗ trợ"):
        st.markdown(
            """
        **Các loại file ảnh y tế mà ứng dụng có thể đọc:**
        
        - **NIfTI** (.nii, .nii.gz) - Thường dùng cho ảnh não, MRI
        - **DICOM** (.dcm) - Định dạng tiêu chuẩn trong y tế
        - **NRRD** (.nrrd) - Định dạng dùng trong nghiên cứu
        - **MetaImage** (.mha, .mhd) - Định dạng ITK toolkit
        - **NumPy** (.npy) - Mảng số liệu Python
        
        **Giải thích:** Mỗi định dạng có cách lưu trữ khác nhau
        nhưng đều chứa thông tin ảnh y tế 2D hoặc 3D.
        """
        )

    st.markdown("---")
    st.markdown(
        """
        ### Hướng dẫn sử dụng ứng dụng
        
        **Bước 1: Tải ảnh**
        - Nhấn nút "Browse files" phía trên
        - Chọn file ảnh y tế từ máy tính
        - Chờ ứng dụng tải và phân tích
        
        **Bước 2: Xem thông tin**
        - Kiểm tra kích thước ảnh (width × height × depth)
        - Xem thống kê giá trị pixel
        - Quan sát biểu đồ phân bố
        
        **Bước 3: Xử lý ảnh**
        - Chọn công cụ ở menu bên trái
        - Làm theo hướng dẫn trong từng công cụ
        
        ### Các công cụ xử lý ảnh
        
        1. **Anonymization** - Xóa thông tin cá nhân trong ảnh DICOM
        2. **Segmentation** - Tách vùng quan tâm (ví dụ: vùng não)
        3. **CT Reconstruction** - Tái tạo ảnh CT từ dữ liệu thô
        4. **MRI Reconstruction** - Tái tạo ảnh MRI từ K-space
        5. **Preprocessing** - Cải thiện chất lượng ảnh
        
        ---
        
        **Lưu ý:** Ứng dụng này chỉ phục vụ mục đích học tập
        và nghiên cứu, không dùng cho chẩn đoán y khoa thực tế.
        """
    )
