"""
Trang Phân đoạn Não

Phân đoạn các vùng não từ ảnh y tế sử dụng nhiều phương pháp.

Tác giả: HaiSGU
Ngày: 2025-10-28
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src/ modules
from src.segmentation.brain_segmentation import BrainSegmentation
from utils.file_io import MedicalImageIO

# Page config
st.set_page_config(page_title="🧠 Phân đoạn Não", layout="wide")

# Initialize session state
if "seg_image_data" not in st.session_state:
    st.session_state.seg_image_data = None
if "seg_mask" not in st.session_state:
    st.session_state.seg_mask = None
if "seg_metadata" not in st.session_state:
    st.session_state.seg_metadata = {}

# Header
st.title("🧠 Phân đoạn Não")
st.markdown("Trích xuất các vùng não từ ảnh y tế")

# Info
with st.expander("📚 Phương pháp Phân đoạn"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Ngưỡng (Threshold):**
        - Phân đoạn dựa trên cường độ đơn giản
        - Nhanh và dễ hiểu
        - Tốt cho ảnh có độ tương phản cao
        
        **Phương pháp Otsu:**
        - Tự động chọn ngưỡng
        - Không cần tham số thủ công
        - Hoạt động tốt với histogram hai đỉnh
        """
        )

    with col2:
        st.markdown(
            """
        **Tăng trưởng vùng (Region Growing):**
        - Phát triển từ điểm khởi đầu
        - Ranh giới chính xác hơn
        - Cần chọn điểm khởi đầu
        
        **Tự động:**
        - Kết hợp nhiều phương pháp
        - Kết quả tổng thể tốt nhất
        - Khuyến nghị cho người mới bắt đầu
        """
        )

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Cài đặt Phân đoạn")

    method = st.selectbox(
        "Phương pháp",
        ["Tự động", "Ngưỡng", "Otsu", "Tăng trưởng vùng"],
        help="Chọn phương pháp phân đoạn",
    )

    # Map Vietnamese to English for processing
    method_map = {
        "Tự động": "Automatic",
        "Ngưỡng": "Threshold",
        "Otsu": "Otsu",
        "Tăng trưởng vùng": "Region Growing",
    }
    method_en = method_map[method]

    st.markdown("---")

    # Method-specific parameters
    if method == "Ngưỡng":
        threshold = st.slider(
            "Giá trị ngưỡng",
            min_value=0,
            max_value=255,
            value=50,
            help="Pixel trên giá trị này sẽ được giữ lại",
        )

    elif method == "Tăng trưởng vùng":
        st.markdown("**Điểm khởi đầu (%):**")
        seed_x = st.slider("Vị trí X", 0, 100, 50)
        seed_y = st.slider("Vị trí Y", 0, 100, 50)
        seed_z = st.slider("Vị trí Z", 0, 100, 50)

        intensity_tolerance = st.slider(
            "Dung sai cường độ",
            min_value=1,
            max_value=50,
            value=10,
            help="Chênh lệch tối đa so với cường độ điểm khởi đầu",
        )

    # Morphological operations
    st.markdown("---")
    st.markdown("**Xử lý sau:**")

    apply_morph = st.checkbox("Áp dụng phép biến đổi hình thái", value=True)

    if apply_morph:
        morph_op = st.selectbox(
            "Phép toán",
            ["đóng (closing)", "mở (opening)", "giãn (dilation)", "xói mòn (erosion)"],
            help="Phép toán hình thái để làm sạch mask",
        )

        # Map to English
        morph_map = {
            "đóng (closing)": "closing",
            "mở (opening)": "opening",
            "giãn (dilation)": "dilation",
            "xói mòn (erosion)": "erosion",
        }
        morph_op_en = morph_map[morph_op]

        kernel_size = st.slider("Kích thước Kernel", 1, 10, 3)

    keep_largest = st.checkbox(
        "Chỉ giữ thành phần lớn nhất",
        value=True,
        help="Loại bỏ các vùng nhỏ không liên kết",
    )

    st.markdown("---")
    st.info("💡 Thử phương pháp 'Tự động' trước để có kết quả tốt nhất")

# File upload
st.subheader("📤 Tải lên Ảnh Y tế")

uploaded_file = st.file_uploader(
    "Chọn file (.nii, .nii.gz, .dcm, .nrrd, .mha, .npy)",
    type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
    help="Tải lên ảnh chụp não",
)

if uploaded_file:
    # Load image
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded_file.name).suffix
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Đang tải ảnh..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

            st.session_state.seg_image_data = image_data
            st.session_state.seg_metadata = metadata

        st.success(f"✅ Đã tải: {uploaded_file.name}")

        # Show image info
        col1, col2, col3 = st.columns(3)
        col1.metric("Kích thước", f"{' × '.join(map(str, metadata['shape']))}")
        col2.metric("Kiểu dữ liệu", metadata["dtype"])
        col3.metric("Chiều", f"{metadata['ndim']}D")

    except Exception as e:
        st.error(f"❌ Lỗi khi tải ảnh: {str(e)}")
        st.stop()

    st.markdown("---")

    # Segmentation button
    if st.button("🧠 Phân đoạn Não", type="primary", use_container_width=True):

        with st.spinner("Đang phân đoạn..."):
            try:
                # Create segmenter
                segmenter = BrainSegmentation()

                # Run segmentation based on method
                if method_en == "Automatic":
                    mask = segmenter.segment_brain(image_data)

                elif method_en == "Threshold":
                    mask = segmenter.threshold_segmentation(
                        image_data, threshold=threshold
                    )

                elif method_en == "Otsu":
                    mask = segmenter.otsu_segmentation(image_data)

                elif method_en == "Region Growing":
                    # Convert percentage to actual coordinates
                    shape = image_data.shape
                    seed = [
                        int(seed_x * shape[0] / 100),
                        int(seed_y * shape[1] / 100),
                        int(seed_z * shape[2] / 100) if len(shape) > 2 else 0,
                    ]

                    # For 2D images
                    if len(shape) == 2:
                        seed = seed[:2]

                    mask = segmenter.region_growing_segmentation(
                        image_data,
                        seed_point=tuple(seed),
                        intensity_tolerance=intensity_tolerance,
                    )

                # Apply morphological operations
                if apply_morph:
                    if morph_op_en == "closing":
                        mask = segmenter.morphological_closing(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op_en == "opening":
                        mask = segmenter.morphological_opening(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op_en == "dilation":
                        mask = segmenter.morphological_dilation(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op_en == "erosion":
                        mask = segmenter.morphological_erosion(
                            mask, kernel_size=kernel_size
                        )

                # Keep largest component
                if keep_largest:
                    mask = segmenter.keep_largest_component(mask)

                # Store in session state
                st.session_state.seg_mask = mask

                st.success("✅ Phân đoạn hoàn tất!")

            except Exception as e:
                st.error(f"❌ Phân đoạn thất bại: {str(e)}")
                st.exception(e)
                st.stop()

    # Display results
    if st.session_state.seg_mask is not None:
        st.markdown("---")
        st.header("📊 Kết quả Phân đoạn")

        image_data = st.session_state.seg_image_data
        mask = st.session_state.seg_mask

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        total_voxels = mask.size
        segmented_voxels = np.sum(mask > 0)
        percentage = (segmented_voxels / total_voxels) * 100

        col1.metric("Tổng Voxel", f"{total_voxels:,}")
        col2.metric("Đã phân đoạn", f"{segmented_voxels:,}")
        col3.metric("Tỷ lệ", f"{percentage:.1f}%")
        col4.metric("Nền", f"{total_voxels - segmented_voxels:,}")

        st.markdown("---")

        # Visualization
        st.subheader("👁️ Trực quan hóa")

        # View controls
        if image_data.ndim == 3:
            col1, col2 = st.columns([3, 1])

            with col1:
                view_mode = st.radio(
                    "Chế độ xem:", ["Gốc", "Mask", "Phủ lớp"], horizontal=True
                )

            with col2:
                opacity = st.slider("Độ mờ phủ lớp", 0.0, 1.0, 0.5)

            # Slice navigation
            axis = st.radio(
                "Trục:",
                ["Trục Z (Axial)", "Trục Y (Coronal)", "Trục X (Sagittal)"],
                horizontal=True,
            )

            if axis == "Trục Z (Axial)":
                max_slice = image_data.shape[2] - 1
                slice_idx = st.slider("Lát cắt", 0, max_slice, max_slice // 2)
                img_slice = image_data[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx]

            elif axis == "Trục Y (Coronal)":
                max_slice = image_data.shape[1] - 1
                slice_idx = st.slider("Lát cắt", 0, max_slice, max_slice // 2)
                img_slice = image_data[:, slice_idx, :]
                mask_slice = mask[:, slice_idx, :]

            else:  # Sagittal
                max_slice = image_data.shape[0] - 1
                slice_idx = st.slider("Lát cắt", 0, max_slice, max_slice // 2)
                img_slice = image_data[slice_idx, :, :]
                mask_slice = mask[slice_idx, :, :]

        else:  # 2D image
            view_mode = st.radio(
                "Chế độ xem:", ["Gốc", "Mask", "Phủ lớp"], horizontal=True
            )
            opacity = st.slider("Độ mờ phủ lớp", 0.0, 1.0, 0.5)

            img_slice = image_data
            mask_slice = mask

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Map view mode
        view_map = {"Gốc": "Original", "Mask": "Mask", "Phủ lớp": "Overlay"}
        view_mode_en = view_map.get(view_mode, view_mode)

        if view_mode_en == "Original":
            ax.imshow(img_slice.T, cmap="gray", origin="lower")
            ax.set_title("Ảnh gốc", fontsize=14, fontweight="bold")

        elif view_mode_en == "Mask":
            ax.imshow(mask_slice.T, cmap="hot", origin="lower")
            ax.set_title("Mask phân đoạn", fontsize=14, fontweight="bold")

        else:  # Overlay
            ax.imshow(img_slice.T, cmap="gray", origin="lower")

            # Create transparent colormap for mask
            colors = [(0, 0, 0, 0), (1, 0, 0, opacity)]
            n_bins = 2
            cmap = ListedColormap(colors)

            ax.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=opacity)
            ax.set_title("Phủ lớp (Đỏ = Đã phân đoạn)", fontsize=14, fontweight="bold")

        ax.axis("off")
        st.pyplot(fig)
        plt.close()

        # Download options
        st.markdown("---")
        st.subheader("📥 Tải về Kết quả")

        col1, col2 = st.columns(2)

        with col1:
            # Download mask as NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, mask)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="📥 Tải Mask (.npy)",
                data=npy_bytes,
                file_name="mask_phan_doan.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Download overlay image
            fig_download, ax_download = plt.subplots(figsize=(10, 10))

            if image_data.ndim == 3:
                mid_slice = image_data.shape[2] // 2
                img_slice = image_data[:, :, mid_slice]
                mask_slice = mask[:, :, mid_slice]
            else:
                img_slice = image_data
                mask_slice = mask

            ax_download.imshow(img_slice.T, cmap="gray", origin="lower")

            colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]
            cmap = ListedColormap(colors)
            ax_download.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=0.5)
            ax_download.axis("off")

            img_buffer = io.BytesIO()
            fig_download.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close(fig_download)

            st.download_button(
                label="📥 Tải Phủ lớp (.png)",
                data=img_buffer,
                file_name="phan_doan_phu_lop.png",
                mime="image/png",
            )

else:
    st.info("👆 Tải lên ảnh não để bắt đầu phân đoạn")

    st.markdown("---")
    st.subheader("📖 Hướng dẫn Nhanh")

    st.markdown(
        """
    **Các bước:**
    1. Tải lên ảnh chụp não (NIfTI, DICOM, v.v.)
    2. Chọn phương pháp phân đoạn
    3. Điều chỉnh tham số (tùy chọn)
    4. Nhấn "Phân đoạn Não"
    5. Xem và tải về kết quả
    
    **Cài đặt Khuyến nghị:**
    - **Phương pháp:** Bắt đầu với "Tự động"
    - **Xử lý sau:** Bật phép đóng hình thái học
    - **Giữ lớn nhất:** Luôn bật
    
    **Mẹo:**
    - Dùng Otsu để tự động chọn ngưỡng
    - Tăng trưởng vùng hoạt động tốt nhất với ranh giới rõ ràng
    - Thử các góc nhìn khác nhau (Axial/Coronal/Sagittal) cho ảnh 3D
    """
    )

# Footer
st.markdown("---")
st.caption(
    "💡 Mẹo: Thử các phương pháp khác nhau và so sánh kết quả để có độ chính xác tốt nhất"
)
