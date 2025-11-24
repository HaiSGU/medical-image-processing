"""
Trang Phân đoạn Não - Brain Segmentation Page

Phân đoạn các vùng não từ ảnh y tế sử dụng nhiều phương pháp khác nhau:
- Ngưỡng (Threshold): Phân đoạn dựa trên cường độ pixel
- Phương pháp Otsu: Tự động chọn ngưỡng tối ưu
- Tăng trưởng vùng (Region Growing): Phát triển từ điểm khởi đầu
- Tự động: Kết hợp nhiều phương pháp để có kết quả tốt nhất
"""

# ==================== IMPORT CÁC THƯ VIỆN ====================
import io  # Xử lý byte streams
import sys  # Tương tác với hệ thống Python
import tempfile  # Tạo file tạm thời
from pathlib import Path  # Xử lý đường dẫn

import matplotlib.pyplot as plt  # Vẽ đồ thị và hình ảnh
import numpy as np  # Xử lý mảng số
import streamlit as st  # Framework web app
from matplotlib.colors import ListedColormap  # Tạo colormap tùy chỉnh

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
# Thêm thư mục gốc vào sys.path để import các module trong project
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import các module từ project
from src.segmentation.brain_segmentation import BrainSegmentation
from utils.file_io import MedicalImageIO
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)
from utils.image_explainer import (
    explain_input_image,
    explain_method_options,
    explain_output_results,
    create_quality_score,
)

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(page_title="Phân đoạn Não", layout="wide")

# ==================== KHỞI TẠO SESSION STATE ====================
# Sử dụng session_state để lưu dữ liệu giữa các lần chạy lại
if "seg_image_data" not in st.session_state:
    st.session_state.seg_image_data = None  # Dữ liệu ảnh gốc
if "seg_mask" not in st.session_state:
    st.session_state.seg_mask = None  # Mask phân đoạn
if "seg_metadata" not in st.session_state:
    st.session_state.seg_metadata = {}  # Thông tin metadata

# ==================== TIÊU ĐỀ TRANG ====================
st.title("Phân đoạn Não")
st.markdown("Trích xuất các vùng não từ ảnh y tế bằng nhiều phương pháp")

# ==================== THÔNG TIN VỀ PHƯƠNG PHÁP ====================
# Expander để hiển thị thông tin về các phương pháp phân đoạn
with st.expander("Phương pháp Phân đoạn"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Ngưỡng (Threshold):**
            - Phân đoạn dựa trên cường độ đơn giản
            - Nhanh và dễ hiểu
            - Tốt cho ảnh có độ tương phản cao
            
            **Phương pháp Otsu:**
            - Tự động chọn ngưỡng tối ưu
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

# ==================== THANH BÊN - CÀI ĐẶT ====================
with st.sidebar:
    st.header("Cài đặt Phân đoạn")

    # Chọn phương pháp phân đoạn
    method = st.selectbox(
        "Phương pháp",
        ["Tự động", "Ngưỡng", "Otsu", "Tăng trưởng vùng"],
        help="Chọn phương pháp phân đoạn phù hợp",
    )

    # Ánh xạ tên tiếng Việt sang tiếng Anh để xử lý
    method_map = {
        "Tự động": "Automatic",
        "Ngưỡng": "Threshold",
        "Otsu": "Otsu",
        "Tăng trưởng vùng": "Region Growing",
    }
    method_en = method_map[method]

    # === GIẢI THÍCH PHƯƠNG PHÁP ===
    with st.expander(f"💡 Giải thích: {method}"):
        if method == "Tự động":
            st.markdown(
                """
            **⭐ Khuyến nghị cho người mới**
            
            **Mô tả:**  
            Tự động kết hợp nhiều phương pháp để cho kết quả tốt nhất.
            
            **Ưu điểm:**
            - ✅ Hoàn toàn tự động, không cần chọn tham số
            - ✅ Kết quả ổn định với nhiều loại ảnh
            - ✅ Phù hợp cho người không chuyên
            
            **Nhược điểm:**
            - ⚠️ Chậm hơn các phương pháp đơn giản
            - ⚠️ Không tùy chỉnh được chi tiết
            
            **Khi nào dùng:**
            - Bạn chưa biết phương pháp nào phù hợp
            - Cần kết quả nhanh và đáng tin cậy
            - Ảnh có chất lượng tốt
            """
            )

        elif method == "Ngưỡng":
            st.markdown(
                """
            **📏 Phân đoạn theo ngưỡng**
            
            **Mô tả:**  
            Giữ lại các pixel có giá trị trên ngưỡng bạn chọn.
            
            **Ưu điểm:**
            - ✅ Đơn giản, dễ hiểu
            - ✅ Rất nhanh
            - ✅ Kiểm soát hoàn toàn
            
            **Nhược điểm:**
            - ⚠️ Phải chọn ngưỡng thủ công
            - ⚠️ Kết quả phụ thuộc nhiều vào ngưỡng
            - ⚠️ Không tốt với ảnh nhiễu
            
            **Khi nào dùng:**
            - Bạn biết rõ giá trị ngưỡng phù hợp
            - Ảnh có độ tương phản rõ rệt
            - Cần xử lý cực nhanh
            
            **Mẹo:** Thử các giá trị 40-80 với ảnh MRI não
            """
            )

        elif method == "Otsu":
            st.markdown(
                """
            **🎯 Otsu - Tự động tìm ngưỡng tối ưu**
            
            **Mô tả:**  
            Tự động tính toán ngưỡng tốt nhất dựa trên histogram.
            
            **Ưu điểm:**
            - ✅ Hoàn toàn tự động
            - ✅ Rất nhanh (~2 giây)
            - ✅ Hiệu quả với ảnh có 2 vùng rõ rệt
            - ✅ Thuật toán kinh điển, đáng tin cậy
            
            **Nhược điểm:**
            - ⚠️ Kém hiệu quả với ảnh có nhiều vùng
            - ⚠️ Nhạy cảm với nhiễu
            - ⚠️ Giả định ảnh có 2 phân bố
            
            **Khi nào dùng:**
            - Ảnh có độ tương phản cao
            - Phân đoạn não từ MRI (não vs nền)
            - Cần tự động nhưng muốn nhanh
            
            **Phù hợp nhất:** Ảnh MRI T1-weighted
            """
            )

        elif method == "Tăng trưởng vùng":
            st.markdown(
                """
            **🌱 Region Growing - Phát triển từ điểm khởi đầu**
            
            **Mô tả:**  
            Bắt đầu từ 1 điểm, lan rộng sang các pixel tương tự.
            
            **Ưu điểm:**
            - ✅ Rất chính xác nếu chọn đúng điểm khởi đầu
            - ✅ Tốt cho vùng có ranh giới rõ ràng
            - ✅ Có thể xử lý ảnh phức tạp
            
            **Nhược điểm:**
            - ⚠️ Phải chọn điểm khởi đầu thủ công
            - ⚠️ Chậm hơn (~5 giây)
            - ⚠️ Nhạy cảm với điểm khởi đầu
            - ⚠️ Có thể bị "rò rỉ" sang vùng khác
            
            **Khi nào dùng:**
            - Cần độ chính xác cao
            - Ranh giới vùng rõ ràng
            - Biết chính xác vị trí vùng cần phân đoạn
            
            **Mẹo:**  
            - Chọn điểm giữa vùng não (50%, 50%, 50%)
            - Dung sai thấp (5-10) cho kết quả chính xác
            """
            )

    st.markdown("---")

    # ===== THAM SỐ RIÊNG CHO TỪNG PHƯƠNG PHÁP =====
    # Nếu chọn phương pháp Ngưỡng
    if method == "Ngưỡng":
        threshold = st.slider(
            "Giá trị ngưỡng",
            min_value=0,
            max_value=255,
            value=50,
            help="Pixel có giá trị trên ngưỡng này sẽ được giữ lại",
        )

    # Nếu chọn phương pháp Tăng trưởng vùng
    elif method == "Tăng trưởng vùng":
        st.markdown("**Điểm khởi đầu (%):**")
        # Ba slider để chọn vị trí điểm khởi đầu theo 3 trục
        seed_x = st.slider("Vị trí X", 0, 100, 50)
        seed_y = st.slider("Vị trí Y", 0, 100, 50)
        seed_z = st.slider("Vị trí Z", 0, 100, 50)

        # Dung sai cường độ - độ chênh lệch cho phép
        intensity_tolerance = st.slider(
            "Dung sai cường độ",
            min_value=1,
            max_value=50,
            value=10,
            help="Chênh lệch cường độ tối đa so với điểm khởi đầu",
        )

    # ===== PHÉP BIẾN ĐỔI HÌNH THÁI (MORPHOLOGICAL OPERATIONS) =====
    st.markdown("---")
    st.markdown("**Xử lý sau phân đoạn:**")

    # Checkbox để bật/tắt phép biến đổi hình thái
    apply_morph = st.checkbox("Áp dụng phép biến đổi hình thái", value=True)

    if apply_morph:
        # Chọn loại phép toán hình thái
        morph_op = st.selectbox(
            "Phép toán",
            ["đóng (closing)", "mở (opening)", "giãn (dilation)", "xói mòn (erosion)"],
            help="Phép toán hình thái để làm sạch mask",
        )

        # Ánh xạ sang tiếng Anh
        morph_map = {
            "đóng (closing)": "closing",
            "mở (opening)": "opening",
            "giãn (dilation)": "dilation",
            "xói mòn (erosion)": "erosion",
        }
        morph_op_en = morph_map[morph_op]

        # Kích thước kernel (ma trận nhân) cho phép toán
        kernel_size = st.slider("Kích thước Kernel", 1, 10, 3)

    # Checkbox để chỉ giữ lại thành phần lớn nhất
    keep_largest = st.checkbox(
        "Chỉ giữ thành phần lớn nhất",
        value=True,
        help="Loại bỏ các vùng nhỏ không liên kết",
    )

    st.markdown("---")
    st.info("Thử phương pháp 'Tự động' trước để có kết quả tốt nhất")

# ==================== TẢI LÊN HÌNH ẢNH ====================
st.subheader("Tải lên Ảnh Y tế")

uploaded_file = st.file_uploader(
    "Chọn file (.nii, .nii.gz, .dcm, .nrrd, .mha, .npy)",
    type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
    help="Tải lên ảnh chụp não để phân đoạn",
)

# ==================== XỬ LÝ KHI CÓ FILE TẢI LÊN ====================
if uploaded_file:
    # Xử lý phần mở rộng file phức hợp như .nii.gz
    if uploaded_file.name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = Path(uploaded_file.name).suffix

    # Tạo file tạm để lưu dữ liệu tải lên
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Đọc ảnh y tế
        with st.spinner("Đang tải ảnh..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

            # Lưu vào session state
            st.session_state.seg_image_data = image_data
            st.session_state.seg_metadata = metadata

        st.success(f"Đã tải: {uploaded_file.name}")

        # Hiển thị thông tin ảnh
        col1, col2, col3 = st.columns(3)
        col1.metric("Kích thước", f"{' × '.join(map(str, metadata['shape']))}")
        col2.metric("Kiểu dữ liệu", metadata["dtype"])
        col3.metric("Số chiều", f"{len(metadata['shape'])}D")

        st.markdown("---")

        # === GIẢI THÍCH ẢNH ĐẦU VÀO ===
        explain_input_image(image_data, metadata)

    except Exception as e:
        st.error(f"Lỗi khi tải ảnh: {str(e)}")
        st.stop()

    st.markdown("---")

    # ==================== NÚT PHÂN ĐOẠN ====================
    if st.button("Phân đoạn Não", type="primary", use_container_width=True):

        with st.spinner("Đang phân đoạn..."):
            try:
                # Tạo đối tượng BrainSegmentation với dữ liệu ảnh
                segmenter = BrainSegmentation(image_data)

                # ===== CHẠY PHÂN ĐOẠN THEO PHƯƠNG PHÁP ĐÃ CHỌN =====
                if method_en == "Automatic":
                    # Phương pháp tự động sử dụng Otsu
                    mask = segmenter.threshold_otsu()

                elif method_en == "Threshold":
                    # Phương pháp ngưỡng thủ công
                    mask = segmenter.threshold_manual(threshold=threshold)

                elif method_en == "Otsu":
                    # Phương pháp Otsu tự động chọn ngưỡng
                    mask = segmenter.threshold_otsu()

                elif method_en == "Region Growing":
                    # Chuyển phần trăm thành tọa độ thực tế
                    shape = image_data.shape
                    seed = [
                        int(seed_x * shape[0] / 100),
                        int(seed_y * shape[1] / 100),
                        int(seed_z * shape[2] / 100) if len(shape) > 2 else 0,
                    ]

                    # Đối với ảnh 2D, chỉ dùng 2 tọa độ
                    if len(shape) == 2:
                        seed = seed[:2]

                    # Chạy thuật toán tăng trưởng vùng
                    mask = segmenter.region_growing(
                        seed=tuple(seed),
                        tolerance=intensity_tolerance,
                    )

                # ===== ÁP DỤNG PHÉP BIẾN ĐỔI HÌNH THÁI =====
                if apply_morph:
                    if morph_op_en == "closing":
                        # Phép đóng: giãn rồi xói mòn
                        mask = segmenter.morphological_closing(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op_en == "opening":
                        # Phép mở: xói mòn rồi giãn
                        mask = segmenter.morphological_opening(
                            mask, kernel_size=kernel_size
                        )
                    elif morph_op_en == "dilation":
                        # Phép giãn nở
                        from skimage import morphology

                        if image_data.ndim == 2:
                            kernel = morphology.disk(kernel_size)
                        else:
                            kernel = morphology.ball(kernel_size)
                        mask = morphology.binary_dilation(mask, kernel).astype(np.uint8)
                    elif morph_op_en == "erosion":
                        # Phép xói mòn
                        from skimage import morphology

                        if image_data.ndim == 2:
                            kernel = morphology.disk(kernel_size)
                        else:
                            kernel = morphology.ball(kernel_size)
                        mask = morphology.binary_erosion(mask, kernel).astype(np.uint8)

                # ===== CHỈ GIỮ THÀNH PHẦN LỚN NHẤT =====
                if keep_largest:
                    mask = segmenter.get_largest_component(mask)

                # Lưu mask vào session state
                st.session_state.seg_mask = mask

                st.success("Phân đoạn hoàn tất!")

            except Exception as e:
                st.error(f"Phân đoạn thất bại: {str(e)}")
                st.exception(e)
                st.stop()

    # ==================== HIỂN THỊ KẾT QUẢ ====================
    if st.session_state.seg_mask is not None:
        st.markdown("---")
        st.header("Kết quả Phân đoạn")

        image_data = st.session_state.seg_image_data
        mask = st.session_state.seg_mask

        # ===== THỐNG KÊ =====
        col1, col2, col3, col4 = st.columns(4)

        total_voxels = mask.size  # Tổng số voxel
        segmented_voxels = np.sum(mask > 0)  # Số voxel đã phân đoạn
        percentage = (segmented_voxels / total_voxels) * 100  # Tỷ lệ phần trăm

        col1.metric("Tổng Voxel", f"{total_voxels:,}")
        col2.metric("Đã phân đoạn", f"{segmented_voxels:,}")
        col3.metric("Tỷ lệ", f"{percentage:.1f}%")
        col4.metric("Nền", f"{total_voxels - segmented_voxels:,}")

        st.markdown("---")

        # ===== TRỰC QUAN HÓA =====
        st.subheader("Trực quan hóa")

        # Điều khiển hiển thị cho ảnh 3D
        if image_data.ndim == 3:
            col1, col2 = st.columns([3, 1])

            with col1:
                # Chế độ xem: Gốc, Mask, hoặc Phủ lớp
                view_mode = st.radio(
                    "Chế độ xem:", ["Gốc", "Mask", "Phủ lớp"], horizontal=True
                )

            with col2:
                # Độ mờ của lớp phủ
                opacity = st.slider("Độ mờ phủ lớp", 0.0, 1.0, 0.5)

            # Chọn trục để xem lát cắt
            axis = st.radio(
                "Trục:",
                ["Trục Z (Axial)", "Trục Y (Coronal)", "Trục X (Sagittal)"],
                horizontal=True,
            )

            # Lấy lát cắt tương ứng
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

        else:  # Ảnh 2D
            view_mode = st.radio(
                "Chế độ xem:", ["Gốc", "Mask", "Phủ lớp"], horizontal=True
            )
            opacity = st.slider("Độ mờ phủ lớp", 0.0, 1.0, 0.5)

            img_slice = image_data
            mask_slice = mask

        # ===== VẼ ẢNH =====
        fig, ax = plt.subplots(figsize=(10, 10))

        # Ánh xạ chế độ xem
        view_map = {"Gốc": "Original", "Mask": "Mask", "Phủ lớp": "Overlay"}
        view_mode_en = view_map.get(view_mode, view_mode)

        if view_mode_en == "Original":
            # Hiển thị ảnh gốc
            ax.imshow(img_slice.T, cmap="gray", origin="lower")
            ax.set_title("Ảnh gốc", fontsize=14, fontweight="bold")

        elif view_mode_en == "Mask":
            # Hiển thị mask phân đoạn
            ax.imshow(mask_slice.T, cmap="hot", origin="lower")
            ax.set_title("Mask phân đoạn", fontsize=14, fontweight="bold")

        else:  # Overlay - Phủ lớp
            # Hiển thị ảnh gốc
            ax.imshow(img_slice.T, cmap="gray", origin="lower")

            # Tạo colormap trong suốt cho mask
            colors = [(0, 0, 0, 0), (1, 0, 0, opacity)]  # Đen trong suốt và đỏ
            n_bins = 2
            cmap = ListedColormap(colors)

            # Phủ mask lên trên
            ax.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=opacity)
            ax.set_title("Phủ lớp (Đỏ = Đã phân đoạn)", fontsize=14, fontweight="bold")

        ax.axis("off")
        st.pyplot(fig)
        plt.close()

        # ===== TẢI VỀ KẾT QUẢ =====
        st.markdown("---")
        st.subheader("Tải về Kết quả")

        col1, col2 = st.columns(2)

        with col1:
            # Tải mask dưới dạng NumPy array
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, mask)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="Tải Mask (.npy)",
                data=npy_bytes,
                file_name="mask_phan_doan.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Tải ảnh phủ lớp dưới dạng PNG
            fig_download, ax_download = plt.subplots(figsize=(10, 10))

            # Lấy lát cắt giữa cho ảnh 3D
            if image_data.ndim == 3:
                mid_slice = image_data.shape[2] // 2
                img_slice = image_data[:, :, mid_slice]
                mask_slice = mask[:, :, mid_slice]
            else:
                img_slice = image_data
                mask_slice = mask

            ax_download.imshow(img_slice.T, cmap="gray", origin="lower")

            # Tạo colormap cho mask
            colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]
            cmap = ListedColormap(colors)
            ax_download.imshow(mask_slice.T, cmap=cmap, origin="lower", alpha=0.5)
            ax_download.axis("off")

            img_buffer = io.BytesIO()
            fig_download.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close(fig_download)

            st.download_button(
                label="Tải Phủ lớp (.png)",
                data=img_buffer,
                file_name="phan_doan_phu_lop.png",
                mime="image/png",
            )

        # ===== PHẦN GIẢI THÍCH KẾT QUẢ =====
        st.markdown("---")
        st.subheader("Giải thích kết quả phân đoạn")

        visualizer = ResultVisualizer()

        # Lấy lát cắt giữa để hiển thị
        if image_data.ndim == 3:
            mid_z = image_data.shape[2] // 2
            display_img = image_data[:, :, mid_z]
            display_mask = mask[:, :, mid_z]
        else:
            display_img = image_data
            display_mask = mask

        # Định nghĩa nhãn cho các vùng não
        labels = {1: "Vùng não đã phân đoạn (Brain Tissue)"}

        # Hiển thị phủ lớp với chú thích màu
        visualizer.show_overlay_with_legend(
            image=display_img,
            mask=display_mask,
            labels=labels,
            title="Kết quả phân đoạn với chú thích màu",
        )

        # Tính toán các chỉ số (metrics)
        metrics = {}

        # Hiển thị phần giải thích chi tiết
        show_interpretation_section(
            task_type="segmentation",
            metrics=metrics,
            image_info={
                "method": method,
                "region_percentage": percentage,
                "segmented_voxels": segmented_voxels,
                "total_voxels": total_voxels,
                "morph_applied": apply_morph,
                "kept_largest": keep_largest,
            },
        )

else:
    # ==================== HƯỚNG DẪN KHI CHƯA TẢI FILE ====================
    st.info("Tải lên ảnh não để bắt đầu phân đoạn")

    st.markdown("---")
    st.subheader("Hướng dẫn Nhanh")

    st.markdown(
        """
    **Các bước thực hiện:**
    1. Tải lên ảnh chụp não (NIfTI, DICOM, v.v.)
    2. Chọn phương pháp phân đoạn từ thanh bên
    3. Điều chỉnh tham số (nếu cần)
    4. Nhấn "Phân đoạn Não"
    5. Xem và tải về kết quả
    
    **Cài đặt Khuyến nghị:**
    - **Phương pháp:** Bắt đầu với "Tự động"
    - **Xử lý sau:** Bật phép đóng hình thái (morphological closing)
    - **Giữ lớn nhất:** Luôn bật để loại bỏ nhiễu
    
    **Mẹo hữu ích:**
    - Dùng Otsu để tự động chọn ngưỡng tối ưu
    - Tăng trưởng vùng hoạt động tốt nhất với ranh giới rõ ràng
    - Thử các góc nhìn khác nhau (Axial/Coronal/Sagittal) cho ảnh 3D
    """
    )

# ==================== FOOTER ====================
st.markdown("---")
st.caption(
    "Mẹo: Thử các phương pháp khác nhau và so sánh kết quả "
    "để có độ chính xác tốt nhất cho từng loại ảnh"
)
