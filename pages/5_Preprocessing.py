"""
Trang Tiền xử lý Ảnh - Image Preprocessing Page

Áp dụng các phép biến đổi và tiền xử lý cho ảnh y tế để:
- Chuẩn hóa cường độ pixel
- Giảm nhiễu (denoising)
- Thay đổi kích thước (resize)
- Tăng cường độ tương phản (contrast enhancement)

Tiền xử lý là bước quan trọng trước khi phân tích hoặc huấn luyện mô hình AI.
"""

# ==================== IMPORT CÁC THƯ VIỆN ====================
import streamlit as st  # Framework web app
import tempfile  # Tạo file tạm
from pathlib import Path  # Xử lý đường dẫn
import sys  # Tương tác hệ thống
import numpy as np  # Xử lý mảng số
import matplotlib.pyplot as plt  # Vẽ đồ thị
import io  # Xử lý byte streams
from skimage import exposure, filters  # Xử lý ảnh
from scipy import ndimage  # Xử lý ảnh khoa học

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import các module từ project
from src.preprocessing.image_transforms import ImageTransforms
from utils.file_io import MedicalImageIO

# Import các công cụ giải thích kết quả
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)
from utils.image_explainer import explain_input_image

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(page_title="🔧 Tiền xử lý Ảnh", layout="wide")

# ==================== KHỞI TẠO SESSION STATE ====================
# Lưu trữ dữ liệu giữa các lần chạy lại
if "prep_image" not in st.session_state:
    st.session_state.prep_image = None  # Ảnh gốc
if "prep_processed" not in st.session_state:
    st.session_state.prep_processed = None  # Ảnh đã xử lý
if "prep_operations" not in st.session_state:
    st.session_state.prep_operations = []  # Danh sách các phép toán đã áp dụng

# ==================== TIÊU ĐỀ ====================
st.title("🔧 Tiền xử lý Ảnh")
st.markdown("Biến đổi và nâng cao chất lượng ảnh y tế để phân tích")

# ==================== THÔNG TIN VỀ TIỀN XỬ LÝ ====================
with st.expander("ℹ Về Tiền xử lý"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Tại sao cần Tiền xử lý?**
        
        Ảnh y tế thô cần chuẩn bị vì:
        - 📊 Dải cường độ khác nhau giữa các ảnh
        - 📏 Kích thước khác nhau
        - 🔊 Nhiễu từ máy quét
        - 🌫️ Độ tương phản thấp
        
        **Các phép toán phổ biến:**
        - 🔢 Chuẩn hóa (Normalization)
        - 🔇 Giảm nhiễu (Denoising)
        - 📐 Thay đổi kích thước (Resize)
        - ✨ Tăng cường độ tương phản (Enhancement)
        """
        )

    with col2:
        st.markdown(
            """
        **Thứ tự Đề xuất:**
        
        1. **Chuẩn hóa** cường độ trước tiên
        2. **Giảm nhiễu** để loại bỏ nhiễu
        3. **Đổi kích thước** về kích cỡ mục tiêu
        4. **Tăng cường** độ tương phản cuối cùng
        
        **Mẹo hữu ích:**
        - ✅ Áp dụng từng phép toán một
        - 👀 Kiểm tra kết quả trước khi tải về
        - 💾 Lưu cấu hình pipeline để tái sử dụng
        """
        )

st.markdown("---")

# ==================== THANH BÊN - CÁC PHÉP TOÁN ====================
with st.sidebar:
    st.header("🔧 Các phép toán")

    # ===== 1. CHUẨN HÓA CƯỜNG ĐỘ =====
    st.markdown("### 📊 Cường độ")

    normalize_enabled = st.checkbox("Chuẩn hóa", value=False)
    if normalize_enabled:
        norm_method = st.selectbox(
            "Phương pháp:",
            ["Min-Max (0-1)", "Z-Score", "Cắt phân vị"],
            help="Phương pháp chuẩn hóa cường độ pixel",
        )

        # Giải thích phương pháp
        with st.expander(f"💡 Giải thích: {norm_method}"):
            if norm_method == "Min-Max (0-1)":
                st.markdown(
                    """
                **📏 Min-Max (Chuẩn)**
                
                Biến đổi tất cả giá trị pixel về khoảng [0, 1]
                
                **Công thức:** `(pixel - min) / (max - min)`
                
                **Ưu điểm:**
                - ✅ Đơn giản nhất
                - ✅ Giữ tỷ lệ tương đối giữa các giá trị
                - ✅ Phù hợp cho hầu hết trường hợp
                - ✅ Chuẩn cho deep learning
                
                **Khi nào dùng:**
                - Training neural networks
                - Chuẩn bị dữ liệu cho model
                - Ảnh không có outliers
                """
                )
            elif norm_method == "Z-Score":
                st.markdown(
                    """
                **📊 Z-Score (Chuẩn hóa thống kê)**
                
                Biến đổi về phân phối chuẩn (mean=0, std=1)
                
                **Công thức:** `(pixel - mean) / std`
                
                **Ưu điểm:**
                - ✅ Tốt cho machine learning
                - ✅ Xử lý outliers tốt hơn
                - ✅ Chuẩn theo phân phối
                
                **Khi nào dùng:**
                - Training CNN, ML models
                - Ảnh có nhiễu/outliers
                - So sánh nhiều ảnh khác nguồn
                """
                )
            else:  # Cắt phân vị
                st.markdown(
                    """
                **✂️ Percentile Clipping (Chống nhiễu)**
                
                Loại bỏ giá trị cực trị trước khi normalize
                
                **Ưu điểm:**
                - ✅ Loại bỏ outliers hiệu quả
                - ✅ Tăng tương phản vùng ROI
                - ✅ Robust với nhiễu
                
                **Khi nào dùng:**
                - Ảnh có điểm sáng/tối cực trị
                - Nhiều vùng nền (background)
                - Cần tập trung vào vùng quan tâm
                
                **Khuyến nghị:** 2%-98%
                """
                )

        # Nếu chọn Cắt phân vị
        if norm_method == "Cắt phân vị":
            lower_p = st.slider("Phân vị dưới", 0, 50, 2)
            upper_p = st.slider("Phân vị trên", 50, 100, 98)

    # ===== 2. TĂNG CƯỜNG TƯƠNG PHẢN =====
    enhance_enabled = st.checkbox("Tăng cường Tương phản", value=False)
    if enhance_enabled:
        enhance_method = st.selectbox(
            "Phương pháp:",
            ["Cân bằng Histogram", "CLAHE", "Hiệu chỉnh Gamma"],
            help="Phương pháp tăng cường độ tương phản",
        )

        # Giải thích phương pháp
        with st.expander(f"💡 Giải thích: {enhance_method}"):
            if enhance_method == "Cân bằng Histogram":
                st.markdown(
                    """
                **📊 Histogram Equalization (Cổ điển)**
                
                Phân bố lại giá trị pixel để histogram đều hơn
                
                **Ưu điểm:**
                - ✅ Đơn giản, không cần tham số
                - ✅ Tăng tương phản toàn cục
                - ✅ Nhanh
                
                **Nhược điểm:**
                - ⚠️ Có thể over-enhance
                - ⚠️ Kém với ảnh nhiều vùng
                
                **Khi nào dùng:**
                - Ảnh độ tương phản thấp toàn bộ
                - Cần giải pháp nhanh
                """
                )
            elif enhance_method == "CLAHE":
                st.markdown(
                    """
                **⭐ CLAHE - Adaptive Histogram (Khuyến nghị)**
                
                Cân bằng histogram cục bộ có giới hạn
                
                **Ưu điểm:**
                - ✅ Tăng cường cục bộ
                - ✅ Tránh over-enhance
                - ✅ Hiệu quả với ảnh y tế
                - ✅ Giữ chi tiết tốt
                
                **Khi nào dùng:**
                - Ảnh y tế (MRI, CT, X-ray)  
                - Tăng chi tiết vùng ROI
                - Độ tương phản không đồng nhất
                
                **Tham số:**
                - 1-2: Nhẹ, tự nhiên
                - 2-3: Chuẩn ⭐
                - 3-5: Mạnh, nhiều chi tiết
                """
                )
            else:  # Gamma
                st.markdown(
                    """
                **🌓 Gamma Correction (Điều chỉnh độ sáng)**
                
                Điều chỉnh phi tuyến độ sáng
                
                **Ưu điểm:**
                - ✅ Kiểm soát chính xác
                - ✅ Không mất thông tin
                - ✅ Đơn giản, trực quan
                
                **Khi nào dùng:**
                - Ảnh quá sáng/tối
                - Điều chỉnh độ sáng nhẹ
                
                **Tham số:**
                - < 1.0: Làm sáng ảnh
                - = 1.0: Không đổi
                - > 1.0: Làm tối ảnh
                
                **Mẹo:** 0.7-0.9 cho ảnh tối, 1.1-1.5 cho ảnh sáng
                """
                )

        # Tham số cho CLAHE
        if enhance_method == "CLAHE":
            clip_limit = st.slider(
                "Giới hạn cắt",
                0.5,
                5.0,
                2.0,
                step=0.5,
                help="Giới hạn cắt cho CLAHE (càng cao càng tăng cường mạnh)",
            )
        # Tham số cho Gamma
        elif enhance_method == "Hiệu chỉnh Gamma":
            gamma = st.slider(
                "Gamma",
                0.1,
                3.0,
                1.0,
                step=0.1,
                help="Giá trị gamma (<1: sáng hơn, >1: tối hơn)",
            )

    # ===== 3. PHÉP TOÁN KHÔNG GIAN =====
    st.markdown("---")
    st.markdown("###  Không gian")

    resize_enabled = st.checkbox("Thay đổi kích thước", value=False)
    if resize_enabled:
        target_size = st.slider(
            "Kích thước đích", 64, 512, 256, step=64, help="Kích thước mục tiêu (vuông)"
        )

    crop_enabled = st.checkbox(
        "Cắt theo Nội dung",
        value=False,
        help="Tự động cắt vùng có nội dung (bỏ vùng đen)",
    )

    # ===== 4. KHỬ NHIỄU =====
    st.markdown("---")
    st.markdown("### 🔇 Khử nhiễu")

    denoise_enabled = st.checkbox("Khử nhiễu", value=False)
    if denoise_enabled:
        denoise_method = st.selectbox(
            "Phương pháp:", ["Gaussian", "Median"], help="Phương pháp khử nhiễu"
        )

        # Giải thích phương pháp
        with st.expander(f"💡 Giải thích: {denoise_method}"):
            if denoise_method == "Gaussian":
                st.markdown(
                    """
                **🔵 Gaussian Blur (Làm mờ Gaussian)**

                Áp dụng bộ lọc Gaussian để làm mịn và giảm nhiễu

                **Công thức:** Tích chập với kernel Gaussian 2D
                ```
                G(x, y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
                ```

                **Ưu điểm:**
                - ✅ Làm mịn đồng đều
                - ✅ Giảm nhiễu Gaussian hiệu quả
                - ✅ Không tạo artifact
                - ✅ Nhanh và ổn định

                **Nhược điểm:**
                - ⚠️ Làm mờ cạnh
                - ⚠️ Mất chi tiết sắc nét

                **Khi nào dùng:**
                - Nhiễu Gaussian từ máy quét
                - Cần kết quả mịn, tự nhiên
                - Tiền xử lý cho CNN

                **Tham số σ (Sigma):**
                - 0.5-1.0: Giảm nhiễu nhẹ ⭐
                - 1.0-2.0: Chuẩn cho ảnh y tế
                - 2.0-5.0: Làm mịn mạnh (mất chi tiết)
                """
                )
            else:  # Median
                st.markdown(
                    """
                **🟢 Median Filter (Bộ lọc Trung vị)**

                Thay thế mỗi pixel bằng giá trị trung vị vùng lân cận

                **Công thức:**
                ```
                pixel_mới = median(các pixel trong kernel)
                ```

                **Ưu điểm:**
                - ✅ Loại bỏ salt-and-pepper noise tốt nhất
                - ✅ Bảo toàn cạnh tốt hơn Gaussian
                - ✅ Không làm mờ cạnh
                - ✅ Robust với outliers

                **Nhược điểm:**
                - ⚠️ Chậm hơn Gaussian
                - ⚠️ Có thể làm mất chi tiết nhỏ

                **Khi nào dùng:**
                - Nhiễu muối tiêu (salt-and-pepper)
                - Nhiễu điểm đơn lẻ
                - Cần giữ cạnh sắc nét
                - Ảnh có nhiều outliers

                **Tham số Kernel Size:**
                - 3×3: Giảm nhiễu nhẹ, giữ chi tiết ⭐
                - 5×5: Chuẩn cho ảnh y tế
                - 7×7+: Giảm nhiễu mạnh (mất chi tiết)

                **Lưu ý:** Kernel size luôn là số lẻ
                """
                )

        # Tham số cho Gaussian
        if denoise_method == "Gaussian":
            sigma = st.slider(
                "Sigma",
                0.1,
                5.0,
                1.0,
                step=0.1,
                help="Độ lớn của bộ lọc Gaussian (càng cao càng mờ)",
            )
        # Tham số cho Median
        else:
            kernel_size = st.slider(
                "Kích thước Kernel",
                3,
                11,
                5,
                step=2,
                help="Kích thước kernel median filter",
            )

    st.markdown("---")
    st.info(" Bật các phép toán theo thứ tự khuyến nghị")

# ==================== TẢI LÊN ẢNH ====================
st.subheader(" Tải lên Ảnh")

uploaded_file = st.file_uploader(
    "Chọn file ảnh (.nii, .nii.gz, .dcm, .nrrd, .mha, .npy)",
    type=["nii", "gz", "dcm", "nrrd", "mha", "npy"],
    help="Tải lên ảnh y tế để tiền xử lý",
)

# ==================== XỬ LÝ KHI CÓ FILE TẢI LÊN ====================
if uploaded_file:
    # Xử lý phần mở rộng file phức hợp như .nii.gz
    if uploaded_file.name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = Path(uploaded_file.name).suffix

    # Tạo file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner("⏳ Đang tải ảnh..."):
            io_handler = MedicalImageIO()
            image_data, metadata = io_handler.read_image(tmp_path)

            # Dùng lát cắt giữa nếu là ảnh 3D
            if image_data.ndim == 3:
                slice_idx = image_data.shape[2] // 2
                image_2d = image_data[:, :, slice_idx]
                st.info(f" Dùng lát cắt giữa ({slice_idx}) từ khối 3D")
            else:
                image_2d = image_data

            st.session_state.prep_image = image_2d

        st.success(f" Đã tải: {image_2d.shape}")

        # Hiển thị thông tin ảnh gốc
        col1, col2, col3 = st.columns(3)
        col1.metric("Kích thước", f"{image_2d.shape[0]}×{image_2d.shape[1]}")
        col2.metric("Dải giá trị", f"{image_2d.min():.1f} - {image_2d.max():.1f}")
        col3.metric("Trung bình", f"{image_2d.mean():.1f}")

        st.markdown("---")

        # === GIẢI THÍCH ẢNH ĐẦU VÀO ===
        explain_input_image(image_2d)

    except Exception as e:
        st.error(f"❌ Lỗi khi tải ảnh: {str(e)}")
        st.stop()

    st.markdown("---")

    # ==================== NÚT ÁP DỤNG XỬ LÝ ====================
    if st.button(" Áp dụng Xử lý", type="primary", use_container_width=True):

        with st.spinner(" Đang xử lý..."):
            try:
                processed = image_2d.copy()
                operations = []  # Danh sách các phép toán đã áp dụng

                # Khởi tạo transformer với ảnh
                transformer = ImageTransforms(processed)

                # ===== 1. CHUẨN HÓA =====
                if normalize_enabled:
                    # Tạo lại transformer với ảnh hiện tại
                    transformer = ImageTransforms(processed)

                    if norm_method == "Min-Max (0-1)":
                        processed = transformer.normalize_minmax(0.0, 1.0)
                        operations.append("Chuẩn hóa (Min-Max)")

                    elif norm_method == "Z-Score":
                        processed = transformer.normalize_zscore()
                        operations.append("Chuẩn hóa (Z-Score)")

                    else:  # Cắt phân vị
                        processed = transformer.normalize_percentile(
                            lower_percentile=lower_p, upper_percentile=upper_p
                        )
                        operations.append(f"Chuẩn hóa (Phân vị {lower_p}-{upper_p})")

                # ===== 2. GIẢM NHIỄU =====
                if denoise_enabled:
                    transformer = ImageTransforms(processed)

                    if denoise_method == "Gaussian":
                        processed = transformer.denoise_gaussian(sigma=sigma)
                        operations.append(f"Gaussian Blur (σ={sigma})")
                    else:  # Median
                        processed = transformer.denoise_median(size=kernel_size)
                        operations.append(f"Median Filter (k={kernel_size})")

                # ===== 3. THAY ĐỔI KÍCH THƯỚC =====
                if resize_enabled:
                    from skimage.transform import resize

                    processed = resize(
                        processed,
                        (target_size, target_size),
                        anti_aliasing=True,  # Chống răng cưa
                        preserve_range=True,  # Giữ nguyên dải giá trị
                    )
                    operations.append(f"Resize ({target_size}×{target_size})")

                # ===== 4. CẮT THEO NỘI DUNG =====
                if crop_enabled:
                    # Tìm hộp giới hạn của vùng có nội dung (non-zero)
                    coords = np.argwhere(processed > np.percentile(processed, 5))
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        processed = processed[y_min : y_max + 1, x_min : x_max + 1]
                        operations.append("Cắt theo Nội dung")

                # ===== 5. TĂNG CƯỜNG TƯƠNG PHẢN =====
                if enhance_enabled:
                    transformer = ImageTransforms(processed)

                    if enhance_method == "Cân bằng Histogram":
                        processed = transformer.histogram_equalization()
                        operations.append("Cân bằng Histogram")

                    elif enhance_method == "CLAHE":
                        processed = transformer.adaptive_histogram_equalization(
                            clip_limit=clip_limit
                        )
                        operations.append(f"CLAHE (clip={clip_limit})")

                    else:  # Gamma - dùng skimage vì không có trong transformer
                        processed = exposure.adjust_gamma(processed, gamma)
                        operations.append(f"Hiệu chỉnh Gamma (γ={gamma})")

                # Lưu kết quả vào session state
                st.session_state.prep_processed = processed
                st.session_state.prep_operations = operations

                st.success(" Hoàn tất!")

            except Exception as e:
                st.error(f" Lỗi trong quá trình xử lý: {str(e)}")
                st.exception(e)

    # ==================== HIỂN THỊ KẾT QUẢ ====================
    if st.session_state.prep_processed is not None:
        st.markdown("---")
        st.header(" Kết quả")

        original = st.session_state.prep_image
        processed = st.session_state.prep_processed

        # ===== SO SÁNH THỐNG KÊ =====
        st.subheader(" So sánh Thống kê")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Kích thước",
            f"{processed.shape[0]}×{processed.shape[1]}",
            delta=f"Gốc: {original.shape[0]}×{original.shape[1]}",
        )

        col2.metric(
            "Min",
            f"{processed.min():.2f}",
            delta=f"{processed.min() - original.min():.2f}",
        )

        col3.metric(
            "Max",
            f"{processed.max():.2f}",
            delta=f"{processed.max() - original.max():.2f}",
        )

        col4.metric(
            "Trung bình",
            f"{processed.mean():.2f}",
            delta=f"{processed.mean() - original.mean():.2f}",
        )

        # ===== TRỰC QUAN HÓA TRƯỚC/SAU =====
        st.markdown("---")
        st.subheader(" So sánh Trước/Sau")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ảnh Gốc**")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(original, cmap="gray")
            ax.set_title("Original", fontsize=14, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"Kích thước: {original.shape} | "
                f"Dải: [{original.min():.1f}, {original.max():.1f}]"
            )

        with col2:
            st.markdown("**Ảnh Đã Xử Lý**")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(processed, cmap="gray")
            ax.set_title("Processed", fontsize=14, fontweight="bold")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"Kích thước: {processed.shape} | "
                f"Dải: [{processed.min():.1f}, {processed.max():.1f}]"
            )

        # ===== ĐÁNH GIÁ CHẤT LƯỢNG =====
        st.markdown("---")
        st.subheader(" Đánh giá Chất lượng và Giải thích")

        # Tính các chỉ số chất lượng
        from skimage.metrics import (
            peak_signal_noise_ratio as psnr_calc,
            structural_similarity as ssim_calc,
            mean_squared_error as mse_calc,
        )

        try:
            # Chuẩn hóa cả hai ảnh về cùng dải [0,1] để so sánh công bằng
            orig_norm = (original - original.min()) / (original.max() - original.min())
            proc_norm = (processed - processed.min()) / (
                processed.max() - processed.min()
            )

            # Tính các chỉ số
            psnr = psnr_calc(orig_norm, proc_norm, data_range=1.0)
            ssim = ssim_calc(orig_norm, proc_norm, data_range=1.0)
            mse = mse_calc(orig_norm, proc_norm)

            # Tính SNR (Signal-to-Noise Ratio)
            signal_power = np.mean(proc_norm**2)
            noise_power = np.mean((proc_norm - orig_norm) ** 2)
            snr = (
                10 * np.log10(signal_power / noise_power)
                if noise_power > 0
                else float("inf")
            )

            metrics = {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "SNR": snr}

            # Hiển thị bảng chỉ số
            explainer = MetricsExplainer()
            explainer.show_metrics_dashboard(metrics)

            # Hiển thị giải thích
            st.markdown("---")
            show_interpretation_section(
                task_type="preprocessing",
                metrics=metrics,
                image_info={
                    "operations": st.session_state.prep_operations,
                    "shape": processed.shape,
                    "dtype": str(processed.dtype),
                },
            )

        except Exception as e:
            st.warning(f"⚠️ Không thể tính một số chỉ số: {str(e)}")

        # ===== SO SÁNH HISTOGRAM =====
        st.markdown("---")
        st.subheader(" Phân bố Cường độ")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Histogram ảnh gốc
        ax1.hist(original.flatten(), bins=50, color="steelblue", alpha=0.7)
        ax1.set_title("Histogram Gốc", fontweight="bold")
        ax1.set_xlabel("Cường độ")
        ax1.set_ylabel("Tần suất")
        ax1.grid(alpha=0.3)

        # Histogram ảnh đã xử lý
        ax2.hist(processed.flatten(), bins=50, color="green", alpha=0.7)
        ax2.set_title("Histogram Đã Xử Lý", fontweight="bold")
        ax2.set_xlabel("Cường độ")
        ax2.set_ylabel("Tần suất")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ===== TỔNG KẾT QUY TRÌNH =====
        st.markdown("---")
        st.subheader(" Tổng kết quy trình")

        if st.session_state.prep_operations:
            for i, op in enumerate(st.session_state.prep_operations, 1):
                st.markdown(f"{i}.  {op}")
        else:
            st.info("Không có phép toán nào được áp dụng")

        # ===== TẢI VỀ KẾT QUẢ =====
        st.markdown("---")
        st.subheader(" Tải về Kết quả")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Tải về dưới dạng NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, processed)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label=" Tải ảnh (.npy)",
                data=npy_bytes,
                file_name="preprocessed_image.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Tải về dưới dạng PNG
            fig_save = plt.figure(figsize=(8, 8))
            plt.imshow(processed, cmap="gray")
            plt.axis("off")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close()

            st.download_button(
                label=" Tải ảnh (.png)",
                data=img_buffer,
                file_name="preprocessed_image.png",
                mime="image/png",
            )

        with col3:
            # Tải cấu hình quy trình dưới dạng JSON
            import json

            config = {"operations": st.session_state.prep_operations, "parameters": {}}

            # Thêm tham số từng phép toán vào config
            if normalize_enabled:
                config["parameters"]["normalize"] = {"method": norm_method}
                if norm_method == "Percentile Clipping":
                    config["parameters"]["normalize"]["percentiles"] = (
                        lower_p,
                        upper_p,
                    )

            if denoise_enabled:
                config["parameters"]["denoise"] = {"method": denoise_method}
                if denoise_method == "Gaussian":
                    config["parameters"]["denoise"]["sigma"] = sigma
                else:
                    config["parameters"]["denoise"]["kernel_size"] = kernel_size

            if resize_enabled:
                config["parameters"]["resize"] = {"size": target_size}

            if enhance_enabled:
                config["parameters"]["enhance"] = {"method": enhance_method}
                if enhance_method == "CLAHE":
                    config["parameters"]["enhance"]["clip_limit"] = clip_limit
                elif enhance_method == "Gamma Correction":
                    config["parameters"]["enhance"]["gamma"] = gamma

            json_str = json.dumps(config, indent=2)

            st.download_button(
                label=" Tải Config (.json)",
                data=json_str,
                file_name="pipeline_config.json",
                mime="application/json",
            )

else:
    # ==================== HƯỚNG DẪN KHI CHƯA TẢI FILE ====================
    st.info(" Tải ảnh lên để bắt đầu tiền xử lý")

    st.markdown("---")
    st.subheader(" Hướng dẫn Sử dụng")

    st.markdown(
        """
    **Cách sử dụng:**
    1.  Tải lên ảnh y tế
    2.  Bật các phép toán mong muốn (thanh bên)
    3.  Điều chỉnh tham số cho mỗi phép toán
    4.  Nhấn "Áp dụng Xử lý"
    5.  So sánh kết quả trước/sau
    6.  Tải về ảnh đã xử lý
    
    **Quy trình đề xuất:**
    - 1️ Bắt đầu với **Chuẩn hóa** (Min-Max)
    - 2️ Thêm **Giảm nhiễu** nếu ảnh nhiễu
    - 3️ Dùng **Thay đổi kích thước** để chuẩn hóa kích cỡ
    - 4️ Áp dụng **CLAHE** để tăng độ tương phản
    
    **Mẹo hữu ích:**
    -  Áp dụng từng phép toán một để thấy hiệu quả
    -  Kiểm tra histogram để xác minh chuẩn hóa
    -  Lưu cấu hình quy trình để tái sử dụng
    -  Dùng Cắt phân vị cho ảnh có nhiều giá trị ngoại lai
    """
    )

# ==================== FOOTER ====================
st.markdown("---")
st.caption(
    " Mẹo: Áp dụng các phép toán theo thứ tự - "
    "Chuẩn hóa → Giảm nhiễu → Đổi kích thước → Tăng cường"
)
