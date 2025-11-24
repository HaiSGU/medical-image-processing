"""
Trang Tái tạo CT - CT Reconstruction Page

Tái tạo ảnh CT từ sinogram (dữ liệu chiếu) sử dụng các thuật toán:
- FBP (Filtered Back Projection): Chiếu ngược có lọc - Nhanh, tiêu chuẩn lâm sàng
- SART (Simultaneous Algebraic Reconstruction Technique): Phương pháp lặp - Chất lượng cao

Sinogram là tập hợp tất cả các hình chiếu X-ray được chụp ở các góc khác nhau
khi máy CT quay quanh bệnh nhân.
"""

# ==================== IMPORT CÁC THƯ VIỆN ====================
import io  # Xử lý byte streams
import sys  # Tương tác với hệ thống
from pathlib import Path  # Xử lý đường dẫn

import matplotlib.pyplot as plt  # Vẽ đồ thị
import numpy as np  # Xử lý mảng số
import streamlit as st  # Framework web app

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import các module từ project
from src.reconstruction.ct_reconstruction import CTReconstructor
from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    show_interpretation_section,
)
from utils.image_explainer import explain_input_image


# ==================== HÀM TIỆN ÍCH ====================
def create_shepp_logan_phantom(size=256):
    """
    Tạo Shepp-Logan phantom - ảnh test chuẩn cho CT.

    Phantom này chứa các hình elip với mật độ khác nhau,
    mô phỏng các mô trong đầu người.

    Args:
        size: Kích thước ảnh phantom (size x size)

    Returns:
        Mảng 2D chứa ảnh phantom
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = shepp_logan_phantom()
    if phantom.shape[0] != size:
        phantom = resize(phantom, (size, size), anti_aliasing=True)

    return phantom


def calculate_psnr(original, reconstructed):
    """
    Tính PSNR (Peak Signal-to-Noise Ratio) - Tỷ lệ tín hiệu trên nhiễu.

    PSNR cao = chất lượng tái tạo tốt.

    Args:
        original: Ảnh gốc
        reconstructed: Ảnh tái tạo

    Returns:
        Giá trị PSNR (dB)
    """
    mse = np.mean((original - reconstructed) ** 2)  # Sai số bình phương trung bình
    if mse == 0:
        return float("inf")  # Tái tạo hoàn hảo
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, reconstructed):
    """
    Tính SSIM (Structural Similarity Index) - Chỉ số tương đồng cấu trúc.

    SSIM đo lường độ tương đồng về cấu trúc giữa hai ảnh.
    Giá trị từ -1 đến 1, càng gần 1 càng giống.

    Args:
        original: Ảnh gốc
        reconstructed: Ảnh tái tạo

    Returns:
        Giá trị SSIM
    """
    # Tính SSIM đơn giản hóa
    mu1 = np.mean(original)
    mu2 = np.mean(reconstructed)
    sigma1 = np.std(original)
    sigma2 = np.std(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))

    # Các hằng số để ổn định
    c1 = (0.01 * np.max(original)) ** 2
    c2 = (0.03 * np.max(original)) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
    )

    return ssim


# ==================== KHỞI TẠO SESSION STATE ====================
# Lưu trữ dữ liệu giữa các lần chạy lại
if "ct_sinogram" not in st.session_state:
    st.session_state.ct_sinogram = None  # Dữ liệu chiếu (sinogram)
if "ct_phantom" not in st.session_state:
    st.session_state.ct_phantom = None  # Ảnh phantom gốc
if "ct_reconstructed" not in st.session_state:
    st.session_state.ct_reconstructed = None  # Ảnh CT tái tạo

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(page_title="Tái tạo CT", layout="wide")

# ==================== TIÊU ĐỀ ====================
st.title("Tái tạo CT")
st.markdown("Tái tạo ảnh CT từ dữ liệu chiếu (sinogram)")

# ==================== THÔNG TIN VỀ TÁI TẠO CT ====================
with st.expander("Về Tái tạo CT"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Tái tạo CT là gì?**
        
        Máy CT quay nguồn tia X quanh bệnh nhân,
        chụp hình chiếu ở các góc khác nhau.
        
        **Sinogram:** Tập hợp tất cả hình chiếu
        - Mỗi hàng = một góc quét
        - Chứa dữ liệu cường độ tia X
        
        **Tái tạo:** Chuyển sinogram → ảnh CT 2D
        """
        )

    with col2:
        st.markdown(
            """
        **Thuật toán:**
        
        **FBP (Chiếu ngược có lọc):**
        - Nhanh (tiêu chuẩn lâm sàng)
        - Nhiều bộ lọc khả dụng
        - Tốt cho dữ liệu đầy đủ
        
        **SART (Phương pháp lặp):**
        - Chậm hơn nhưng chất lượng tốt hơn
        - Tốt cho dữ liệu thưa (ít góc quét)
        - Giảm nhiễu hiệu quả
        """
        )

st.markdown("---")

# ==================== THANH BÊN - CÀI ĐẶT ====================
with st.sidebar:
    st.header("Cài đặt")

    # Chọn nguồn dữ liệu
    data_source = st.radio(
        "Nguồn dữ liệu:",
        ["Tạo Phantom", "Tải lên Sinogram"],
        help="Dùng phantom cho demo hoặc tải lên dữ liệu thật",
    )

    st.markdown("---")

    # Chọn phương pháp tái tạo
    method = st.selectbox("Phương pháp:", ["FBP", "SART"], help="Thuật toán tái tạo")

    # === GIẢI THÍCH PHƯƠNG PHÁP ===
    with st.expander(f"💡 Giải thích: {method}"):
        if method == "FBP":
            st.markdown(
                """
            **⚡ FBP - Filtered Back Projection (Tiêu chuẩn lâm sàng)**
            
            **Mô tả:**  
            Chiếu ngược có lọc - Thuật toán tái tạo CT phổ biến nhất trong y tế.
            
            **Ưu điểm:**
            - ✅ Rất nhanh (~1-2 giây)
            - ✅ Tiêu chuẩn lâm sàng được sử dụng rộng rãi
            - ✅ Kết quả ổn định, đáng tin cậy
            - ✅ Nhiều bộ lọc để tùy chỉnh
            
            **Nhược điểm:**
            - ⚠️ Cần dữ liệu chiếu đầy đủ (nhiều góc)
            - ⚠️ Nhạy cảm với nhiễu
            - ⚠️ Có thể xuất hiện artifacts với dữ liệu thưa
            
            **Khi nào dùng:**
            - Dữ liệu từ máy CT thực tế
            - Cần tốc độ xử lý nhanh
            - Có đầy đủ các góc quét (>90 góc)
            
            **Bộ lọc phổ biến:**
            - **ramp** ⭐: Chuẩn, cân bằng giữa độ nét và nhiễu
            - **shepp-logan**: Giảm nhiễu, mềm hơn
            - **cosine**: Mượt mà, ít artifacts
            - **hamming**: Giảm nhiễu mạnh, hơi mờ
            
            **Khuyến nghị:** Dùng 'ramp' trước, nếu nhiễu quá thì chuyển sang 'shepp-logan'
            """
            )
        else:  # SART
            st.markdown(
                """
            **🔄 SART - Simultaneous Algebraic Reconstruction (Chất lượng cao)**
            
            **Mô tả:**  
            Thuật toán lặp - Cải thiện dần ảnh qua nhiều vòng lặp.
            
            **Ưu điểm:**
            - ✅ Chất lượng cao hơn FBP
            - ✅ Tốt với dữ liệu thưa (ít góc quét)
            - ✅ Giảm nhiễu hiệu quả
            - ✅ Giảm artifacts
            - ✅ Linh hoạt với dữ liệu không hoàn hảo
            
            **Nhược điểm:**
            - ⚠️ Chậm hơn nhiều (~10-30 giây)
            - ⚠️ Cần chọn số lần lặp phù hợp
            - ⚠️ Có thể bị over-smoothing nếu lặp quá nhiều
            
            **Khi nào dùng:**
            - Dữ liệu thưa (ít góc quét <90)
            - Cần chất lượng cao nhất
            - Giảm liều xạ (ít chiếu hơn)
            - Không cần tốc độ xử lý nhanh
            
            **Tham số:**
            - **Số lần lặp**: 5-20 (10 là tốt)
            - **Hệ số thư giãn**: 0.3-0.7 (0.5 là tốt)
            
            **Mẹo:**  
            - Bắt đầu với 10 lần lặp
            - Nếu kết quả còn nhiễu → tăng lên 15-20
            - Nếu quá mịn/mờ → giảm xuống 5-8
            """
            )

    # Tham số riêng cho FBP
    if method == "FBP":
        filter_type = st.selectbox(
            "Bộ lọc:",
            ["ramp", "shepp-logan", "cosine", "hamming"],
            help="Bộ lọc cho tái tạo FBP (ramp là phổ biến nhất)",
        )
    # Tham số riêng cho SART
    else:
        num_iterations = st.slider(
            "Số lần lặp:",
            min_value=1,
            max_value=50,
            value=10,
            help="Số lần lặp SART (càng nhiều càng tốt nhưng chậm hơn)",
        )

        relaxation = st.slider(
            "Hệ số thư giãn:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Hệ số thư giãn SART (0.5 là tốt)",
        )

    st.markdown("---")
    st.info("Thử FBP với bộ lọc 'ramp' trước để có kết quả tốt nhất")

# ==================== NỘI DUNG CHÍNH ====================
# Chế độ 1: Tạo Phantom và Sinogram
if data_source == "Tạo Phantom":
    st.subheader("Shepp-Logan Phantom")

    col1, col2 = st.columns(2)

    with col1:
        # Cài đặt phantom
        phantom_size = st.slider("Kích thước Phantom:", 64, 512, 256, step=64)
        num_angles = st.slider("Số góc quét:", 30, 360, 180, step=30)

    with col2:
        st.markdown(
            """
        **Shepp-Logan Phantom:**
        - Ảnh test chuẩn cho CT
        - Chứa các hình elip có mật độ khác nhau
        - Hoàn hảo để test thuật toán
        - Mô phỏng cấu trúc bên trong đầu
        """
        )

    # Nút tạo và tái tạo
    if st.button("Tạo & Tái tạo", type="primary", use_container_width=True):

        with st.spinner("Đang tạo phantom và sinogram..."):
            # Tạo phantom
            phantom = create_shepp_logan_phantom(phantom_size)
            st.session_state.ct_phantom = phantom

            # Tạo sinogram bằng phép biến đổi Radon
            from skimage.transform import radon

            angles = np.linspace(0, 180, num_angles, endpoint=False)
            sinogram = radon(phantom, theta=angles)
            # radon() returns (num_detectors, num_angles) - correct format!
            # CTReconstructor expects this format, no transpose needed
            st.session_state.ct_sinogram = sinogram

        with st.spinner(f"Đang tái tạo sử dụng {method}..."):
            # Tạo đối tượng reconstructor
            reconstructor = CTReconstructor(sinogram, theta=angles)

            # Tái tạo theo phương pháp đã chọn
            if method == "FBP":
                reconstructed = reconstructor.reconstruct_fbp(filter_name=filter_type)
            else:  # SART
                reconstructed = reconstructor.reconstruct_sart(
                    iterations=num_iterations,
                    relaxation=relaxation,
                    image_size=phantom_size,
                )

            st.session_state.ct_reconstructed = reconstructed

        st.success("Tái tạo hoàn tất!")

# Chế độ 2: Tải lên Sinogram
else:  # Upload Sinogram
    st.subheader("Tải lên Sinogram")

    uploaded_file = st.file_uploader(
        "Chọn file sinogram (.npy)",
        type=["npy"],
        help="Mảng NumPy chứa dữ liệu chiếu",
    )

    if uploaded_file:
        try:
            # Đọc sinogram từ file
            sinogram = np.load(io.BytesIO(uploaded_file.getvalue()))

            # Kiểm tra shape của sinogram
            if sinogram.ndim == 1:
                st.error(
                    f"❌ Lỗi: Sinogram phải là mảng 2D, nhưng file có shape 1D: {sinogram.shape}\n\n"
                    "**Sinogram cần có dạng:**\n"
                    "- Shape: `(num_detectors, num_angles)`\n"
                    "- Ví dụ: `(256, 180)` = 256 detectors, 180 góc quét\n\n"
                    "**Gợi ý:**\n"
                    "- File này không phải sinogram, có thể là projection 1 chiều\n"
                    "- Dùng chế độ 'Tạo Phantom' để xem ví dụ\n"
                    "- Hoặc reshape mảng 1D thành 2D nếu biết số góc"
                )
                st.stop()
            elif sinogram.ndim != 2:
                st.error(
                    f"❌ Lỗi: Sinogram phải là mảng 2D, nhưng nhận được {sinogram.ndim}D: {sinogram.shape}"
                )
                st.stop()

            st.session_state.ct_sinogram = sinogram

            st.success(f"✅ Đã tải sinogram: {sinogram.shape}")

            # Giải thích ảnh đầu vào
            with st.expander("📊 Thông tin về Sinogram đã tải"):
                explain_input_image(sinogram)

            # Nút tái tạo
            if st.button("Tái tạo", type="primary", use_container_width=True):

                with st.spinner(f"Đang tái tạo sử dụng {method}..."):
                    # Tạo các góc quét từ số cột của sinogram (num_angles)
                    num_angles = sinogram.shape[1]
                    angles = np.linspace(0, 180, num_angles, endpoint=False)

                    # Tạo đối tượng reconstructor
                    reconstructor = CTReconstructor(sinogram, theta=angles)

                    if method == "FBP":
                        reconstructed = reconstructor.reconstruct_fbp(
                            filter_name=filter_type
                        )
                    else:  # SART
                        image_size = sinogram.shape[0]  # num_detectors
                        reconstructed = reconstructor.reconstruct_sart(
                            iterations=num_iterations,
                            relaxation=relaxation,
                            image_size=image_size,
                        )

                    st.session_state.ct_reconstructed = reconstructed

                st.success("✅ Tái tạo hoàn tất!")

        except ValueError as e:
            if "shape" in str(e).lower():
                st.error(
                    f"❌ Lỗi shape: {str(e)}\n\n"
                    "Sinogram cần là mảng 2D (num_detectors, num_angles)"
                )
            else:
                st.error(f"❌ Lỗi: {str(e)}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải sinogram: {str(e)}")

# ==================== HIỂN THỊ KẾT QUẢ ====================
if st.session_state.ct_sinogram is not None:
    st.markdown("---")
    st.header("Kết quả")

    sinogram = st.session_state.ct_sinogram

    # ===== HIỂN THỊ SINOGRAM =====
    st.subheader("Sinogram (Dữ liệu chiếu)")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sinogram, cmap="gray", aspect="auto")
    ax.set_xlabel("Vị trí Detector", fontsize=12)
    ax.set_ylabel("Góc chiếu", fontsize=12)
    ax.set_title("Sinogram", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Cường độ")
    st.pyplot(fig)
    plt.close()

    st.caption(f"Kích thước: {sinogram.shape[0]} góc × {sinogram.shape[1]} detectors")

    # ===== HIỂN THỊ ẢNH TÁI TẠO =====
    if st.session_state.ct_reconstructed is not None:
        st.markdown("---")
        st.subheader("Ảnh CT Tái tạo")

        reconstructed = st.session_state.ct_reconstructed

        # Điều khiển hiển thị
        col1, col2 = st.columns([3, 1])

        with col2:
            colormap = st.selectbox("Colormap:", ["gray", "bone", "hot"], index=1)
            show_colorbar = st.checkbox("Hiển thị thanh màu", value=True)

        # Vẽ ảnh tái tạo
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(reconstructed, cmap=colormap)
        ax.axis("off")
        ax.set_title(f"Ảnh tái tạo ({method})", fontsize=14, fontweight="bold")

        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        st.pyplot(fig)
        plt.close()

        # ===== CHẤT LƯỢNG TÁI TẠO (nếu có phantom) =====
        if st.session_state.ct_phantom is not None:
            st.markdown("---")
            st.subheader("Chỉ số Chất lượng")

            phantom = st.session_state.ct_phantom

            # Đảm bảo cùng kích thước
            if phantom.shape != reconstructed.shape:
                from skimage.transform import resize

                phantom = resize(phantom, reconstructed.shape, anti_aliasing=True)

            # Tính các chỉ số
            psnr = calculate_psnr(phantom, reconstructed)
            ssim = calculate_ssim(phantom, reconstructed)

            col1, col2, col3 = st.columns(3)

            col1.metric("PSNR (dB)", f"{psnr:.2f}")
            col2.metric("SSIM", f"{ssim:.4f}")
            col3.metric("Sai số Max", f"{np.max(np.abs(phantom - reconstructed)):.4f}")

            # ===== SO SÁNH GỐC VÀ TÁI TẠO =====
            st.markdown("---")
            st.subheader("So sánh: Gốc và Tái tạo")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Ảnh gốc
            axes[0].imshow(phantom, cmap="gray")
            axes[0].set_title("Phantom gốc", fontweight="bold")
            axes[0].axis("off")

            # Ảnh tái tạo
            axes[1].imshow(reconstructed, cmap="gray")
            axes[1].set_title(f"Tái tạo ({method})", fontweight="bold")
            axes[1].axis("off")

            # Ảnh sai số (difference)
            diff = np.abs(phantom - reconstructed)
            im = axes[2].imshow(diff, cmap="hot")
            axes[2].set_title("Sai số tuyệt đối", fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ===== TẢI VỀ KẾT QUẢ =====
        st.markdown("---")
        st.subheader("Tải về Kết quả")

        col1, col2 = st.columns(2)

        with col1:
            # Tải về dưới dạng NumPy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, reconstructed)
            npy_bytes = npy_buffer.getvalue()

            st.download_button(
                label="Tải ảnh (.npy)",
                data=npy_bytes,
                file_name=f"ct_reconstructed_{method.lower()}.npy",
                mime="application/octet-stream",
            )

        with col2:
            # Tải về dưới dạng PNG
            fig_save = plt.figure(figsize=(8, 8))
            plt.imshow(reconstructed, cmap="gray")
            plt.axis("off")

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
            img_buffer.seek(0)
            plt.close()

            st.download_button(
                label="Tải ảnh (.png)",
                data=img_buffer,
                file_name=f"ct_reconstructed_{method.lower()}.png",
                mime="image/png",
            )

        # ===== GIẢI THÍCH KẾT QUẢ =====
        st.markdown("---")
        st.subheader("Giải thích kết quả tái tạo CT")

        # So sánh với phantom nếu có
        if st.session_state.ct_phantom is not None:
            visualizer = ResultVisualizer()

            phantom = st.session_state.ct_phantom

            # Resize nếu cần
            if phantom.shape != reconstructed.shape:
                from skimage.transform import resize

                phantom = resize(phantom, reconstructed.shape, anti_aliasing=True)

            # Chuẩn hóa ảnh về [0, 1]
            phantom_norm = (phantom - phantom.min()) / (
                phantom.max() - phantom.min() + 1e-8
            )
            recon_norm = (reconstructed - reconstructed.min()) / (
                reconstructed.max() - reconstructed.min() + 1e-8
            )

            # Hiển thị so sánh
            visualizer.compare_images(
                phantom_norm,
                recon_norm,
                title_before="Phantom gốc",
                title_after=f"CT tái tạo ({method})",
                description=(
                    f"Tái tạo từ {sinogram.shape[0]} góc quét. "
                    f"Phương pháp {method}: "
                    f"{'Nhanh nhưng có thể có artifacts' if method == 'FBP' else 'Chất lượng cao hơn nhưng chậm hơn'}."
                ),
            )

            # Tính các chỉ số chất lượng
            from skimage.metrics import (
                peak_signal_noise_ratio,
                structural_similarity,
                mean_squared_error,
            )

            psnr = peak_signal_noise_ratio(phantom_norm, recon_norm, data_range=1.0)
            ssim = structural_similarity(phantom_norm, recon_norm, data_range=1.0)
            mse = mean_squared_error(phantom_norm, recon_norm)
            snr = psnr - 10  # Xấp xỉ

            metrics = {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "SNR": snr}

            # Hiển thị bảng chỉ số
            explainer = MetricsExplainer()
            explainer.show_metrics_dashboard(metrics)

            # Hiển thị giải thích
            info_dict = {"method": method, "num_angles": sinogram.shape[0]}

            # Thêm tham số riêng của từng phương pháp
            if method == "FBP" and "filter_type" in locals():
                info_dict["filter"] = filter_type
            elif method == "SART" and "num_iterations" in locals():
                info_dict["iterations"] = num_iterations

            show_interpretation_section(
                task_type="reconstruction", metrics=metrics, image_info=info_dict
            )

else:
    # ==================== HƯỚNG DẪN KHI CHƯA CÓ DỮ LIỆU ====================
    st.info("Tạo phantom hoặc tải sinogram lên để bắt đầu")

    st.markdown("---")
    st.subheader("📖 Hướng dẫn Sử dụng Chi tiết")

    st.markdown(
        """
    **Các bước thực hiện - Sử dụng Phantom (Khuyến nghị cho người mới):**
    1. **Chuẩn bị**: Giữ "Tạo Phantom" được chọn từ sidebar
    2. **Cài đặt Phantom**:
       - Kích thước: 256x256 (cân bằng tốc độ/chất lượng)
       - Số góc quét: 180 góc (đủ cho tái tạo tốt)
    3. **Chọn phương pháp tái tạo**:
       - **FBP** (Filtered Back Projection): Nhanh, phù hợp demo
       - **SART** (Algebraic): Chất lượng cao hơn, chậm hơn
    4. **Điều chỉnh tham số**:
       - FBP: Chọn bộ lọc (ramp khuyến nghị)
       - SART: Số lần lặp (10-20), relaxation (0.5)
    5. **Thực hiện**: Nhấn "Tạo & Tái tạo"
    6. **Đánh giá**:
       - So sánh ảnh gốc vs ảnh tái tạo
       - Xem chỉ số: PSNR, SSIM, MSE, SNR
    7. **Thử nghiệm**: Thay đổi tham số và so sánh kết quả
    
    **Sử dụng Dữ liệu Thực (Sinogram):**
    1. Chọn "Tải lên Sinogram" từ sidebar
    2. Tải file .npy chứa dữ liệu sinogram
    3. Chọn phương pháp và tham số
    4. Nhấn "Tái tạo"
    5. Tải về ảnh đã tái tạo (.npy)
    
    **Mẹo hữu ích:**
    - ⭐ **Bắt đầu với FBP**: Nhanh, dễ hiểu
    - ⭐ **Nhiều góc = Tốt hơn**: 180 góc > 90 góc
    - ⭐ **SART cho ít góc**: Nếu < 90 góc, dùng SART
    - ⭐ **Thử bộ lọc**: ramp → shepp-logan → hamming
    - ⭐ **FBP vs SART**: FBP nhanh x10-20
    - ⭐ **Dùng phantom**: Để học và thử nghiệm
    """
    )

# Footer
st.markdown("---")
st.caption(
    "Mẹo: Dùng Shepp-Logan phantom để thử nghiệm "
    "các tham số tái tạo khác nhau trước khi áp dụng vào dữ liệu thật"
)
