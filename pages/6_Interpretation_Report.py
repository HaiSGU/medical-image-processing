"""
🧠 6. Báo cáo Giải thích Kết quả
===================================

Trang hiển thị kết quả xử lý với giải thích cho người không chuyên y học.

Features:
- Trực quan hóa kết quả rõ ràng
- Giải thích bằng ngôn ngữ đơn giản
- Chỉ số định lượng dễ hiểu
- Tạo báo cáo PDF/HTML tự động
- So sánh trước/sau xử lý

Author: Medical Image Processing Team
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.interpretation import (
    ResultVisualizer,
    MetricsExplainer,
    InterpretationGenerator,
    ReportBuilder,
    show_interpretation_section,
)
from utils.file_io import MedicalImageIO
from utils.image_utils import normalize_array
import SimpleITK as sitk
from datetime import datetime

st.set_page_config(page_title="Interpretation Report", page_icon="📊", layout="wide")

st.title("📊 Báo cáo Giải thích Kết quả")
st.markdown(
    """
Trang này giúp bạn hiểu rõ kết quả xử lý ảnh y tế thông qua:
- 🖼️ **Trực quan hóa:** So sánh ảnh trước/sau, overlay phân đoạn
- 📊 **Chỉ số đơn giản:** Giải thích các metrics kỹ thuật
- 💡 **Diễn giải tự động:** Báo cáo bằng ngôn ngữ dễ hiểu
- 📄 **Export báo cáo:** Tải về PDF hoặc HTML
"""
)

st.markdown("---")

# Initialize visualizers
visualizer = ResultVisualizer()
metrics_explainer = MetricsExplainer()

# ============================================================================
# SECTION 1: Demo với dữ liệu mẫu
# ============================================================================

st.header("🎯 Demo: Xem trước các tính năng")

demo_tab1, demo_tab2, demo_tab3, demo_tab4 = st.tabs(
    ["📸 So sánh ảnh", "🎨 Overlay phân đoạn", "📊 Metrics Dashboard", "📄 Tạo báo cáo"]
)

with demo_tab1:
    st.subheader("So sánh ảnh trước và sau xử lý")

    # Load sample data
    sample_path = Path("data/sitk/training_001_mr_T1.mha")

    if sample_path.exists():
        try:
            # Load image
            sitk_img = sitk.ReadImage(str(sample_path))
            img_array = sitk.GetArrayFromImage(sitk_img)

            # Get middle slice
            mid_slice = img_array.shape[0] // 2
            original = img_array[mid_slice]

            # Simulate processing (normalize + slight blur)
            from scipy import ndimage

            processed = normalize_array(original)
            processed = ndimage.gaussian_filter(processed, sigma=0.5)

            # Display comparison
            visualizer.compare_images(
                original,
                processed,
                title_before="Ảnh MRI gốc",
                title_after="Ảnh sau tiền xử lý",
                description="Ảnh đã được chuẩn hóa và giảm nhiễu nhẹ để cải thiện độ rõ nét.",
            )

        except Exception as e:
            st.error(f"Lỗi khi load demo: {e}")
    else:
        st.warning("Không tìm thấy file demo. Vui lòng upload ảnh ở section bên dưới.")

with demo_tab2:
    st.subheader("Overlay vùng phân đoạn")

    if sample_path.exists():
        try:
            # Load and prepare image
            sitk_img = sitk.ReadImage(str(sample_path))
            img_array = sitk.GetArrayFromImage(sitk_img)
            mid_slice = img_array.shape[0] // 2
            image = normalize_array(img_array[mid_slice])

            # Create synthetic mask (circular region)
            mask = np.zeros_like(image)
            center = np.array(mask.shape) // 2
            y, x = np.ogrid[: mask.shape[0], : mask.shape[1]]
            r = min(mask.shape) // 4
            circle_mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= r**2
            mask[circle_mask] = 1

            # Add another region
            r2 = min(mask.shape) // 6
            circle_mask2 = (x - center[1] - 50) ** 2 + (
                y - center[0] + 30
            ) ** 2 <= r2**2
            mask[circle_mask2] = 2

            # Display with legend
            labels = {0: "Nền", 1: "Vùng quan tâm chính (ROI)", 2: "Vùng phụ"}

            visualizer.show_overlay_with_legend(
                image, mask, labels, title="Demo: Overlay phân đoạn tự động"
            )

        except Exception as e:
            st.error(f"Lỗi: {e}")

with demo_tab3:
    st.subheader("Dashboard Chỉ số chất lượng")

    # Sample metrics
    sample_metrics = {
        "PSNR": 35.2,
        "SSIM": 0.94,
        "Dice": 0.87,
        "IoU": 0.76,
        "SNR": 28.5,
        "MSE": 42.3,
    }

    metrics_explainer.show_metrics_dashboard(
        sample_metrics, title="Ví dụ: Chỉ số chất lượng sau xử lý"
    )

    st.markdown("---")
    st.markdown("### 📖 Cách đọc chỉ số")

    cols = st.columns(2)

    with cols[0]:
        st.markdown(
            """
        **🟢 Xuất sắc** - Chất lượng rất tốt
        - PSNR > 40 dB
        - SSIM > 0.95
        - Dice > 0.90
        """
        )

        st.markdown(
            """
        **🟡 Tốt** - Chất lượng đạt yêu cầu
        - PSNR: 30-40 dB
        - SSIM: 0.90-0.95
        - Dice: 0.80-0.90
        """
        )

    with cols[1]:
        st.markdown(
            """
        **🟠 Chấp nhận được** - Cần cải thiện
        - PSNR: 20-30 dB
        - SSIM: 0.80-0.90
        - Dice: 0.70-0.80
        """
        )

        st.markdown(
            """
        **🔴 Kém** - Không đạt yêu cầu
        - PSNR < 20 dB
        - SSIM < 0.80
        - Dice < 0.70
        """
        )

with demo_tab4:
    st.subheader("Tạo báo cáo tự động")

    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.markdown("#### 📄 Báo cáo PDF")
        st.markdown(
            """
        Báo cáo PDF bao gồm:
        - ✅ Thông tin tổng quan
        - ✅ Bảng chỉ số kỹ thuật
        - ✅ Giải thích chi tiết
        - ✅ Hình ảnh minh họa
        - ✅ Lưu ý và khuyến cáo
        """
        )

        if st.button("🔮 Tạo báo cáo PDF mẫu", key="demo_pdf"):
            with st.spinner("Đang tạo báo cáo..."):
                try:
                    # Create sample report
                    pdf_bytes = ReportBuilder.create_interpretation_report(
                        title="Báo cáo Xử lý Ảnh Y tế - Demo",
                        task_type="preprocessing",
                        images={
                            "Ảnh gốc": (
                                original
                                if "original" in locals()
                                else np.random.rand(256, 256)
                            ),
                            "Ảnh xử lý": (
                                processed
                                if "processed" in locals()
                                else np.random.rand(256, 256)
                            ),
                        },
                        metrics=sample_metrics,
                        interpretation=InterpretationGenerator.generate_interpretation(
                            "preprocessing",
                            sample_metrics,
                            {"operations": ["normalize", "denoise", "enhance"]},
                        ),
                        output_format="pdf",
                    )

                    st.download_button(
                        label="📥 Tải báo cáo PDF",
                        data=pdf_bytes,
                        file_name=f"medical_report_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )

                    st.success("✅ Báo cáo PDF đã được tạo!")

                except Exception as e:
                    st.error(f"Lỗi khi tạo PDF: {e}")

    with report_col2:
        st.markdown("#### 🌐 Báo cáo HTML")
        st.markdown(
            """
        Báo cáo HTML bao gồm:
        - ✅ Giao diện đẹp, responsive
        - ✅ Metrics dashboard interactive
        - ✅ Hình ảnh chất lượng cao
        - ✅ Dễ chia sẻ qua web
        - ✅ Có thể in trực tiếp
        """
        )

        if st.button("🔮 Tạo báo cáo HTML mẫu", key="demo_html"):
            with st.spinner("Đang tạo báo cáo..."):
                try:
                    # Create sample report
                    html_bytes = ReportBuilder.create_interpretation_report(
                        title="Báo cáo Xử lý Ảnh Y tế - Demo",
                        task_type="segmentation",
                        images={
                            "Ảnh gốc": (
                                image
                                if "image" in locals()
                                else np.random.rand(256, 256)
                            ),
                            "Overlay phân đoạn": visualizer.overlay_segmentation(
                                (
                                    image
                                    if "image" in locals()
                                    else np.random.rand(256, 256)
                                ),
                                (
                                    mask
                                    if "mask" in locals()
                                    else np.random.randint(0, 3, (256, 256))
                                ),
                            ),
                        },
                        metrics={"Dice": 0.87, "IoU": 0.76, "PSNR": 35.2},
                        interpretation=InterpretationGenerator.generate_interpretation(
                            "segmentation", {"Dice": 0.87}, {"region_percentage": 15.3}
                        ),
                        output_format="html",
                    )

                    st.download_button(
                        label="📥 Tải báo cáo HTML",
                        data=html_bytes,
                        file_name=f"medical_report_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                    )

                    st.success("✅ Báo cáo HTML đã được tạo!")

                except Exception as e:
                    st.error(f"Lỗi khi tạo HTML: {e}")

st.markdown("---")

# ============================================================================
# SECTION 2: Upload và phân tích ảnh của bạn
# ============================================================================

st.header("📤 Upload và phân tích ảnh của bạn")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "Chọn file ảnh y tế",
        type=["dcm", "nii", "nii.gz", "mha", "png", "jpg"],
        help="Hỗ trợ DICOM, NIfTI, MetaImage, PNG, JPG",
    )

with upload_col2:
    task_type = st.selectbox(
        "Loại xử lý",
        ["preprocessing", "segmentation", "reconstruction", "anonymization"],
        format_func=lambda x: {
            "preprocessing": "🎨 Tiền xử lý",
            "segmentation": "🧠 Phân đoạn",
            "reconstruction": "🔄 Tái tạo",
            "anonymization": "🔒 Ẩn danh hóa",
        }[x],
    )

if uploaded_file is not None:
    st.markdown("---")

    try:
        # Initialize IO
        medical_io = MedicalImageIO()

        # Save uploaded file to temp
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load image based on type
        if uploaded_file.name.endswith((".dcm", ".nii", ".nii.gz", ".mha", ".nrrd")):
            image_array, metadata = medical_io.read_image(tmp_path)
            # Get middle slice if 3D
            if len(image_array.shape) == 3:
                display_image = image_array[image_array.shape[0] // 2]
            else:
                display_image = image_array
        else:
            from PIL import Image

            pil_img = Image.open(tmp_path)
            display_image = np.array(pil_img)
            if len(display_image.shape) == 3:
                display_image = display_image[:, :, 0]

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

        # Normalize
        display_image = normalize_array(display_image)

        st.success("✅ Đã load ảnh thành công!")

        # Analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
            ["🖼️ Xem ảnh", "📊 Phân tích", "📄 Báo cáo"]
        )

        with analysis_tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ảnh gốc")
                st.image(display_image, use_container_width=True)

                st.markdown(
                    f"""
                **Thông tin:**
                - Kích thước: {display_image.shape}
                - Kiểu dữ liệu: {display_image.dtype}
                - Giá trị min/max: {display_image.min():.2f} / {display_image.max():.2f}
                """
                )

            with col2:
                st.subheader("Xử lý mẫu")

                # Apply sample processing
                from scipy import ndimage

                processed_sample = normalize_array(display_image)
                processed_sample = ndimage.gaussian_filter(processed_sample, sigma=1.0)

                st.image(processed_sample, use_container_width=True)

                st.info(
                    "💡 Đây là ví dụ xử lý đơn giản (chuẩn hóa + giảm nhiễu). "
                    "Bạn có thể tùy chỉnh các tham số ở các trang khác."
                )

        with analysis_tab2:
            st.subheader("Phân tích chất lượng")

            # Calculate sample metrics
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            psnr = peak_signal_noise_ratio(display_image, processed_sample)
            ssim = structural_similarity(
                display_image, processed_sample, data_range=1.0
            )
            mse = np.mean((display_image - processed_sample) ** 2)

            analysis_metrics = {"PSNR": psnr, "SSIM": ssim, "MSE": mse}

            # Show metrics dashboard
            metrics_explainer.show_metrics_dashboard(
                analysis_metrics, title="Chỉ số chất lượng sau xử lý"
            )

            # Show interpretation
            show_interpretation_section(
                task_type, analysis_metrics, {"operations": ["normalize", "denoise"]}
            )

        with analysis_tab3:
            st.subheader("Tạo báo cáo chi tiết")

            report_format = st.radio(
                "Chọn định dạng báo cáo", ["PDF", "HTML"], horizontal=True
            )

            if st.button("🚀 Tạo báo cáo đầy đủ", type="primary"):
                with st.spinner("Đang tạo báo cáo..."):
                    try:
                        report_bytes = ReportBuilder.create_interpretation_report(
                            title=f"Báo cáo Xử lý Ảnh Y tế - {uploaded_file.name}",
                            task_type=task_type,
                            images={
                                "Ảnh gốc": display_image,
                                "Ảnh sau xử lý": processed_sample,
                            },
                            metrics=analysis_metrics,
                            interpretation=InterpretationGenerator.generate_interpretation(
                                task_type,
                                analysis_metrics,
                                {"operations": ["normalize", "denoise"]},
                            ),
                            output_format=report_format.lower(),
                        )

                        file_ext = "pdf" if report_format == "PDF" else "html"
                        mime_type = (
                            "application/pdf" if report_format == "PDF" else "text/html"
                        )

                        st.download_button(
                            label=f"📥 Tải báo cáo {report_format}",
                            data=report_bytes,
                            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type,
                        )

                        st.success(
                            f"✅ Báo cáo {report_format} đã được tạo thành công!"
                        )

                    except Exception as e:
                        st.error(f"Lỗi khi tạo báo cáo: {e}")

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")

st.markdown("---")

# ============================================================================
# SECTION 3: Hướng dẫn sử dụng
# ============================================================================

with st.expander("📚 Hướng dẫn sử dụng chi tiết"):
    st.markdown(
        """
    ## 🎯 Cách sử dụng trang này
    
    ### 1️⃣ Xem Demo
    - Khám phá các tính năng trực quan hóa
    - Hiểu cách đọc các chỉ số kỹ thuật
    - Xem ví dụ báo cáo PDF/HTML
    
    ### 2️⃣ Upload ảnh của bạn
    - Chọn file ảnh y tế (DICOM, NIfTI, etc.)
    - Chọn loại xử lý phù hợp
    - Xem kết quả phân tích tự động
    
    ### 3️⃣ Tạo báo cáo
    - Chọn định dạng (PDF hoặc HTML)
    - Click nút "Tạo báo cáo"
    - Tải về và lưu trữ
    
    ---
    
    ## 📊 Giải thích các chỉ số
    
    ### PSNR (Peak Signal-to-Noise Ratio)
    - **Ý nghĩa:** Đo độ rõ nét của ảnh
    - **Đơn vị:** dB (decibel)
    - **Cách đọc:** Càng cao càng tốt (> 30 dB là tốt)
    
    ### SSIM (Structural Similarity Index)
    - **Ý nghĩa:** Đo độ giống cấu trúc giữa 2 ảnh
    - **Phạm vi:** 0 đến 1
    - **Cách đọc:** Càng gần 1 càng giống (> 0.9 là tốt)
    
    ### Dice Coefficient
    - **Ý nghĩa:** Đo độ chính xác của phân đoạn
    - **Phạm vi:** 0 đến 1
    - **Cách đọc:** > 0.8 là phân đoạn tốt
    
    ### IoU (Intersection over Union)
    - **Ý nghĩa:** Đo độ trùng khớp giữa 2 vùng
    - **Phạm vi:** 0 đến 1
    - **Cách đọc:** > 0.7 là trùng khớp tốt
    
    ---
    
    ## ⚠️ Lưu ý quan trọng
    
    1. **Không thay thế chẩn đoán y khoa:**
       - Công cụ này chỉ hỗ trợ kỹ thuật
       - Cần ý kiến bác sĩ chuyên khoa
       
    2. **Bảo mật dữ liệu:**
       - Không upload ảnh có thông tin nhận dạng
       - Sử dụng tính năng ẩn danh hóa trước
       
    3. **Chất lượng ảnh đầu vào:**
       - Ảnh rõ nét cho kết quả tốt hơn
       - Kiểm tra định dạng file
       
    4. **Giới hạn kỹ thuật:**
       - Một số thuật toán có thể sai
       - Luôn kiểm tra kết quả thủ công
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>🏥 <b>Medical Image Processing Platform</b></p>
    <p>Công cụ hỗ trợ xử lý và phân tích ảnh y tế</p>
    <p><small>⚠️ Chỉ dùng cho mục đích nghiên cứu và giảng dạy</small></p>
</div>
""",
    unsafe_allow_html=True,
)
