"""
Module giải thích hình ảnh và các phương pháp xử lý cho người dùng không chuyên
"""

import numpy as np
import streamlit as st


def explain_input_image(image_data, metadata=None):
    """
    Phân tích và giải thích ảnh đầu vào cho người dùng

    Args:
        image_data: Dữ liệu ảnh (numpy array)
        metadata: Dict chứa thông tin bổ sung về ảnh
    """
    st.markdown("### 📸 Phân tích ảnh đầu vào")

    # Phân tích cơ bản
    shape = image_data.shape
    dtype = image_data.dtype
    value_range = (image_data.min(), image_data.max())
    mean_val = image_data.mean()
    std_val = image_data.std()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Thông tin kỹ thuật:**")
        st.write(f"• Kích thước: {' × '.join(map(str, shape))}")
        st.write(f"• Số chiều: {len(shape)}D")
        st.write(f"• Kiểu dữ liệu: {dtype}")
        st.write(f"• Dải giá trị: {value_range[0]:.1f} - {value_range[1]:.1f}")

    with col2:
        st.markdown("**🔍 Đánh giá chất lượng:**")

        # Đánh giá độ tương phản
        contrast_ratio = std_val / (mean_val + 1e-8)
        if contrast_ratio > 0.5:
            st.success("✅ Độ tương phản: Tốt")
        elif contrast_ratio > 0.3:
            st.info("ℹ️ Độ tương phản: Trung bình")
        else:
            st.warning("⚠️ Độ tương phản: Thấp")

        # Đánh giá nhiễu (dựa vào độ biến thiên)
        noise_estimate = np.percentile(std_val, 95)
        if noise_estimate < mean_val * 0.1:
            st.success("✅ Nhiễu: Thấp")
        elif noise_estimate < mean_val * 0.2:
            st.info("ℹ️ Nhiễu: Trung bình")
        else:
            st.warning("⚠️ Nhiễu: Cao")

        # Kiểm tra giá trị bất thường
        zero_percent = (image_data == 0).sum() / image_data.size * 100
        if zero_percent > 30:
            st.warning(f"⚠️ Có {zero_percent:.1f}% pixel = 0 (có thể là vùng nền)")
        else:
            st.success("✅ Phân bố giá trị: Hợp lý")

    # Khuyến nghị
    st.markdown("**💡 Nhận xét & Khuyến nghị:**")

    recommendations = []

    if len(shape) == 3:
        recommendations.append(
            "Ảnh 3D - Phù hợp cho phân đoạn, tái tạo, và xử lý đầy đủ"
        )
    elif len(shape) == 2:
        recommendations.append("Ảnh 2D - Phù hợp cho xử lý nhanh và tái tạo đơn giản")

    if contrast_ratio < 0.3:
        recommendations.append(
            "⚠️ Độ tương phản thấp → Khuyến nghị dùng CLAHE để tăng cường"
        )

    if noise_estimate > mean_val * 0.2:
        recommendations.append(
            "⚠️ Nhiễu cao → Khuyến nghị dùng Gaussian blur để khử nhiễu"
        )

    if zero_percent > 30:
        recommendations.append(
            "ℹ️ Có nhiều vùng nền → Có thể cần crop để tập trung vào vùng quan tâm"
        )

    if not recommendations:
        recommendations.append("✅ Ảnh chất lượng tốt, phù hợp cho xử lý")

    for rec in recommendations:
        st.markdown(f"• {rec}")


def explain_method_options(task_type, methods_info):
    """
    Giải thích các phương pháp/options để người dùng lựa chọn

    Args:
        task_type: Loại task ('segmentation', 'reconstruction', 'preprocessing')
        methods_info: Dict chứa thông tin về các phương pháp
    """
    st.markdown("### 🎯 So sánh các phương pháp")
    st.markdown("*Chọn phương pháp phù hợp với nhu cầu của bạn:*")

    # Tạo bảng so sánh
    for method_name, info in methods_info.items():
        with st.expander(
            f"**{info.get('emoji', '🔹')} {method_name}** {info.get('recommended', '')}"
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Mô tả:**  \n{info.get('description', '')}")

                st.markdown("**Ưu điểm:**")
                for pro in info.get("pros", []):
                    st.markdown(f"• ✅ {pro}")

                st.markdown("**Nhược điểm:**")
                for con in info.get("cons", []):
                    st.markdown(f"• ⚠️ {con}")

                st.markdown("**Phù hợp khi:**")
                for use_case in info.get("use_cases", []):
                    st.markdown(f"• {use_case}")

            with col2:
                st.metric("Tốc độ", info.get("speed", "N/A"))
                st.metric("Độ chính xác", info.get("accuracy", "N/A"))
                st.metric("Độ khó", info.get("difficulty", "N/A"))


def explain_output_results(task_type, results, comparison_data=None):
    """
    Giải thích kết quả sau khi xử lý

    Args:
        task_type: Loại task đã thực hiện
        results: Dict chứa kết quả
        comparison_data: Dict chứa dữ liệu để so sánh (optional)
    """
    st.markdown("### 📊 Phân tích kết quả")

    # Đánh giá tổng quan
    quality_score = results.get("quality_score", 0)

    if quality_score >= 90:
        st.success(f"✅ **KẾT QUẢ XUẤT SẮC** (Điểm: {quality_score}/100)")
        st.markdown("Kết quả có chất lượng rất cao, bạn có thể sử dụng an toàn.")
    elif quality_score >= 75:
        st.info(f"ℹ️ **KẾT QUẢ TỐT** (Điểm: {quality_score}/100)")
        st.markdown("Kết quả chấp nhận được, có thể cần kiểm tra thêm một số vùng.")
    elif quality_score >= 60:
        st.warning(f"⚠️ **KẾT QUẢ TRUNG BÌNH** (Điểm: {quality_score}/100)")
        st.markdown("Kết quả có thể sử dụng nhưng cần xác nhận lại.")
    else:
        st.error(f"❌ **KẾT QUẢ CHƯA TỐT** (Điểm: {quality_score}/100)")
        st.markdown("Khuyến nghị thử lại với tham số khác hoặc ảnh chất lượng cao hơn.")

    # Chi tiết kết quả
    st.markdown("---")
    st.markdown("**🔍 Chi tiết kết quả:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Kết quả chính:**")
        for key, value in results.get("main_results", {}).items():
            st.write(f"• {key}: {value}")

    with col2:
        st.markdown("**Chỉ số chất lượng:**")
        for key, value in results.get("quality_metrics", {}).items():
            st.write(f"• {key}: {value}")

    # So sánh với chuẩn (nếu có)
    if comparison_data:
        st.markdown("---")
        st.markdown("**📏 So sánh với giá trị chuẩn:**")

        for metric, data in comparison_data.items():
            actual = data.get("actual")
            normal_range = data.get("normal_range")

            if normal_range:
                min_val, max_val = normal_range
                if min_val <= actual <= max_val:
                    st.success(
                        f"✅ {metric}: {actual} (Trong khoảng chuẩn: {min_val}-{max_val})"
                    )
                elif actual < min_val:
                    st.warning(
                        f"⚠️ {metric}: {actual} (Thấp hơn chuẩn: {min_val}-{max_val})"
                    )
                else:
                    st.warning(
                        f"⚠️ {metric}: {actual} (Cao hơn chuẩn: {min_val}-{max_val})"
                    )

    # Gợi ý cải thiện
    if results.get("suggestions"):
        st.markdown("---")
        st.markdown("**💡 Gợi ý cải thiện:**")
        for suggestion in results.get("suggestions", []):
            st.markdown(f"• {suggestion}")

    # Cảnh báo (nếu có)
    if results.get("warnings"):
        st.markdown("---")
        st.markdown("**⚠️ Lưu ý quan trọng:**")
        for warning in results.get("warnings", []):
            st.warning(warning)


def create_quality_score(metrics):
    """
    Tính điểm chất lượng tổng hợp từ các metrics

    Args:
        metrics: Dict chứa các chỉ số

    Returns:
        int: Điểm từ 0-100
    """
    score = 0
    count = 0

    # PSNR: 20-50 dB
    if "PSNR" in metrics:
        psnr = metrics["PSNR"]
        if psnr >= 40:
            score += 100
        elif psnr >= 30:
            score += 80
        elif psnr >= 25:
            score += 60
        elif psnr >= 20:
            score += 40
        else:
            score += 20
        count += 1

    # SSIM: 0-1
    if "SSIM" in metrics:
        ssim = metrics["SSIM"]
        score += ssim * 100
        count += 1

    # MSE: Càng thấp càng tốt
    if "MSE" in metrics:
        mse = metrics["MSE"]
        if mse < 0.001:
            score += 100
        elif mse < 0.01:
            score += 80
        elif mse < 0.05:
            score += 60
        else:
            score += 40
        count += 1

    return int(score / count) if count > 0 else 50
