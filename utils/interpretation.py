"""
🧠 Medical Image Interpretation Components
===========================================

Các công cụ để giải thích kết quả xử lý ảnh y khoa cho người không chuyên.

Components:
- ResultVisualizer: Trực quan hóa kết quả với so sánh, overlay, 3D
- MetricsExplainer: Giải thích các chỉ số kỹ thuật bằng ngôn ngữ đơn giản
- InterpretationGenerator: Tạo báo cáo giải thích tự động
- ReportBuilder: Tạo báo cáo PDF/HTML đầy đủ

Author: Medical Image Processing Team
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class ResultVisualizer:
    """Trực quan hóa kết quả xử lý ảnh y khoa"""

    @staticmethod
    def compare_images(
        img_before: np.ndarray,
        img_after: np.ndarray,
        title_before: str = "Ảnh gốc",
        title_after: str = "Ảnh đã xử lý",
        description: str = "",
    ):
        """
        So sánh 2 ảnh trước và sau xử lý

        Args:
            img_before: Ảnh trước xử lý
            img_after: Ảnh sau xử lý
            title_before: Tiêu đề ảnh trước
            title_after: Tiêu đề ảnh sau
            description: Mô tả giải thích
        """
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_before, caption=title_before, use_container_width=True)

        with col2:
            st.image(img_after, caption=title_after, use_container_width=True)

        if description:
            st.info(f"**Giải thích:** {description}")

    @staticmethod
    def overlay_segmentation(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
        labels: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Tạo overlay của mask phân đoạn lên ảnh gốc

        Args:
            image: Ảnh gốc (grayscale hoặc RGB)
            mask: Mask phân đoạn (0 = background, >0 = regions)
            alpha: Độ trong suốt (0-1)
            colormap: Bảng màu ('jet', 'hot', 'rainbow')
            labels: Dict mapping mask values to labels

        Returns:
            Ảnh đã overlay
        """
        # Chuẩn hóa ảnh về 0-1
        if image.max() > 1:
            image = image.astype(float) / image.max()

        # Chuyển grayscale thành RGB
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image.copy()

        # Tạo colormap cho mask
        cmap = cm.get_cmap(colormap)

        # Normalize mask
        if mask.max() > 0:
            mask_norm = mask.astype(float) / mask.max()
        else:
            mask_norm = mask.astype(float)

        # Apply colormap
        mask_colored = cmap(mask_norm)[..., :3]

        # Blend
        overlay = image_rgb.copy()
        mask_region = mask > 0
        overlay[mask_region] = (
            alpha * mask_colored[mask_region] + (1 - alpha) * image_rgb[mask_region]
        )

        return overlay

    @staticmethod
    def show_overlay_with_legend(
        image: np.ndarray,
        mask: np.ndarray,
        labels: Dict[int, str],
        title: str = "Kết quả phân đoạn",
    ):
        """
        Hiển thị overlay với chú thích đầy đủ

        Args:
            image: Ảnh gốc
            mask: Mask phân đoạn
            labels: {value: description} - ví dụ {1: "Khối u", 2: "Mô bình thường"}
            title: Tiêu đề
        """
        st.subheader(title)

        # Tạo overlay
        overlay = ResultVisualizer.overlay_segmentation(image, mask)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(overlay, use_container_width=True)

        with col2:
            st.markdown("**📍 Chú thích:**")

            # Tính % diện tích của mỗi vùng
            total_pixels = mask.size

            for value, label in labels.items():
                if value == 0:
                    continue

                region_pixels = np.sum(mask == value)
                percentage = (region_pixels / total_pixels) * 100

                # Color indicator
                cmap = cm.get_cmap("jet")
                color_rgb = cmap(value / max(labels.keys()))[:3]
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(color_rgb[0] * 255),
                    int(color_rgb[1] * 255),
                    int(color_rgb[2] * 255),
                )

                st.markdown(
                    f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                    f'<div style="width: 20px; height: 20px; background-color: {color_hex}; '
                    f'margin-right: 10px; border: 1px solid #ccc;"></div>'
                    f"<div><b>{label}</b><br/><small>{percentage:.1f}% diện tích</small></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    @staticmethod
    def show_3d_slices(
        volume: np.ndarray,
        axis: int = 2,
        num_slices: int = 9,
        title: str = "Các lát cắt 3D",
    ):
        """
        Hiển thị nhiều slices của volume 3D

        Args:
            volume: Volume 3D (depth, height, width) hoặc (height, width, depth)
            axis: Trục để cắt (0, 1, hoặc 2)
            num_slices: Số lượng slices hiển thị
            title: Tiêu đề
        """
        st.subheader(title)

        # Chọn slices đều nhau
        slice_indices = np.linspace(0, volume.shape[axis] - 1, num_slices, dtype=int)

        # Tạo grid
        cols = st.columns(3)

        for idx, slice_idx in enumerate(slice_indices):
            # Lấy slice
            if axis == 0:
                slice_img = volume[slice_idx, :, :]
            elif axis == 1:
                slice_img = volume[:, slice_idx, :]
            else:
                slice_img = volume[:, :, slice_idx]

            # Hiển thị
            col_idx = idx % 3
            with cols[col_idx]:
                st.image(
                    slice_img,
                    caption=f"Slice {slice_idx + 1}",
                    use_container_width=True,
                )


class MetricsExplainer:
    """Giải thích các chỉ số kỹ thuật bằng ngôn ngữ đơn giản"""

    # Dictionary ánh xạ metrics -> giải thích
    EXPLANATIONS = {
        "PSNR": {
            "name": "Độ rõ nét (PSNR)",
            "unit": "dB",
            "good_threshold": 30,
            "description": "Đo mức độ nhiễu trong ảnh. Càng cao càng tốt.",
            "interpretation": {
                "excellent": "> 40 dB: Chất lượng xuất sắc",
                "good": "30-40 dB: Chất lượng tốt",
                "fair": "20-30 dB: Chất lượng chấp nhận được",
                "poor": "< 20 dB: Chất lượng kém",
            },
        },
        "SSIM": {
            "name": "Độ tương đồng cấu trúc (SSIM)",
            "unit": "",
            "good_threshold": 0.9,
            "description": "Đo mức độ giống nhau giữa ảnh gốc và ảnh xử lý (0-1).",
            "interpretation": {
                "excellent": "> 0.95: Rất giống ảnh gốc",
                "good": "0.90-0.95: Giống ảnh gốc",
                "fair": "0.80-0.90: Tương đối giống",
                "poor": "< 0.80: Khác biệt đáng kể",
            },
        },
        "Dice": {
            "name": "Độ chính xác phân đoạn (Dice)",
            "unit": "",
            "good_threshold": 0.8,
            "description": "Đo mức độ trùng khớp giữa vùng phân đoạn và vùng thực tế (0-1).",
            "interpretation": {
                "excellent": "> 0.90: Phân đoạn rất chính xác",
                "good": "0.80-0.90: Phân đoạn tốt",
                "fair": "0.70-0.80: Phân đoạn chấp nhận được",
                "poor": "< 0.70: Phân đoạn kém",
            },
        },
        "IoU": {
            "name": "Độ trùng khớp (IoU)",
            "unit": "",
            "good_threshold": 0.7,
            "description": "Đo phần giao và hợp của 2 vùng (0-1).",
            "interpretation": {
                "excellent": "> 0.80: Trùng khớp rất tốt",
                "good": "0.70-0.80: Trùng khớp tốt",
                "fair": "0.50-0.70: Trùng khớp chấp nhận được",
                "poor": "< 0.50: Trùng khớp kém",
            },
        },
        "MSE": {
            "name": "Sai số bình phương trung bình (MSE)",
            "unit": "",
            "good_threshold": 100,
            "description": "Đo sự khác biệt giữa 2 ảnh. Càng thấp càng tốt.",
            "interpretation": {
                "excellent": "< 50: Sai số rất nhỏ",
                "good": "50-100: Sai số nhỏ",
                "fair": "100-500: Sai số trung bình",
                "poor": "> 500: Sai số lớn",
            },
        },
        "SNR": {
            "name": "Tỷ lệ tín hiệu/nhiễu (SNR)",
            "unit": "dB",
            "good_threshold": 20,
            "description": "Đo mức độ tín hiệu so với nhiễu. Càng cao càng tốt.",
            "interpretation": {
                "excellent": "> 30 dB: Tín hiệu rất mạnh",
                "good": "20-30 dB: Tín hiệu tốt",
                "fair": "10-20 dB: Tín hiệu trung bình",
                "poor": "< 10 dB: Nhiễu cao",
            },
        },
    }

    @staticmethod
    def explain_metric(metric_name: str, value: float) -> Dict[str, Any]:
        """
        Giải thích ý nghĩa của một metric

        Args:
            metric_name: Tên metric ('PSNR', 'SSIM', etc.)
            value: Giá trị của metric

        Returns:
            Dict chứa tên, giá trị, đánh giá, mô tả
        """
        if metric_name not in MetricsExplainer.EXPLANATIONS:
            return {
                "name": metric_name,
                "value": value,
                "assessment": "unknown",
                "description": "Chỉ số kỹ thuật",
            }

        info = MetricsExplainer.EXPLANATIONS[metric_name]

        # Đánh giá chất lượng
        if metric_name in ["PSNR", "SSIM", "Dice", "IoU", "SNR"]:
            # Càng cao càng tốt
            if metric_name == "PSNR" or metric_name == "SNR":
                if value > 40:
                    assessment = "excellent"
                elif value > 30:
                    assessment = "good"
                elif value > 20:
                    assessment = "fair"
                else:
                    assessment = "poor"
            else:  # SSIM, Dice, IoU
                if value > 0.9:
                    assessment = "excellent"
                elif value > 0.8:
                    assessment = "good"
                elif value > 0.7:
                    assessment = "fair"
                else:
                    assessment = "poor"
        else:  # MSE
            # Càng thấp càng tốt
            if value < 50:
                assessment = "excellent"
            elif value < 100:
                assessment = "good"
            elif value < 500:
                assessment = "fair"
            else:
                assessment = "poor"

        return {
            "name": info["name"],
            "value": value,
            "unit": info["unit"],
            "assessment": assessment,
            "description": info["description"],
            "interpretation": info["interpretation"][assessment],
        }

    @staticmethod
    def show_metrics_dashboard(
        metrics: Dict[str, float], title: str = "Chỉ số chất lượng"
    ):
        """
        Hiển thị dashboard các metrics với giải thích

        Args:
            metrics: Dict {metric_name: value}
            title: Tiêu đề dashboard
        """
        st.subheader(title)

        # Tạo columns cho metrics
        num_metrics = len(metrics)
        cols = st.columns(min(num_metrics, 4))

        for idx, (metric_name, value) in enumerate(metrics.items()):
            col_idx = idx % 4

            with cols[col_idx]:
                explanation = MetricsExplainer.explain_metric(metric_name, value)

                # Color based on assessment
                color_map = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "fair": "🟠",
                    "poor": "🔴",
                    "unknown": "⚪",
                }

                icon = color_map.get(explanation["assessment"], "⚪")

                # Format value
                if explanation["unit"]:
                    value_str = f"{value:.2f} {explanation['unit']}"
                else:
                    value_str = f"{value:.3f}"

                st.metric(label=f"{icon} {explanation['name']}", value=value_str)

                with st.expander("ℹ️ Giải thích"):
                    st.markdown(f"**Ý nghĩa:** {explanation['description']}")
                    st.markdown(f"**Đánh giá:** {explanation['interpretation']}")


class InterpretationGenerator:
    """Tạo báo cáo giải thích tự động từ kết quả xử lý"""

    @staticmethod
    def generate_interpretation(
        task_type: str,
        metrics: Dict[str, float],
        image_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Tạo đoạn giải thích tự động cho kết quả

        Args:
            task_type: Loại task ('anonymization', 'segmentation', 'reconstruction', 'preprocessing')
            metrics: Dict các metrics
            image_info: Thông tin bổ sung về ảnh

        Returns:
            Đoạn text giải thích
        """
        if task_type == "anonymization":
            return InterpretationGenerator._interpret_anonymization(metrics, image_info)
        elif task_type == "segmentation":
            return InterpretationGenerator._interpret_segmentation(metrics, image_info)
        elif task_type == "reconstruction":
            return InterpretationGenerator._interpret_reconstruction(
                metrics, image_info
            )
        elif task_type == "preprocessing":
            return InterpretationGenerator._interpret_preprocessing(metrics, image_info)
        else:
            return "Kết quả xử lý ảnh đã hoàn tất."

    @staticmethod
    def _interpret_anonymization(metrics: Dict, info: Optional[Dict]) -> str:
        """Giải thích kết quả anonymization"""
        text = "### Kết quả Ẩn danh hóa DICOM\n\n"
        text += "**Hoàn tất:** Tất cả thông tin nhận dạng cá nhân đã được xóa khỏi ảnh y tế.\n\n"

        if info and "fields_removed" in info:
            text += f"**Các trường đã xóa:** {', '.join(info['fields_removed'])}\n\n"

        text += "**Ý nghĩa:** Ảnh này giờ đây an toàn để chia sẻ cho mục đích nghiên cứu hoặc giảng dạy "
        text += "mà không vi phạm quyền riêng tư của bệnh nhân.\n\n"
        text += "**Lưu ý:** Luôn kiểm tra kỹ trước khi chia sẻ dữ liệu y tế."

        return text

    @staticmethod
    def _interpret_segmentation(metrics: Dict, info: Optional[Dict]) -> str:
        """Giải thích kết quả segmentation"""
        text = "### Kết quả Phân đoạn ảnh y tế\n\n"

        if "Dice" in metrics:
            dice = metrics["Dice"]
            if dice > 0.9:
                quality = "xuất sắc"
            elif dice > 0.8:
                quality = "tốt"
            elif dice > 0.7:
                quality = "chấp nhận được"
            else:
                quality = "cần cải thiện"

            text += (
                f"**Độ chính xác:** {dice:.3f} - Chất lượng phân đoạn {quality}.\n\n"
            )

        if info and "region_percentage" in info:
            pct = info["region_percentage"]
            text += f"**Vùng phát hiện:** Chiếm {pct:.1f}% tổng thể tích ảnh.\n\n"

        text += "**Ý nghĩa:** Hệ thống đã tự động xác định và tách vùng quan tâm "
        text += (
            "(ví dụ: khối u, mô não) khỏi nền. Vùng được tô màu giúp bác sĩ dễ dàng "
        )
        text += "xác định vị trí và kích thước bất thường.\n\n"
        text += "**Lưu ý:** Đây chỉ là công cụ hỗ trợ, không thay thế chẩn đoán y khoa."

        return text

    @staticmethod
    def _interpret_reconstruction(metrics: Dict, info: Optional[Dict]) -> str:
        """Giải thích kết quả reconstruction"""
        text = "### Kết quả Tái tạo ảnh\n\n"

        if "PSNR" in metrics:
            psnr = metrics["PSNR"]
            if psnr > 35:
                quality = "rất cao"
            elif psnr > 30:
                quality = "cao"
            elif psnr > 25:
                quality = "trung bình"
            else:
                quality = "thấp"

            text += f"**Chất lượng tái tạo:** PSNR = {psnr:.2f} dB - Chất lượng {quality}.\n\n"

        if "SSIM" in metrics:
            ssim = metrics["SSIM"]
            text += f"**Độ tương đồng:** SSIM = {ssim:.3f} - "
            text += f"Ảnh tái tạo {'rất giống' if ssim > 0.95 else 'tương đối giống'} ảnh gốc.\n\n"

        text += (
            "**Ý nghĩa:** Từ dữ liệu thô của máy quét (CT/MRI), hệ thống đã tái tạo "
        )
        text += (
            "thành hình ảnh có thể nhìn thấy được. Chất lượng tốt giúp bác sĩ quan sát "
        )
        text += "rõ các chi tiết mô, xương, cơ quan nội tạng.\n\n"

        if info and "method" in info:
            text += f"**Phương pháp:** {info['method']}\n\n"

        text += "**Lưu ý:** Các thông số kỹ thuật (góc quét, độ phân giải) ảnh hưởng đến chất lượng."

        return text

    @staticmethod
    def _interpret_preprocessing(metrics: Dict, info: Optional[Dict]) -> str:
        """Giải thích kết quả preprocessing"""
        text = "### Kết quả Tiền xử lý ảnh\n\n"

        operations = info.get("operations", []) if info else []

        if operations:
            text += "**Các bước đã thực hiện:**\n"
            for op in operations:
                if op == "normalize":
                    text += "- Chuẩn hóa độ sáng (giúp ảnh đồng đều)\n"
                elif op == "denoise":
                    text += "- Giảm nhiễu (làm rõ ảnh)\n"
                elif op == "enhance":
                    text += "- Tăng độ tương phản (nổi bật chi tiết)\n"
                elif op == "resize":
                    text += "- Thay đổi kích thước\n"
            text += "\n"

        if "PSNR" in metrics:
            psnr = metrics["PSNR"]
            text += f"**Chất lượng:** PSNR = {psnr:.2f} dB\n\n"

        text += "**Ý nghĩa:** Ảnh đã được làm sạch và tối ưu để phục vụ các bước phân tích tiếp theo. "
        text += "Các mô, khối u, hay bất thường sẽ nổi bật rõ ràng hơn.\n\n"
        text += (
            "**Lưu ý:** Tiền xử lý giúp cải thiện độ chính xác của các thuật toán AI."
        )

        return text


class ReportBuilder:
    """Tạo báo cáo PDF/HTML đầy đủ với giải thích"""

    @staticmethod
    def create_interpretation_report(
        title: str,
        task_type: str,
        images: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        interpretation: str,
        output_format: str = "pdf",
    ) -> bytes:
        """
        Tạo báo cáo giải thích đầy đủ

        Args:
            title: Tiêu đề báo cáo
            task_type: Loại task
            images: Dict {name: image_array}
            metrics: Dict {metric_name: value}
            interpretation: Đoạn giải thích
            output_format: 'pdf' hoặc 'html'

        Returns:
            Bytes của file báo cáo
        """
        if output_format == "pdf":
            return ReportBuilder._create_pdf_report(
                title, task_type, images, metrics, interpretation
            )
        else:
            return ReportBuilder._create_html_report(
                title, task_type, images, metrics, interpretation
            )

    @staticmethod
    def _create_pdf_report(
        title: str,
        task_type: str,
        images: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        interpretation: str,
    ) -> bytes:
        """Tạo PDF report"""
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Image as RLImage,
            Table,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#1f77b4"),
            spaceAfter=30,
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Timestamp
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        story.append(Paragraph(f"<b>Thời gian:</b> {timestamp}", styles["Normal"]))
        story.append(
            Paragraph(f"<b>Loại xử lý:</b> {task_type.capitalize()}", styles["Normal"])
        )
        story.append(Spacer(1, 0.3 * inch))

        # Metrics table
        if metrics:
            story.append(Paragraph("<b>Chỉ số kỹ thuật:</b>", styles["Heading2"]))

            table_data = [["Chỉ số", "Giá trị", "Đánh giá"]]

            for metric_name, value in metrics.items():
                explanation = MetricsExplainer.explain_metric(metric_name, value)

                assessment_map = {
                    "excellent": "Xuất sắc",
                    "good": "Tốt",
                    "fair": "Chấp nhận được",
                    "poor": "Kém",
                }

                if explanation["unit"]:
                    value_str = f"{value:.2f} {explanation['unit']}"
                else:
                    value_str = f"{value:.3f}"

                table_data.append(
                    [
                        explanation["name"],
                        value_str,
                        assessment_map.get(explanation["assessment"], "-"),
                    ]
                )

            table = Table(table_data)
            table.setStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )

            story.append(table)
            story.append(Spacer(1, 0.3 * inch))

        # Interpretation
        story.append(Paragraph("<b>Giải thích kết quả:</b>", styles["Heading2"]))

        # Clean markdown formatting
        interp_clean = interpretation.replace("###", "").replace("**", "")
        for line in interp_clean.split("\n"):
            if line.strip():
                story.append(Paragraph(line, styles["Normal"]))

        story.append(Spacer(1, 0.3 * inch))

        # Images
        if images:
            story.append(Paragraph("<b>Hình ảnh:</b>", styles["Heading2"]))

            for img_name, img_array in images.items():
                # Convert to PIL Image
                if img_array.max() > 1:
                    img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                else:
                    img_array = (img_array * 255).astype(np.uint8)

                pil_img = Image.fromarray(img_array)

                # Save to buffer
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # Add to PDF
                rl_img = RLImage(img_buffer, width=4 * inch, height=3 * inch)
                story.append(Paragraph(img_name, styles["Normal"]))
                story.append(rl_img)
                story.append(Spacer(1, 0.2 * inch))

        # Disclaimer
        story.append(Spacer(1, 0.5 * inch))
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.red,
            leftIndent=20,
            rightIndent=20,
        )
        story.append(
            Paragraph(
                "<b>Lưu ý:</b> Báo cáo này chỉ mang tính chất tham khảo kỹ thuật. "
                "Không thay thế cho ý kiến chẩn đoán của bác sĩ chuyên khoa.",
                disclaimer_style,
            )
        )

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def _create_html_report(
        title: str,
        task_type: str,
        images: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        interpretation: str,
    ) -> bytes:
        """Tạo HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2em;
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .metric-card .value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .images-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 15px;
                }}
                .image-card {{
                    text-align: center;
                }}
                .image-card img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .disclaimer {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin-top: 30px;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Loại xử lý: {task_type.capitalize()}</p>
                <p>Thời gian: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
            </div>
        """

        # Metrics
        if metrics:
            html += '<div class="section"><h2>Chỉ số kỹ thuật</h2><div class="metrics-grid">'

            for metric_name, value in metrics.items():
                explanation = MetricsExplainer.explain_metric(metric_name, value)

                if explanation["unit"]:
                    value_str = f"{value:.2f} {explanation['unit']}"
                else:
                    value_str = f"{value:.3f}"

                html += f"""
                <div class="metric-card">
                    <div><strong>{explanation['name']}</strong></div>
                    <div class="value">{value_str}</div>
                    <div><small>{explanation['interpretation']}</small></div>
                </div>
                """

            html += "</div></div>"

        # Interpretation
        html += (
            f'<div class="section"><h2>Giải thích kết quả</h2>{interpretation}</div>'
        )

        # Images
        if images:
            html += '<div class="section"><h2>🖼️ Hình ảnh</h2><div class="images-grid">'

            for img_name, img_array in images.items():
                # Convert to base64
                if img_array.max() > 1:
                    img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                else:
                    img_array = (img_array * 255).astype(np.uint8)

                pil_img = Image.fromarray(img_array)
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                import base64

                img_base64 = base64.b64encode(img_buffer.read()).decode()

                html += f"""
                <div class="image-card">
                    <img src="data:image/png;base64,{img_base64}" alt="{img_name}">
                    <p><strong>{img_name}</strong></p>
                </div>
                """

            html += "</div></div>"

        # Disclaimer
        html += """
        <div class="disclaimer">
            <strong>Lưu ý:</strong> Báo cáo này chỉ mang tính chất tham khảo kỹ thuật.
            Không thay thế cho ý kiến chẩn đoán của bác sĩ chuyên khoa.
        </div>
        </body>
        </html>
        """

        return html.encode("utf-8")


# Helper functions
def show_interpretation_section(
    task_type: str,
    metrics: Dict[str, float],
    image_info: Optional[Dict[str, Any]] = None,
):
    """
    Hiển thị section giải thích trong Streamlit

    Args:
        task_type: Loại task
        metrics: Metrics
        image_info: Thông tin bổ sung
    """
    st.markdown("---")
    st.subheader("Giải thích kết quả")

    interpretation = InterpretationGenerator.generate_interpretation(
        task_type, metrics, image_info
    )

    st.markdown(interpretation)
