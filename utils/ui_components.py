"""
UI Components for Medical Image Processing

Provides reusable UI components for Streamlit pages:
- Progress bars with status
- Image comparison sliders
- Batch upload handlers
- Export functionality

Author: HaiSGU
Date: 2025-11-11
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import time
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import zipfile
from datetime import datetime


class ProgressTracker:
    """
    Progress tracker with status messages

    Examples:
        >>> tracker = ProgressTracker("Processing images", total_steps=5)
        >>> tracker.update(1, "Loading image...")
        >>> tracker.update(2, "Applying filters...")
        >>> tracker.complete("Done!")
    """

    def __init__(self, title: str = "Processing", total_steps: int = 100):
        """
        Initialize progress tracker

        Args:
            title: Title to display
            total_steps: Total number of steps
        """
        self.title = title
        self.total_steps = total_steps
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()

    def update(self, current_step: int, message: str = ""):
        """
        Update progress

        Args:
            current_step: Current step number (0 to total_steps)
            message: Status message to display
        """
        progress = min(current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)

        elapsed = time.time() - self.start_time
        eta = (
            (elapsed / current_step) * (self.total_steps - current_step)
            if current_step > 0
            else 0
        )

        status = f"**{self.title}** - {message}"
        if eta > 0:
            status += f" | ETA: {eta:.1f}s"

        self.status_text.markdown(status)

    def complete(self, message: str = "✅ Hoàn thành!"):
        """Mark as complete"""
        self.progress_bar.progress(1.0)
        elapsed = time.time() - self.start_time
        self.status_text.success(f"{message} (Thời gian: {elapsed:.2f}s)")

    def error(self, message: str = "❌ Có lỗi xảy ra"):
        """Mark as error"""
        self.status_text.error(message)


class ImageComparer:
    """
    Image comparison slider

    Examples:
        >>> comparer = ImageComparer()
        >>> comparer.show(original_image, processed_image, "Original", "Processed")
    """

    @staticmethod
    def show(
        image1: np.ndarray,
        image2: np.ndarray,
        label1: str = "Trước xử lý",
        label2: str = "Sau xử lý",
        slider_position: float = 0.5,
    ):
        """
        Show image comparison with slider

        Args:
            image1: First image
            image2: Second image
            label1: Label for first image
            label2: Label for second image
            slider_position: Initial slider position (0-1)
        """
        st.subheader("📊 So sánh Ảnh")

        # Method selection
        comparison_method = st.radio(
            "Phương pháp so sánh:",
            ["Side by Side", "Overlay", "Difference Map"],
            horizontal=True,
        )

        if comparison_method == "Side by Side":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{label1}**")
                st.image(image1, use_container_width=True, clamp=True)
                st.caption(f"Shape: {image1.shape}")

            with col2:
                st.markdown(f"**{label2}**")
                st.image(image2, use_container_width=True, clamp=True)
                st.caption(f"Shape: {image2.shape}")

        elif comparison_method == "Overlay":
            # Slider-based overlay
            alpha = st.slider(
                "Độ trong suốt ảnh 1 (0 = ảnh 2, 1 = ảnh 1)",
                0.0,
                1.0,
                slider_position,
                0.05,
            )

            # Normalize images
            img1_norm = ImageComparer._normalize(image1)
            img2_norm = ImageComparer._normalize(image2)

            # Blend images
            blended = img1_norm * alpha + img2_norm * (1 - alpha)

            st.image(blended, use_container_width=True, clamp=True)
            st.caption(f"Alpha = {alpha:.2f} ({label1} → {label2})")

        else:  # Difference Map
            # Calculate difference
            img1_norm = ImageComparer._normalize(image1)
            img2_norm = ImageComparer._normalize(image2)

            diff = np.abs(img1_norm - img2_norm)

            # Colormap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(diff, cmap="hot")
            ax.set_title("Difference Map (Màu sáng = Khác biệt lớn)")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            st.pyplot(fig)
            plt.close()

            # Statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Diff trung bình", f"{diff.mean():.4f}")
            col2.metric("Diff max", f"{diff.max():.4f}")
            col3.metric("% khác biệt", f"{(diff > 0.1).mean() * 100:.1f}%")

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        img = image.astype(float)
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return img


class BatchProcessor:
    """
    Batch file upload and processing

    Examples:
        >>> processor = BatchProcessor()
        >>> files = processor.upload_multiple("Upload DICOM files", ["dcm"])
        >>> for file, data in processor.process_files(files, load_func):
        >>>     # Process each file
    """

    @staticmethod
    def upload_multiple(
        label: str = "Upload nhiều file",
        accepted_types: List[str] = ["dcm", "nii", "npy"],
        max_files: int = 50,
    ) -> List:
        """
        Upload multiple files

        Args:
            label: Upload label
            accepted_types: Accepted file extensions
            max_files: Maximum number of files

        Returns:
            List of uploaded files
        """
        uploaded_files = st.file_uploader(
            label,
            type=accepted_types,
            accept_multiple_files=True,
            help=f"Chọn tối đa {max_files} files ({', '.join(accepted_types)})",
        )

        if uploaded_files:
            if len(uploaded_files) > max_files:
                st.warning(f"⚠️ Chỉ xử lý {max_files} files đầu tiên")
                uploaded_files = uploaded_files[:max_files]

            st.info(f"📁 Đã chọn {len(uploaded_files)} files")

            # Show file list in expander
            with st.expander("📋 Danh sách files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.text(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")

        return uploaded_files

    @staticmethod
    def process_files(
        files: List, process_func, show_progress: bool = True
    ) -> List[Tuple[str, Any]]:
        """
        Process multiple files with progress tracking

        Args:
            files: List of uploaded files
            process_func: Function to process each file (takes file, returns result)
            show_progress: Show progress bar

        Returns:
            List of (filename, result) tuples
        """
        results = []

        if show_progress:
            tracker = ProgressTracker("Xử lý batch", len(files))

        for i, file in enumerate(files, 1):
            try:
                if show_progress:
                    tracker.update(i, f"Đang xử lý: {file.name}")

                # Process file
                result = process_func(file)
                results.append((file.name, result))

            except Exception as e:
                st.error(f"❌ Lỗi xử lý {file.name}: {str(e)}")
                results.append((file.name, None))

        if show_progress:
            tracker.complete(f"✅ Đã xử lý {len(files)} files")

        return results


class ResultExporter:
    """
    Export results to PDF or ZIP

    Examples:
        >>> exporter = ResultExporter()
        >>> pdf_bytes = exporter.create_pdf_report(images, metrics, description)
        >>> st.download_button("Download PDF", pdf_bytes, "report.pdf")
    """

    @staticmethod
    def create_pdf_report(
        images: Dict[str, np.ndarray],
        metrics: Dict[str, Any],
        description: str = "",
        title: str = "Medical Image Processing Report",
    ) -> bytes:
        """
        Create PDF report

        Args:
            images: Dict of {name: image_array}
            metrics: Dict of {metric_name: value}
            description: Report description
            title: Report title

        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()

        with PdfPages(buffer) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.8, title, ha="center", fontsize=24, weight="bold")
            fig.text(
                0.5,
                0.7,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha="center",
                fontsize=12,
            )

            if description:
                fig.text(
                    0.5,
                    0.5,
                    description,
                    ha="center",
                    fontsize=10,
                    wrap=True,
                    bbox=dict(boxstyle="round", facecolor="wheat"),
                )

            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

            # Metrics page
            if metrics:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")

                metrics_text = "METRICS\n" + "=" * 50 + "\n\n"
                for key, value in metrics.items():
                    metrics_text += f"{key}: {value}\n"

                ax.text(
                    0.1,
                    0.9,
                    metrics_text,
                    fontsize=12,
                    verticalalignment="top",
                    family="monospace",
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

            # Images
            for name, image in images.items():
                fig, ax = plt.subplots(figsize=(8.5, 11))

                # Normalize for display
                img_display = ResultExporter._normalize_for_display(image)

                ax.imshow(img_display, cmap="gray")
                ax.set_title(name, fontsize=16, weight="bold")
                ax.axis("off")

                # Add shape info
                fig.text(
                    0.5,
                    0.05,
                    f"Shape: {image.shape} | dtype: {image.dtype}",
                    ha="center",
                    fontsize=10,
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def create_zip_archive(
        files_dict: Dict[str, bytes], archive_name: str = "results.zip"
    ) -> bytes:
        """
        Create ZIP archive

        Args:
            files_dict: Dict of {filename: file_bytes}
            archive_name: Archive name

        Returns:
            ZIP bytes
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, file_bytes in files_dict.items():
                zip_file.writestr(filename, file_bytes)

        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def save_image_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
        """
        Convert numpy array to image bytes

        Args:
            image: Image array
            format: Image format (PNG, JPEG)

        Returns:
            Image bytes
        """
        buffer = io.BytesIO()

        # Normalize and convert to uint8
        img_display = ResultExporter._normalize_for_display(image)
        img_pil = Image.fromarray(img_display)

        img_pil.save(buffer, format=format)
        buffer.seek(0)

        return buffer.getvalue()

    @staticmethod
    def _normalize_for_display(image: np.ndarray) -> np.ndarray:
        """Normalize image for display (0-255 uint8)"""
        img = image.astype(float)
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img, dtype=np.uint8)

        return img_norm


def show_metrics_dashboard(metrics: Dict[str, Any], title: str = "📊 Thống kê"):
    """
    Show metrics dashboard with cards

    Args:
        metrics: Dict of {metric_name: value}
        title: Dashboard title
    """
    st.subheader(title)

    # Calculate number of columns
    n_metrics = len(metrics)
    n_cols = min(n_metrics, 4)

    cols = st.columns(n_cols)

    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % n_cols]:
            # Format value
            if isinstance(value, float):
                display_value = f"{value:.4f}"
            elif isinstance(value, int):
                display_value = f"{value:,}"
            else:
                display_value = str(value)

            st.metric(key, display_value)


def show_preview_gallery(
    images: Dict[str, np.ndarray], columns: int = 3, title: str = "🖼️ Preview Gallery"
):
    """
    Show image gallery with thumbnails

    Args:
        images: Dict of {name: image_array}
        columns: Number of columns
        title: Gallery title
    """
    st.subheader(title)

    image_items = list(images.items())

    for i in range(0, len(image_items), columns):
        cols = st.columns(columns)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(image_items):
                name, image = image_items[idx]

                with col:
                    st.image(image, caption=name, use_container_width=True, clamp=True)
                    st.caption(f"Shape: {image.shape}")


def create_download_section(results: Dict[str, Any], page_name: str = "results"):
    """
    Create download section with multiple formats

    Args:
        results: Dict with 'images', 'metrics', 'description'
        page_name: Page name for filename
    """
    st.markdown("---")
    st.subheader("💾 Download Kết quả")

    col1, col2, col3 = st.columns(3)

    # PDF Report
    with col1:
        if st.button("📄 Tạo PDF Report", use_container_width=True):
            with st.spinner("Đang tạo PDF..."):
                pdf_bytes = ResultExporter.create_pdf_report(
                    images=results.get("images", {}),
                    metrics=results.get("metrics", {}),
                    description=results.get("description", ""),
                    title=f"{page_name.upper()} Report",
                )

            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{page_name}_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ZIP Archive
    with col2:
        if st.button("📦 Tạo ZIP Archive", use_container_width=True):
            with st.spinner("Đang tạo ZIP..."):
                files_dict = {}

                # Add images
                for name, image in results.get("images", {}).items():
                    img_bytes = ResultExporter.save_image_bytes(image)
                    files_dict[f"{name}.png"] = img_bytes

                # Add metrics as text
                if results.get("metrics"):
                    metrics_text = "\n".join(
                        f"{k}: {v}" for k, v in results["metrics"].items()
                    )
                    files_dict["metrics.txt"] = metrics_text.encode("utf-8")

                zip_bytes = ResultExporter.create_zip_archive(files_dict)

            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_bytes,
                file_name=f"{page_name}_results_{datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # Individual Images
    with col3:
        st.write("**Download từng ảnh:**")
        for name, image in results.get("images", {}).items():
            img_bytes = ResultExporter.save_image_bytes(image)
            st.download_button(
                label=f"⬇️ {name}",
                data=img_bytes,
                file_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_{name}",
            )
