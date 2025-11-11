"""
Demo Script - Test UI Components

Test các tính năng UX mới:
- Progress bars
- Image comparison
- Batch processing
- PDF/ZIP export

Run: streamlit run demo_ui_features.py
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.ui_components import (
    ProgressTracker,
    ImageComparer,
    BatchProcessor,
    show_metrics_dashboard,
    show_preview_gallery,
    create_download_section,
)

st.set_page_config(page_title="🎨 UI Components Demo", layout="wide")

st.title("🎨 Demo: UI Components")
st.markdown("Test các tính năng UX mới")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Progress Bars", "🔄 Image Comparison", "📦 Batch Processing", "💾 Export"]
)

# Tab 1: Progress Bars
with tab1:
    st.header("Progress Bars & Status")

    if st.button("Test Progress Bar"):
        import time

        tracker = ProgressTracker("Đang xử lý", total_steps=5)

        steps = [
            "Đang tải dữ liệu...",
            "Chuẩn hóa ảnh...",
            "Áp dụng filters...",
            "Tính toán metrics...",
            "Hoàn tất!",
        ]

        for i, step in enumerate(steps, 1):
            tracker.update(i, step)
            time.sleep(1)

        tracker.complete("✅ Xong rồi!")

# Tab 2: Image Comparison
with tab2:
    st.header("Image Comparison Slider")

    # Create sample images
    if st.button("Tạo ảnh mẫu"):
        # Original
        original = np.random.rand(256, 256)

        # Processed (add some changes)
        processed = original.copy()
        processed = processed * 1.5 + 0.2
        processed = np.clip(processed, 0, 1)

        # Store
        st.session_state.demo_original = original
        st.session_state.demo_processed = processed

    if "demo_original" in st.session_state:
        comparer = ImageComparer()
        comparer.show(
            st.session_state.demo_original,
            st.session_state.demo_processed,
            "Ảnh gốc",
            "Đã xử lý",
        )

# Tab 3: Batch Processing
with tab3:
    st.header("Batch File Upload")

    batch_processor = BatchProcessor()

    uploaded_files = batch_processor.upload_multiple(
        "Upload nhiều files (.npy)", ["npy"], max_files=10
    )

    if uploaded_files:

        def process_file(file):
            """Dummy processor"""
            import time

            time.sleep(0.5)
            data = np.load(file)
            return data

        if st.button("Xử lý batch"):
            results = batch_processor.process_files(uploaded_files, process_file)

            st.success(f"✅ Đã xử lý {len(results)} files")

            for filename, data in results:
                if data is not None:
                    st.write(f"✓ {filename}: Shape {data.shape}")

# Tab 4: Export
with tab4:
    st.header("Export Results")

    # Create sample data
    if st.button("Tạo dữ liệu mẫu"):
        images = {
            "image_1": np.random.rand(256, 256),
            "image_2": np.random.rand(256, 256),
            "image_3": np.random.rand(256, 256),
        }

        metrics = {
            "Total Images": 3,
            "Average Size": "256×256",
            "Processing Time": "2.5s",
            "Success Rate": "100%",
        }

        st.session_state.demo_results = {
            "images": images,
            "metrics": metrics,
            "description": "Demo report with 3 sample images",
        }

        st.success("✅ Đã tạo dữ liệu mẫu")

    if "demo_results" in st.session_state:
        # Show metrics
        show_metrics_dashboard(st.session_state.demo_results["metrics"])

        # Show gallery
        show_preview_gallery(st.session_state.demo_results["images"])

        # Download section
        create_download_section(st.session_state.demo_results, "demo")

# Footer
st.markdown("---")
st.caption("💡 Test tất cả tính năng trước khi integrate vào pages thật")
