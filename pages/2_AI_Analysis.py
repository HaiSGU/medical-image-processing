"""
AI Analysis Pipeline - Comprehensive Integration

Tích hợp đầy đủ các chức năng Computer Vision và AI:
- Classification (Phân loại khối u)
- Detection (Phát hiện tổn thương)
- Feature Extraction (Trích xuất đặc trưng)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.file_io import MedicalImageIO
from src.computer_vision.feature_extraction import FeatureExtractor
from src.computer_vision.detection import LesionDetector

# Try to import classification (requires TensorFlow)
try:
    from src.computer_vision.classification import BrainTumorClassifier

    CLASSIFICATION_AVAILABLE = True
except ImportError:
    CLASSIFICATION_AVAILABLE = False

# Page config
st.set_page_config(page_title="AI Analysis Pipeline", page_icon="🧠", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🏥 Navigation")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/1_Processing_Pipeline.py", label="🔧 CORE Processing")
    st.page_link("pages/2_AI_Analysis.py", label="🧠 AI Analysis")

    st.markdown("---")
    st.info(
        """
    💡 **AI Analysis**
    
    Trang này tích hợp tất cả các công cụ Computer Vision và AI.
    
    Sử dụng ảnh đã qua CORE Processing để phân tích.
    """
    )

# Title
st.title("🧠 AI Analysis Pipeline")
st.markdown(
    """
Integrated Computer Vision & AI workflow với các công cụ phân tích:
**Classification** | **Detection** | **Feature Extraction**
"""
)

# Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "🏷️ Classification (Phân loại)",
        "🎯 Detection (Phát hiện)",
        "📊 Feature Extraction (Trích xuất đặc trưng)",
    ]
)

# ==================== TAB 1: CLASSIFICATION ====================
with tab1:
    st.header("🏷️ Brain Tumor Classification")
    st.markdown(
        """
    Phân loại các loại khối u não sử dụng Deep Learning (CNN).
    
    **Các loại khối u:**
    - **Glioma**: Khối u não ác tính
    - **Meningioma**: Khối u màng não (thường lành tính)
    - **Pituitary**: Khối u tuyến yên
    - **Normal**: Não bình thường (không có khối u)
    """
    )

    if not CLASSIFICATION_AVAILABLE:
        st.warning(
            """
        ⚠️ **Classification không khả dụng**
        
        TensorFlow chưa được cài đặt. Để sử dụng tính năng này, chạy:
        ```bash
        pip install tensorflow>=2.13.0
        ```
        """
        )
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Upload Brain Scan")

            # File upload
            uploaded_file = st.file_uploader(
                "Tải lên ảnh MRI não",
                type=["nii", "nii.gz", "nrrd", "mha", "dcm", "png", "jpg"],
                key="classification_upload",
            )

            if uploaded_file:
                # Load image
                io_handler = MedicalImageIO()

                try:
                    # Save temp file
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Read image
                    image, metadata = io_handler.read_image(tmp_path)

                    # Get 2D slice if 3D
                    if image.ndim == 3:
                        slice_idx = image.shape[2] // 2
                        image_2d = image[:, :, slice_idx]
                    else:
                        image_2d = image

                    # Normalize to 0-1
                    image_2d = (image_2d - image_2d.min()) / (
                        image_2d.max() - image_2d.min() + 1e-10
                    )

                    # Display original image
                    st.image(
                        image_2d,
                        caption="Original Brain Scan",
                        use_container_width=True,
                        clamp=True,
                    )

                    # Model selection
                    st.subheader("⚙️ Model Settings")
                    model_name = st.selectbox(
                        "Chọn model",
                        options=["resnet50", "vgg16", "efficientnet"],
                        help="ResNet50 thường cho kết quả tốt nhất",
                    )

                    show_gradcam = st.checkbox(
                        "Hiển thị Grad-CAM",
                        value=True,
                        help="Grad-CAM hiển thị vùng ảnh hưởng đến quyết định",
                    )

                    # Classify button
                    if st.button(
                        "🔍 Classify", type="primary", use_container_width=True
                    ):
                        with st.spinner("Đang phân tích..."):
                            try:
                                # Create classifier
                                classifier = BrainTumorClassifier(model_name=model_name)

                                # Predict
                                result = classifier.predict(image_2d)

                                # Generate Grad-CAM if requested
                                if show_gradcam:
                                    heatmap = classifier.get_gradcam(image_2d)

                                # Store results in session state
                                st.session_state["classification_result"] = result
                                st.session_state["classification_image"] = image_2d
                                if show_gradcam:
                                    st.session_state["classification_heatmap"] = heatmap

                            except Exception as e:
                                st.error(f"Lỗi phân loại: {e}")

                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

        with col2:
            st.subheader("📊 Results")

            # Display results if available
            if "classification_result" in st.session_state:
                result = st.session_state["classification_result"]

                # Main prediction
                st.success(
                    f"""
                ### 🎯 Prediction: **{result['class'].upper()}**
                **Confidence:** {result['confidence']:.1%}
                """
                )

                # Top 3 predictions
                st.markdown("**Top 3 Predictions:**")
                for i, (cls, prob) in enumerate(result["top_k"][:3], 1):
                    st.progress(prob, text=f"{i}. {cls.capitalize()}: {prob:.1%}")

                # All probabilities
                with st.expander("📈 All Probabilities"):
                    prob_df = pd.DataFrame(
                        [
                            {"Class": cls.capitalize(), "Probability": f"{prob:.2%}"}
                            for cls, prob in result["probabilities"].items()
                        ]
                    )
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)

                # Grad-CAM visualization
                if "classification_heatmap" in st.session_state:
                    st.markdown("### 🔥 Grad-CAM Heatmap")
                    st.markdown("Vùng đỏ = Ảnh hưởng nhiều đến quyết định")

                    heatmap = st.session_state["classification_heatmap"]
                    original = st.session_state["classification_image"]

                    # Create overlay
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(original, cmap="gray")
                    ax.imshow(heatmap, alpha=0.5, cmap="jet")
                    ax.axis("off")
                    ax.set_title(f"Grad-CAM: {result['class'].capitalize()}")
                    st.pyplot(fig)
                    plt.close()

# ==================== TAB 2: DETECTION ====================
with tab2:
    st.header("🎯 Lesion Detection")
    st.markdown(
        """
    Phát hiện và định vị khối u, tổn thương trong ảnh não.
    
    **Phương pháp:**
    - Threshold-based detection (dựa trên ngưỡng độ sáng)
    - Blob detection (phát hiện vùng tròn, oval)
    - Bounding box visualization
    """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Brain Scan")

        uploaded_file = st.file_uploader(
            "Tải lên ảnh chứa tổn thương",
            type=["nii", "nii.gz", "nrrd", "mha", "dcm", "png", "jpg"],
            key="detection_upload",
        )

        if uploaded_file:
            io_handler = MedicalImageIO()

            try:
                # Load image (same as classification)
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                image, metadata = io_handler.read_image(tmp_path)

                if image.ndim == 3:
                    slice_idx = image.shape[2] // 2
                    image_2d = image[:, :, slice_idx]
                else:
                    image_2d = image

                # Normalize
                image_2d = (image_2d - image_2d.min()) / (
                    image_2d.max() - image_2d.min() + 1e-10
                )

                st.image(
                    image_2d,
                    caption="Original Image",
                    use_container_width=True,
                    clamp=True,
                )

                # Detection parameters
                st.subheader("⚙️ Detection Settings")

                threshold = st.slider(
                    "Detection Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Ngưỡng càng thấp = phát hiện nhiều hơn",
                )

                min_area = st.slider(
                    "Minimum Lesion Area (pixels)",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                )

                use_blobs = st.checkbox(
                    "Enable Blob Detection", value=True, help="Phát hiện vùng tròn/oval"
                )

                # Detect button
                if st.button(
                    "🔍 Detect Lesions", type="primary", use_container_width=True
                ):
                    with st.spinner("Đang phát hiện..."):
                        try:
                            # Create detector
                            detector = LesionDetector(min_area=min_area, max_area=10000)

                            # Detect
                            detections = detector.detect_lesions(
                                image_2d, threshold=threshold, use_blobs=use_blobs
                            )

                            # Visualize
                            vis_image = detector.visualize_detections(
                                image_2d, detections
                            )

                            # Store results
                            st.session_state["detection_results"] = detections
                            st.session_state["detection_vis"] = vis_image
                            st.session_state["detection_original"] = image_2d

                        except Exception as e:
                            st.error(f"Lỗi phát hiện: {e}")

            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")

    with col2:
        st.subheader("📊 Detection Results")

        if "detection_results" in st.session_state:
            detections = st.session_state["detection_results"]
            vis_image = st.session_state["detection_vis"]

            # Summary
            st.success(f"✅ Phát hiện **{len(detections)}** tổn thương")

            # Visualize
            st.image(
                vis_image,
                caption="Detected Lesions",
                use_container_width=True,
                clamp=True,
            )

            # Statistics
            if len(detections) > 0:
                st.markdown("### 📈 Statistics")

                areas = [d.area for d in detections]
                confidences = [d.confidence for d in detections]

                stats_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Total Lesions",
                            "Total Area",
                            "Mean Area",
                            "Max Confidence",
                        ],
                        "Value": [
                            len(detections),
                            f"{sum(areas):.0f} pixels",
                            f"{np.mean(areas):.1f} pixels",
                            f"{max(confidences):.1%}",
                        ],
                    }
                )
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

                # Detailed table
                with st.expander("🔍 Detailed Detections"):
                    details = []
                    for i, det in enumerate(detections, 1):
                        details.append(
                            {
                                "#": i,
                                "Area": f"{det.area:.0f}",
                                "Centroid": f"({det.centroid[0]:.0f}, {det.centroid[1]:.0f})",
                                "Confidence": f"{det.confidence:.2%}",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(details), hide_index=True, use_container_width=True
                    )

# ==================== TAB 3: FEATURE EXTRACTION ====================
with tab3:
    st.header("📊 Feature Extraction")
    st.markdown(
        """
    Trích xuất đặc trưng định lượng từ ảnh y tế.
    
    **Loại đặc trưng:**
    - **Texture**: GLCM (contrast, homogeneity), LBP
    - **Shape**: Area, perimeter, circularity, eccentricity
    - **Intensity**: Mean, std, skewness, kurtosis
    """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Tải lên ảnh để trích xuất đặc trưng",
            type=["nii", "nii.gz", "nrrd", "mha", "dcm", "png", "jpg"],
            key="features_upload",
        )

        if uploaded_file:
            io_handler = MedicalImageIO()

            try:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                image, metadata = io_handler.read_image(tmp_path)

                if image.ndim == 3:
                    slice_idx = image.shape[2] // 2
                    image_2d = image[:, :, slice_idx]
                else:
                    image_2d = image

                # Normalize
                image_2d = (image_2d - image_2d.min()) / (
                    image_2d.max() - image_2d.min() + 1e-10
                )

                st.image(
                    image_2d,
                    caption="Input Image",
                    use_container_width=True,
                    clamp=True,
                )

                # Feature selection
                st.subheader("⚙️ Feature Settings")

                extract_texture = st.checkbox(
                    "Texture Features (GLCM, LBP)", value=True
                )
                extract_intensity = st.checkbox("Intensity Features", value=True)

                # Optional: create mask for shape features
                create_mask = st.checkbox(
                    "Create Binary Mask for Shape Features", value=False
                )

                mask = None
                if create_mask:
                    mask_threshold = st.slider("Mask Threshold", 0.0, 1.0, 0.5, 0.05)
                    mask = image_2d > mask_threshold
                    st.image(
                        mask.astype(float),
                        caption="Binary Mask",
                        use_container_width=True,
                        clamp=True,
                    )
                    extract_shape = True
                else:
                    extract_shape = False

                # Extract button
                if st.button(
                    "📊 Extract Features", type="primary", use_container_width=True
                ):
                    with st.spinner("Đang trích xuất..."):
                        try:
                            extractor = FeatureExtractor()

                            features = extractor.extract_all_features(
                                image_2d,
                                mask=mask if extract_shape else None,
                                extract_texture=extract_texture,
                                extract_shape=extract_shape,
                                extract_intensity=extract_intensity,
                            )

                            st.session_state["features"] = features
                            st.session_state["features_image"] = image_2d

                        except Exception as e:
                            st.error(f"Lỗi trích xuất: {e}")

            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")

    with col2:
        st.subheader("📊 Extracted Features")

        if "features" in st.session_state:
            features = st.session_state["features"]

            st.success(f"✅ Extracted **{len(features)}** features")

            # Group features by type
            texture_features = {
                k: v for k, v in features.items() if k.startswith(("glcm_", "lbp_"))
            }
            shape_features = {
                k: v
                for k, v in features.items()
                if k
                in [
                    "area",
                    "perimeter",
                    "circularity",
                    "eccentricity",
                    "solidity",
                    "major_axis_length",
                    "minor_axis_length",
                ]
            }
            intensity_features = {
                k: v
                for k, v in features.items()
                if k in ["mean", "std", "min", "max", "skewness", "kurtosis", "energy"]
            }

            # Display by category
            if texture_features:
                with st.expander("🎨 Texture Features", expanded=True):
                    df = pd.DataFrame(
                        [
                            {"Feature": k, "Value": f"{v:.4f}"}
                            for k, v in texture_features.items()
                        ]
                    )
                    st.dataframe(df, hide_index=True, use_container_width=True)

            if shape_features:
                with st.expander("⭕ Shape Features", expanded=True):
                    df = pd.DataFrame(
                        [
                            {"Feature": k, "Value": f"{v:.4f}"}
                            for k, v in shape_features.items()
                        ]
                    )
                    st.dataframe(df, hide_index=True, use_container_width=True)

            if intensity_features:
                with st.expander("💡 Intensity Features", expanded=True):
                    df = pd.DataFrame(
                        [
                            {"Feature": k, "Value": f"{v:.4f}"}
                            for k, v in intensity_features.items()
                        ]
                    )
                    st.dataframe(df, hide_index=True, use_container_width=True)

            # Download CSV
            st.markdown("### 💾 Export")
            features_df = pd.DataFrame([features])
            csv = features_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Features (CSV)",
                csv,
                "features.csv",
                "text/csv",
                use_container_width=True,
            )

# Footer
st.markdown("---")
st.caption(
    "🧠 AI Analysis Pipeline - Medical Image Processing System | Computer Vision + Deep Learning"
)
