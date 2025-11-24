"""
Medical Image Processing & AI Analysis System

2-Stage Pipeline Architecture:
- A. CORE Processing: Preprocessing, reconstruction, segmentation
- B. AI Analysis: Classification, detection, feature extraction
"""

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_io import MedicalImageIO

# Page config
st.set_page_config(
    page_title="Medical Image Processing & AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("🏥 Medical Imaging System")
    st.caption("CORE Processing + AI Analysis")
    st.markdown("---")

    st.info(
        """💡 **Tip**: Use the 2-stage workflow for best results:  
    1. CORE Processing → Clean your images  
    2. AI Analysis → Get insights"""
    )

    # Current file info
    if "uploaded_filename" in st.session_state:
        st.markdown("---")
        st.markdown("### 📄 Current File")
        st.info(f"**{st.session_state['uploaded_filename']}**")
        if "preprocessed" in st.session_state:
            st.success("✅ Preprocessed")
        if "mask" in st.session_state:
            st.success("✅ Segmented")

# Main content
st.title("🏥 Medical Image Processing & AI Analysis")
st.markdown(
    """
---
### Welcome to the Medical Imaging System

This system provides **two integrated workflows** for comprehensive medical image analysis:
"""
)

# Workflow diagram
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
    ### 🔧 A. CORE Processing
    
    **Medical Image Processing Pipeline**
    
    Transform raw medical images into clean, standardized data ready for analysis.
    
    **Steps:**
    1. 📤 **Upload** - Load medical images
    2. 🔐 **Anonymize** - Remove patient info (DICOM)
    3. 🎨 **Preprocess** - Normalize, denoise, enhance
    4. 🔬 **Reconstruct** - CT/M RI reconstruction
    5. 📐 **Register** - Align multiple scans
    6. ✂️ **Segment** - Extract regions of interest
    
    **Output:** Clean, preprocessed images
    """
    )

    if st.button(
        "🔧 Start CORE Processing →", type="primary", use_container_width=True
    ):
        st.switch_page("pages/1_Processing_Pipeline.py")

with col2:
    st.markdown(
        """
    ### 🧠 B. AI Analysis
    
    **Computer Vision & AI Pipeline**
    
    Apply deep learning and computer vision to analyze preprocessed images.
    
    **Steps:**
    1. 🏷️ **Classify** - Brain tumor classification
    2. 🎯 **Detect** - Lesion and abnormality detection
    3. 📊 **Extract** - Radiomics feature extraction
    4. 📄 **Report** - Comprehensive analysis report
    
    **Input:** Preprocessed images from CORE
    **Output:** AI-powered insights
    """
    )

    if st.button("🧠 Start AI Analysis →", type="primary", use_container_width=True):
        st.switch_page("pages/2_AI_Analysis.py")

# Workflow visualization
st.markdown("---")
st.markdown("### 📊 Complete Workflow")

st.markdown(
    """
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Upload    │ ──▶ │ CORE Processing  │ ──▶ │  AI Analysis    │ ──▶ │   Report    │
│   Images    │     │  (Stage A)       │     │   (Stage B)     │     │  & Export   │
└─────────────┘     └──────────────────┘     └─────────────────┘     └─────────────┘
                    • Preprocess                • Classify
                    • Reconstruct               • Detect
                    • Register                  • Extract Features
                    • Segment
```
"""
)

# Quick Start section
st.markdown("---")
st.markdown("### 🚀 Quick Start")

st.info(
    """
**New to the system?** Follow these steps:

1. **Start with CORE Processing** to prepare your medical images
2. **Continue to AI Analysis** to get intelligent insights
3. **Download results** for further analysis or reporting

**Already have preprocessed images?** Jump directly to AI Analysis!
"""
)

# Features grid
st.markdown("---")
st.markdown("### ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    #### 📥 Multi-Format Support
    - NIfTI (.nii, .nii.gz)
    - DICOM (.dcm)
    - NRRD (.nrrd)
    - MetaImage (.mha)
    - NumPy (.npy)
    """
    )

with col2:
    st.markdown(
        """
    #### 🔬 CORE Processing
    - Anonymization
    - Preprocessing
    - CT/MRI Reconstruction
    - Registration
    - Segmentation
    """
    )

with col3:
    st.markdown(
        """
    #### 🧠 AI Analysis
    - Tumor Classification
    - Lesion Detection
    - Feature Extraction
    - Grad-CAM Visualization
    """
    )

# Status section
if "preprocessed" in st.session_state or "classification_result" in st.session_state:
    st.markdown("---")
    st.markdown("### 📈 Current Session Status")

    cols = st.columns(4)

    with cols[0]:
        if "original_image" in st.session_state:
            st.success("✅ Image Uploaded")
        else:
            st.info("⏸️ No Image")

    with cols[1]:
        if "preprocessed" in st.session_state:
            st.success("✅ CORE Complete")
        else:
            st.info("⏸️ Not Processed")

    with cols[2]:
        if (
            "classification_result" in st.session_state
            or "detections" in st.session_state
            or "features" in st.session_state
        ):
            st.success("✅ AI Complete")
        else:
            st.info("⏸️ No AI Analysis")

    with cols[3]:
        if st.session_state.get("classification_result") or st.session_state.get(
            "features"
        ):
            st.success("✅ Results Ready")
        else:
            st.info("⏸️ No Results")

# Documentation
st.markdown("---")
with st.expander("📖 Documentation & Help"):
    st.markdown(
        """
    ### System Architecture
    
    This system uses a **2-stage pipeline architecture**:
    
    **Stage A: CORE Processing (Medical Image Processing)**
    - Handles raw medical images
    - Performs preprocessing, reconstruction, registration
    - Outputs clean, standardized images
    
    **Stage B: AI Analysis (Computer Vision & ML)**
    - Uses preprocessed images from Stage A
    - Applies deep learning for classification
    - Performs detection and feature extraction
    - Generates comprehensive reports
    
    ### Supported Analyses
    
    **Classification:**
    - Glioma (aggressive brain tumor)
    - Meningioma (usually benign)
    - Pituitary (pituitary gland tumor)
    - Normal (no tumor)
    
    **Detection:**
    - Lesion detection
    - Abnormality localization
    - Bounding box visualization
    
    **Feature Extraction:**
    - Texture features (GLCM, LBP)
    - Shape features (area, circularity, etc.)
    - Intensity features (mean, std, skewness, etc.)
    
    ### Data Flow
    
    Data is automatically shared between stages using session state:
    - CORE Processing saves `preprocessed` image
    - AI Analysis reads from session state
    - No manual export/import needed!
    
    ### Tips
    
    - **First time?** Start with CORE Processing
    - **Have preprocessed data?** Go straight to AI Analysis
    - **Need fine control?** Use Advanced Tools
    - **Stuck?** Check the sidebar for current file status
    """
    )

# Footer
st.markdown("---")
st.caption(
    """
Medical Image Processing & AI Analysis System | Version 2.0  
**Note:** This system is for research and educational purposes only.  
Do not use for clinical diagnosis without proper validation.
"""
)
