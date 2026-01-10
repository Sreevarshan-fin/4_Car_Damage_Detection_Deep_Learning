import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from model_helper import predict

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>🚗 Vehicle Damage Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Upload a vehicle image or short video</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        **Vehicle Damage Classification System**

        - Model: **ResNet-50 (Image Classifier)**
        - Task: **Damage classification**
        - Validation: **Image sanity check**
        """
    )
    st.success("🟢 App Ready")

# --------------------------------------------------
# VEHICLE VALIDATION (NO CASCADE, NO BOX)
# --------------------------------------------------
def is_likely_car(image_path):
    """
    Lightweight heuristic to block non-car images.
    This is NOT detection — only validation.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w, _ = img.shape

    # Rule 1: minimum resolution
    if h < 200 or w < 200:
        return False

    # Rule 2: cars are usually wider than tall
    aspect_ratio = w / h
    if aspect_ratio < 0.8 or aspect_ratio > 3.0:
        return False

    # Rule 3: cars have structural edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.mean(edges > 0)

    if edge_ratio < 0.02:
        return False

    return True

# --------------------------------------------------
# Upload Type Selector
# --------------------------------------------------
upload_type = st.radio(
    "Select input type:",
    ("📷 Image", "🎥 Video")
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
if upload_type == "📷 Image":
    uploaded_image = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:
        image_path = "temp_image.jpg"

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.markdown("### 🖼️ Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

        # ✅ VALIDATION
        if not is_likely_car(image_path):
            st.error("❌ Invalid image. Please upload a clear vehicle image.")
            os.remove(image_path)
        else:
            with st.spinner("🔍 Analyzing damage..."):
                prediction = predict(image_path)

            st.success(f"✅ **Predicted Damage Class:** {prediction}")
            os.remove(image_path)

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
if upload_type == "🎥 Video":
    uploaded_video = st.file_uploader(
        "Upload a short vehicle video (≤45 sec)",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:
        st.markdown("### 🎬 Uploaded Video")
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.getbuffer())
            video_path = tmp_video.name

        cap = cv2.VideoCapture(video_path)
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        )
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)

            st.markdown("### 🖼️ Extracted Frame")
            st.image(frame_path, use_container_width=True)

            # ✅ VALIDATION
            if not is_likely_car(frame_path):
                st.error("❌ Invalid frame. No clear vehicle detected.")
            else:
                with st.spinner("🔍 Analyzing damage..."):
                    prediction = predict(frame_path)

                st.success(f"✅ **Predicted Damage Class:** {prediction}")

            os.remove(frame_path)

        os.remove(video_path)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Demo Project | Deep Learning • Computer Vision • Streamlit</p>",
    unsafe_allow_html=True
)















