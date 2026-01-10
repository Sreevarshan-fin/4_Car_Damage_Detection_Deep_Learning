import streamlit as st
import cv2
import os
import tempfile
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
    "<p style='text-align: center; color: gray;'>Upload a vehicle image or video to detect damage</p>",
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
        This application uses a **deep learning classification model (ResNet-50)**  
        to identify **vehicle damage types**.

        ⚠️ The model performs **classification**, not object detection.
        """
    )

    st.markdown("**Supported Classes:**")
    st.markdown(
        """
        - Front Breakage  
        - Front Crushed  
        - Front Normal  
        - Rear Breakage  
        - Rear Crushed  
        - Rear Normal  
        """
    )

    st.success("🟢 Model Loaded")

# --------------------------------------------------
# SOFT CAR VALIDATION (FIXED)
# --------------------------------------------------
def car_validation_score(image_path):
    """
    Returns a confidence score (0–1) indicating
    likelihood of a vehicle being present.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    h, w, _ = img.shape
    score = 0.0

    # Size check
    if h > 200 and w > 200:
        score += 0.4

    # Aspect ratio (loose)
    aspect_ratio = w / h
    if 0.7 < aspect_ratio < 2.5:
        score += 0.4

    # Edge density (cars have many edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255

    if edge_density > 0.02:
        score += 0.2

    return score

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

        # 🔍 Soft validation
        score = car_validation_score(image_path)

        if score < 0.4:
            st.warning(
                "⚠️ The uploaded image may not clearly contain a vehicle. "
                "Prediction may be unreliable."
            )

        with st.spinner("🔍 Analyzing image..."):
            prediction = predict(image_path)

        st.success(f"✅ **Predicted Damage Class:** {prediction}")

        st.info(
            "ℹ️ Damage localization is not shown because the model "
            "is a classifier. Accurate localization requires object detection models."
        )

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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)

            st.markdown("### 🖼️ Extracted Frame")
            st.image(frame_path, use_container_width=True)

            score = car_validation_score(frame_path)

            if score < 0.4:
                st.warning(
                    "⚠️ The video frame may not clearly show a vehicle."
                )

            with st.spinner("🔍 Analyzing video frame..."):
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





