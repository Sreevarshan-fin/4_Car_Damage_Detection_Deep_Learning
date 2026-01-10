import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from model_helper import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🚗 Vehicle Damage Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload a vehicle image or video</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIMPLE VEHICLE VALIDATION ----------------
def is_likely_vehicle(image_path):
    """
    Lightweight heuristic:
    - Rejects icons, logos, text images
    - Allows most real vehicle photos
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w, _ = img.shape

    # Rule 1: size
    if h < 200 or w < 200:
        return False

    # Rule 2: aspect ratio (cars are wider)
    ratio = w / h
    if ratio < 0.8 or ratio > 3.0:
        return False

    # Rule 3: edge density (cars have structure)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.mean(edges > 0)

    if edge_ratio < 0.02:
        return False

    return True

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    - Model: **ResNet-50 (Classifier)**
    - Task: **Damage classification**
    - Validation: **Image sanity check**
    """)
    st.success("🟢 Ready")

# ---------------- INPUT TYPE ----------------
upload_type = st.radio("Select input type", ["📷 Image", "🎥 Video"])

# =====================================================
# IMAGE
# =====================================================
if upload_type == "📷 Image":
    uploaded_image = st.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # ✅ VALIDATION
        if not is_likely_vehicle(image_path):
            st.error("❌ Invalid image. Please upload a vehicle image.")
            os.remove(image_path)
        else:
            with st.spinner("🔍 Analyzing damage..."):
                prediction = predict(image_path)

            st.success(f"✅ **Predicted Damage Class:** {prediction}")
            os.remove(image_path)

# =====================================================
# VIDEO
# =====================================================
if upload_type == "🎥 Video":
    uploaded_video = st.file_uploader("Upload vehicle video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getbuffer())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)

            if not is_likely_vehicle(frame_path):
                st.error("❌ Invalid video frame. No clear vehicle detected.")
            else:
                with st.spinner("🔍 Analyzing damage..."):
                    prediction = predict(frame_path)

                st.success(f"✅ **Predicted Damage Class:** {prediction}")

            os.remove(frame_path)

        os.remove(video_path)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Deep Learning • Computer Vision • Streamlit</p>",
    unsafe_allow_html=True
)










