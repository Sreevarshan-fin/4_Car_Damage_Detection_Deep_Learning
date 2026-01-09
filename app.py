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
    "<p style='text-align: center; color: gray;'>Upload an image or a short video to detect vehicle damage</p>",
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
        Deep Learning based **Vehicle Damage Detection**
        using **ResNet-50**.
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
# Damage Localization (Bounding Box - Demo)
# --------------------------------------------------
def draw_damage_box(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img

    # Largest contour assumed as damage area
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw bounding box
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    cv2.putText(
        img,
        "Detected Damage Area",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    return img

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
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:
        image_path = "temp_image.jpg"

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.markdown("### 🖼️ Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

        with st.spinner("🔍 Analyzing image..."):
            prediction = predict(image_path)

        st.success(f"✅ **Predicted Damage Class:** {prediction}")

        # Damage localization
        boxed_image = draw_damage_box(image_path)
        st.markdown("### 📍 Damage Localization")
        st.image(boxed_image, channels="BGR", use_container_width=True)

        os.remove(image_path)

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
if upload_type == "🎥 Video":
    uploaded_video = st.file_uploader(
        "Upload a short video (≤45 sec, ~1MB)",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video is not None:
        st.markdown("### 🎬 Uploaded Video")
        st.video(uploaded_video)

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.getbuffer())
            video_path = tmp_video.name

        # Extract middle frame
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

            with st.spinner("🔍 Analyzing video frame..."):
                prediction = predict(frame_path)

            st.success(f"✅ **Predicted Damage Class:** {prediction}")

            # Damage localization
            boxed_frame = draw_damage_box(frame_path)
            st.markdown("### 📍 Damage Localization (Frame)")
            st.image(boxed_frame, channels="BGR", use_container_width=True)

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
