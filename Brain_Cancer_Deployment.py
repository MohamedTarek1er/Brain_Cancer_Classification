import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = r"C:\Users\moham\Downloads\Amit_Course\Graduation_Project\Brain_Cancer_Classification\best_model.h5"

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ===================== GRAD-CAM =====================
def grad_cam(model, img_array, layer_name="top_activation"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap

# ===================== SIDEBAR =====================
st.sidebar.title("🧠 Brain Tumor Classifier")
st.sidebar.markdown("EfficientNet-based MRI classification")
show_cam = st.sidebar.checkbox("Show Grad-CAM Explanation", value=True)

# ===================== MAIN UI =====================
st.title("Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI image to classify it into one of the tumor categories.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    # -------- Load Image --------
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    # -------- Preprocess --------
    img_resized = cv2.resize(img_np, (224, 224))
    img_array = preprocess_input(img_resized.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    # -------- Prediction --------
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = np.max(preds)
    pred_class = CLASS_NAMES[pred_idx]

    with col2:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", pred_class)
        st.metric("Confidence", f"{confidence:.2f}")

        st.markdown("### Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(float(preds[0][i]), text=f"{cls}: {preds[0][i]:.2f}")

    # -------- Grad-CAM --------
    if show_cam:
        st.markdown("---")
        st.subheader("Model Explanation (Grad-CAM)")

        heatmap = grad_cam(model, img_array)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            img_resized, 0.6,
            heatmap_color, 0.4,
            0
        )

        # -------- Zoom Region --------
        h, w = heatmap_resized.shape
        y, x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        size = int(0.15 * min(h, w))

        x1, y1 = max(0, x - size), max(0, y - size)
        x2, y2 = min(w, x + size), min(h, y + size)
        zoom = img_resized[y1:y2, x1:x2]

        cam_col1, cam_col2, cam_col3 = st.columns(3)

        with cam_col1:
            st.image(img_resized, caption="Original", use_container_width=True)

        with cam_col2:
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

        with cam_col3:
            st.image(zoom, caption="Important Region", use_container_width=True)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("Developed for Brain Tumor Classification using EfficientNet & Streamlit")