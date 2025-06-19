import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# === Load model ===
MODEL_PATH = '../models/densenet121_mask_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['mask_worn_incorrectly', 'with_mask', 'without_mask']

# === Page Config ===
st.set_page_config(page_title="Face Mask Detector", layout="centered")

# === App Title ===
st.markdown("<h1 style='text-align: center; font-size: 40px;'>😷 Face Mask Detection Web App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload or capture a face image to check mask status</h4>", unsafe_allow_html=True)
st.write("---")

# === Option: Upload or Webcam ===
option = st.radio("📸 Choose image input method:", ["Upload an image", "Use webcam"])

image_data = None

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = Image.open(uploaded_file)

elif option == "Use webcam":
    capture = st.camera_input("Take a picture")
    if capture:
        image_data = Image.open(capture)

# === Prediction Section ===
if image_data:
    st.image(image_data, caption="Input Image", use_container_width=True)

    # Preprocess
    img = image_data.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.write("---")
    st.markdown(f"<h2 style='text-align: center;'>🎯 Result:</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='text-align: center; color: purple;'>Class: <b>{class_names[class_index].replace('_', ' ').title()}</b> — Confidence Score: <b>{confidence:.2%}</b></h3>", unsafe_allow_html=True)

