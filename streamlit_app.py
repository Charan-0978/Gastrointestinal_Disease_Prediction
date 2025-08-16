import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("model.h5")
class_names = sorted(os.listdir("kvasir-dataset"))

def preprocess_image(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image.reshape(1, 128, 128, 3), image

st.title("🩺 Gastrointestinal Disease Detection")
uploaded_file = st.file_uploader("Upload an Endoscopy Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_input, image_display = preprocess_image(uploaded_file)
    prediction = model.predict(image_input)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(image_display, caption="Uploaded Image", channels="BGR")
    st.success(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
