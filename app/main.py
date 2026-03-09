import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# -----------------------------------------------------------
# 1️⃣ PATHS — Fixed for your actual folder structure
# -----------------------------------------------------------

# Get the folder where main.py is located
working_dir = os.path.dirname(os.path.abspath(__file__))

# Model and JSON are inside: .venv/app/trained_model/
model_dir = os.path.join(working_dir, "..", "..", "app", "trained_model")


# ✅ Your actual filenames
model_path = r"C:\Users\Sathw\PycharmProjects\PythonProject4\.venv\app\trained_model\plant_disease_model.h5"

class_indices_path = r"C:\Users\Sathw\PycharmProjects\PythonProject4\.venv\app\class_indices (1).json"

# Strip out any hidden characters (like '\r' or '\n')
model_path = model_path.strip()
class_indices_path = class_indices_path.strip()
print("🔍 Checking model file:", model_path)
print("🔍 File exists?", os.path.exists(model_path))

# -----------------------------------------------------------
# 2️⃣ LOAD MODEL AND CLASS LABELS
# -----------------------------------------------------------

st.write("🔍 Looking for model at:", model_path)
st.write("🔍 Looking for classes at:", class_indices_path)

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at:\n{model_path}")
    st.stop()

if not os.path.exists(class_indices_path):
    st.error(f"❌ class_indices.json not found at:\n{class_indices_path}")
    st.stop()

# Load trained model
model = tf.keras.models.load_model(model_path)

# Load class labels
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

st.success("✅ Model and class indices loaded successfully!")

# -----------------------------------------------------------
# 3️⃣ IMAGE PREPROCESSING
# -----------------------------------------------------------

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """Preprocess image for prediction"""
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(model, image_file, class_indices):
    """Run model prediction"""
    img_array = load_and_preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_index)]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

# -----------------------------------------------------------
# 4️⃣ STREAMLIT WEB UI
# -----------------------------------------------------------

st.title("🌿  Plant Disease Detection")
st.markdown("Detect plant leaf diseases by **uploading an image** or using your **camera**.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Capture from Camera"])

# ----------- TAB 1: UPLOAD IMAGE -----------
with tab1:
    uploaded_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image.resize((200, 200)), caption="Uploaded Image", use_column_width=False)

        with col2:
            if st.button("🔍 Classify Uploaded Image"):
                predicted_class, confidence = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"✅ Prediction: {predicted_class}")
                st.info(f"📊 Confidence: {confidence:.2f}%")

# ----------- TAB 2: CAMERA INPUT -----------
with tab2:
    st.markdown("Take a photo using your device camera:")
    camera_image = st.camera_input("📷 Capture a photo")

    if camera_image is not None:
        image = Image.open(camera_image)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image.resize((200, 200)), caption="Captured Image", use_column_width=False)

        with col2:
            if st.button("🔍 Classify Captured Image"):
                predicted_class, confidence = predict_image_class(model, camera_image, class_indices)
                st.success(f"✅ Prediction: {predicted_class}")
                st.info(f"📊 Confidence: {confidence:.2f}%")
