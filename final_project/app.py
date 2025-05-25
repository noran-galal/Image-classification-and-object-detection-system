
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Load models
@st.cache_resource
def load_models():
    yolo = YOLO("models/yolov8.pt")
    resnet = load_model("models/resnet50.h5")
    return yolo, resnet

yolo_model, resnet_model = load_models()

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit UI
st.set_page_config(page_title="Object Detection & Classification", layout="wide")
st.title("🧠 Integrated Object Detection and Classification System")
st.write("Upload an image to detect and classify objects using YOLOv8 and ResNet50.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Save the uploaded image temporarily and detect
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = yolo_model(temp.name)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    annotated = image_np.copy()

    st.subheader("🔍 Detection and Classification Results")
    col1, col2 = st.columns([2, 1])

    with col1:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (32, 32))
            crop_normalized = crop_resized / 255.0
            crop_input = np.expand_dims(crop_normalized, axis=0)

            preds = resnet_model.predict(crop_input, verbose=0)
            class_idx = np.argmax(preds)
            label = class_names[class_idx]
            confidence = float(np.max(preds))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        st.image(annotated, caption="Detected and Classified Objects", use_column_width=True)

    with col2:
        st.info("Detected Object Summary")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (32, 32))
            crop_normalized = crop_resized / 255.0
            crop_input = np.expand_dims(crop_normalized, axis=0)
            preds = resnet_model.predict(crop_input, verbose=0)
            class_idx = np.argmax(preds)
            label = class_names[class_idx]
            confidence = float(np.max(preds))
            st.write(f"📦 **{label}** - Confidence: {confidence:.2f}")
