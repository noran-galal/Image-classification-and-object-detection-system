import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import os
import torch
from pathlib import Path

# Load models
@st.cache_resource
def load_models():
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    
    # Construct absolute paths to the model files
    yolo_path = script_dir / 'yolov8_trained.pt'
    resnet_path = script_dir / 'best_resnet50_cifar10.keras'
    
    # Verify files exist before loading
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model file not found at: {yolo_path}")
    if not resnet_path.exists():
        raise FileNotFoundError(f"ResNet model file not found at: {resnet_path}")
    
    yolo_model = YOLO(str(yolo_path))
    resnet_model = tf.keras.models.load_model(str(resnet_path))
    return yolo_model, resnet_model
yolo_model, resnet_model = load_models()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app
st.title("Object Detection and Image Classification App")
st.write("Upload an image and choose a task to perform: Object Detection (YOLOv8), Image Classification (ResNet50), or Both.")

# Task selection
task = st.radio("Select Task:", ["Object Detection", "Image Classification", "Both"], index=2)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert PIL image to OpenCV and NumPy formats
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Object Detection with YOLOv8
    if task in ["Object Detection", "Both"]:
        st.subheader("Object Detection Results (YOLOv8)")
        results = yolo_model.predict(source=img_array, save=False, conf=0.25)
        
        # Display detected objects
        detected_img = results[0].plot()  # Plot bounding boxes
        detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        st.image(detected_img_rgb, caption='Detected Objects', use_column_width=True)
        
        # Extract detected class names
        detected_classes = [yolo_model.names[int(cls)] for cls in results[0].boxes.cls]
        st.write("Detected Objects:", ", ".join(detected_classes) if detected_classes else "None")
    
    # Image Classification with ResNet50
    if task in ["Image Classification", "Both"]:
        st.subheader("Image Classification Results (ResNet50)")
        # Resize image to 32x32 for ResNet50
        img_resized = cv2.resize(img_array, (32, 32))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        pred_probs = resnet_model.predict(img_input)
        predicted_class = np.argmax(pred_probs, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence = pred_probs[0][predicted_class]
        
        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Confidence: {confidence:.4f}")

