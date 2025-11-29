import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('brain_tumor_classifier.h5')
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit app
st.title("Brain Tumor Classifier")
st.write("Upload an image to classify the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        bytes_data = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            st.error("Failed to load the image. Please try a different image.")
        else:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for prediction
        img_resized = cv2.resize(image, (128, 128)) / 255.0
        img_reshaped = img_resized.reshape(1, 128, 128, 1)

        # Make the prediction
        prediction = model.predict(img_reshaped)
        class_index = np.argmax(prediction)
        tumor_type = categories[class_index]
        confidence = prediction[0][class_index]

        # Display the prediction
        st.write(f"Predicted Tumor Type: {tumor_type}")
        st.write(f"Confidence: {confidence:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")