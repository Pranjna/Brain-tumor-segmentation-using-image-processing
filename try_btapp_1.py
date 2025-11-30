import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pretrained ViT model
model = load_model('brain_tumor_classifier.h5')
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit app setup
st.title("Brain Tumor Classifier with Patient Details")

# Patient details form
with st.form(key='patient_form'):
    st.header("Patient Information")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    contact = st.text_input("Contact Information (Phone/Email)")
    symptoms = st.text_area("Symptoms")
    duration = st.text_input("Duration of Symptoms")
    medical_history = st.text_area("Medical History")
    uploaded_file = st.file_uploader("Upload MRI Image...", type=["jpg", "jpeg", "png"])

    submit_button = st.form_submit_button(label='Submit and Predict')

# Handle form submission
if submit_button:
    if uploaded_file is not None:
        try:
            # Display uploaded image
            bytes_data = uploaded_file.read()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                st.error("Failed to load the image. Please try a different one.")
            else:
                st.image(image, caption='Uploaded MRI Image', use_column_width=True)

                img_resized = cv2.resize(image, (128, 128)) / 255.0
                img_reshaped = img_resized.reshape(1, 128, 128, 1)

        # Make the prediction
                prediction = model.predict(img_reshaped)
                class_index = np.argmax(prediction)
                tumor_type = categories[class_index]
                confidence = prediction[0][class_index]

                # Show patient details and results
                st.subheader("Patient Details")
                st.write(f"**Name:** {name}")
                st.write(f"**Age:** {age}")
                st.write(f"**Gender:** {gender}")
                st.write(f"**Contact:** {contact}")
                st.write(f"**Symptoms:** {symptoms}")
                st.write(f"**Duration of Symptoms:** {duration}")
                st.write(f"**Medical History:** {medical_history}")
                st.subheader("Prediction Result")
                st.write(f"**Predicted Tumor Type:** {tumor_type}")
                st.write(f"**Confidence:** {confidence:.2f}")
                report = f"""Patient Details
---------------
Name: {name}
Age: {age}
Gender: {gender}
Contact: {contact}
Symptoms: {symptoms}
Duration of Symptoms: {duration}
Medical History: {medical_history}

Prediction Result
-----------------
Predicted Tumor Type: {tumor_type}
Confidence: {confidence:.2f}
"""

# Create a downloadable text file
                st.download_button(
    label="Download Report",
    data=report,
    file_name=f"{name}_report.txt",
    mime="text/plain"
)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload an MRI image.")