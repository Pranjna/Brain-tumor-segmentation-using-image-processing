import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

# Compatibility shim for models saved with a standalone `keras` DTypePolicy
class DTypePolicy:
    def __init__(self, name=None, **kwargs):
        self.name = name or 'float32'
        self.compute_dtype = tf.float32
        self.variable_dtype = tf.float32

    @classmethod
    def from_config(cls, config):
        return cls(**(config or {}))

    def get_config(self):
        return {'name': self.name}

get_custom_objects()['DTypePolicy'] = DTypePolicy

# Compatibility shim for InputLayer that accepts 'batch_shape' kwarg
from tensorflow.keras.layers import InputLayer as _KInputLayer

class CompatInputLayer(_KInputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None and 'batch_input_shape' not in kwargs:
            kwargs['batch_input_shape'] = tuple(batch_shape)
        super().__init__(**kwargs)

get_custom_objects()['InputLayer'] = CompatInputLayer

# Load the pretrained ViT model with diagnostics
model = None
try:
    model = load_model('brain_tumor_classifier.h5', compile=False)
except Exception as e:
    import traceback, h5py
    tb = traceback.format_exc()
    # If run under Streamlit show diagnostic info to the app
    try:
        st.error("Model load failed — see traceback below")
        st.text(tb)
    except Exception:
        print("Model load failed:\n", tb)

    # Try to inspect the HDF5 file to give clues
    try:
        with h5py.File('brain_tumor_classifier.h5', 'r') as f:
            keys = list(f.keys())
            try:
                st.write("HDF5 top-level keys:", keys)
            except Exception:
                print("HDF5 top-level keys:", keys)
            # show whether model config is present
            model_config = f.attrs.get('model_config') or f.attrs.get('model_config'.encode())
            if model_config is not None:
                try:
                    st.write("HDF5 contains 'model_config' attribute (model architecture saved)")
                except Exception:
                    print("HDF5 contains 'model_config' attribute (model architecture saved)")
            else:
                try:
                    st.write("No 'model_config' attr found — file may only contain weights")
                except Exception:
                    print("No 'model_config' attr found — file may only contain weights")
    except Exception as e2:
        try:
            st.write(f"Failed to inspect HDF5: {e2}")
        except Exception:
            print("Failed to inspect HDF5:", e2)
    # If the failure is due to an unexpected 'batch_shape' kwarg in InputLayer,
    # provide a compatibility shim and retry loading.
    try:
        from tensorflow.keras.layers import InputLayer as _KInputLayer
        from tensorflow.keras.utils import get_custom_objects

        class CompatInputLayer(_KInputLayer):
            def __init__(self, batch_shape=None, **kwargs):
                if batch_shape is not None and 'batch_input_shape' not in kwargs:
                    kwargs['batch_input_shape'] = tuple(batch_shape)
                super().__init__(**kwargs)

        get_custom_objects()['InputLayer'] = CompatInputLayer
        try:
            model = load_model('brain_tumor_classifier.h5', compile=False)
            try:
                st.success('Model loaded successfully with CompatInputLayer shim')
            except Exception:
                print('Model loaded successfully with CompatInputLayer shim')
        except Exception as e3:
            try:
                st.error('Retry with CompatInputLayer failed — see traceback')
                st.text(traceback.format_exc())
            except Exception:
                print('Retry with CompatInputLayer failed')
                print(traceback.format_exc())
    except Exception as e_shim:
        # If we can't create the shim, just continue — diagnostics already printed
        print('Could not apply CompatInputLayer shim:', e_shim)
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

                # Make the prediction (only if model loaded)
                if model is None:
                    st.error("Model not loaded — prediction unavailable. Check diagnostics above.")
                else:
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