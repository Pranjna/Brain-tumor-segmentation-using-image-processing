import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import traceback
import h5py

# --- Keras Compatibility Shims ---
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

class CompatInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None and 'batch_input_shape' not in kwargs:
            kwargs['batch_input_shape'] = tuple(batch_shape)
        super().__init__(**kwargs)

get_custom_objects()['InputLayer'] = CompatInputLayer

# --- Page Config ---
st.set_page_config(
    page_title="Tumor Vue",
    page_icon="brain.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Glassmorphism Theme ---
st.markdown("""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }
        
        /* Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Inputs */
        .stTextInput > div > div > input, 
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stTextArea > div > div > textarea {
            background-color: rgba(255, 255, 255, 0.05);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, #6c5ce7, #a29bfe);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
            border: none;
        }

        /* Headers */
        h1, h2, h3 {
            color: transparent;
            background: linear-gradient(to right, #fff, #a29bfe);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: 700;
        }

        /* Card/Container effect for main area */
        .block-container {
            padding-top: 2rem;
        }
        
        /* Logout Button Style */
        .logout-btn {
            display: inline-block;
            text-decoration: none;
            background-color: rgba(255, 76, 76, 0.2);
            color: #ff7675;
            border: 1px solid #ff7675;
            padding: 8px 16px;
            border-radius: 6px;
            text-align: center;
            transition: all 0.3s;
            width: 100%;
            margin-top: 20px;
        }
        .logout-btn:hover {
            background-color: #ff7675;
            color: white;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Branding & Logout ---
with st.sidebar:
    try:
        st.image("brain.png", width=100)
    except:
        st.write("🧠") # Fallback if image missing
        
    st.markdown("<h1>Tumor Vue</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b2bec3; font-style: italic; margin-top: -15px;'>An AI Based Brain Tumor Detector</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Navigation")
    
    # Logout Logic
    # Since we can't easily redirect via button click in pure streamlit without rerun hack,
    # we use a styled link which is cleanest for 'Logout' to external Flask app.
    st.markdown('<a href="http://localhost:5000/logout" class="logout-btn" target="_self">Logout</a>', unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_tumor_model():
    try:
        model = load_model('brain_tumor_classifier.h5', compile=False)
        return model, None
    except Exception:
        # Retry with shim
        try:
             model = load_model('brain_tumor_classifier.h5', custom_objects={'InputLayer': CompatInputLayer}, compile=False)
             return model, None
        except Exception as e:
            return None, traceback.format_exc()

model, error_trace = load_tumor_model()
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

if error_trace:
    st.error("Failed to load model. Application provided for data entry only.")
    with st.expander("Show Error Details"):
        st.code(error_trace)

# --- Main Content ---
st.markdown("## Patient Diagnosis & MRI Analysis")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.form(key='patient_form'):
        st.subheader("Patient Information")
        name = st.text_input("Full Name", placeholder="e.g. John Doe")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        contact = st.text_input("Contact", placeholder="Phone or Email")
        symptoms = st.text_area("Symptoms", height=100)
        duration = st.text_input("Duration of Symptoms")
        medical_history = st.text_area("Medical History", height=100)
        
        st.markdown("### MRI Upload")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        submit_button = st.form_submit_button(label='Analyze & Generate Report')

with col2:
    if submit_button:
        if uploaded_file is not None:
            try:
                # Process Image
                bytes_data = uploaded_file.read()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    # Show Image
                    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
                    
                    # Predict
                    tumor_type = "Unknown"
                    confidence = 0.0
                    
                    if model:
                        img_resized = cv2.resize(image, (128, 128)) / 255.0
                        img_reshaped = img_resized.reshape(1, 128, 128, 1)
                        prediction = model.predict(img_reshaped)
                        class_index = np.argmax(prediction)
                        tumor_type = categories[class_index]
                        confidence = prediction[0][class_index]
                        
                        # Result Display
                        st.success("Analysis Complete")
                        st.metric(label="Predicted Classification", value=tumor_type.title(), delta=f"{confidence*100:.1f}% Confidence")
                        
                        # Report Logic
                        report = f"""TUMOR VUE - PATIENT REPORT
================================
Date: {np.datetime64('today')}

PATIENT DETAILS
---------------
Name:     {name}
Age:      {age}
Gender:   {gender}
Contact:  {contact}

CLINICAL NOTES
--------------
Symptoms: {symptoms}
Duration: {duration}
History:  {medical_history}

ANALYSIS RESULT
---------------
Prediction: {tumor_type.upper()}
Confidence: {confidence:.2f}

--------------------------------
Generated by Tumor Vue AI
"""
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name=f"{name}_report.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("Model not loaded. Prediction skipped.")
                else:
                    st.error("Could not read image file.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("👆 Please upload an MRI scan to start the analysis.")
    else:
        st.markdown("""
        ### Instructions
        1. Fill in the **Patient Information** on the left.
        2. Upload a clear **MRI Scan** (JPG/PNG).
        3. Click **Analyze** to get AI predictions.
        
        The AI will classify the tumor type and generate a downloadable report.
        """)