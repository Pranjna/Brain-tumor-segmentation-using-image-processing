import cv2
import numpy as np
import streamlit as st
import sqlite3
import os
from datetime import datetime
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
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

# --- Session State for Theme ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    if st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'

# --- Custom CSS (Dynamic) ---
dark_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #ffffff; }
        .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
        section[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea { background-color: rgba(255, 255, 255, 0.05); color: white; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; }
        .stButton>button { background: linear-gradient(45deg, #6c5ce7, #a29bfe); color: white; border: none; border-radius: 8px; }
        h1, h2, h3 { color: transparent; background: linear-gradient(to right, #fff, #a29bfe); -webkit-background-clip: text; background-clip: text; font-weight: 700; }
        .history-card { background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
"""

light_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1a1a1a; }
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        section[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px); border-right: 1px solid rgba(0, 0, 0, 0.1); }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea { background-color: white; color: #333; border: 1px solid #ccc; border-radius: 8px; }
        .stButton>button { background: linear-gradient(45deg, #6c5ce7, #a29bfe); color: white; border: none; border-radius: 8px; }
        h1, h2, h3 { color: #6c5ce7; font-weight: 700; }
        .history-card { background: rgba(255, 255, 255, 0.6); padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(0,0,0,0.1); color: #333; }
        p, span, div { color: #333; }
    </style>
"""

st.markdown(dark_css if st.session_state.theme == 'dark' else light_css, unsafe_allow_html=True)

# --- Database Helper ---
DB_PATH = 'users.db'

def fetch_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Order by ID desc
        c.execute("SELECT patient_name, tumor_type, confidence, scan_date FROM scan_history ORDER BY id DESC LIMIT 5")
        rows = c.fetchall()
        conn.close()
        return rows
    except:
        return []

def save_history_record(patient_name, tumor_type, confidence):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Dummy user_id=1 for demo
        c.execute("INSERT INTO scan_history (user_id, patient_name, tumor_type, confidence, scan_date) VALUES (?, ?, ?, ?, ?)", 
                  (1, patient_name, tumor_type, confidence, ts))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# --- Sidebar ---
with st.sidebar:
    try:
        st.image("brain.png", width=80)
    except:
        st.write("🧠")
        
    st.markdown("### Tumor Vue")
    st.caption("AI Brain Tumor Detector")
    
    # Theme Toggle
    st.markdown("---")
    st.write("Display Mode")
    col_t1, col_t2 = st.columns([1,1])
    with col_t1:
        if st.button("☀ Light"):
            st.session_state.theme = 'light'
            st.rerun()
    with col_t2:
        if st.button("🌙 Dark"):
            st.session_state.theme = 'dark'
            st.rerun()

    st.markdown("---")
    st.markdown("### 🕒 Recent History")
    history = fetch_history()
    if history:
        for row in history:
            st.markdown(f"""
            <div class="history-card">
                <b>{row[0]}</b><br>
                <span style="font-size:0.8em">{row[1]} ({row[2]*100:.0f}%)</span><br>
                <span style="font-size:0.7em; opacity:0.7">{row[3]}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent scans.")

    st.markdown("---")
    st.markdown('<a href="http://localhost:5000/logout" style="text-decoration:none; color:inherit;"><button style="width:100%; padding:8px; border-radius:6px; background:#ff7675; color:white; border:none; cursor:pointer;">Logout</button></a>', unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_tumor_model():
    try:
        model = load_model('brain_tumor_classifier.h5', compile=False)
        return model, None
    except:
        try:
             model = load_model('brain_tumor_classifier.h5', custom_objects={'InputLayer': CompatInputLayer}, compile=False)
             return model, None
        except Exception as e:
            return None, traceback.format_exc()

model, error_trace = load_tumor_model()
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- PDF Generation ---
def generate_pdf(name, age, gender, contact, symptoms, duration, history, tumor_type, confidence, img_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="TUMOR VUE - Diagnostic Report", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=1)
    pdf.cell(200, 10, txt=f"Age: {age} | Gender: {gender}", ln=1)
    pdf.cell(200, 10, txt=f"Contact: {contact}", ln=1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Analysis Results", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Class: {tumor_type.upper()}", ln=1)
    pdf.cell(200, 10, txt=f"AI Confidence: {confidence*100:.2f}%", ln=1)
    
    # Add Image
    if img_path and os.path.exists(img_path):
        pdf.image(img_path, x=10, y=120, w=100)
    
    pdf.ln(110)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Clinical Notes", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=f"Symptoms: {symptoms}\nDuration: {duration}\nMedical History: {history}")
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This report is generated by an AI system (Tumor Vue) and should be verified by a certified radiologist.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- Main Layout ---
st.title("Patient Diagnosis & Analysis")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.form(key='patient_form'):
        st.subheader("Patient Input")
        name = st.text_input("Name")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 0, 120)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        contact = st.text_input("Contact Info")
        symptoms = st.text_area("Symptoms")
        duration = st.text_input("Duration of Symptoms")
        history = st.text_area("Medical History")
        
        uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])
        btn = st.form_submit_button("Run Analysis")

with col2:
    if btn:
        # Input Validation
        missing = []
        if not name.strip(): missing.append("Name")
        if age <= 0: missing.append("Valid Age")
        if not contact.strip(): missing.append("Contact Info")
        if not symptoms.strip(): missing.append("Symptoms")
        if not duration.strip(): missing.append("Duration")
        if not history.strip(): missing.append("Medical History")
        if not uploaded_file: missing.append("MRI Scan")
        
        if missing:
            st.error(f"⚠️ Missing Mandatory Fields: {', '.join(missing)}")
        else:
            bytes_data = uploaded_file.read()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                st.error("Invalid Image File")
            else:
                # 1. Prediction
                if model:
                    img_resized = cv2.resize(image, (128, 128)) / 255.0
                    img_reshaped = img_resized.reshape(1, 128, 128, 1)
                    pred = model.predict(img_reshaped)
                    idx = np.argmax(pred)
                    tumor_type = categories[idx]
                    conf = pred[0][idx]
                    
                    # Display Original Image (No Highlight)
                    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                    
                    # Display Result
                    st.success(f"Prediction: **{tumor_type.upper()}**")
                    
                    # 3. Save to DB
                    save_history_record(name, tumor_type, float(conf))
                    
                    # 4. Generate PDF
                    # Save temp image for PDF (Original)
                    temp_img_path = "temp_scan.jpg"
                    cv2.imwrite(temp_img_path, image)
                    
                    pdf_bytes = generate_pdf(name, age, gender, contact, symptoms, duration, history, tumor_type, conf, temp_img_path)
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{name}_Diagnostic_Report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("Model not loaded.")