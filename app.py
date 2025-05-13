import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Load model and scaler
try:
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    top_features = ['trestbps', 'slope_2.0', 'chol', 'slope_1.0', 'thalach', 'oldpeak', 'ca', 'age',
                    'slope_3.0', 'cp_2.0', 'restecg_2.0', 'cp_4.0']
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Ensure rf_model.pkl and scaler.pkl are in {os.getcwd()}")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="CardioPredict AI", layout="wide")

# Custom CSS for dark theme, animations, and responsiveness
st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Helvetica', sans-serif;
    }
    .main {
        background-color: #121212;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 5px;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }
    .stTextInput>div>input, .stSelectbox>div>div {
        border: 1px solid #1976d2;
        border-radius: 8px;
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 8px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput>div>input:focus, .stSelectbox>div>div:focus {
        border-color: #1976d2;
        box-shadow: 0 0 5px #1976d2;
    }
    h1, h2, h3 {
        color: #1976d2;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        border-right: 1px solid #1976d2;
    }
    .stForm {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #1976d2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .progress-container {
        width: 100%;
        background-color: #2e3b3e;
        border-radius: 8px;
        overflow: hidden;
    }
    .progress-bar {
        height: 30px;
        transition: width 0.5s ease-in-out;
    }
    .prediction-result {
        display: flex;
        align-items: center;
        gap: 10px;
        animation: fadeIn 0.5s ease-in;
    }
    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-left-color: #d32f2f;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stSuccess, .stInfo, .stWarning {
        border-radius: 8px;
        animation: fadeIn 0.5s ease-in;
    }
    .input-label {
        display: flex;
        align-items: center;
        gap: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    @media (max-width: 768px) {
        .stColumns > div {
            width: 100% !important;
        }
        .stButton>button {
            width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Disclaimer at the top
st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #d32f2f;'>
        <h3 style='color: #d32f2f;'>Disclaimer</h3>
        <p>This tool is for educational purposes only. It uses a Random Forest model trained on UCI Statlog and other datasets to estimate heart disease risk. Results are not a substitute for professional medical advice. Consult a healthcare provider for accurate diagnosis and treatment.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("CardioPredict AI")
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <i class='fas fa-heartbeat' style='color: #d32f2f; font-size: 24px;'></i>
            <span>A machine learning tool to assess heart disease risk.</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        - **Key Features**: Resting BP, Cholesterol, Max HR, ST Depression, Vessels, Age
        - **Data Source**: UCI Statlog and other heart disease datasets
        - **How to Use**: Enter patient details and click 'Predict' to assess risk.
    """)
    with st.expander("Learn About Heart Disease"):
        st.markdown("""
            **What is Heart Disease?**
            Heart disease is the leading cause of death globally, encompassing conditions like coronary artery disease and heart failure. Risk factors include high blood pressure, high cholesterol, smoking, and family history.

            **Prevention Tips:**
            - Maintain a healthy diet low in saturated fats.
            - Exercise regularly (at least 150 minutes/week).
            - Avoid smoking and limit alcohol intake.
            - Monitor blood pressure and cholesterol levels.

            **Resources:**
            - [CDC Heart Disease](https://www.cdc.gov/heartdisease/)
            - [American Heart Association](https://www.heart.org/)
        """)
    st.markdown("[Contact Us](mailto:debangshubhattacharya4@gmail.com)")

# Main App
st.title("CardioPredict AI")
st.markdown("Evaluate your heart disease risk with our advanced machine learning tool.")

# Form inputs
with st.form("patient_form"):
    st.markdown("### Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-label'><i class='fas fa-user' style='color: #1976d2;'></i> Age (years)</div>", unsafe_allow_html=True)
        age = st.number_input("", min_value=1, max_value=120, value=None, placeholder="e.g., 45", key="age")
        st.markdown("<div class='input-label'><i class='fas fa-tachometer-alt' style='color: #1976d2;'></i> Resting BP (mm Hg)</div>", unsafe_allow_html=True)
        trestbps = st.number_input("", min_value=50, max_value=200, value=None, placeholder="e.g., 120", key="trestbps")
        st.markdown("<div class='input-label'><i class='fas fa-flask' style='color: #1976d2;'></i> Cholesterol (mg/dl)</div>", unsafe_allow_html=True)
        chol = st.number_input("", min_value=100, max_value=600, value=None, placeholder="e.g., 200", key="chol")
    with col2:
        st.markdown("<div class='input-label'><i class='fas fa-heartbeat' style='color: #1976d2;'></i> Max Heart Rate</div>", unsafe_allow_html=True)
        thalach = st.number_input("", min_value=60, max_value=220, value=None, placeholder="e.g., 150", key="thalach")
        st.markdown("<div class='input-label'><i class='fas fa-chart-line' style='color: #1976d2;'></i> ST Depression</div>", unsafe_allow_html=True)
        oldpeak = st.number_input("", min_value=0.0, max_value=6.0, step=0.1, value=None, placeholder="e.g., 1.0", key="oldpeak")
        st.markdown("<div class='input-label'><i class='fas fa-tint' style='color: #1976d2;'></i> Major Vessels (0-3)</div>", unsafe_allow_html=True)
        ca = st.number_input("", min_value=0, max_value=3, value=None, placeholder="e.g., 1", key="ca")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='input-label'><i class='fas fa-stethoscope' style='color: #1976d2;'></i> Chest Pain Type</div>", unsafe_allow_html=True)
        cp = st.selectbox("", ["Select", 1, 2, 3, 4], index=0, help="1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic", key="cp")
        st.markdown("<div class='input-label'><i class='fas fa-wave-square' style='color: #1976d2;'></i> Resting ECG</div>", unsafe_allow_html=True)
        restecg = st.selectbox("", ["Select", 0, 1, 2], index=0, help="0: Normal, 1: ST-T abnormality, 2: LV hypertrophy", key="restecg")
    with col4:
        st.markdown("<div class='input-label'><i class='fas fa-chart-area' style='color: #1976d2;'></i> ST Slope</div>", unsafe_allow_html=True)
        slope = st.selectbox("", ["Select", 1, 2, 3], index=0, help="1: Upsloping, 2: Flat, 3: Downsloping", key="slope")
        st.markdown("<p style='font-size: 14px; color: #e0e0e0;'>All fields are required.</p>", unsafe_allow_html=True)

    col_submit, col_clear = st.columns(2)
    with col_submit:
        submitted = st.form_submit_button("Predict")
    with col_clear:
        cleared = st.form_submit_button("Clear Form")

# Reset form
if cleared:
    st.experimental_rerun()

# Function to generate PDF report
def generate_pdf_report(inputs, prediction, prob):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CardioPredict AI Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Input Parameters:", styles['Heading2']))
    for key, value in inputs.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Prediction Results:", styles['Heading2']))
    story.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    story.append(Paragraph(f"Risk Probability: {prob:.2%}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Disclaimer: This is an educational tool. Consult a healthcare professional for medical advice.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Prediction
if submitted:
    if any([age is None, trestbps is None, chol is None, thalach is None, oldpeak is None, ca is None,
            cp == "Select", restecg == "Select", slope == "Select"]):
        st.error("Please complete all fields.")
    else:
        st.markdown("<div class='spinner'></div>", unsafe_allow_html=True)
        time.sleep(1)  # Simulate processing
        input_data = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak, 'ca': ca,
            'cp_2.0': 1 if cp == 2 else 0, 'cp_4.0': 1 if cp == 4 else 0, 'restecg_2.0': 1 if restecg == 2 else 0,
            'slope_1.0': 1 if slope == 1 else 0, 'slope_2.0': 1 if slope == 2 else 0, 'slope_3.0': 1 if slope == 3 else 0
        }
        input_df = pd.DataFrame([input_data])
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        input_df = input_df[top_features]

        prediction = rf_model.predict(input_df)[0]
        prob = rf_model.predict_proba(input_df)[0][1]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        # Color based on risk
        color = "#4CAF50" if prob < 0.3 else "#FFC107" if prob < 0.7 else "#d32f2f"

        # Results
        st.markdown("### Prediction Results")
        st.markdown(f"""
            <div class='prediction-result'>
                <span style='font-size: 24px; color: {color};'>{'❌' if prediction == 1 else '✅'}</span>
                <span>{result}</span>
            </div>
        """, unsafe_allow_html=True)
        st.info(f"**Risk Probability**: {prob:.2%}")

        # Progress bar
        st.markdown("### Risk Level")
        st.markdown(f"""
            <div class='progress-container'>
                <div class='progress-bar' style='width: {prob * 100}%; background-color: {color};'></div>
            </div>
        """, unsafe_allow_html=True)

        # Interpretation
        st.markdown("### What This Means")
        if prob > 0.7:
            st.warning("**High Concern**: Immediate consultation with a cardiologist is recommended.")
        elif prob > 0.3:
            st.info("**Moderate Concern**: Monitor your health and consult a doctor if symptoms persist.")
        else:
            st.success("**Low Concern**: Maintain a healthy lifestyle.")

        # Downloadable report
        input_summary = {
            "Age": age, "Resting BP (mm Hg)": trestbps, "Cholesterol (mg/dl)": chol,
            "Max Heart Rate": thalach, "ST Depression": oldpeak, "Major Vessels": ca,
            "Chest Pain Type": cp, "Resting ECG": restecg, "ST Slope": slope
        }
        pdf_buffer = generate_pdf_report(input_summary, result, prob)
        st.markdown("### Download Your Report")
        st.download_button(
            label="Download Prediction Report",
            data=pdf_buffer,
            file_name="cardiopredict_ai_report.pdf",
            mime="application/pdf",
            key="download_button"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>Developed by Group - 23 </a></p>
    </div>
""", unsafe_allow_html=True)