# 📄 Full Enhanced Streamlit Web App for Employability Prediction

import streamlit as st
import pandas as pd
import joblib
import base64
from fpdf import FPDF
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Employability Predictor", page_icon="🎓", layout="centered")

# --- Set Background ---
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.85);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Now call the function with your image file name
set_background("background.jpg")  # or "background.png" if that's the format

# --- Custom CSS ---
st.markdown("""
    <style>
    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #2B5DFF;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    div.stButton > button:first-child {
        background-color: #2B5DFF;
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1a3eb5;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
model = joblib.load("best_mod.pkl")
scaler = joblib.load("scale.pkl")

# --- App Title ---
st.markdown('<h1 class="title">🎓 Predicting Employability</h1>', unsafe_allow_html=True)

# --- Introduction ---
with st.expander("ℹ️ About This Tool"):
    st.write("""
        This web application predicts a graduate's likelihood of being employed based on academic, personal, and experiential attributes using a machine learning model trained on real graduate data.
    """)

# --- Sidebar Info ---
with st.sidebar:
    st.header("📊 Dataset Info")
    st.write("**Training Set:** 540 Employed, 140 Unemployed")
    st.write("**Test Split:** 20% of total data")
    st.caption("Built for final year project presentation and demo.")

# --- Candidate Form ---
st.subheader("🗘️ Candidate Profile")
candidate_name = st.text_input("Full Name of the Candidate")
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.selectbox("Age", ["21", "22", "23", "24", "25"])
CGPA = st.selectbox("CGPA Range", [
    "4.00 - 3.75", "3.74 - 3.50", "3.49 - 3.00", "2.99 - 2.50",
    "2.49 - 2.00", "1.99 - 1.50", "1.49 - 1.00", "Below 1.00"
])
Race = st.selectbox("Race / Ethnicity", ["Malay", "Chinese", "Indian", "Others"])
Area_Field = st.selectbox("Programme / Field of Study", [
    "Bachelor of Information Technology (Hons)",
    "Bachelor of Science (Hons) Statistics",
    "Bachelor of Science (Hons) Actuarial Science",
    "Bachelor of Science (Hons) Management Mathematics",
    "Bachelor of Science (Hons) Mathematics",
    "Bachelor of Computer Science (Hons) Netcentric Computing",
    "Bachelor of Computer Science (Hons) Multimedia Computing",
    "Bachelor of Computer Science (Hons) Computer Networks",
    "Bachelor of Information Systems (Hons) Business Computing",
    "Bachelor of Information Systems (Hons) Information Systems Engineering",
    "Bachelor of Science in Mathematical Modelling and Analytics (Hons)",
    "Bachelor of Library Management (Hons)",
    "Bachelor of Records Management (Hons)",
    "Bachelor of Information Management (Hons)",
    "Bachelor of Content Management (Hons)"
])
Income = st.selectbox("Monthly Household Income", [
    "Less than RM1,000", "RM1,000 - RM1,499", "RM1,500 - RM1,999",
    "RM2,000 - RM2,499", "RM2,500 - RM2,999", "RM3,000 - RM3,499",
    "RM3,500 - RM3,999", "RM4,000 - RM4,499", "RM4,500 - RM4,999",
    "RM5,000 - RM5,499", "RM5,500 - RM5,999", "RM6,000 - RM6,499",
    "RM6,500 - RM6,999", "RM7,000 - RM7,499", "RM7,500 - RM7,999",
    "RM8,000 - RM8,499", "RM8,500 - RM8,999", "RM9,000 - RM9,499",
    "RM9,500 - RM9,999"
])
Internship = st.selectbox("Completed Internship", ["Yes", "No"])
Industry_Match = st.selectbox("Internship Related to Field", ["Yes", "No"])
Skill = st.selectbox("Technical Skill Level", ["Low", "Medium", "High"])
Extra_curricular = st.selectbox("Involved in Extra-curricular Activities", ["Yes", "No"])

# --- Encoding ---
def manual_encode(val, mapping):
    return mapping.get(val, 0)

# --- Input Transformation ---
input_data = pd.DataFrame([{
    "Gender": manual_encode(Gender, {"Male": 1, "Female": 0}),
    "Age": manual_encode(Age, {"21": 0, "22": 1, "23": 2, "24": 3, "25": 4}),
    "CGPA": manual_encode(CGPA, {
        "4.00 - 3.75": 0, "3.74 - 3.50": 1, "3.49 - 3.00": 2, "2.99 - 2.50": 3,
        "2.49 - 2.00": 4, "1.99 - 1.50": 5, "1.49 - 1.00": 6, "Below 1.00": 7
    }),
    "Race": manual_encode(Race, {"Malay": 0, "Chinese": 1, "Indian": 2, "Others": 3}),
    "Area Field": manual_encode(Area_Field, {name: idx for idx, name in enumerate([
        "Bachelor of Information Technology (Hons)",
        "Bachelor of Science (Hons) Statistics",
        "Bachelor of Science (Hons) Actuarial Science",
        "Bachelor of Science (Hons) Management Mathematics",
        "Bachelor of Science (Hons) Mathematics",
        "Bachelor of Computer Science (Hons) Netcentric Computing",
        "Bachelor of Computer Science (Hons) Multimedia Computing",
        "Bachelor of Computer Science (Hons) Computer Networks",
        "Bachelor of Information Systems (Hons) Business Computing",
        "Bachelor of Information Systems (Hons) Information Systems Engineering",
        "Bachelor of Science in Mathematical Modelling and Analytics (Hons)",
        "Bachelor of Library Management (Hons)",
        "Bachelor of Records Management (Hons)",
        "Bachelor of Information Management (Hons)",
        "Bachelor of Content Management (Hons)"
    ])}),
    "Income": manual_encode(Income, {name: idx for idx, name in enumerate([
        "Less than RM1,000", "RM1,000 - RM1,499", "RM1,500 - RM1,999",
        "RM2,000 - RM2,499", "RM2,500 - RM2,999", "RM3,000 - RM3,499",
        "RM3,500 - RM3,999", "RM4,000 - RM4,499", "RM4,500 - RM4,999",
        "RM5,000 - RM5,499", "RM5,500 - RM5,999", "RM6,000 - RM6,499",
        "RM6,500 - RM6,999", "RM7,000 - RM7,499", "RM7,500 - RM7,999",
        "RM8,000 - RM8,499", "RM8,500 - RM8,999", "RM9,000 - RM9,499",
        "RM9,500 - RM9,999"
    ])}),
    "Internship": manual_encode(Internship, {"Yes": 1, "No": 0}),
    "Industry Match": manual_encode(Industry_Match, {"Yes": 1, "No": 0}),
    "Skill": manual_encode(Skill, {"Low": 0, "Medium": 1, "High": 2}),
    "Extra-curricular": manual_encode(Extra_curricular, {"Yes": 1, "No": 0})
}])

# --- Match training columns ---
input_data = input_data[scaler.feature_names_in_]
input_scaled = scaler.transform(input_data)

# --- PDF Generator ---
def generate_pdf(name, result_label, confidence_percent):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Employability Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Candidate Name: {name}", ln=True)
    pdf.cell(0, 10, f"Prediction Date: {datetime.date.today()}", ln=True)
    pdf.cell(0, 10, f"Predicted Status: {result_label}", ln=True)
    pdf.cell(0, 10, f"Confidence Level: {confidence_percent:.2f}%", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "This result is based on a trained machine learning model. It reflects statistical patterns and does not guarantee actual employment outcomes.")
    return pdf.output(dest="S").encode("latin1")

# --- Prediction Button ---
if st.button("🎯 Predict Employment Status"):
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # FIXED: 1 = Employed, 0 = Unemployed
    if prediction == 1:
        label = "Employed"
        st.balloons()
        st.success(f"🎉 Congratulations {candidate_name}! The model predicts you are **Employed**.")
        confidence = probabilities[1] * 100
    else:
        label = "Unemployed"
        st.warning(f"🔍 The model predicts {candidate_name} is **Unemployed**.")
        confidence = probabilities[0] * 100

    st.metric(
        f"{label} Probability",
        f"{confidence:.2f}%"
    )

    st.subheader("📈 Probability Distribution")
    prob_df = pd.DataFrame({
        'Employment Status': ['Unemployed', 'Employed'],
        'Probability': [probabilities[0], probabilities[1]]
    })
    st.bar_chart(prob_df.set_index("Employment Status"))

    # PDF download
    pdf_bytes = generate_pdf(candidate_name, label, confidence)
    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=pdf_bytes,
        file_name="employability_report.pdf",
        mime="application/pdf"
    )

# --- Footer ---
st.markdown("""
---
<center>
© 2025 Predicting Employability | Final Year Project by Syamimi – UiTM
</center>
""", unsafe_allow_html=True)
