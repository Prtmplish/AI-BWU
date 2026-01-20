import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# --- LOGO SECTION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("BWU_logo.png", width=200)

st.title("Student Performance & Interview Preparedness Dashboard")

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("Interview_Preparedness.pkl", "rb") as f:
        interview_model = pickle.load(f)
    with open("Student_Grading_Model.pkl", "rb") as f:
        grading_model = pickle.load(f)
    return interview_model, grading_model

# --- Sidebar ---
st.sidebar.image("BWU_logo.png", width=150)
st.sidebar.header("Student Academic Inputs")

marks = [
    st.sidebar.slider(f"Semester {i} Marks", 0, 100, 60)
    for i in range(1, 9)
]
soft_skills = st.sidebar.slider("Soft Skills Score", 0, 100, 70)
aptitude = st.sidebar.slider("Aptitude Score", 0, 100, 65)

input_data = np.array(marks + [soft_skills, aptitude]).reshape(1, -1)

# --- Prediction Section ---
if st.sidebar.button("Predict"):
    interview_model, grading_model = load_models()

    interview_pred = interview_model["model"].predict(input_data)[0]
    interview_label = interview_model["label_encoder"].inverse_transform(
        [interview_pred]
    )[0]

    grade_pred = grading_model.predict(input_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interview Preparedness")
        st.success(f"Prediction: {interview_label}")

    with col2:
        st.subheader("Academic Performance")
        st.info(f"Predicted Grade Category: {grade_pred}")

# --- Data Overview ---
st.divider()
st.subheader("Dataset Overview")

uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    st.bar_chart(df.select_dtypes(include="number").mean())

st.caption("Brainware University | Student performance Dashboard")
