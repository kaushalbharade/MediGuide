import streamlit as st
import pickle
import pandas as pd
import numpy as np
from thefuzz import process
import ast
import base64

# ------------------------ -------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Healthcare Intelligence Network",
    layout="wide"
)

# -------------------------------------------------
# Background (ONLY visual layer)
# -------------------------------------------------
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        textarea, input {{
            background-color: #f3f4f6 !important;
            color: #111827 !important;
        }}

        textarea::placeholder, input::placeholder {{
            color: #6b7280 !important;
            opacity: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("utils/medical_bg.jpg")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.markdown("<h2>Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph3.png", use_container_width=True)
st.sidebar.markdown(
    "The Disease Prediction & Medical Recommendation system uses AI to analyze "
    "symptoms, predict diseases, assess health risks, and suggest personalized treatments."
)

# -------------------------------------------------
# Data Loader
# -------------------------------------------------
@st.cache_resource
def load_data():
    sym_des = pd.read_csv("data/Disease-Prediction-and-Medical dataset/symptoms_df.csv")
    precautions = pd.read_csv("data/Disease-Prediction-and-Medical dataset/precautions_df.csv")
    workout = pd.read_csv("data/Disease-Prediction-and-Medical dataset/workout_df.csv")
    description = pd.read_csv("data/Disease-Prediction-and-Medical dataset/description.csv")
    medications = pd.read_csv("data/Disease-Prediction-and-Medical dataset/medications.csv")
    diets = pd.read_csv("data/Disease-Prediction-and-Medical dataset/diets.csv")
    model = pickle.load(open('models/first_feature_models/RandomForest.pkl', 'rb'))
    return sym_des, precautions, workout, description, medications, diets, model

sym_des, precautions, workout, description, medications, diets, model = load_data()
disease_names = list(description['Disease'].unique())

# -------------------------------------------------
# KEEP ALL YOUR EXISTING:
# - symptoms_list
# - diseases_list
# - symptoms_list_processed
# - information()
# - predicted_value()
# - correct_spelling()
# -------------------------------------------------

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Disease Prediction & Medical Recommendation")

st.markdown("### Disease Prediction Based on Symptoms")
st.markdown("_To get the best and most accurate results, provide as many symptoms as possible._")

user_input = st.text_area(
    "Enter symptoms (comma-separated):",
    placeholder="e.g., headache, constipation, nausea"
)

if st.button("Predict Disease"):
    if user_input:
        patient_symptoms = [s.strip().lower() for s in user_input.split(',')]
        patient_symptoms = [correct_spelling(s) for s in patient_symptoms if correct_spelling(s)]

        if patient_symptoms:
            predicted_disease = predicted_value(patient_symptoms)
            dis_des, prec, meds, diet, work = information(predicted_disease)

            st.success(f"**Predicted Disease:** {predicted_disease}")
            st.write(f"**Description:** {dis_des}")
            st.write("**Precautions:**", ", ".join(p for p in prec if p))
            st.write("**Medications:**", ", ".join(m for m in meds if m))
            st.write("**Recommended Diet:**", ", ".join(d for d in diet if d))
            st.write("**Recommended Workout:**", ", ".join(w for w in work if w))
        else:
            st.error("Invalid symptoms detected. Please check and try again.")
    else:
        st.warning("Please enter at least one symptom.")

st.markdown("---")

# -------------------------------------------------
# Search Section (ALWAYS VISIBLE)
# -------------------------------------------------
st.markdown("### Search for Disease Description")

disease_query = st.text_input(
    "Type a disease name to get recommendations:",
    placeholder="Start typing..."
)

if disease_query:
    matches = [d for d in disease_names if d.lower().startswith(disease_query.lower())]
    if matches:
        selected = matches[0]
        dis_des, prec, meds, diet, work = information(selected)

        st.subheader(f"Recommendations for {selected}")
        st.write(f"**Description:** {dis_des}")
        st.write("**Precautions:**", ", ".join(p for p in prec if p))
        st.write("**Medications:**", ", ".join(m for m in meds if m))
        st.write("**Recommended Diet:**", ", ".join(d for d in diet if d))
        st.write("**Recommended Workout:**", ", ".join(w for w in work if w))
    else:
        st.warning("No matching disease found. Try a different name.")
