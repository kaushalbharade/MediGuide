import streamlit as st
import pandas as pd
import pickle
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Drug Recommendation System",
    layout="wide"
)

# -------------------------------------------------
# Theme Toggle
# -------------------------------------------------
theme = st.toggle("Dark Mode")

overlay = "rgba(15,23,42,0.85)" if theme else "rgba(255,255,255,0.85)"
text_color = "#e5e7eb" if theme else "#111827"
card_bg = "#020617" if theme else "#ffffff"

# -------------------------------------------------
# Background Loader
# -------------------------------------------------
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient({overlay}, {overlay}),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: {text_color};
        }}

        .card {{
            background: {card_bg};
            padding: 22px;
            border-radius: 16px;
            margin-bottom: 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        }}

        h1, h2, h3 {{
            color: #1f77ff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("utils/medical_bg.jpg")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Drug Recommendation")
    st.image("utils/ph4.png", use_container_width=True)
    st.markdown(
        """
        This system recommends **alternative medicines**
        using **TF-IDF + cosine similarity**.
        """
    )

# -------------------------------------------------
# Load Artifacts (CORRECT WAY)
# -------------------------------------------------
@st.cache_resource
def load_models():
    medicine = pickle.load(
        open("models/second_feature_models/medicine_df.pkl", "rb")
    )
    vectors = pickle.load(
        open("models/second_feature_models/tfidf_vectors.pkl", "rb")
    )
    return medicine, vectors

@st.cache_data
def load_description_data():
    return pd.read_csv("data/Drug reccomendation/medicine.csv")

medicine, vectors = load_models()
description_data = load_description_data()

# -------------------------------------------------
# Recommendation Logic (ON-DEMAND)
# -------------------------------------------------
def recommend(drug_name, top_n=5):
    if drug_name not in medicine["Drug_Name"].values:
        return []

    idx = medicine.index[medicine["Drug_Name"] == drug_name][0]
    scores = cosine_similarity(vectors[idx], vectors).flatten()
    top_indices = np.argsort(scores)[::-1][1:top_n + 1]

    return medicine.iloc[top_indices]["Drug_Name"].tolist()

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<h1>Drug Recommendation System</h1>", unsafe_allow_html=True)

# -------------------------------------------------
# Search Section
# -------------------------------------------------
st.markdown("<div class='card'><h3>Find Similar Drugs</h3></div>",
            unsafe_allow_html=True)

selected_medicine = st.selectbox(
    "Select a medicine",
    sorted(medicine["Drug_Name"].values)
)

recommend_btn = st.button("Recommend")

# -------------------------------------------------
# Description Section
# -------------------------------------------------
desc = description_data.loc[
    description_data["Drug_Name"] == selected_medicine, "Description"
]

if not desc.empty:
    st.markdown(
        f"""
        <div class='card'>
            <h3>Description</h3>
            {desc.values[0]}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Results
# -------------------------------------------------
if recommend_btn:
    results = recommend(selected_medicine)

    if results:
        st.markdown(
            "<div class='card'><h3>Recommended Alternatives</h3></div>",
            unsafe_allow_html=True
        )

        for drug in results:
            buy_link = f"https://pharmeasy.in/search/all?name={drug}"

            st.markdown(
                f"""
                <div class='card'>
                    <b>{drug}</b><br><br>
                    <a href="{buy_link}" target="_blank"
                       style="
                       background:#28a745;
                       color:white;
                       padding:8px 14px;
                       border-radius:8px;
                       text-decoration:none;">
                       Buy Now
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error("No recommendations found.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div style="margin-top:40px; font-size:14px;">
        This application is for educational purposes only
        and does not replace professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
