import streamlit as st
import base64

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Medical Learning and Recommendation System",
    layout="wide"
)

# -------------------------------------------------
# Theme Toggle
# -------------------------------------------------
theme = st.toggle("Dark Mode")

overlay = "rgba(15,23,42,0.85)" if theme else "rgba(255,255,255,0.78)"
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

        .hero-title {{
            font-size: 46px;
            font-weight: 600;
            text-align: center;
            color: #1f77ff;
            margin-bottom: 10px;
        }}

        .section-card {{
            background: {card_bg};
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            transition: all 0.3s ease;
            height: 100%;
            color: {text_color};
        }}

        .section-card:hover {{
            transform: translateY(-6px) scale(1.01);
            box-shadow: 0 22px 40px rgba(0,0,0,0.25);
        }}

        .section-card h3 {{
            color: #1f77ff;
            margin-bottom: 10px;
        }}

        .footer {{
            text-align: center;
            font-size: 14px;
            color: #9ca3af;
            margin-top: 40px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# APPLY BACKGROUND
set_bg_from_local("utils/medical_bg.jpg")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Navigation")
    st.write("Explore healthcare intelligence modules.")
    st.image("utils/ph6.jpeg", use_container_width=True)

# -------------------------------------------------
# Hero Section
# -------------------------------------------------
st.markdown(
    "<div class='hero-title'>AI-Powered Medical Learning and Recommendation System</div>",
    unsafe_allow_html=True
)

col_img1, col_img2, col_img3 = st.columns([1, 6, 1])
with col_img2:
    st.image("utils/ph1.jpeg", use_container_width=True)

st.markdown("---")

# -------------------------------------------------
# Core Features
# -------------------------------------------------
st.markdown("## Core Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class='section-card'>
            <h3>Disease Prediction and Information</h3>
            <p>
            Predicts possible diseases from user-entered symptoms and provides structured medical details such as overview, symptoms, causes, and precautions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class='section-card'>
            <h3>Drug Recommendation System</h3>
            <p>
            Suggests similar or alternative medicines using NLP and similarity-based matching on curated pharmaceutical datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class='section-card' style='margin-top:20px;'>
        <h3>Medibot – Medical Knowledge Assistant</h3>
        <p>
        A RAG-based chatbot that answers medical questions strictly from verified textbooks and PDFs. Works fully offline and avoids hallucinated responses.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# Technologies Used
# -------------------------------------------------
st.markdown("## Technologies Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class='section-card'>
            <h3>Machine Learning</h3>
            <p>
            Classical ML models for symptom-based disease prediction, feature encoding, and model evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class='section-card'>
            <h3>Natural Language Processing</h3>
            <p>
            Sentence embeddings and cosine similarity for semantic understanding and text comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class='section-card'>
            <h3>RAG and Vector Search</h3>
            <p>
            FAISS vector indexing combined with transformer models for grounded medical question answering.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------------------------------
# Purpose Section
# -------------------------------------------------
st.markdown(
    """
    <div class='section-card'>
        <h3 style="text-align:center;">Purpose of This System</h3>
        <p>
        This system is designed as a learning assistant for medical students.
It helps users understand diseases, explore medicines, and ask medical questions using verified textbook knowledge, with an emphasis on offline use and responsible AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div class='footer'>
        !!This application is intended for educational and academic use only.
        It does not replace professional medical consultation.
    </div>
    """,
    unsafe_allow_html=True
)
