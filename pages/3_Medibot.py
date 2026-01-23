# ============================
# OFFLINE SAFETY (MUST BE FIRST)
# ============================
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ============================
# STANDARD IMPORTS
# ============================
import asyncio
import nest_asyncio
import streamlit as st
import base64
from dotenv import load_dotenv, find_dotenv

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Medibot | Medical Knowledge Assistant",
    layout="wide"
)

# ============================
# THEME TOGGLE
# ============================
theme = st.toggle("Dark Mode")

overlay = "rgba(15,23,42,0.85)" if theme else "rgba(255,255,255,0.78)"
text_color = "#e5e7eb" if theme else "#111827"
card_bg = "#020617" if theme else "#ffffff"

# ============================
# BACKGROUND
# ============================
def set_bg(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

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
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            margin-bottom: 18px;
            color: {text_color};
        }}

        h1, h2, h3 {{
            color: #1f77ff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("utils/medical_bg.jpg")

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("## Medibot")
    st.image("utils/ph3.png", use_container_width=True)
    st.markdown(
        """
        Medibot is a **retrieval-based medical assistant**.

        - Uses verified medical textbooks  
        - Answers strictly from retrieved context  
        - No hallucinations  
        """
    )

# ============================
# LANGCHAIN IMPORTS
# ============================
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# ============================
# ENV SETUP
# ============================
load_dotenv(find_dotenv())
nest_asyncio.apply()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please set it in your environment.")
    st.stop()

# ============================
# VECTORSTORE (OFFLINE SAFE)
# ============================
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True}
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error("Failed to load FAISS vectorstore.")
    st.error(str(e))
    st.stop()

# ============================
# PROMPT
# ============================
def get_prompt_template():
    return PromptTemplate(
        template="""
Answer the medical question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

# ============================
# LLM (GROQ)
# ============================
def load_llm():
    return ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

# ============================
# MAIN APP
# ============================
def main():
    st.title("Medibot – AI Health Assistant")

    st.markdown(
        "**Ask medical questions and get answers grounded in verified textbooks.**  \n"
        "_No guessing. No hallucinations._"
    )

    # ---------- SESSION STATE ----------
    if "history" not in st.session_state:
        st.session_state.history = []

    # ---------- CHAT HISTORY ----------
    for chat in st.session_state.history:
        st.markdown(
            f"<div class='card'><b>You</b><br>{chat['question']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='card'><b>Medibot</b><br>{chat['answer']}</div>",
            unsafe_allow_html=True
        )

        with st.expander("Source Medical Text Used"):
            for i, src in enumerate(chat["sources"], 1):
                st.markdown(
                    f"<div class='card'><b>Source {i}</b><br>{src}</div>",
                    unsafe_allow_html=True
                )

    # ---------- USER INPUT ----------
    user_query = st.chat_input("Type your medical query...")

    if user_query:
        with st.spinner("Searching medical textbooks..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": get_prompt_template()}
                )

                response = qa_chain.invoke({"query": user_query})

                answer = response.get("result", "No response generated.")
                source_docs = response.get("source_documents", [])

                sources_text = [doc.page_content for doc in source_docs]

                st.session_state.history.append({
                    "question": user_query,
                    "answer": answer,
                    "sources": sources_text
                })

                st.rerun()

            except Exception as e:
                st.error("Error while generating response")
                st.error(str(e))


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
