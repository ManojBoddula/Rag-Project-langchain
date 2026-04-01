import streamlit as st
import tempfile
import os

from rag_pipeline import RAG

st.set_page_config(page_title="RAG", layout="wide")

st.title("🧠 Universal RAG Assistant (Low Hallucination)")

if "rag" not in st.session_state:
    st.session_state.rag = RAG()

uploaded_file = st.file_uploader("📄 Upload PDF or TXT")
url = st.text_input("🌐 Or Enter Website URL")

if st.button("🚀 Build Knowledge Base"):

    if not uploaded_file and not url:
        st.warning("⚠️ Please upload a file or enter a URL")

    else:
        file_path = None

        if uploaded_file:
            file_extension = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

        try:
            st.session_state.rag.build(file_path=file_path, url=url)
            st.success("✅ Knowledge base built successfully!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

query = st.text_input("💬 Ask a question")

if query:
    with st.spinner("Thinking..."):
        answer = st.session_state.rag.ask(query)
        st.write("🤖", answer)