import streamlit as st
import tempfile
import os
from rag_pipeline import RAG

st.set_page_config(page_title="InsightBot", page_icon="🤖")

if "rag" not in st.session_state:
    st.session_state.rag = RAG()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("Data Source")
    up_file = st.file_uploader("Upload PDF", type=["pdf", "txt"])
    site_url = st.text_input("Website URL")
    
    if st.button("Build Knowledge Base"):
        if up_file or site_url:
            with st.spinner("Building..."):
                f_path = None
                f_name = None
                if up_file:
                    f_name = up_file.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                        t.write(up_file.read())
                        f_path = t.name
                
                try:
                    st.session_state.rag.build(file_path=f_path, url=site_url, file_name=f_name)
                    st.success("Success!")
                except Exception as e:
                    st.error(f"Build Failed: {e}")
        else:
            st.warning("Upload something first!")

# Chat
st.title("🤖 InsightBot RAG")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We pass only the content of previous messages to avoid recursive nesting
        ans, sources = st.session_state.rag.ask(prompt, st.session_state.messages[:-1])
        
        display_text = ans
        if sources:
            display_text += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
            
        st.markdown(display_text)
        st.session_state.messages.append({"role": "assistant", "content": ans}) # Save only the text