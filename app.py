import streamlit as st
import tempfile
import os
from rag_pipeline import RAG

st.set_page_config(page_title="RAG low Hallucination", layout="wide")

# --- Session State Management ---
if "rag" not in st.session_state:
    st.session_state.rag = RAG()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = True

st.title("RAG low Hallucination")
st.caption("Expert Chatbot ")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("App Controls")
    if st.button("Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_active = True
        st.rerun()
    st.divider()
    db_status = "Online" if st.session_state.rag.vectorstore else "Offline"
    st.write(f"System Knowledge: **{db_status}**")

# --- Main UI Tabs ---
tab_setup, tab_chat = st.tabs(["Knowledge Setup", "Expert Chat"])

with tab_setup:
    st.subheader("Ingest Knowledge Sources")
    c1, c2 = st.columns(2)
    with c1:
        f = st.file_uploader("Upload Python PDF/Doc", type=["pdf", "txt"], key="file_up")
    with c2:
        u = st.text_input("Enter Website URL", key="url_in")
    
    if st.button("Synchronize Database", use_container_width=True):
        if f or u:
            with st.status("Building Vector Database...", expanded=True) as s:
                path, name = None, None
                if f:
                    name = f.name
                    suffix = os.path.splitext(name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        path = tmp.name
                try:
                    st.session_state.rag.build(path, u, name)
                    s.update(label="Database Ready!", state="complete")
                except Exception as e:
                    st.error(f"Sync Failed: {e}")
                finally:
                    if path and os.path.exists(path): os.remove(path)
        else:
            st.warning("Please provide at least one source.")

# --- Chat Interface (Tab 2) ---
with tab_chat:
    # This container holds all the messages
    chat_container = st.container()
    
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("src"):
                    with st.expander("Sources used"):
                        for s in m["src"]: st.caption(s)

    # The Input Box is placed AFTER the history container 
    # to ensure it stays at the bottom of the "Expert Chat" tab.
    if st.session_state.chat_active:
        if query := st.chat_input("Ask about your data...", key="main_chat_input"):
            
            # 1. Immediately show user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # 2. Process Assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        ans, src = st.session_state.rag.ask(query)
                        
                        if ans == "EXIT_SIGNAL":
                            final_msg = "Goodbye! Refresh the page or click 'Reset' to start a new session."
                            st.info(final_msg)
                            st.session_state.messages.append({"role": "assistant", "content": final_msg})
                            st.session_state.chat_active = False
                            st.rerun()
                        else:
                            st.markdown(ans)
                            if src:
                                with st.expander("Sources used"):
                                    for s in src: st.caption(s)
                            
                            st.session_state.messages.append({
                                "role": "assistant", "content": ans, "src": src
                            })
    else:
        st.warning("Chat session ended. Please reset using the sidebar to chat again.")