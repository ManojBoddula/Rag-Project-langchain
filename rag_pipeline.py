import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configuration
os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

class RAG:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None

    def load_data(self, file_path=None, url=None, file_name=None):
        docs = []
        try:
            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = file_name or "uploaded_file"
                docs.extend(loaded_docs)
            if url:
                loader = WebBaseLoader(url)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = url
                docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error loading data: {e}")
        return docs

    def create_vectorstore(self, docs):
        # Smaller chunks often work better for basic GPT-3.5
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        # Ensure your OpenRouter key supports OpenAI Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def setup_qa(self):
        # Temperature 0 is CRITICAL for factual consistency
        llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free", 
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://localhost:8501",
                "X-Title": "Strict RAG"
            }
        )

        def qa(query, chat_history_str):
            # 1. Retrieve documents with a similarity threshold (optional but helpful)
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.5} # Only take good matches
            )
            
            try:
                docs = retriever.invoke(query)
            except:
                # Fallback if thresholding isn't supported by the vectorstore version
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)

            # 2. If NO context is found, stop immediately
            if not docs or len(docs) == 0:
                return "I'm sorry, but that information is not available in the documents provided.", []

            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 3. Use a "Strict Guardrail" Prompt
            prompt = f"""You are a strict technical assistant. Use ONLY the provided context to answer.

RULES:
- If the answer is not contained within the Context below, say: "I cannot find this in the provided data."
- Do NOT use outside knowledge.
- Do NOT explain that you are an AI.
- Use the Chat History only to understand what 'it' or 'they' refers to.

Chat History:
{chat_history_str}

Context from Documents:
{context}

Question: {query}

Answer:"""
            
            try:
                response = llm.invoke(prompt).content
                sources = list(set([str(doc.metadata.get("source", "unknown")) for doc in docs]))
                return response, sources
            except Exception as e:
                return f"Connection Error: {str(e)}", []

        self.qa_chain = qa

        def qa(query, chat_history_str):
            retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
            docs = retriever.invoke(query)

            context = "\n".join([doc.page_content for doc in docs]) if docs else "No context found."
            
            prompt = f"""You are a helpful AI assistant. Use the context and history.
            
History:
{chat_history_str}
            
Context:
{context}
            
Question: {query}
Answer:"""
            
            try:
                response = llm.invoke(prompt).content
                sources = list(set([str(doc.metadata.get("source", "unknown")) for doc in docs]))
                return response, sources
            except Exception as e:
                return f"API Error: {str(e)}", []

        self.qa_chain = qa

    def build(self, file_path=None, url=None, file_name=None):
        docs = self.load_data(file_path, url, file_name)
        if not docs:
            raise ValueError("Document loading failed. No text extracted.")
        self.create_vectorstore(docs)
        self.setup_qa()

    def ask(self, query, history):
        if not self.qa_chain:
            return "Knowledge base not initialized. Please upload a file and click 'Build'.", []
        
        # Format the history as a string
        history_str = ""
        for m in history[-3:]: # Only last 3 exchanges to save tokens
            history_str += f"{m['role'].capitalize()}: {m['content']}\n"
            
        return self.qa_chain(query, history_str)