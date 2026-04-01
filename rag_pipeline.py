import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


class RAG:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None

    # Load Data
    def load_data(self, file_path=None, url=None, file_name=None):
        docs = []

        # FILE
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_name if file_name else "uploaded_file"

            docs.extend(loaded_docs)

        # URL
        if url:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = url

            docs.extend(loaded_docs)

        return docs

    # Vector Store
    def create_vectorstore(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80
        )

        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    # 🔹 QA System (Low Hallucination + MMR)
    def setup_qa(self):
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo", 
            temperature=0
        )

        def qa(query):
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )

            docs = retriever.get_relevant_documents(query)

            if not docs:
                return "I don't know based on the provided data"

            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the context
- If not found, say: "I don't know based on the provided data"
- Do NOT hallucinate

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt).content

            sources = "\n".join([
                str(doc.metadata.get("source", "unknown"))
                for doc in docs
            ])

            return f"{response}\n\n📌 Sources:\n{sources}"

        self.qa_chain = qa

    # Build Pipeline
    def build(self, file_path=None, url=None, file_name=None):
        docs = self.load_data(file_path, url, file_name)

        print("DEBUG: Loaded docs =", len(docs))

        if not docs:
            raise ValueError("No data provided")

        self.create_vectorstore(docs)
        self.setup_qa()

    # Ask
    def ask(self, query):
        if not self.qa_chain:
            return "Please build knowledge base first."

        return self.qa_chain(query)