import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RAG:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None

    def load_data(self, file_path=None, url=None, file_name=None):
        docs = []
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file_name if file_name else "Local Document"
            docs.extend(loaded_docs)
        if url:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = url
            docs.extend(loaded_docs)
        return docs

    def create_vectorstore(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Lowered chunk_size for embeddings to prevent OpenRouter "No data" errors
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            chunk_size=100, 
        )
        
        if not chunks:
            raise ValueError("No text chunks created. Check your data sources.")
            
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def setup_qa(self):
        model_id = "nvidia/nemotron-3-super-120b-a12b:free" 
        llm = ChatOpenAI(
            model=model_id, 
            temperature=0.2, 
            max_tokens=1000, 
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={"X-Title": "Manoj Technical RAG"}
        )

        def qa(query):
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])
            sources = list(set([str(d.metadata.get("source", "Unknown")) for d in docs]))

            prompt = f"""You are a Technical Expert specializing in Python and Machine Learning.
            TASK: Use the Context below to explain the concepts. 
            RULES:
            1. Use bullet points for readability. (No bolding in final explanation).
            2. If the answer is not in the context, say: "I'm sorry, my current database doesn't contain information about that specific topic."
            3. Do not mention "the context" specifically.

            Context: {context}
            Question: {query}
            Expert Explanation:"""
            
            response = llm.invoke(prompt).content
            return response, sources

        self.qa_chain = qa

    def build(self, file_path=None, url=None, file_name=None):
        docs = self.load_data(file_path, url, file_name)
        if not docs: raise ValueError("No valid data sources found.")
        self.create_vectorstore(docs)
        self.setup_qa()

    def ask(self, query):
        q = query.lower().strip()
        if q in ["hi", "hello", "hey"]:
            return "Hello! Please upload your data in the Setup tab so I can assist you.", []
        if q in ["exit", "bye", "quit", "goodbye"]:
            return "EXIT_SIGNAL", [] # Special signal for the UI

        if not self.vectorstore:
            return "Knowledge base is empty. Please use the 'Setup' tab.", []
        if not self.qa_chain:
            self.setup_qa()
        return self.qa_chain(query)