# 🤖 RAG-Low-Hallucination: Technical Assistant

A Retrieval-Augmented Generation (RAG) chatbot designed to answer technical questions about Python and Machine Learning using specific uploaded documents and web sources.

## 🚀 Features
- **Zero-Cost Architecture**: Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and OpenRouter's free LLM tier.
- **Low Hallucination**: Strict system prompting ensures the bot only answers based on provided context.
- **Multi-Source Ingestion**: Supports PDF uploads and real-time Web Scraping.
- **Streamlit UI**: Clean, dual-tab interface for setup and chat.

## 🛠️ Tech Stack
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace (Local)
- **LLM**: NVIDIA Nemotron-3 (via OpenRouter)
- **Frontend**: Streamlit

## ⚙️ Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your `OPENROUTER_API_KEY` to `.streamlit/secrets.toml`.
4. Run the app: `streamlit run app.py`.