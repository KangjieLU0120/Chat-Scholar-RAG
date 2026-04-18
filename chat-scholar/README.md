# 📘 Chat Scholar — AI Academic Assistant

Chat Scholar is an AI-powered academic assistant that enables students to interact with documents and receive intelligent academic support.  
The system combines Retrieval-Augmented Generation (RAG), local Large Language Models, and semantic vector search to provide accurate, grounded answers from uploaded PDFs along with automated essay evaluation.

This project demonstrates a production-style AI architecture using local models, vector databases, and real-time streaming responses.

---

## 🚀 Features

### 📄 PDF Chat (RAG-Based Question Answering)
- Upload academic PDFs and ask questions naturally.
- Semantic search retrieves relevant document sections.
- AI answers strictly using document context.
- Source citation included in responses.
- Supports multiple PDFs in a shared knowledge base.

### 🧠 Semantic Retrieval
- Text chunking and embedding generation.
- Vector similarity search using FAISS.
- Meaning-based retrieval instead of keyword matching.

### ⚡ Streaming AI Responses
- ChatGPT-style live typing responses.
- Real-time token streaming from local LLM.

### 📝 Essay Grading System
- Automatic academic essay evaluation.
- Provides:
  - Overall score
  - Strengths & weaknesses
  - Grammar feedback
  - Improvement suggestions

### 💾 Persistent Knowledge Base
- Vector database saved locally.
- Knowledge survives server restarts.
- Incremental indexing when new PDFs are added.

---

## 🏗️ System Architecture

User Uploads PDF  
        ↓  
Text Extraction (PyPDF2)  
        ↓  
Text Chunking  
        ↓  
Embeddings (Ollama - nomic-embed-text)  
        ↓  
FAISS Vector Database  
        ↓  
Semantic Retrieval  
        ↓  
TinyLlama LLM (Ollama)  
        ↓  
Grounded Answer + Source Citation  

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|------------|
| Backend | Flask (Python) |
| LLM Runtime | Ollama |
| Language Model | TinyLlama |
| Embeddings | nomic-embed-text |
| Vector Database | FAISS |
| PDF Processing | PyPDF2 |
| Frontend | HTML, CSS, JavaScript |
| Streaming | Fetch Streaming API |

---

## 📂 Project Structure

Chat-Scholar/

├── app/  
│   ├── routes/  
│   │   └── main_routes.py  
│   ├── services/  
│   │   ├── ai_service.py  
│   │   └── embedding_service.py  
│   ├── utils/  
│   │   ├── pdf_reader.py  
│   │   ├── text_chunker.py  
│   │   ├── vector_store.py  
│   │   └── document_registry.py  

├── templates/  
│   ├── pdf_chat.html  
│   └── essay_grading.html  

├── vector_db/  
├── data/  
├── app.py  
└── README.md  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
git clone https://github.com/your-username/chat-scholar.git  
cd chat-scholar

### 2️⃣ Create Virtual Environment
python -m venv venv  
venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Install Ollama
Download from:  
https://ollama.com

### 5️⃣ Pull Required Models
ollama pull tinyllama  
ollama pull nomic-embed-text

### 6️⃣ Start Ollama
ollama serve

### 7️⃣ Run Application
python app.py

Open browser:

http://127.0.0.1:5000

---

## 🧪 How to Use

### PDF Chat
1. Navigate to PDF Chat page.
2. Upload a PDF document.
3. Ask questions related to the document.
4. Receive grounded answers with citations.

### Essay Grading
1. Open Essay Grading page.
2. Paste student essay.
3. Click Grade Essay.
4. View structured evaluation feedback.

---

## 🎯 Key AI Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings & Semantic Search
- Local LLM Deployment
- Context Grounding
- Streaming Token Responses
- Persistent Vector Databases

---

## 💡 Why This Project Matters

Most AI chat applications rely on cloud APIs.  
Chat Scholar demonstrates how to build a fully local, privacy-friendly AI assistant using open-source models and modern AI engineering practices.

---

## 📈 Future Improvements

- Multi-user authentication
- Document highlighting for citations
- PDF page-level referencing
- Async indexing for large documents
- Advanced evaluation rubrics for essays

---

## 👨‍💻 Author

Charan Kamalakara

AI & Machine Learning Developer  
Focused on building intelligent systems using LLMs, Retrieval Systems, and Applied AI Engineering.

---

## ⭐ Acknowledgements

- Ollama
- FAISS (Meta AI)
- Open-source LLM community

---

## 📜 License

This project is for educational and research purposes.
