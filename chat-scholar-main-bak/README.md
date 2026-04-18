# 📘 Chat-Scholar-RAG: A Lightweight RAG System on Personal Laptop

**Chat-Scholar-RAG** is a high-privacy, low-cost academic assistant designed to run entirely on personal hardware. By integrating **Layout-aware Parsing**, **Hybrid Retrieval**, and an **Agentic Self-Correction Loop**, the system overcomes the common limitations of traditional RAG systems when handling complex academic PDFs, such as structural loss, low retrieval precision, and model hallucinations.

---

## 👥 Member List & Responsibilities

| Name | Student ID | Core Responsibilities (Sections Covered) |
| :--- | :--- | :--- |
| **CHENG Yiyang** | 25048768G | **Ingestion Module**: Layout-aware document parsing pipeline, structure-based semantic chunking, and metadata indexing. |
| **WEI Zixian** | 25108555G | **Retrieval & Generation**: Hybrid retrieval pipeline (BM25 + FAISS), Agentic self-correction loop, and final report consolidation. |
| **LU Kangjie** | 25118211G | **Evaluation & Data**: Academic testing dataset pre-processing, quantitative evaluation framework (RAGAS), and performance analysis. |

---

## 🚀 Key Technical Innovations

### 1. Layout-Aware Ingestion
Unlike standard text splitters, our system recognizes the structural hierarchy of academic papers:
- **Hierarchical Extraction**: Automatically identifies headers, sub-headers, paragraphs, and lists.
- **Semantic Chunking**: Segments text based on logical structure rather than arbitrary character counts.
- **Context Preservation**: Retains section-level metadata to ensure high-quality grounding and accurate citations.

### 2. Hybrid Retrieval Pipeline
Combines the strengths of traditional keyword matching and modern semantic search:
- **BM25 + FAISS**: Uses BM25 for precise term matching and FAISS for conceptual similarity.
- **Reciprocal Rank Fusion (RRF)**: Merges results from both streams to provide the most relevant context to the LLM.

### 3. Agentic Self-Correction Loop
Inspired by the Self-RAG framework, the system includes a reflection mechanism:
- **Relevance Check**: Evaluates if the retrieved context is sufficient to answer the query.
- **Hallucination Detection**: Audits the generated response against the source text to prevent logical errors.

### 4. Quantitative Evaluation (RAGAS)
The system was rigorously tested using the **RAGAS** framework across multiple dimensions:
- **Faithfulness**: Ensuring answers are derived solely from the context.
- **Answer Relevance**: Measuring how well the response addresses the user's prompt.
- **Context Precision**: Evaluating the quality of the retrieval engine.

---

## 🏗️ System Architecture

```text
User Query
    ↓
Hybrid Retrieval ──→ [BM25 Keyword Search] + [FAISS Vector Search]
    ↓
RRF Re-ranking (Merging results)
    ↓
Agentic Reasoning ←── [Ollama: Llama3 / TinyLlama]
    ↓
Self-Reflection & Correction
    ↓
Final Grounded Answer + Source Citations
