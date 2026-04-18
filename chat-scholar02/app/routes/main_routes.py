import os
import json
from flask import Blueprint, render_template, request, session, Response
from app.utils.document_registry import add_document, load_documents

from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
# from app.utils.pdf_reader import extract_text_from_pdf
from app.utils.pdf_reader import extract_pages_from_pdf
from app.utils.text_chunker import split_text_into_chunks
from app.utils.vector_store import VectorStore
from app.utils.bm25_store import BM25Store
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService

main = Blueprint("main", __name__)

embedding_service = EmbeddingService()
reranker_service = RerankerService()

# load persistent vector DB
vector_store = VectorStore.load()

# bm25_store = BM25Store()

# load persistent BM25 DB
bm25_store = BM25Store.load()

# ---------------------------------------------------
# Helper to rebuild BM25 on app startup if VectorStore exists
# ---------------------------------------------------
# if vector_store and vector_store.metadata:
#     print("🔄 Rebuilding BM25 index from persistent Vector DB metadata...")
#     # Extract text chunks and their sources from the loaded FAISS metadata
#     existing_chunks = [item for item in vector_store.metadata]
#     existing_sources = [item["source"] for item in vector_store.metadata]
#     # We feed them one by one to keep the source alignment
#     for chunk, source in zip(existing_chunks, existing_sources):
#         bm25_store.add_documents([chunk], source)
#     print("✅ BM25 index rebuilt successfully.")

def _process_pdf_file(pdf_path, filename):
    """Process a single PDF file and add embeddings to vector store."""
    global vector_store, bm25_store
    
    try:
        print(f"📄 Processing: {filename}")
        # pdf_text = extract_text_from_pdf(pdf_path)
        # chunks = split_text_into_chunks(pdf_text)

        pages = extract_pages_from_pdf(pdf_path)
        chunks = split_text_into_chunks(
            pages,
            chunk_size=500,
            overlap=100,
            return_metadata=True,
        )
        
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            emb = embedding_service.get_embedding(chunk["text"])
            if emb is not None:
                embeddings.append(emb)
                valid_chunks.append(chunk)
        
        if embeddings:
            if vector_store is None:
                dimension = len(embeddings[0])
                vector_store = VectorStore(dimension)
            
            vector_store.add_embeddings(
                embeddings,
                valid_chunks,
                source_name=filename,
            )
            
            if bm25_store is None:
                bm25_store = BM25Store()

            bm25_store.add_documents(
                valid_chunks,
                source_name=filename
            )
            
            vector_store.save()
            bm25_store.save()
            add_document(filename)
            print(f"✅ Processed {filename}: {len(embeddings)} embeddings added")
            return True
        else:
            print(f"⚠️ No embeddings generated for {filename}")
            return False
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False


def _auto_load_pdfs_from_data_dir():
    """Auto-load all PDF files from data/ directory on startup."""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"ℹ️ {data_dir}/ directory does not exist")
        return
    
    existing_docs = set(load_documents())
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"ℹ️ No PDF files found in {data_dir}/")
        return
    
    print(f"\n🔍 Found {len(pdf_files)} PDF(s) in {data_dir}/")
    
    for filename in pdf_files:
        if filename in existing_docs:
            print(f"⏭️ Skipping {filename} (already processed)")
            continue
        
        pdf_path = os.path.join(data_dir, filename)
        _process_pdf_file(pdf_path, filename)
    
    print("✨ Auto-load complete\n")


# Auto-load PDF files from data directory on startup
_auto_load_pdfs_from_data_dir()

retrieval_service = RetrievalService(
    vector_store=vector_store,
    bm25_store=bm25_store,
    embedding_service=embedding_service,
    reranker_service=reranker_service
)

ai_service = AIService(retrieval_service=retrieval_service)

@main.route("/")
def home():
    return render_template("home.html")


@main.route("/clear-chat")
def clear_chat():
    session.pop("chat_history", None)
    return render_template("pdf_chat.html", chat_history=[])


# ---------------------------------------------------
# PDF CHAT PAGE
# ---------------------------------------------------
@main.route("/pdf-chat", methods=["GET", "POST"])
def pdf_chat():

    global vector_store
    global bm25_store

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":

        # ---------- PDF Upload ----------
        if "pdf_file" in request.files:
            pdf = request.files["pdf_file"]

            if pdf and pdf.filename.endswith(".pdf"):

                os.makedirs("data", exist_ok=True)
                save_path = os.path.join("data", pdf.filename)
                pdf.save(save_path)

                # pdf_text = extract_text_from_pdf(save_path)
                pages = extract_pages_from_pdf(save_path)
                # chunks = split_text_into_chunks(pdf_text)
                chunks = split_text_into_chunks(
                    pages,
                    chunk_size=500,
                    overlap=100,
                    return_metadata=True,
                )

                embeddings = []
                valid_chunks = []

                for chunk in chunks:
                    emb = embedding_service.get_embedding(chunk["text"])
                    if emb is not None:
                        embeddings.append(emb)
                        valid_chunks.append(chunk)

                if embeddings:

                    if vector_store is None:
                        dimension = len(embeddings[0])
                        vector_store = VectorStore(dimension)


                    vector_store.add_embeddings(
                        embeddings,
                        valid_chunks,
                        source_name=pdf.filename
                    )

                    if bm25_store is None:
                        bm25_store = BM25Store()  

                    bm25_store.add_documents(
                        valid_chunks, 
                        source_name=pdf.filename
                    )

                    vector_store.save()
                    bm25_store.save()
                    add_document(pdf.filename)


                    print(f"✅ Added document to Hybrid knowledge base: {pdf.filename}")


    return render_template(
        "pdf_chat.html",
        chat_history=session.get("chat_history", []),
        documents=load_documents()
    )

# ---------------------------------------------------
# STREAM CHAT (LIVE AI TYPING)
# ---------------------------------------------------
@main.route("/stream-chat", methods=["POST"])
def stream_chat():

    data = request.get_json()
    user_message = data.get("message")

    # ---- Prepare chat history BEFORE streaming ----
    chat_history = session.get("chat_history", [])

    chat_history.append({
        "role": "user",
        "content": user_message
    })

    # Save immediately (inside request context)
    session["chat_history"] = chat_history
    session.modified = True

    # ---- STREAM GENERATOR ----
    return Response(
        ai_service.agentic_stream_response(chat_history),
        # ai_service.multi_query_stream_response(chat_history),
        mimetype='text/event-stream'
    )


