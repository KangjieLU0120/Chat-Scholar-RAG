import os
import uuid
from flask import Blueprint, render_template, request, session, Response
from app.utils.document_registry import add_document, load_documents

from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
from app.utils.pdf_reader import extract_pages_from_pdf
from app.utils.text_chunker import split_text_into_chunks
from app.utils.vector_store import VectorStore


main = Blueprint("main", __name__)

ai_service = AIService()
embedding_service = EmbeddingService()
vector_store = VectorStore.load()



def safe_text(value):
    return value if isinstance(value, str) else ""



def _write_debug_pages(pages):
    with open("debug_pages.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n===== Page {page.get('page')} =====\n")
            blocks = page.get("blocks", [])
            if blocks:
                for idx, block in enumerate(blocks, start=1):
                    f.write(
                        f"[Block {idx} | Type: {block.get('block_type')} | "
                        f"Font: {block.get('font_size')} | BBox: {block.get('bbox')}]\n"
                    )
                    f.write(safe_text(block.get("text", "")))
                    f.write("\n\n")
            else:
                f.write(safe_text(page.get("text", "")))
                f.write("\n")



def _write_debug_chunks(chunk_items):
    with open("debug_chunks.txt", "w", encoding="utf-8") as f:
        for item in chunk_items:
            f.write(
                f"\n===== Chunk {item.get('chunk_id')} | Page {item.get('page')} | "
                f"Section {item.get('section')} | Type {item.get('chunk_type')} =====\n"
            )
            f.write(safe_text(item.get("text", "")))
            f.write("\n")



def _write_debug_retrieved(retrieved_chunks):
    with open("debug_retrieved.txt", "w", encoding="utf-8") as f:
        if not retrieved_chunks:
            f.write("No retrieved chunks.\n")
            return

        for i, chunk in enumerate(retrieved_chunks, start=1):
            f.write(f"\n===== Retrieved {i} =====\n")
            f.write(f"Source: {chunk.get('source')}\n")
            f.write(f"Section: {chunk.get('section')}\n")
            f.write(f"Page: {chunk.get('page') or chunk.get('page_start')}\n")
            f.write(f"Distance: {chunk.get('distance')}\n")
            f.write(safe_text(chunk.get("text", "")))
            f.write("\n")



def _enrich_chunk_metadata(chunk_items, filename, document_id):
    enriched = []
    for item in chunk_items:
        text = safe_text(item.get("text", "")).strip()
        if not text:
            continue
        enriched_item = dict(item)
        enriched_item["source"] = filename
        enriched_item["filename"] = filename
        enriched_item["document_id"] = document_id
        enriched_item["parser_version"] = "v2_block_parser"
        enriched_item["chunker_version"] = "v5_inline_heading_section_chunker"
        enriched.append(enriched_item)
    return enriched



def _dedup_retrieved_chunks(retrieved_chunks, max_items=3):
    if not retrieved_chunks:
        return []

    filtered = []
    seen = set()
    for chunk in retrieved_chunks:
        text = safe_text(chunk.get("text", "")).strip()
        if not text:
            continue
        signature = text[:180]
        chunk_type = chunk.get("chunk_type")
        if signature in seen:
            continue
        if chunk_type in {"title", "reference"} and len(text) < 120:
            continue
        seen.add(signature)
        filtered.append(chunk)
        if len(filtered) >= max_items:
            break

    return filtered


@main.route("/")
def home():
    return render_template("home.html")


@main.route("/clear-chat")
def clear_chat():
    session.pop("chat_history", None)
    session.pop("current_document", None)
    session.pop("current_document_id", None)
    return render_template("pdf_chat.html", chat_history=[])


@main.route("/pdf-chat", methods=["GET", "POST"])
def pdf_chat():
    global vector_store

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf = request.files["pdf_file"]

            if pdf and pdf.filename.endswith(".pdf"):
                os.makedirs("data", exist_ok=True)
                save_path = os.path.join("data", pdf.filename)
                pdf.save(save_path)

                pages = extract_pages_from_pdf(save_path)
                _write_debug_pages(pages)

                chunk_items = split_text_into_chunks(
                    pages,
                    chunk_size=500,
                    overlap=100,
                    return_metadata=True,
                )
                _write_debug_chunks(chunk_items)

                document_id = str(uuid.uuid4())
                chunk_items = _enrich_chunk_metadata(chunk_items, pdf.filename, document_id)

                embeddings = []
                valid_chunks = []
                for chunk in chunk_items:
                    emb = embedding_service.get_embedding(chunk["text"])
                    if emb:
                        embeddings.append(emb)
                        valid_chunks.append(chunk)

                if embeddings:
                    if vector_store is None:
                        dimension = len(embeddings[0])
                        vector_store = VectorStore(dimension)

                    vector_store.add_embeddings(
                        embeddings,
                        valid_chunks,
                        source_name=pdf.filename,
                        document_id=document_id,
                    )
                    vector_store.save()
                    add_document(pdf.filename)

                    session["current_document"] = pdf.filename
                    session["current_document_id"] = document_id
                    session.modified = True

                    print(f"Added document to knowledge base: {pdf.filename}")

    return render_template(
        "pdf_chat.html",
        chat_history=session.get("chat_history", []),
        documents=load_documents(),
    )


@main.route("/stream-chat", methods=["POST"])
def stream_chat():
    global vector_store

    data = request.get_json()
    user_message = data.get("message")

    chat_history = session.get("chat_history", [])
    chat_history.append({"role": "user", "content": user_message})
    session["chat_history"] = chat_history
    session.modified = True

    retrieved_chunks = None
    if vector_store:
        query_embedding = embedding_service.get_embedding(user_message)
        if query_embedding:
            current_document = session.get("current_document")
            current_document_id = session.get("current_document_id")

            retrieved_chunks = vector_store.search(
                query_embedding,
                top_k=5,
                source_name=current_document,
                document_id=current_document_id,
            )

            if not retrieved_chunks:
                retrieved_chunks = vector_store.search(query_embedding, top_k=5)

            retrieved_chunks = _dedup_retrieved_chunks(retrieved_chunks, max_items=3)

    _write_debug_retrieved(retrieved_chunks)

    def generate():
        for token in ai_service.stream_response(chat_history, document_chunks=retrieved_chunks):
            yield f"data:{token}"

    return Response(generate(), mimetype="text/event-stream")


@main.route("/essay-grading", methods=["GET", "POST"])
def essay_grading():
    result = None
    if request.method == "POST":
        essay_text = request.form.get("essay_text")
        if essay_text:
            result = ai_service.grade_essay(essay_text)

    return render_template("essay_grading.html", result=result)
