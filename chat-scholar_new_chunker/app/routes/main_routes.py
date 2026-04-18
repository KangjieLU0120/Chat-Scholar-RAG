import os
import json
from flask import Blueprint, render_template, request, session, Response
from app.utils.document_registry import add_document, load_documents

from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
from app.utils.pdf_reader import extract_text_from_pdf
from app.utils.text_chunker import split_text_into_chunks
from app.utils.vector_store import VectorStore


main = Blueprint("main", __name__)

ai_service = AIService()
embedding_service = EmbeddingService()

# load persistent vector DB
vector_store = VectorStore.load()


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

                pdf_text = extract_text_from_pdf(save_path)
                chunks = split_text_into_chunks(pdf_text)

                embeddings = []
                valid_chunks = []

                for chunk in chunks:
                    emb = embedding_service.get_embedding(chunk)
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
                        source_name=pdf.filename
                    )

                    vector_store.save()
                    add_document(pdf.filename)


                    print(f"✅ Added document to knowledge base: {pdf.filename}")


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

    global vector_store

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

    # ---- Retrieve relevant chunks ----
    retrieved_chunks = None

    if vector_store:
        query_embedding = embedding_service.get_embedding(user_message)
        if query_embedding:
            retrieved_chunks = vector_store.search(
                query_embedding,
                top_k=3
            )

    # ---- STREAM GENERATOR ----
    def generate():

        # IMPORTANT: use local variables ONLY
        for token in ai_service.stream_response(
            chat_history,
            document_chunks=retrieved_chunks
        ):
            yield f"data:{token}\n\n"

    return Response(generate(), mimetype="text/event-stream")



@main.route("/essay-grading", methods=["GET", "POST"])
def essay_grading():

    result = None

    if request.method == "POST":
        essay_text = request.form.get("essay_text")

        if essay_text:
            result = ai_service.grade_essay(essay_text)

    return render_template(
        "essay_grading.html",
        result=result
    )

