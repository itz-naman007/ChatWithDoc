import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, session
import main

logger = logging.getLogger(__name__)
app = Flask(__name__)
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    secret_key = secrets.token_hex(32)
    logger.warning("SECRET_KEY is missing; generated a temporary key for this startup.")
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

_sessions = OrderedDict()
_sessions_lock = threading.Lock()


def _get_state():
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid4())
        session["session_id"] = session_id
    with _sessions_lock:
        state = _sessions.pop(session_id, {"index": None, "chunks": [], "chat_history": []})
        _sessions[session_id] = state
        while len(_sessions) > 50:
            _sessions.popitem(last=False)
        return state


def _save_state(state):
    with _sessions_lock:
        session_id = session["session_id"]
        _sessions.pop(session_id, None)
        _sessions[session_id] = state

@app.route("/")
def home():
    return render_template("index.html")


#  File Upload Route
@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("file")
    documents = []
    failures = []
    allowed = {".pdf", ".docx", ".txt", ".md", ".csv"}
    parse_start = time.perf_counter()
    for file in files:
        filename = file.filename or ""
        extension = os.path.splitext(filename.lower())[1]
        if extension not in allowed:
            failures.append(f"{filename}: unsupported file type")
            continue
        try:
            if extension == ".pdf":
                documents.extend(main.read_pdf_documents([file.stream], filename))
            elif extension == ".docx":
                documents.extend(main.read_docx_documents(file.stream, filename))
            elif extension == ".csv":
                documents.extend(main.read_csv_file(file.stream, filename))
            else:
                documents.extend(main.read_text_file(file.stream, filename))
        except Exception as error:
            logger.exception("Failed to parse %s", filename)
            failures.append(f"{filename}: {error}")
    parse_ms = round((time.perf_counter() - parse_start) * 1000, 2)
    if not documents:
        if failures and all("unsupported" in failure for failure in failures):
            return jsonify({"status": "Unsupported file type. Supported types: PDF, DOCX, TXT, MD, CSV."}), 400
        return jsonify({"status": "No readable text found. Scanned/image-only PDFs are not supported yet."}), 400
    chunk_start = time.perf_counter()
    chunks = main.chunk_documents(documents)
    chunk_ms = round((time.perf_counter() - chunk_start) * 1000, 2)
    index_start = time.perf_counter()
    state = _get_state()
    state.update({"index": main.build_text_index(chunks), "chunks": chunks, "chat_history": []})
    index_ms = round((time.perf_counter() - index_start) * 1000, 2)
    _save_state(state)
    logger.info("Upload timings_ms parsing=%s chunking=%s embedding_indexing=%s", parse_ms, chunk_ms, index_ms)

    return jsonify({"status": "Documents processed successfully"})


#  Chat Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question")

    if not question:
        return jsonify({"answer": "No question provided."})

    state = _get_state()
    if state["index"] is None or len(state["chunks"]) == 0:
        return jsonify({"answer": "Please upload documents first."})

    # Initialize memory
    history = state["chat_history"]
    retrieval_question = main.rewrite_question(question, history)
    mode = data.get("mode", "student")
    top_k = 8 if mode == "summarize" else int(os.getenv("TOP_K", "4"))
    retrieval_timings = {}
    results = main.search_text(retrieval_question, state["index"], state["chunks"], k=top_k, final_k=top_k, return_scores=True, timings=retrieval_timings)

    # Stop if no relevant chunks are found
    if not results:
        return jsonify({"answer": "Not found in document.", "sources": [], "timings_ms": retrieval_timings})

    context = "\n\n".join(f"[Source: {r['chunk']['source']}]\n{r['chunk']['text']}" for r in results)
    sources = [{"filename": r["chunk"]["source"], "page": r["chunk"].get("page")} for r in results]

    # Generate answer with memory
    groq_start = time.perf_counter()
    answer = main.ask_groq(
        context,
        question,
        history,
        mode
    )
    groq_ms = round((time.perf_counter() - groq_start) * 1000, 2)

    # Save memory
    history.append({"question": question, "answer": answer})
    state["chat_history"] = history[-5:]
    _save_state(state)

    timings = {**retrieval_timings, "groq_ms": groq_ms}
    logger.info("Chat timings_ms=%s", timings)
    return jsonify({"answer": answer, "sources": sources, "timings_ms": timings})


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"})
