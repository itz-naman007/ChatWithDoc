from io import BytesIO

import app
import main


class FakeIndex:
    ntotal = 2

    def search(self, _embedding, _limit):
        return [[0.9, 0.8]], [[0, 1]]


def setup_fake_retrieval(monkeypatch):
    monkeypatch.setattr(main, "build_text_index", lambda chunks: FakeIndex())
    monkeypatch.setattr(main, "_cached_query_embedding", lambda query: (1.0, 0.0))
    monkeypatch.setattr(
        main,
        "search_text",
        lambda query, index, chunks, **kwargs: [
            {"chunk": chunk, "score": 0.9} for chunk in chunks
        ],
    )


def test_chunking_falls_back_to_character_chunks():
    chunks = main.chunk_text("abcdef", size=4, overlap=1)
    assert chunks == ["abcd", "def"]


def test_search_clamps_k_and_skips_negative_indices(monkeypatch):
    monkeypatch.setattr(main, "_cached_query_embedding", lambda query: (1.0, 0.0))
    monkeypatch.setattr(main, "MIN_SCORE", 0.0)
    results = main.search_text("question", FakeIndex(), ["first", "second"], k=10)
    assert results == ["first", "second"]


def test_memory_append_and_session_isolation(monkeypatch):
    setup_fake_retrieval(monkeypatch)
    monkeypatch.setattr(main, "read_text_file", lambda file, source: [{"text": source, "source": source, "page": None}])

    def fake_ask(context, question, history, mode):
        return f"history={len(history)} context={context}"

    monkeypatch.setattr(main, "ask_groq", fake_ask)
    app._sessions.clear()
    client_a = app.app.test_client()
    client_b = app.app.test_client()

    assert client_a.post("/upload", data={"file": (BytesIO(b"a"), "a.txt")}).status_code == 200
    assert client_b.post("/upload", data={"file": (BytesIO(b"b"), "b.txt")}).status_code == 200
    first = client_a.post("/chat", json={"question": "one", "mode": "student"}).get_json()
    second = client_a.post("/chat", json={"question": "two", "mode": "student"}).get_json()
    other = client_b.post("/chat", json={"question": "one", "mode": "student"}).get_json()

    assert "history=0" in first["answer"]
    assert "history=1" in second["answer"]
    assert "a.txt" in second["answer"]
    assert "history=0" in other["answer"]
    assert "b.txt" in other["answer"]


def test_empty_upload_returns_clear_400(monkeypatch):
    monkeypatch.setattr(main, "read_pdf_documents", lambda files, source: [])
    response = app.app.test_client().post(
        "/upload", data={"file": (BytesIO(b"not a readable pdf"), "scan.pdf")}
    )
    assert response.status_code == 400
    assert response.get_json()["status"] == "No readable text found. Scanned/image-only PDFs are not supported yet."
