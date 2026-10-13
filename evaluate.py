import json
import numpy as np
import main
from docx import Document
from PyPDF2 import PdfReader

DOCUMENT_PATH = "document.pdf"
QA_PATH = "evaluation_qa.json"
K_VALUES = [1, 3, 5]

# Load the same document used to create the QA labels
if DOCUMENT_PATH.lower().endswith(".pdf"):
    with open(DOCUMENT_PATH, "rb") as f:
        text = main.read_pdfs([f])
elif DOCUMENT_PATH.lower().endswith(".docx"):
    with open(DOCUMENT_PATH, "rb") as f:
        text = main.read_docx(f)
else:
    raise ValueError("Use a PDF or DOCX document.")

chunks = main.chunk_text(text)
index = main.build_text_index(chunks)

with open(QA_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

def evaluate_retrieval(qa_data, chunks, index, k):
    hits, recalls, reciprocal_ranks = [], [], []

    # Map chunk text to its index
    chunk_to_id = {chunk: i for i, chunk in enumerate(chunks)}

    for item in qa_data:
        relevant = set(item["relevant_chunk_ids"])

        # Recall/MRR are evaluated on answerable questions
        if not item["answerable"] or not relevant:
            continue

        retrieved = main.search_text(
            item["question"],
            index,
            chunks,
            k=10,
            final_k=k
        )

        retrieved_ids = [
            chunk_to_id[c] for c in retrieved
            if c in chunk_to_id
        ]

        hits.append(
            int(bool(relevant.intersection(retrieved_ids)))
        )

        recalls.append(
            len(relevant.intersection(retrieved_ids))
            / len(relevant)
        )

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    if not hits:
        return None

    return {
        "Hit Rate": round(float(np.mean(hits)), 3),
        f"Recall@{k}": round(float(np.mean(recalls)), 3),
        "MRR": round(float(np.mean(reciprocal_ranks)), 3),
        "Questions evaluated": len(hits)
    }

for k in K_VALUES:
    print(f"\nResults at k={k}")
    print(evaluate_retrieval(qa_data, chunks, index, k))
