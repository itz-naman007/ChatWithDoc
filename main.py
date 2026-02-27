import os
import numpy as np
import fitz
import faiss
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor,AutoModel

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load models ONCE
text_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# image_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1.5")

# ---------------- TEXT ----------------
def read_pdfs(files):
    text = ""
    for pdf in files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def read_docx(file):
    text = ""
    document = Document(file)
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# ---------------- IMAGE ----------------
def extract_images_from_pdf(file):
    images = []
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            images.append(image)
    return images


# ---------------- FAISS ----------------
def build_text_index(chunks):

    if not chunks:
        raise ValueError("No text chunks found. Check PDF/DOCX reading.")

    embeddings = text_model.encode(chunks, normalize_embeddings=True)

    embeddings = np.array(embeddings)

    # 🔥 Ensure 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    return index



def build_image_index(images):
    embeddings = image_model.encode(images, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# ---------------- RETRIEVAL ----------------
def search_text(query, index, chunks, k=4):
    q_emb = text_model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

def search_images(query, index, images, k=2):
    q_emb = image_model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_emb, k)
    return [images[i] for i in indices[0]]

# ---------------- Model_Prompts ----------------

MODE_PROMPTS = {

    "student": """
- Explain in simple language.
- Avoid complex terminology.
- Use bullet points.
- Keep answer short and clear.
- Use examples if possible.
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in English, answer in English.
- Do not add speaker labels, roles, or commentary (e.g., Bot, Assistant).
- Maintain a neutral, factual, and professional tone.
""",

    "medical": """
- Use clinical tone.
- Focus on findings, interpretation, conclusion.
- Use medical terminology.
- Be concise and precise.
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in English, answer in English.
-Medical Report Format:
1. Patient Information
   - Name:
    - Age:
    - Gender:
2. Clinical Findings
   - Symptoms:
    - Test Results:
3. Interpretation
4. Conclusion
- Recommendations
- Do not add speaker labels, roles, or commentary (e.g., Bot, Assistant).
- Maintain a neutral, factual, and professional tone.
- Make sure to follow the exact format for medical reports.
- Do not include any information that is not explicitly stated in the provided context.
- Do not give examples or explanations unless they are directly supported by the context.
""",

    "legal": """
- Use formal legal language.
- Focus on clauses, obligations, liabilities.
- Be precise and structured.
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in English, answer in English.
- Legal Analysis Format:
1. Issue Identification
2. Relevant Clauses
3. Obligations and Liabilities
4. Conclusion
- Do not add speaker labels, roles, or commentary (e.g., Bot, Assistant).
- Maintain a neutral, factual, and professional tone.
- Do not include any information that is not explicitly stated in the provided context.
- Strictly adhere to the legal analysis format outlined above.
- Support your analysis only with information directly extracted from the provided documents. Do not introduce external legal principles or precedents unless they are explicitly mentioned in the context.
- Make sure to clearly identify the issue, relevant clauses, obligations, liabilities, and conclusion based solely on the information available in the provided documents.
""",

    "research": """
- Use academic tone.
- Structure as: Introduction, Key Points, Conclusion.
- Be analytical.
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in English, answer in English.
- Do not add speaker labels, roles, or commentary (e.g., Bot, Assistant).
- Maintain a neutral, factual, and professional tone.
- Do not include any information that is not explicitly stated in the provided context.
- Strictly follow the structure of Introduction, Key Points, and Conclusion.
- If the context provides multiple perspectives or findings, make sure to analyze and present them in a balanced manner without introducing any bias or assumptions.
- Try to catch any nuances in the provided context and reflect them accurately in your answer, while avoiding any form of speculation or unsupported claims.
- Plagiarism Warning: Ensure that your answer is original and does not copy any sentences or phrases directly from the provided context. Paraphrase the information in your own words while maintaining the original meaning, and do not include any direct quotes unless they are explicitly cited in the context.
"""
}

model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
processor = AutoProcessor.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ---------------- GROQ ----------------
def ask_groq(context, question, chat_history=None, mode="student"):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    if mode not in MODE_PROMPTS:
        mode = "student"

    mode_instruction = MODE_PROMPTS[mode]

    # Build conversation history
    conversation = ""
    if chat_history:
        for chat in chat_history[-5:]:  # limit memory
            conversation += f"User: {chat['question']}\n"
            conversation += f"Assistant: {chat['answer']}\n"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": """
Rules:
- Use ONLY the provided context (text + diagram information).
- If the answer cannot be found in the context, respond exactly with:
  "Not found in document."
- Provide a clear, detailed explanation using complete sentences.
- Organize the answer into paragraphs and bullet points where appropriate.
- Do not introduce external knowledge, assumptions, or hallucinations.
- Do not add speaker labels, roles, or commentary (e.g., Bot, Assistant).
- Maintain a neutral, factual, and professional tone.
- If the question is in Hindi, answer in Hindi.
- If the question is in Hinglish, answer in Hinglish.
- If the question is in English, answer in English.
- If both text and diagram information are provided:
   - Cross-reference them.
   - Prefer explicit textual statements.
   - Use diagram information only if clearly relevant.
- Do NOT infer hidden meaning from diagrams unless explicitly described.
"""
 }, {
                "role": "user",
                "content": f"""
Previous Conversation:
{conversation}

Mode Instructions:
{mode_instruction}

Document Context:
{context}

Current Question:
{question}
"""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    r = requests.post(url, headers=headers, json=payload)

    if r.status_code != 200:
        return "Error generating response."

    return r.json()["choices"][0]["message"]["content"]