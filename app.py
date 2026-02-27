from flask import Flask, render_template, request, jsonify, session
import main

app = Flask(__name__)
app.secret_key = "supersecretkey"

text_index = None
text_chunks = []

@app.route("/")
def home():
    return render_template("index.html")


# 🔹 File Upload Route
@app.route("/upload", methods=["POST"])
def upload_files():
    global text_index, text_chunks

    files = request.files.getlist("file")
    text_data = ""

    for file in files:
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            text_data += main.read_pdfs([file.stream])

        elif filename.endswith(".docx"):
            text_data += main.read_docx(file.stream)

    text_chunks = main.chunk_text(text_data)
    text_index = main.build_text_index(text_chunks)

    return jsonify({"status": "Documents processed successfully"})


# 🔹 Chat Route (For GPT UI)
@app.route("/chat", methods=["POST"])
def chat():
    global text_index, text_chunks

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"answer": "No question provided."})

    if not text_index:
        return jsonify({"answer": "Please upload documents first."})

    # Initialize memory
    if "chat_history" not in session:
        session["chat_history"] = []

    # Retrieve context
    results = main.search_text(question, text_index, text_chunks)[:3]
    context = "\n\n".join(results)

    mode = data.get("mode", "student")

    # Generate answer with memory
    answer = main.ask_groq(
        context,
        question,
        session["chat_history"],
        mode
    )

    # Save memory
    session["chat_history"] = session["chat_history"][-5:]
    session.modified = True

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)