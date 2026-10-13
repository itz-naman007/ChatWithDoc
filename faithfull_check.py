
import csv

with open("answer_evaluation.csv", "w", newline="",encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question",
            "reference_answer",
            "retrieved_context",
            "generated_answer",
            "faithfulness_1_to_5",
            "relevance_1_to_5"
        ]
    )
    writer.writeheader()

    for item in qa_data:
        results = main.search_text(
            item["question"], index, chunks,
            k=10, final_k=3
        )

        context = "\n\n".join(results)

        if not context:
            answer = "Not found in document."
        else:
            answer = main.ask_groq(
                context,
                item["question"],
                chat_history=[],
                mode="student"
            )

        writer.writerow({
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "retrieved_context": context,
            "generated_answer": answer,
            "faithfulness_1_to_5": "",
            "relevance_1_to_5": ""
        })

print("Saved answers to answer_evaluation.csv")
