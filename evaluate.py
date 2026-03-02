import json
from main import answer_question

# Ground truth questions based on the PDF instructions
EVAL_QUESTIONS = [
    {"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
    {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
    {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
    {"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
    {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
    {"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
    {"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
    {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
    {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
    {"question_id": 10, "question": "What is the purpose of Teslas 'lease pass-through fund arrangements'?"},
    {"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
    {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
    {"question_id": 13, "question": "What color is Teslas headquarters painted?"}
]

def run_evaluation():
    results = []
    print("Starting Evaluation Pipeline...")

    for item in EVAL_QUESTIONS:
        q_id = item["question_id"]
        q_text = item["question"]
        print(f"\n--- Q{q_id}: {q_text} ---")

        answer_dict = answer_question(q_text)

        answer = answer_dict.get("answer", "Error processing")
        sources = answer_dict.get("sources", [])

        result_item = {
            "question_id": q_id,
            "question": q_text,
            "answer": answer,
            "sources": sources,
        }
        results.append(result_item)

        print(f"A: {answer}")
        print(f"Num sources: {len(sources)}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Complete. Saved to results.json")

if __name__ == "__main__":
    run_evaluation()
