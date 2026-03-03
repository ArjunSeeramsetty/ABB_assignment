# Financial 10-K RAG System (Apple & Tesla)

A robust Retrieval-Augmented Generation (RAG) system built to answer financial questions accurately and strictly based on provided SEC 10-K documents (Apple Q4 2024, Tesla 2023).

## Key Features
*   **Open Source LLM:** Powered entirely by `Qwen/Qwen2.5-1.5B-Instruct` running locally via HuggingFace Transformers. No proprietary APIs (like OpenAI) are used.
*   **Hybrid Retrieval:** Combines Dense Vector Search (ChromaDB + `BAAI/bge-small-en-v1.5`) with Sparse Keyword Search (`BM25Okapi`).
*   **Cross-Encoder Re-ranking:** Uses `ms-marco-MiniLM-L-6-v2` to aggressively re-rank combined dense/sparse hits for maximum accuracy.
*   **Strict Guardrails:** Deterministic generation (`temperature=0.1`) combined with highly explicit system prompts to forcefully refuse out-of-scope questions and prevent hallucination.

## Project Structure
*   `ingest.py`: Parses the raw PDF documents into chunked texts with section metadata.
*   `retriever.py`: Hybrid Retrieval logic (Vector DB indexing, BM25 indexing, and Re-ranking).
*   `main.py` / `evaluate.py`: The core LLM Question-Answering logic and evaluation loops.
*   `colab_evaluation.ipynb`: A plug-in-and-play Jupyter Notebook configured to smoothly test the entire pipeline in Kaggle or Colab environments.
*   `design_report.md`: Detailed architectural design choices regarding chunking size, LLM selection, and Out-of-Scope strategies.

## Setup & Evaluation

### 1. Hosted Notebook (Easiest)
You can seamlessly test this entire RAG pipeline end-to-end using the provided notebook:
1. Open [colab_evaluation.ipynb](https://www.kaggle.com/code/arjunseeramsetty/notebook0860611f71/edit) in Google Colab or Kaggle.
2. The notebook handles fetching PDFs, installing requirements, building the indices, and formatting the final validation questions into `results.json`.

### 2. Local Setup
If you want to run this locally:

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Indexer & Retriever Builder:**
```bash
python retriever.py
```

**Run the Evaluation Suite:**
```bash
python evaluate.py
```

This will run all validation questions through the pipeline and write strictly formatted outputs to `results.json`.
