# ABB Assignment - RAG System Design Report

## 1. Chunking Strategy
The data ingestion pipeline utilizes the `RecursiveCharacterTextSplitter` from LangChain to process the Apple and Tesla 10-K PDF documents.
- **Chunk Size (1024 tokens):** Financial 10-K filings are highly dense documents where context (such as a table's description or a preceding paragraph explaining a metric) is often spread across several lines. A chunk size of 1024 allows the retriever to capture complete, coherent semantic blocks (e.g., an entire financial subsection or "Item" header) without truncating necessary surrounding context.
- **Chunk Overlap (100 tokens):** A 100-token overlap acts as a sliding window to prevent strict boundary cutoffs from splitting related sentences in half, ensuring that cross-paragraph concepts are preserved.

## 2. LLM Selection
The core generation engine utilizes the **`Qwen/Qwen2.5-1.5B-Instruct`** model. This decision was driven by the strict requirement to utilize a lightweight, open-access model capable of running entirely natively without relying on proprietary or closed-source APIs (such as OpenAI's GPT-4 or Anthropic's Claude).
- **Justification:** At 1.5 billion parameters, Qwen 2.5 strikes a perfect balance between robust instruction-following capabilities and low memory footprint. It natively supports the hardware constraints typical of Kaggle or Google Colab environments (fitting completely within modest GPU VRAM constraints or operating efficiently on CPU fallback). Furthermore, its `Instruct` tuning makes it highly disciplined at adhering to rigid prompt constraints.

## 3. Out-of-Scope Handling & Hallucination Prevention
A critical assignment objective is ensuring the system completely refuses to invent knowledge. The system employs a "Defense in Depth" strategy with two distinct layers:
1. **Strict System Prompt Constraints:** The LLM's system prompt explicitly enforces a unified restriction: *"Answer the user's question using ONLY the provided context blocks. If the answer is not contained in the context, respond EXACTLY with: 'Not specified in the document.' If the question is completely out of scope of Apple or Tesla 10-K filings, respond EXACTLY with: 'This question cannot be answered based on the provided documents.'"*
2. **Determinism and Grounding:** The generation logic specifies a `temperature=0.1` parameter. By stripping the model of "creativity," it behaves deterministically, significantly lowering the probability of hallucinating external world knowledge (e.g., citing a known CEO or color when the retrieved chunk vectors return empty or irrelevant for the respective question).

## 4. Retrieval Architecture (Bonus Context)
The agent operates via a **Hybrid/Ensemble Retriever**:
- **Semantic Dense Search:** Computes vector similarities utilizing `BAAI/bge-small-en-v1.5` embeddings against a persistent ChromaDB store.
- **Sparse Keyword Search:** Uses `BM25Okapi` to strictly match crucial exact keywords or numbers.
- **Cross-Encoder Re-ranking:** Results from both spaces are merged and heavily re-scored via `cross-encoder/ms-marco-MiniLM-L-6-v2`. This ensures that the context ultimately fed to the LLM has the absolute highest relevance to the specific nuance of the prompt.
