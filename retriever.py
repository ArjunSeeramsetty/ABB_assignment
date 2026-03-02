import os
import pickle
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, collection_name="10k_filings", persist_directory="./chroma_db", embedding_model="BAAI/bge-small-en-v1.5", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        import torch
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device} for Retriever")
        
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': device})
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Setup BM25 and chunk metadata storage
        self.bm25 = None
        self.all_chunks = [] # store the raw text chunks to map back for BM25
        self.bm25_path = os.path.join(persist_directory, "bm25_index.pkl")
        self.chunks_path = os.path.join(persist_directory, "all_chunks.pkl")
        
        # Setup Reranker
        print("Loading cross encoder for reranking...")
        self.reranker = CrossEncoder(reranker_model, max_length=512, device=device)
        
        self.load_local_indices()

    def load_local_indices(self):
        if os.path.exists(self.bm25_path) and os.path.exists(self.chunks_path):
            print("Loading BM25 index from disk...")
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.chunks_path, 'rb') as f:
                self.all_chunks = pickle.load(f)

    def populate(self, documents):
        """
        documents is a list of dicts: {"page_content": str, "metadata": dict}
        """
        if self.collection.count() > 0:
            print("ChromaDB collection already populated. Skipping populate step.")
            return

        print(f"Adding {len(documents)} chunks to Vector DB and BM25 index...")
        ids = [str(i) for i in range(len(documents))]
        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Add to Chroma (this will compute embeddings via its own function or we pass embeddings)
        # Note: langchain's HuggingFaceEmbeddings doesn't directly plug into chromadb collection.add
        # We need to compute embeddings first
        print("Computing embeddings... this may take a moment.")
        embedded_texts = self.embeddings.embed_documents(texts)
        
        self.collection.add(
            ids=ids,
            embeddings=embedded_texts,
            documents=texts,
            metadatas=metadatas
        )

        # Setup BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.all_chunks = documents
        
        # Save BM25 and chunks to disk
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.all_chunks, f)
            
        print("Populate complete.")

    def search(self, query: str, top_k: int = 5):
        # 1. Vector Search (Top 15)
        query_embedding = self.embeddings.embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=15
        )
        
        vector_hits = []
        if vector_results['documents'] and len(vector_results['documents']) > 0:
            for i in range(len(vector_results['documents'][0])):
                vector_hits.append({
                    "id": vector_results['ids'][0][i],
                    "text": vector_results['documents'][0][i],
                    "metadata": vector_results['metadatas'][0][i]
                })

        # 2. BM25 Keyword Search (Top 15)
        keyword_hits = []
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # manual argsort to get top 15 indices
            top_15_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]
            
            for idx in top_15_bm25_idx:
                chunk = self.all_chunks[idx]
                keyword_hits.append({
                    "id": str(idx),
                    "text": chunk["page_content"],
                    "metadata": chunk["metadata"]
                })

        # 3. Merge and Deduplicate
        seen_ids = set()
        combined_hits = []
        for hit in vector_hits + keyword_hits:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                combined_hits.append(hit)
                
        if not combined_hits:
            return []

        # 4. Rerank 
        # Reranker takes pairs: [query, doc_text]
        pairs = [[query, hit["text"]] for hit in combined_hits]
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for hit, score in zip(combined_hits, scores):
            hit["score"] = score
            
        combined_hits.sort(key=lambda x: x["score"], reverse=True)
        return combined_hits[:top_k]

if __name__ == "__main__":
    from ingest import get_text_chunks
    retriever = HybridRetriever()
    if retriever.collection.count() == 0:
        docs = get_text_chunks()
        retriever.populate(docs)
    
    # Test query
    sample_q = "What was Apples total revenue for the fiscal year ended September 28, 2024?"
    print(f"\nQuerying: {sample_q}")
    res = retriever.search(sample_q)
    for i, r in enumerate(res):
        print(f"\n--- Rank {i+1} --- Score: {r['score']:.4f}")
        print(f"Metadata: {r['metadata']}")
        print(r["text"][:200] + "...")
