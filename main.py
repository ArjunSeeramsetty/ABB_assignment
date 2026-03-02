import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from retriever import HybridRetriever
import os

class RAGPipeline:
    def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
        print("Initializing Retriever...")
        self.retriever = HybridRetriever()
        
        # In a real environment, you might need bitsandbytes 4-bit config here.
        # We load in bfloat16 to save memory.
        print(f"Loading LLM {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # We use device_map="auto" which requires accelerate, fallback to CPU gracefully
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
                trust_remote_code=True
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True
        )

    def format_sources(self, hits):
        # As per instructions: ["Apple 10-K", "Item 8", "p. 28"]
        sources = []
        for hit in hits:
            meta = hit['metadata']
            doc = meta.get('document', 'Unknown')
            sec = meta.get('section', 'Unknown')
            page = f"p. {meta.get('page_number', '?')}"
            sources.append(f'["{doc}", "{sec}", "{page}"]')
        return sources

    def answer_question(self, query: str) -> dict:
        print(f"\nProcessing Query: {query}")
        
        # 1. Retrieve Candidate Chunks
        hits = self.retriever.search(query, top_k=5)
        
        # 2. Format Context
        context_blocks = []
        for i, hit in enumerate(hits):
            meta = hit['metadata']
            context_blocks.append(f"--- Chunk {i+1} ---\nSource: {meta['document']}, {meta['section']}, p. {meta['page_number']}\nText: {hit['text']}\n")
            
        context_str = "\n".join(context_blocks)
        
        # 3. Prompt Construction
        system_prompt = (
            "You are an expert financial analyst. Answer the user's question using ONLY the provided context blocks. "
            "If the answer is not contained in the context, respond EXACTLY with: 'Not specified in the document.' "
            "If the question is completely out of scope of Apple or Tesla 10-K filings, respond EXACTLY with: 'This question cannot be answered based on the provided documents.' "
            "Keep your answer concise and accurate. Do not add conversational filler."
        )
        
        user_prompt = f"Context Blocks:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # 4. LLM Generation
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=150, temperature=0.1, return_full_text=False)
        answer_text = outputs[0]["generated_text"].strip()
        
        # 5. Extract sources if the answer is valid
        sources = []
        if "cannot be answered" not in answer_text.lower() and "not specified" not in answer_text.lower():
            # In a strict implementation, we would extract the exact cited sources from the text.
            # Here, we attach the context blocks that were highly relevant.
            sources_list = []
            for hit in hits:
                meta = hit['metadata']
                doc = meta.get('document', 'Unknown')
                sec = meta.get('section', 'Unknown')
                page = f"p. {meta.get('page_number', '?')}"
                sources_list.append([doc, sec, page])
            sources = sources_list
            
        return {
            "answer": answer_text,
            "sources": sources
        }

# Global instance for the "One Function to Rule Them All" requirement
_rag_pipeline = None

def answer_question(query: str) -> dict:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline("Qwen/Qwen2.5-1.5B-Instruct") # using a smaller model to avoid immense download & memory issues
    return _rag_pipeline.answer_question(query)

if __name__ == "__main__":
    test_queries = [
        "What was Apples total revenue for the fiscal year ended September 28, 2024?",
        "What color is Teslas headquarters painted?"
    ]
    for q in test_queries:
        res = answer_question(q)
        print(json.dumps(res, indent=2))
