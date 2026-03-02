import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path, company_name):
    """
    Extracts text from a PDF file using PyMuPDF and formats it with metadata.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    
    # We will try to map sections by naive heuristics or just keep them generic 
    # since exact section boundaries in 10-K need massive regex parsing.
    # The requirement is to preserve metadata: document, section, page number.
    
    current_section = "General"
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Simple heuristic for section: if a line starts with "Item " or "ITEM "
        lines = text.split('\n')
        for line in lines:
            if line.strip().upper().startswith("ITEM ") and len(line.strip()) < 100:
                current_section = line.strip()
                break # Just take the first one found on page
                
        if text.strip():
            metadata = {
                "document": company_name,
                "section": current_section,
                "page_number": page_num + 1
            }
            chunks.append({"text": text, "metadata": metadata})
            
    return chunks

def get_text_chunks():
    """
    Gets text from both PDFs, splits them into smaller chunks.
    """
    apple_pdf = os.path.join("input_pdfs", "10-Q4-2024-As-Filed.pdf")
    tesla_pdf = os.path.join("input_pdfs", "tsla-20231231-gen.pdf")
    
    apple_data = extract_text_from_pdf(apple_pdf, "Apple 10-K")
    tesla_data = extract_text_from_pdf(tesla_pdf, "Tesla 10-K")
    
    all_data = apple_data + tesla_data
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []
    for item in all_data:
        splits = text_splitter.split_text(item["text"])
        for split in splits:
            final_chunks.append({
                "page_content": split,
                "metadata": item["metadata"]
            })
            
    print(f"Total extracted documents chunks: {len(final_chunks)}")
    return final_chunks

if __name__ == "__main__":
    chunks = get_text_chunks()
    print(f"Sample chunk metadata: {chunks[0]['metadata']}")
