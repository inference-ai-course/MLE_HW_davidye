import fitz  # PyMuPDF
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import json

# ============================================
# STEP 1: TEXT EXTRACTION
# ============================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text


# ============================================
# STEP 2: CHUNKING
# ============================================
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks


# ============================================
# STEP 3: EMBEDDING GENERATION
# ============================================
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks: List[str]):
    """Generate embeddings for a list of text chunks."""
    embeddings = model.encode(chunks)  # embeds each text chunk into a 384-d vector
    return embeddings


# ============================================
# STEP 4: FAISS INDEXING AND SEARCH
# ============================================
def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings."""
    # Assume embeddings is a 2D numpy array of shape (num_chunks, dim)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(np.array(embeddings))  # add all chunk vectors
    return index


def search_index(index, query_embedding, k=3):
    """Search the FAISS index for top-k results."""
    # query_embedding shape should be [1, dim]
    distances, indices = index.search(query_embedding, k)
    # indices[0] holds the top-k chunk indices
    return distances, indices


# ============================================
# PROCESS ALL 50 PAPERS
# ============================================
if __name__ == "__main__":
    papers_dir = "pdfs"  # Change to your folder
    pdf_files = list(Path(papers_dir).glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        print(f"  Added {len(chunks)} chunks")
    
    print(f"\n✓ Total chunks from all papers: {len(all_chunks)}")
    
    # Generate embeddings for all chunks
    print("\nGenerating embeddings (this may take a few minutes)...")
    embeddings = generate_embeddings(all_chunks)
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    print(f"✓ Built FAISS index with {index.ntotal} vectors")
    
    # Save index and chunks
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.json", "w") as f:
        json.dump(all_chunks, f)
    
    print("✓ Saved faiss_index.bin")
    print("✓ Saved chunks.json")
    print("\n--- All 50 papers processed! Ready for FastAPI! ---")