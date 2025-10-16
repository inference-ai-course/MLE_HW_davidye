from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# ============================================
# LOAD MODEL, INDEX, AND CHUNKS
# ============================================
print("Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("faiss_index.bin")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

print(f"✓ Loaded index with {faiss_index.ntotal} vectors")
print(f"✓ Loaded {len(chunks)} chunks")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI()

@app.get("/search")
async def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # Embed the query 'q' using your embedding model
    query_vector = model.encode([q])
    
    # Perform FAISS search
    k = 3
    distances, indices = faiss_index.search(np.array(query_vector), k)
    
    # Retrieve the corresponding chunks
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    
    return {"query": q, "results": results}
