from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: list of chunk texts
chunks = ["chunk 1 text", "chunk 2 text", ...]

# Generate embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# To search:
query = "your search query"
query_emb = model.encode([query], convert_to_numpy=True)
D, I = index.search(query_emb, k=5)  # k = number of results