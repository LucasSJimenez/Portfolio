import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. SETUP: Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def run_embedding_pipeline(data_path):
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}.")
        return

    df = pd.read_csv(data_path)
    
    # Auto-pick the text column
    column_name = 'text' if 'text' in df.columns else df.columns[0]
    
    # THE FIX: Fill empty cells with an empty string, THEN convert everything to string
    chunks = df[column_name].fillna("").astype(str).tolist()

    print(f"Embedding {len(chunks)} chunks from column '{column_name}'...")
    
    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    # ... (rest of your build/save code)

    # Build and Save the FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save files locally
    faiss.write_index(index, "docs.index")
    
    # Map IDs back to text for retrieval
    with open("chunk_map.json", "w") as f:
        json.dump({i: text for i, text in enumerate(chunks)}, f)
    
    print("Embedding complete. Saved 'docs.index' and 'chunk_map.json'.")

def retrieve(query, k=3):
    """ Loads the index and returns the top matches. """
    idx = faiss.read_index("docs.index")
    with open("chunk_map.json", "r") as f:
        chunk_map = json.load(f)
    
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = idx.search(query_vector, k)
    
    return [chunk_map[str(i)] for i in indices[0]]

if __name__ == "__main__":
    # Force script to run relative to its own folder (processing)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Updated path to look for the CSV file in the data folder
    real_data_path = os.path.join("..", "data", "tatqa_chunks.csv")
    
    # 1. Build the database
    run_embedding_pipeline(data_path=real_data_path)
    
    # 2. Test the retrieval
    if os.path.exists("docs.index"):
        print("\n--- Running Test Query ---")
        test_results = retrieve("How does the system handle data?")
        print("Top matches found:", test_results)
    else:
        print(f"Error: Still couldn't find the file at {real_data_path}")