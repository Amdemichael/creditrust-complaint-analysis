import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os

# Set paths
input_file = '../data/processed/filtered_complaints.csv'
chunks_file = '../data/processed/complaint_chunks.csv'
index_file = '../vector_store/faiss_index.bin'
metadata_file = '../vector_store/metadata.csv'

# Create directories
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../vector_store', exist_ok=True)

# Load filtered dataset
print("Loading filtered complaints...")
df = pd.read_csv(input_file)

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Based on typical narrative length
    chunk_overlap=50,
    length_function=lambda x: len(x.split())
)

# Split narratives into chunks
print("Chunking narratives...")
chunks = []
for idx, row in df.iterrows():
    splits = splitter.split_text(row['Consumer complaint narrative'])
    for split in splits:
        chunks.append({
            'complaint_id': row['Complaint ID'],  # Adjust if column name differs
            'product': row['Product'],
            'chunk': split
        })

# Save chunks
df_chunks = pd.DataFrame(chunks)
df_chunks.to_csv(chunks_file, index=False)
print(f"Created {len(df_chunks)} chunks, saved to {chunks_file}")

# Generate embeddings
print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df_chunks['chunk'].tolist(), batch_size=32, show_progress_bar=True)
df_chunks['embedding'] = embeddings.tolist()

# Create FAISS index
print("Building FAISS index...")
embeddings_np = np.array(df_chunks['embedding'].tolist(), dtype=np.float32)
dimension = embeddings_np.shape[1]  # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Save index and metadata
faiss.write_index(index, index_file)
df_chunks[['complaint_id', 'product', 'chunk']].to_csv(metadata_file, index=False)
print(f"Saved FAISS index with {index.ntotal} vectors to {index_file}")
print(f"Saved metadata to {metadata_file}")