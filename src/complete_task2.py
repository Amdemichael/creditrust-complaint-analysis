#!/usr/bin/env python3
"""
Complete Task 2: Generate embeddings and create FAISS vector store
This script takes the existing complaint_chunks.csv and creates the vector store needed for Task 3.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

def main():
    print("🚀 Completing Task 2: Generating Vector Store")
    print("=" * 50)
    
    # Paths
    chunks_file = '../data/processed/complaint_chunks.csv'
    index_file = '../vector_store/faiss_index.bin'
    metadata_file = '../vector_store/metadata.csv'
    
    # Check if chunks file exists
    if not os.path.exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please run the chunking step in the notebook first.")
        return
    
    # Load chunks
    print("📖 Loading complaint chunks...")
    df_chunks = pd.read_csv(chunks_file)
    print(f"✅ Loaded {len(df_chunks)} chunks")
    
    # Create vector store directory
    os.makedirs('../vector_store', exist_ok=True)
    
    # Generate embeddings
    print("\n🔧 Generating embeddings...")
    print("This may take several minutes depending on your system...")
    
    start_time = time.time()
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings in batches to avoid memory issues
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(df_chunks), batch_size):
        batch = df_chunks['chunk'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(batch, batch_size=32, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
        
        if (i + batch_size) % 10000 == 0:
            print(f"   Processed {i + batch_size}/{len(df_chunks)} chunks...")
    
    # Add embeddings to dataframe
    df_chunks['embedding'] = all_embeddings
    
    embedding_time = time.time() - start_time
    print(f"✅ Generated embeddings in {embedding_time:.1f} seconds")
    
    # Create FAISS index
    print("\n🏗️ Building FAISS index...")
    
    # Convert embeddings to numpy
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]  # 384 for all-MiniLM-L6-v2
    
    print(f"   Embedding dimension: {dimension}")
    print(f"   Total vectors: {len(embeddings_np)}")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save index and metadata
    print("\n💾 Saving vector store...")
    faiss.write_index(index, index_file)
    df_chunks[['complaint_id', 'product', 'chunk']].to_csv(metadata_file, index=False)
    
    print(f"✅ Saved FAISS index with {index.ntotal} vectors to {index_file}")
    print(f"✅ Saved metadata to {metadata_file}")
    
    # Verify the files
    print("\n🔍 Verifying vector store...")
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        # Test loading
        test_index = faiss.read_index(index_file)
        test_metadata = pd.read_csv(metadata_file)
        
        print(f"✅ Verified FAISS index: {test_index.ntotal} vectors, dimension {test_index.d}")
        print(f"✅ Verified metadata: {len(test_metadata)} rows")
        
        print("\n🎉 Task 2 completed successfully!")
        print("You can now proceed with Task 3.")
        
    else:
        print("❌ Error: Vector store files not created properly.")

if __name__ == "__main__":
    main() 