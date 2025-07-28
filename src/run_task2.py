#!/usr/bin/env python3
"""
Task 2: Complete Text Chunking, Embedding, and Vector Store Indexing
This script completes Task 2 by generating embeddings and creating the FAISS vector store.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

def main():
    print("🚀 Task 2: Complete Text Chunking, Embedding, and Vector Store Indexing")
    print("=" * 70)
    
    # Step 1: Check prerequisites
    print("\n📋 Step 1: Checking prerequisites...")
    chunks_file = '../data/processed/complaint_chunks.csv'
    filtered_file = '../data/processed/filtered_complaints.csv'
    
    if not os.path.exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please complete the chunking step first.")
        return
    
    if not os.path.exists(filtered_file):
        print(f"❌ Filtered complaints file not found: {filtered_file}")
        print("Please complete Task 1 first.")
        return
    
    print("✅ All prerequisites met!")
    
    # Step 2: Load chunked data
    print("\n📖 Step 2: Loading chunked data...")
    df_chunks = pd.read_csv(chunks_file)
    print(f"✅ Loaded {len(df_chunks):,} chunks")
    print(f"Columns: {df_chunks.columns.tolist()}")
    
    # Step 3: Initialize embedding model
    print("\n🔧 Step 3: Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    
    # Step 4: Generate embeddings
    print("\n🚀 Step 4: Generating embeddings...")
    print("This may take several minutes depending on your system...")
    
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    all_embeddings = []
    total_batches = (len(df_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(df_chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = df_chunks['chunk'].iloc[i:i+batch_size].tolist()
        
        print(f"   Processing batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, len(df_chunks)):,} chunks)...")
        
        batch_embeddings = model.encode(batch, batch_size=32, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    embedding_time = time.time() - start_time
    print(f"✅ Generated embeddings in {embedding_time:.1f} seconds")
    print(f"Total embeddings: {len(all_embeddings):,}")
    print(f"Embedding dimension: {len(all_embeddings[0])}")
    
    # Step 5: Create FAISS index
    print("\n🏗️ Step 5: Building FAISS vector store...")
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    print(f"✅ FAISS index created with {index.ntotal:,} vectors")
    print(f"Index dimension: {index.d}")
    
    # Step 6: Save vector store
    print("\n💾 Step 6: Saving vector store...")
    vector_store_dir = '../vector_store'
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Save FAISS index
    index_file = os.path.join(vector_store_dir, 'faiss_index.bin')
    faiss.write_index(index, index_file)
    print(f"✅ FAISS index saved ({os.path.getsize(index_file) / (1024*1024):.1f} MB)")
    
    # Save metadata
    metadata_file = os.path.join(vector_store_dir, 'metadata.csv')
    metadata_df = df_chunks[['complaint_id', 'product', 'chunk']].copy()
    metadata_df.to_csv(metadata_file, index=False)
    print(f"✅ Metadata saved ({os.path.getsize(metadata_file) / (1024*1024):.1f} MB)")
    
    # Step 7: Verify vector store
    print("\n🔍 Step 7: Verifying vector store...")
    test_index = faiss.read_index(index_file)
    test_metadata = pd.read_csv(metadata_file)
    
    print(f"✅ FAISS index verified: {test_index.ntotal:,} vectors, dimension {test_index.d}")
    print(f"✅ Metadata verified: {len(test_metadata):,} rows")
    
    # Test similarity search
    print("\n🧪 Testing similarity search...")
    test_question = "What are common credit card issues?"
    test_embedding = model.encode([test_question])
    distances, indices = test_index.search(test_embedding, k=3)
    
    print(f"Test question: {test_question}")
    print(f"Retrieved {len(indices[0])} similar chunks")
    
    # Step 8: Summary
    print("\n📊 Task 2 Completion Summary")
    print("=" * 50)
    print(f"✅ Chunks processed: {len(df_chunks):,}")
    print(f"✅ Embeddings generated: {len(all_embeddings):,}")
    print(f"✅ Embedding dimension: {len(all_embeddings[0])}")
    print(f"✅ FAISS vectors: {test_index.ntotal:,}")
    print(f"✅ Processing time: {embedding_time:.1f} seconds")
    print(f"✅ Vector store size: {os.path.getsize(index_file) / (1024*1024):.1f} MB")
    print(f"✅ Metadata size: {os.path.getsize(metadata_file) / (1024*1024):.1f} MB")
    
    print("\n🎉 Task 2 Status: COMPLETED")
    print("You can now proceed with Task 3: RAG Core Logic and Evaluation")
    
    # List vector store files
    print(f"\n📁 Vector store contents:")
    for file in os.listdir(vector_store_dir):
        file_path = os.path.join(vector_store_dir, file)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  {file} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main() 