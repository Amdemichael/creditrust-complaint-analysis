#!/usr/bin/env python3
"""
Test script for the RAG pipeline.
Run this to verify everything is working before the full evaluation.
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from rag_pipeline import RAGPipeline

def test_rag_pipeline():
    """Test the RAG pipeline with a simple question."""
    print("🧪 Testing RAG Pipeline...")
    
    try:
        # Initialize RAG pipeline
        print("1. Initializing RAG pipeline...")
        rag = RAGPipeline()
        print("✅ RAG pipeline initialized successfully")
        
        # Test retrieval
        print("\n2. Testing retrieval...")
        test_question = "What are common credit card issues?"
        retrieved_chunks = rag.retrieve(test_question, k=3)
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"   Chunk {i}: {chunk['text'][:100]}...")
        
        # Test full RAG pipeline
        print("\n3. Testing full RAG pipeline...")
        result = rag.answer_question(test_question)
        
        print(f"✅ Generated answer: {result['answer'][:200]}...")
        print(f"✅ Used {len(result['sources'])} sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_vector_store():
    """Test if vector store files exist and are accessible."""
    print("\n🔍 Testing Vector Store...")
    
    vector_store_path = '../vector_store/'
    required_files = ['faiss_index.bin', 'metadata.csv']
    
    for file in required_files:
        file_path = os.path.join(vector_store_path, file)
        if os.path.exists(file_path):
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
            return False
    
    # Check metadata file
    try:
        metadata = pd.read_csv(os.path.join(vector_store_path, 'metadata.csv'))
        print(f"✅ Metadata file loaded with {len(metadata)} records")
        print(f"   Columns: {metadata.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error loading metadata: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting RAG System Tests\n")
    
    # Test vector store
    if not test_vector_store():
        print("\n❌ Vector store test failed. Please run Task 2 first.")
        return
    
    # Test RAG pipeline
    if test_rag_pipeline():
        print("\n🎉 All tests passed! RAG system is ready.")
        print("\nNext steps:")
        print("1. Run evaluation: python src/evaluation.py")
        print("2. Launch interface: python app.py")
    else:
        print("\n❌ RAG pipeline test failed. Check the error messages above.")

if __name__ == "__main__":
    main() 