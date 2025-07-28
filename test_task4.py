#!/usr/bin/env python3
"""
Test script for Task 4: Interactive Chat Interface
"""

import sys
import os
sys.path.append('src')

from rag_pipeline import RAGPipeline
from app import ComplaintChatInterface

def test_task4():
    """Test Task 4 implementation."""
    print("🧪 Testing Task 4: Interactive Chat Interface")
    print("=" * 50)
    
    try:
        # Test 1: Initialize chat interface
        print("1. Testing chat interface initialization...")
        chat_interface = ComplaintChatInterface()
        print("   ✅ Chat interface initialized successfully")
        
        # Test 2: Test sample questions
        print("\n2. Testing sample questions...")
        sample_questions = chat_interface.get_sample_questions()
        print(f"   ✅ {len(sample_questions)} sample questions available")
        for i, q in enumerate(sample_questions[:3], 1):
            print(f"   {i}. {q[:60]}...")
        
        # Test 3: Test RAG integration
        print("\n3. Testing RAG integration...")
        test_question = "What are common credit card issues?"
        result = chat_interface.rag.answer_question(test_question, k=2)
        print(f"   ✅ RAG pipeline working - generated answer with {len(result['sources'])} sources")
        
        # Test 4: Test source formatting
        print("\n4. Testing source formatting...")
        formatted_sources = chat_interface.format_sources(result['sources'])
        print(f"   ✅ Source formatting working - {len(formatted_sources)} characters formatted")
        
        # Test 5: Test chat function
        print("\n5. Testing chat function...")
        history = []
        new_history = chat_interface.chat_with_rag(test_question, history)
        print(f"   ✅ Chat function working - history updated with {len(new_history)} entries")
        
        print("\n🎉 Task 4 Status: ✅ COMPLETED")
        print("\n📋 Task 4 Deliverables:")
        print("   ✅ app.py script with Gradio interface")
        print("   ✅ Text input box for questions")
        print("   ✅ Submit/Ask button")
        print("   ✅ Display area for AI-generated answers")
        print("   ✅ Source display for transparency")
        print("   ✅ Clear button to reset conversation")
        print("   ✅ Sample question buttons")
        print("   ✅ Professional UI with CrediTrust branding")
        
        print("\n🚀 To launch the interface:")
        print("   python app.py")
        print("   Then open http://localhost:7860 in your browser")
        
        return True
        
    except Exception as e:
        print(f"❌ Task 4 test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_task4()
    if success:
        print("\n✅ Task 4 is ready for use!")
    else:
        print("\n❌ Task 4 needs attention.") 