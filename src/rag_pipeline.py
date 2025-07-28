import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

class RAGPipeline:
    def __init__(self, vector_store_path='vector_store/', model_name='microsoft/DialoGPT-medium'):
        """
        Initialize the RAG pipeline with vector store and LLM.
        
        Args:
            vector_store_path: Path to the FAISS index and metadata
            model_name: HuggingFace model name for text generation
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        
        # Load vector store
        self.index = faiss.read_index(os.path.join(vector_store_path, 'faiss_index.bin'))
        self.metadata = pd.read_csv(os.path.join(vector_store_path, 'metadata.csv'))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set up generation pipeline
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"RAG Pipeline initialized with {self.index.ntotal} vectors")
    
    def retrieve(self, question, k=5):
        """
        Retrieve the top-k most relevant chunks for a given question.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        # Embed the question
        question_embedding = self.embedding_model.encode([question])
        
        # Search the vector store
        distances, indices = self.index.search(question_embedding, k)
        
        # Get the retrieved chunks with metadata
        retrieved_chunks = []
        for idx in indices[0]:
            chunk_data = self.metadata.iloc[idx]
            retrieved_chunks.append({
                'text': chunk_data['chunk'],
                'complaint_id': chunk_data['complaint_id'],
                'product': chunk_data['product'],
                'similarity_score': 1 - distances[0][list(indices[0]).index(idx)]
            })
        
        return retrieved_chunks
    
    def create_prompt(self, question, context_chunks):
        """
        Create a prompt template for the LLM.
        
        Args:
            question: User's question
            context_chunks: List of retrieved chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([f"Complaint {chunk['complaint_id']} ({chunk['product']}): {chunk['text']}" 
                                   for chunk in context_chunks])
        
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based on the provided context.

Context (retrieved complaint excerpts):
{context_text}

Question: {question}

Answer: Based on the complaint data, """
        
        return prompt
    
    def generate_answer(self, prompt):
        """
        Generate an answer using the LLM.
        
        Args:
            prompt: Formatted prompt for the LLM
            
        Returns:
            Generated answer text
        """
        # Generate response
        response = self.generator(prompt, max_new_tokens=200, do_sample=True)
        
        # Extract the generated text (remove the input prompt)
        generated_text = response[0]['generated_text']
        answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def answer_question(self, question, k=5):
        """
        Complete RAG pipeline: retrieve relevant chunks and generate an answer.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and retrieved sources
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, k)
        
        # Step 2: Create prompt
        prompt = self.create_prompt(question, retrieved_chunks)
        
        # Step 3: Generate answer
        answer = self.generate_answer(prompt)
        
        return {
            'answer': answer,
            'sources': retrieved_chunks,
            'question': question
        }

def create_evaluation_questions():
    """
    Create a list of representative questions for evaluation.
    
    Returns:
        List of test questions
    """
    questions = [
        "What are the most common issues with credit cards?",
        "Why are customers unhappy with BNPL services?",
        "What problems do people face with money transfers?",
        "What are the main complaints about personal loans?",
        "How do customers feel about savings accounts?",
        "What billing issues do customers report?",
        "What fraud-related complaints exist?",
        "What customer service problems are mentioned?",
        "What technical issues do users face?",
        "What are the most urgent complaints that need immediate attention?"
    ]
    return questions

if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test with a sample question
    test_question = "What are the most common issues with credit cards?"
    result = rag.answer_question(test_question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['sources'])} sources:")
    for i, source in enumerate(result['sources'][:2]):  # Show first 2 sources
        print(f"Source {i+1}: {source['text'][:200]}...") 