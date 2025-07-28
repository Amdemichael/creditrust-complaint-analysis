import gradio as gr
import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline

class ComplaintChatInterface:
    def __init__(self):
        """Initialize the chat interface with RAG pipeline."""
        self.rag = RAGPipeline()
        self.chat_history = []
        
    def format_sources(self, sources):
        """Format source chunks for display."""
        formatted_sources = []
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            formatted_sources.append(
                f"**Source {i}** (Complaint {source['complaint_id']}, {source['product']})\n"
                f"Similarity: {source['similarity_score']:.3f}\n"
                f"Text: {source['text'][:300]}{'...' if len(source['text']) > 300 else ''}\n"
            )
        return "\n".join(formatted_sources)
    
    def chat_with_rag(self, message, history):
        """Process user message and generate response using RAG."""
        if not message.strip():
            return "", history
        
        try:
            # Get RAG response
            result = self.rag.answer_question(message)
            
            # Format the response
            answer = result['answer']
            sources = result['sources']
            
            # Create detailed response with sources
            response = f"**Answer:**\n{answer}\n\n"
            response += f"**Sources Used:**\n{self.format_sources(sources)}"
            
            # Add to chat history
            history.append((message, response))
            
            return "", history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def clear_chat(self):
        """Clear the chat history."""
        return [], []
    
    def get_sample_questions(self):
        """Get sample questions for users."""
        return [
            "What are the most common issues with credit cards?",
            "Why are customers unhappy with BNPL services?",
            "What problems do people face with money transfers?",
            "What are the main complaints about personal loans?",
            "How do customers feel about savings accounts?",
            "What billing issues do customers report?",
            "What fraud-related complaints exist?",
            "What customer service problems are mentioned?"
        ]

def create_interface():
    """Create the Gradio interface."""
    chat_interface = ComplaintChatInterface()
    
    # Sample questions
    sample_questions = chat_interface.get_sample_questions()
    
    with gr.Blocks(
        title="CrediTrust Complaint Analysis Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .bot-message {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏦 CrediTrust Complaint Analysis Chatbot
        
        **Welcome to the intelligent complaint analysis system!** 
        
        This AI-powered chatbot helps you understand customer complaints across CrediTrust's financial products:
        - Credit Cards
        - Personal Loans  
        - Buy Now, Pay Later (BNPL)
        - Savings Accounts
        - Money Transfers
        
        Ask questions in plain English and get evidence-backed insights from thousands of real customer complaints.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a question about customer complaints...",
                        placeholder="e.g., What are the most common issues with credit cards?",
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    info_btn = gr.Button("ℹ️ About", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Sample Questions")
                
                # Sample questions buttons
                sample_btns = []
                for question in sample_questions:
                    btn = gr.Button(
                        question[:50] + "..." if len(question) > 50 else question,
                        size="sm",
                        variant="outline"
                    )
                    sample_btns.append(btn)
                
                gr.Markdown("""
                ### 📊 System Info
                - **Vector Store**: FAISS with 392K+ complaint chunks
                - **Embedding Model**: all-MiniLM-L6-v2
                - **LLM**: DialoGPT-medium
                - **Coverage**: 5 financial products
                
                ### 🔍 How it works
                1. Your question is embedded and matched against complaint chunks
                2. Most relevant complaints are retrieved
                3. LLM generates answer based on retrieved evidence
                4. Sources are displayed for transparency
                """)
        
        # Event handlers
        submit_btn.click(
            chat_interface.chat_with_rag,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            chat_interface.chat_with_rag,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            chat_interface.clear_chat,
            outputs=[msg, chatbot]
        )
        
        # Sample question buttons
        for btn in sample_btns:
            btn.click(
                lambda q: (q, []),
                inputs=[btn],
                outputs=[msg, chatbot]
            )
        
        # Info button
        def show_info():
            return """
            ## About This System
            
            **Purpose**: Transform customer complaints into actionable insights for CrediTrust teams.
            
            **Data Source**: Consumer Financial Protection Bureau (CFPB) complaint database
            - 355K+ filtered complaints across 5 product categories
            - Real customer narratives and feedback
            
            **Technology Stack**:
            - **Vector Search**: FAISS with sentence-transformers embeddings
            - **Text Generation**: DialoGPT-medium for natural language responses
            - **Interface**: Gradio for user-friendly web interface
            
            **Key Features**:
            - Semantic search across complaint narratives
            - Evidence-backed answers with source transparency
            - Multi-product analysis capabilities
            - Real-time question answering
            
            **Use Cases**:
            - Product Managers identifying trends
            - Support teams understanding customer pain points
            - Compliance teams monitoring issues
            - Executive insights into customer experience
            """
        
        info_btn.click(
            show_info,
            outputs=[chatbot]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


