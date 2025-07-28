import gradio as gr
import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline

# Fix path for vector store
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
                f"**📄 Source {i}**\n"
                f"**Complaint ID:** {source['complaint_id']}\n"
                f"**Product:** {source['product']}\n"
                f"**Relevance:** {source['similarity_score']:.1%}\n"
                f"**Text:** {source['text'][:250]}{'...' if len(source['text']) > 250 else ''}\n"
            )
        return "\n---\n".join(formatted_sources)
    
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
            response = f"🤖 **AI Analysis:**\n{answer}\n\n"
            response += f"📚 **Evidence Sources:**\n{self.format_sources(sources)}"
            
            # Add to chat history
            history.append((message, response))
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ **Error:** Sorry, I encountered an issue: {str(e)}"
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
    """Create the polished Gradio interface."""
    chat_interface = ComplaintChatInterface()
    
    # Sample questions
    sample_questions = chat_interface.get_sample_questions()
    
    # Custom CSS for polished design
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .sidebar {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        height: fit-content;
    }
    
    .sample-question {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .sample-question:hover {
        background: #2196f3;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-section {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: white;
        border: 2px solid #e0e0e0;
        margin-right: 2rem;
    }
    
    .input-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    .submit-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .submit-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .clear-btn {
        background: #f44336 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .clear-btn:hover {
        background: #d32f2f !important;
        transform: translateY(-2px) !important;
    }
    
    .info-btn {
        background: #2196f3 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .info-btn:hover {
        background: #1976d2 !important;
        transform: translateY(-2px) !important;
    }
    """
    
    with gr.Blocks(
        title="CrediTrust Complaint Analysis Chatbot",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: bold;">🏦 CrediTrust Complaint Analysis</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Intelligent AI-powered insights from customer feedback
            </p>
            <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">💳 Credit Cards</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">💰 Personal Loans</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🛒 BNPL</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🏦 Savings</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">💸 Money Transfers</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                with gr.Group(elem_classes="chat-container"):
                    gr.Markdown("### 💬 Ask Questions About Customer Complaints")
                    chatbot = gr.Chatbot(
                        label="",
                        height=500,
                        show_label=False,
                        container=True,
                        bubble_full_width=False,
                        elem_classes="chat-interface"
                    )
                
                # Input area
                with gr.Group(elem_classes="input-container"):
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="Ask a question about customer complaints... (e.g., What are the most common issues with credit cards?)",
                            scale=4,
                            show_label=False,
                            container=False
                        )
                        submit_btn = gr.Button(
                            "🚀 Ask AI", 
                            variant="primary", 
                            scale=1,
                            elem_classes="submit-btn"
                        )
                    
                    with gr.Row():
                        clear_btn = gr.Button(
                            "🗑️ Clear Chat", 
                            variant="secondary",
                            elem_classes="clear-btn"
                        )
                        info_btn = gr.Button(
                            "ℹ️ About System", 
                            variant="secondary",
                            elem_classes="info-btn"
                        )
            
            with gr.Column(scale=1):
                # Sidebar
                with gr.Group(elem_classes="sidebar"):
                    gr.Markdown("### 💡 Quick Questions")
                    
                    # Sample questions buttons
                    sample_btns = []
                    for question in sample_questions:
                        btn = gr.Button(
                            question[:60] + "..." if len(question) > 60 else question,
                            size="sm",
                            variant="outline",
                            elem_classes="sample-question"
                        )
                        sample_btns.append(btn)
                    
                    # Stats card
                    gr.HTML("""
                    <div class="stats-card">
                        <h4 style="margin: 0 0 0.5rem 0;">📊 System Statistics</h4>
                        <p style="margin: 0.2rem 0;">🔍 392K+ complaint chunks</p>
                        <p style="margin: 0.2rem 0;">🤖 AI-powered analysis</p>
                        <p style="margin: 0.2rem 0;">📈 5 product categories</p>
                        <p style="margin: 0.2rem 0;">⚡ Real-time insights</p>
                    </div>
                    """)
                    
                    # How it works
                    gr.HTML("""
                    <div class="info-section">
                        <h4 style="margin: 0 0 0.5rem 0;">🔍 How It Works</h4>
                        <ol style="margin: 0; padding-left: 1.2rem;">
                            <li>Your question is analyzed</li>
                            <li>Relevant complaints are found</li>
                            <li>AI generates insights</li>
                            <li>Sources are shown for transparency</li>
                        </ol>
                    </div>
                    """)
                    
                    # Use cases
                    gr.HTML("""
                    <div class="info-section">
                        <h4 style="margin: 0 0 0.5rem 0;">🎯 Use Cases</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Product Managers: Identify trends</li>
                            <li>Support Teams: Understand pain points</li>
                            <li>Compliance: Monitor issues</li>
                            <li>Executives: Customer insights</li>
                        </ul>
                    </div>
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
            ## 🏦 About CrediTrust Complaint Analysis System
            
            **Purpose**: Transform customer complaints into actionable insights for CrediTrust teams.
            
            ### 📊 Data Source
            **Consumer Financial Protection Bureau (CFPB)** complaint database
            - 355K+ filtered complaints across 5 product categories
            - Real customer narratives and feedback
            - Comprehensive financial services coverage
            
            ### 🛠️ Technology Stack
            - **Vector Search**: FAISS with sentence-transformers embeddings
            - **Text Generation**: DialoGPT-medium for natural language responses
            - **Interface**: Gradio for user-friendly web interface
            - **Processing**: Advanced NLP and machine learning
            
            ### ✨ Key Features
            - **Semantic Search**: Find relevant complaints instantly
            - **Evidence-Backed Answers**: See source complaints for transparency
            - **Multi-Product Analysis**: Compare issues across financial services
            - **Real-Time Insights**: Get answers in seconds
            - **Professional Interface**: Designed for business users
            
            ### 🎯 Business Impact
            - **Product Managers**: Identify emerging trends quickly
            - **Support Teams**: Understand customer pain points instantly
            - **Compliance Teams**: Monitor issues proactively
            - **Executives**: Get real-time customer experience insights
            
            ### 🔒 Trust & Transparency
            - All answers are backed by real complaint data
            - Source complaints are always displayed
            - No black-box AI - you can verify every insight
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


