# CrediTrust Complaint Analysis
A Retrieval-Augmented Generation (RAG) chatbot to analyze customer complaints for CrediTrust Financial, built for the Intelligent Complaint Analysis Challenge.

## Project Structure
- `data/raw/`: Original CFPB dataset (not tracked due to size).
- `data/processed/`: Filtered and cleaned dataset.
- `notebooks/`: EDA and preprocessing notebooks.
- `src/`: Reusable scripts for RAG pipeline.
- `vector_store/`: Vector database for embeddings.
- `reports/`: Interim and final reports.
- `app.py`: Gradio/Streamlit interface.

## Setup
```bash
python -m venv venv
venv\Scripts\activate  #Windows
pip install -r requirements.txt