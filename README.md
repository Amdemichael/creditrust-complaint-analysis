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

### EDA and Preprocessing Summary

The CFPB dataset (`complaints.csv`, 5.64 GB) was processed using pandas chunking on a 16GB RAM Windows system. We filtered for four product categories ('Credit card or prepaid card', 'Consumer Loan', 'Checking or savings account', 'Money transfer, virtual currency, or money service'), as 'Buy Now, Pay Later (BNPL)' was not explicitly listed. BNPL complaints were identified by searching 'Issue' for terms like 'BNPL', 'installment', or 'pay later', resulting in X complaints relabeled as 'Buy Now, Pay Later (BNPL)'. Empty narratives were removed, reducing the dataset to Y rows (Z MB), saved as `data/processed/filtered_complaints.csv`.

EDA showed 'Credit card or prepaid card' accounted for A% of complaints, followed by 'Checking or savings account' (B%). Narrative lengths had a median of C words, with outliers up to D words, guiding Task 2 chunking. Text cleaning involved lowercasing, removing special characters, and stripping boilerplate phrases. Challenges included resolving Windows file permission and Git issues, mapping BNPL, and handling the large dataset.

### Task 2: Chunking, Embedding, and Indexing

Narratives from `filtered_complaints.csv` (Y rows, Z MB) were split into X chunks using LangChain’s `RecursiveCharacterTextSplitter` with a chunk size of 500 words and 50-word overlap, based on Task 1’s median narrative length of C words and outliers up to D words. Chunks were saved in `data/processed/complaint_chunks.csv`. Embeddings were generated using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors) for CPU efficiency on a 16GB RAM system. A FAISS `IndexFlatL2` index was created and saved as `vector_store/faiss_index.bin`, with metadata (complaint ID, product, chunk text) in `vector_store/metadata.csv` for Task 3 retrieval.