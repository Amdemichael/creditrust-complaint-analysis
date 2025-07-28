# CrediTrust Complaint Analysis - RAG-Powered Chatbot

An intelligent complaint analysis system that transforms customer feedback into actionable insights using Retrieval-Augmented Generation (RAG).

## 🎯 Project Overview

This system helps CrediTrust Financial teams quickly understand customer pain points across five major product categories:
- Credit Cards
- Personal Loans  
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

## 📊 Current Status

✅ **Task 1**: Exploratory Data Analysis and Preprocessing - **COMPLETED**
- Loaded and analyzed 5.2GB CFPB complaint dataset
- Filtered to 355K+ relevant complaints across 5 products
- Cleaned and preprocessed complaint narratives
- Generated EDA visualizations

✅ **Task 2**: Text Chunking, Embedding, and Vector Store - **COMPLETED**
- Implemented recursive text chunking (500 words, 50 overlap)
- Generated embeddings using `all-MiniLM-L6-v2`
- Created FAISS vector store with 392K+ chunks
- Stored metadata for traceability

🔄 **Task 3**: RAG Core Logic and Evaluation - **IN PROGRESS**
- RAG pipeline implementation
- Evaluation framework
- Quality assessment

🔄 **Task 4**: Interactive Chat Interface - **IN PROGRESS**
- Gradio web interface
- Source transparency
- User-friendly design

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Test the System
```bash
cd src
python test_rag.py
```

### 2. Run Evaluation
```bash
cd src
python evaluation.py
```

### 3. Launch Chat Interface
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

## 📁 Project Structure

```
creditrust-complaint-analysis/
├── app.py                          # Gradio chat interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── raw/                       # Original CFPB dataset
│   └── processed/                 # Filtered and cleaned data
├── src/
│   ├── chunking_embedding.py      # Task 2: Vector store creation
│   ├── rag_pipeline.py            # Task 3: RAG core logic
│   ├── evaluation.py              # Task 3: Evaluation framework
│   └── test_rag.py                # System testing
├── notebooks/
│   ├── eda_preprocessing.ipynb    # Task 1: EDA and preprocessing
│   └── chunking_embedding.ipynb   # Task 2: Chunking and embedding
├── vector_store/                  # FAISS index and metadata
└── reports/                       # Evaluation results and visualizations
```

## 🔧 Technical Architecture

### Data Pipeline
1. **Data Loading**: 5.2GB CFPB complaint dataset
2. **Filtering**: 5 product categories, non-empty narratives
3. **Preprocessing**: Text cleaning and normalization
4. **Chunking**: 500-word chunks with 50-word overlap
5. **Embedding**: Sentence transformers (all-MiniLM-L6-v2)
6. **Indexing**: FAISS vector store for fast similarity search

### RAG Pipeline
1. **Query Processing**: User question embedding
2. **Retrieval**: Top-k similar chunks from vector store
3. **Context Assembly**: Format retrieved chunks with metadata
4. **Generation**: LLM (DialoGPT-medium) generates answer
5. **Response**: Answer + source transparency

### Evaluation Framework
- **Quality Metrics**: Relevance, completeness, source quality
- **Test Questions**: 10 representative business questions
- **Scoring**: 1-5 scale with detailed analysis
- **Reporting**: CSV export with insights

## 📈 Key Features

### For Product Managers
- **Trend Identification**: Quick discovery of emerging issues
- **Product Comparison**: Cross-product complaint analysis
- **Evidence-Based Decisions**: Source-backed insights

### For Support Teams
- **Pain Point Understanding**: Real customer feedback analysis
- **Issue Prioritization**: Most frequent problems identification
- **Response Improvement**: Better understanding of customer needs

### For Compliance Teams
- **Risk Monitoring**: Proactive issue detection
- **Pattern Recognition**: Repeated violation identification
- **Regulatory Insights**: Compliance-related complaint trends

## 🎯 Sample Questions

The system can answer questions like:
- "What are the most common issues with credit cards?"
- "Why are customers unhappy with BNPL services?"
- "What problems do people face with money transfers?"
- "What billing issues do customers report?"
- "What fraud-related complaints exist?"

## 📊 Performance Metrics

- **Vector Store**: 392K+ complaint chunks
- **Coverage**: 5 financial product categories
- **Response Time**: <5 seconds for most queries
- **Source Transparency**: Always shows evidence
- **Quality Score**: Target 4.0+ average

## 🔍 Evaluation Results

Run the evaluation to see detailed performance metrics:
```bash
cd src
python evaluation.py
```

This generates:
- Quality scores for each test question
- Source relevance analysis
- System performance insights
- Recommendations for improvement

## 🚀 Next Steps

### For Task 3 Completion:
1. **Run Evaluation**: `python src/evaluation.py`
2. **Review Results**: Check `reports/evaluation_results.csv`
3. **Optimize Prompt**: Adjust prompt template if needed
4. **Test Edge Cases**: Try different question types

### For Task 4 Completion:
1. **Launch Interface**: `python app.py`
2. **Test User Experience**: Try sample questions
3. **Capture Screenshots**: For final report
4. **Document Features**: UI walkthrough

## 📝 Final Report Structure

Your final report should include:

### Technical Section
- **System Architecture**: RAG pipeline overview
- **Data Processing**: EDA findings and preprocessing steps
- **Model Choices**: Embedding and LLM selection rationale
- **Performance Metrics**: Evaluation results and analysis

### Business Section
- **Problem Statement**: CrediTrust's challenges
- **Solution Benefits**: Time savings and insights gained
- **Use Cases**: Real-world applications
- **ROI Impact**: Business value demonstration

### Implementation Section
- **Code Quality**: Clean, documented implementation
- **User Interface**: Screenshots and walkthrough
- **Deployment**: Production readiness assessment
- **Future Improvements**: Enhancement opportunities

## 🤝 Support

For questions or issues:
1. Check the test script: `python src/test_rag.py`
2. Review error messages in the console
3. Verify vector store files exist
4. Ensure all dependencies are installed

## 📄 License

This project is part of the CrediTrust Financial AI Challenge.