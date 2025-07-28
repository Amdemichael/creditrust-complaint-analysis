import pandas as pd
import numpy as np
from rag_pipeline import RAGPipeline, create_evaluation_questions
import os

class RAGEvaluator:
    def __init__(self, rag_pipeline):
        """
        Initialize the evaluator with a RAG pipeline.
        
        Args:
            rag_pipeline: Initialized RAGPipeline instance
        """
        self.rag = rag_pipeline
        self.evaluation_results = []
    
    def evaluate_question(self, question, expected_aspects=None):
        """
        Evaluate a single question and score the response.
        
        Args:
            question: Test question
            expected_aspects: List of aspects that should be covered in the answer
            
        Returns:
            Dictionary with evaluation results
        """
        # Get RAG response
        result = self.rag.answer_question(question)
        
        # Analyze response quality
        answer = result['answer']
        sources = result['sources']
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(answer, sources, expected_aspects)
        
        # Create evaluation result
        evaluation = {
            'question': question,
            'generated_answer': answer,
            'retrieved_sources': sources[:2],  # Top 2 sources for display
            'quality_score': quality_score,
            'source_count': len(sources),
            'avg_similarity': np.mean([s['similarity_score'] for s in sources]),
            'comments': self._generate_comments(answer, sources, quality_score)
        }
        
        return evaluation
    
    def _calculate_quality_score(self, answer, sources, expected_aspects=None):
        """
        Calculate a quality score (1-5) for the generated answer.
        
        Args:
            answer: Generated answer text
            sources: Retrieved source chunks
            expected_aspects: Expected aspects to cover
            
        Returns:
            Quality score from 1-5
        """
        score = 3  # Base score
        
        # Check answer length (not too short, not too long)
        if len(answer) < 50:
            score -= 1
        elif len(answer) > 500:
            score -= 0.5
        
        # Check if answer is relevant to question
        if any(word in answer.lower() for word in ['credit', 'loan', 'payment', 'billing', 'fraud', 'service']):
            score += 0.5
        
        # Check source quality
        avg_similarity = np.mean([s['similarity_score'] for s in sources])
        if avg_similarity > 0.7:
            score += 0.5
        elif avg_similarity < 0.3:
            score -= 0.5
        
        # Check for specific financial terms
        financial_terms = ['complaint', 'issue', 'problem', 'customer', 'service', 'billing']
        if any(term in answer.lower() for term in financial_terms):
            score += 0.5
        
        # Ensure score is within bounds
        return max(1, min(5, score))
    
    def _generate_comments(self, answer, sources, quality_score):
        """
        Generate analysis comments for the evaluation.
        
        Args:
            answer: Generated answer
            sources: Retrieved sources
            quality_score: Calculated quality score
            
        Returns:
            Analysis comments
        """
        comments = []
        
        if quality_score >= 4:
            comments.append("High quality response with relevant sources")
        elif quality_score >= 3:
            comments.append("Good response with some room for improvement")
        else:
            comments.append("Response needs improvement in relevance or detail")
        
        # Check source diversity
        products = set(s['product'] for s in sources)
        if len(products) > 1:
            comments.append("Good source diversity across products")
        else:
            comments.append("Limited source diversity")
        
        # Check answer specificity
        if len(answer) < 100:
            comments.append("Answer could be more detailed")
        elif len(answer) > 300:
            comments.append("Comprehensive answer provided")
        
        return "; ".join(comments)
    
    def run_evaluation(self, questions=None):
        """
        Run evaluation on a set of questions.
        
        Args:
            questions: List of questions to evaluate (uses default if None)
            
        Returns:
            DataFrame with evaluation results
        """
        if questions is None:
            questions = create_evaluation_questions()
        
        print(f"Running evaluation on {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"Evaluating question {i}/{len(questions)}: {question[:50]}...")
            result = self.evaluate_question(question)
            self.evaluation_results.append(result)
        
        return pd.DataFrame(self.evaluation_results)
    
    def generate_evaluation_report(self, output_file='../reports/evaluation_results.csv'):
        """
        Generate and save evaluation report.
        
        Args:
            output_file: Path to save the evaluation results
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluation first.")
            return
        
        # Create detailed report
        report_data = []
        for result in self.evaluation_results:
            report_data.append({
                'Question': result['question'],
                'Generated Answer': result['generated_answer'],
                'Quality Score': result['quality_score'],
                'Source Count': result['source_count'],
                'Avg Similarity': round(result['avg_similarity'], 3),
                'Comments': result['comments']
            })
        
        df_report = pd.DataFrame(report_data)
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_report.to_csv(output_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Questions: {len(self.evaluation_results)}")
        print(f"Average Quality Score: {np.mean([r['quality_score'] for r in self.evaluation_results]):.2f}")
        print(f"Average Source Count: {np.mean([r['source_count'] for r in self.evaluation_results]):.1f}")
        print(f"Average Similarity: {np.mean([r['avg_similarity'] for r in self.evaluation_results]):.3f}")
        
        # Show top and bottom performers
        sorted_results = sorted(self.evaluation_results, key=lambda x: x['quality_score'], reverse=True)
        print(f"\nBest Question: {sorted_results[0]['question'][:50]}... (Score: {sorted_results[0]['quality_score']})")
        print(f"Worst Question: {sorted_results[-1]['question'][:50]}... (Score: {sorted_results[-1]['quality_score']})")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return df_report

def create_custom_evaluation_questions():
    """
    Create additional custom evaluation questions for specific scenarios.
    
    Returns:
        List of custom evaluation questions
    """
    custom_questions = [
        "What are the most urgent complaints that need immediate attention?",
        "How do complaints vary by product category?",
        "What are the common themes in customer service complaints?",
        "What technical issues do users frequently report?",
        "What billing and payment problems are most common?",
        "How do fraud-related complaints differ across products?",
        "What are the most frustrating experiences customers describe?",
        "What improvements do customers suggest?",
        "What are the most expensive complaints to resolve?",
        "How do complaint patterns change over time?"
    ]
    return custom_questions

if __name__ == "__main__":
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    print("Running evaluation...")
    evaluator.run_evaluation()
    
    # Generate report
    evaluator.generate_evaluation_report() 