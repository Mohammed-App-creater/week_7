"""
Qualitative Evaluation of RAG Pipeline
Task 3: Evaluate RAG system with representative questions and quality scoring.

Author: Data & AI Engineer
Date: 2026-01-09
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Qualitative evaluation of RAG system.
    
    Evaluates RAG pipeline with representative questions covering:
    - General product dissatisfaction
    - Product-specific issues
    - Cross-product comparisons
    - Compliance/fraud signals
    
    Quality scoring (1-5 scale):
    - 1 = Poor (irrelevant/hallucinated)
    - 3 = Partial (vague/incomplete)
    - 5 = Clear (grounded/actionable)
    """
    
    def __init__(
        self,
        vector_store_path: str = "vector_store/",
        llm_model: str = "google/flan-t5-base",
        output_path: str = "reports/task3_evaluation.md"
    ):
        """
        Initialize evaluator.
        
        Args:
            vector_store_path: Path to ChromaDB vector store
            llm_model: HuggingFace LLM model
            output_path: Path to save evaluation report
        """
        self.vector_store_path = vector_store_path
        self.llm_model = llm_model
        self.output_path = Path(output_path)
        
        # Initialize RAG pipeline
        self.pipeline = RAGPipeline(
            vector_store_path=vector_store_path,
            llm_model=llm_model,
            device="cpu"
        )
        
        # Evaluation results
        self.results = []
    
    def define_evaluation_questions(self) -> List[Dict[str, str]]:
        """
        Define representative evaluation questions.
        
        Returns:
            List of dicts with 'question' and 'category' keys
        """
        questions = [
            {
                "question": "What are the most common complaints about poor customer service?",
                "category": "General Dissatisfaction"
            },
            {
                "question": "What issues do customers report about being unable to reach their bank or get responses?",
                "category": "General Dissatisfaction"
            },
            {
                "question": "What are the main complaints about credit card billing errors?",
                "category": "Product-Specific: Credit Card"
            },
            {
                "question": "What problems do customers face with unauthorized credit card charges?",
                "category": "Product-Specific: Credit Card"
            },
            {
                "question": "What complaints do customers have about mortgage loan servicing?",
                "category": "Product-Specific: Mortgage"
            },
            {
                "question": "What issues are reported about checking or savings account fees?",
                "category": "Product-Specific: Bank Account"
            },
            {
                "question": "How do credit card complaints compare to checking account complaints?",
                "category": "Cross-Product Comparison"
            },
            {
                "question": "What complaints mention fraud, identity theft, or unauthorized account access?",
                "category": "Compliance/Fraud"
            },
            {
                "question": "What problems do customers report with debt collection practices?",
                "category": "Compliance/Fraud"
            },
            {
                "question": "What are common complaints about account closures or frozen accounts?",
                "category": "Account Management"
            }
        ]
        
        logger.info(f"Defined {len(questions)} evaluation questions")
        return questions
    
    def score_quality(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> tuple[int, str]:
        """
        Manually score answer quality and provide analysis.
        
        Scoring criteria:
        - Relevance: Does answer address the question?
        - Groundedness: Is answer supported by sources?
        - Clarity: Is answer clear and actionable?
        - Completeness: Does answer synthesize multiple sources?
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Retrieved source documents
        
        Returns:
            Tuple of (score, analysis)
        """
        score = 3  # Default: Partial
        analysis_parts = []
        
        # Check for "not enough information" response
        if "not enough information" in answer.lower():
            score = 2
            analysis_parts.append("System correctly identified insufficient context")
        
        # Check relevance (basic keyword matching)
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Extract key terms from question
        key_terms = []
        if "credit card" in question_lower:
            key_terms.append("credit card")
        if "mortgage" in question_lower or "loan" in question_lower:
            key_terms.append("loan")
        if "fraud" in question_lower or "identity theft" in question_lower:
            key_terms.append("fraud")
        if "debt collection" in question_lower:
            key_terms.append("debt")
        if "billing" in question_lower:
            key_terms.append("billing")
        if "fees" in question_lower or "fee" in question_lower:
            key_terms.append("fee")
        if "account" in question_lower:
            key_terms.append("account")
        
        # Check if answer addresses key terms
        relevant_terms = [term for term in key_terms if term in answer_lower]
        
        if len(relevant_terms) >= len(key_terms) * 0.5:
            analysis_parts.append("Answer addresses key question terms")
            score = max(score, 3)
        else:
            analysis_parts.append("Answer may lack relevance to question")
            score = min(score, 2)
        
        # Check groundedness (answer should reference complaint themes)
        grounded_indicators = [
            "complaint", "customer", "issue", "problem", "report",
            "unauthorized", "error", "charge", "service", "account"
        ]
        grounded_count = sum(1 for indicator in grounded_indicators if indicator in answer_lower)
        
        if grounded_count >= 3:
            analysis_parts.append("Answer appears grounded in complaint data")
            score = max(score, 4)
        elif grounded_count >= 1:
            analysis_parts.append("Answer shows some grounding in data")
        else:
            analysis_parts.append("Answer may lack grounding in sources")
            score = min(score, 2)
        
        # Check clarity (length and structure)
        if len(answer) < 20:
            analysis_parts.append("Answer is very brief")
            score = min(score, 2)
        elif len(answer) > 50 and len(answer) < 300:
            analysis_parts.append("Answer has appropriate length and detail")
            score = max(score, 4)
        elif len(answer) >= 300:
            analysis_parts.append("Answer is comprehensive")
            score = max(score, 5)
        
        # Check for hallucination indicators
        hallucination_indicators = [
            "according to", "studies show", "research indicates",
            "it is known that", "experts say", "generally speaking"
        ]
        if any(indicator in answer_lower for indicator in hallucination_indicators):
            analysis_parts.append("WARNING: Possible hallucination detected")
            score = min(score, 2)
        
        # Check source relevance
        if sources:
            source_products = [s['metadata'].get('product', '') for s in sources[:2]]
            source_issues = [s['metadata'].get('issue', '') for s in sources[:2]]
            
            # Check if sources align with question
            source_relevant = False
            for term in key_terms:
                for product in source_products:
                    if term in product.lower():
                        source_relevant = True
                for issue in source_issues:
                    if term in issue.lower():
                        source_relevant = True
            
            if source_relevant:
                analysis_parts.append("Retrieved sources are relevant to question")
                score = max(score, 4)
            else:
                analysis_parts.append("Retrieved sources may not fully match question")
        
        # Final score adjustment
        score = max(1, min(5, score))  # Clamp to 1-5
        
        analysis = "; ".join(analysis_parts)
        
        return score, analysis
    
    def evaluate_question(
        self,
        question: str,
        category: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline on a single question.
        
        Args:
            question: Question to evaluate
            category: Question category
            k: Number of documents to retrieve
        
        Returns:
            Dict with evaluation results
        """
        logger.info(f"\nEvaluating: {question}")
        logger.info(f"Category: {category}")
        
        # Run RAG pipeline
        result = self.pipeline.answer_question(question, k=k)
        
        answer = result['answer']
        sources = result['sources']
        
        # Score quality
        score, analysis = self.score_quality(question, answer, sources)
        
        logger.info(f"Score: {score}/5")
        logger.info(f"Analysis: {analysis}")
        
        # Format evaluation result
        eval_result = {
            'question': question,
            'category': category,
            'answer': answer,
            'sources': sources[:2],  # Top 2 sources
            'score': score,
            'analysis': analysis
        }
        
        return eval_result
    
    def run_evaluation(self):
        """Run full evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING RAG EVALUATION")
        logger.info("=" * 80)
        
        # Load RAG components
        logger.info("\nLoading RAG pipeline...")
        self.pipeline.load_vector_store()
        self.pipeline.load_llm()
        
        # Get evaluation questions
        questions = self.define_evaluation_questions()
        
        # Evaluate each question
        logger.info(f"\nEvaluating {len(questions)} questions...")
        
        for i, q in enumerate(questions, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {i}/{len(questions)}")
            logger.info(f"{'='*80}")
            
            result = self.evaluate_question(
                question=q['question'],
                category=q['category'],
                k=5
            )
            
            self.results.append(result)
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
    
    def generate_report(self):
        """Generate Markdown evaluation report."""
        logger.info(f"\nGenerating evaluation report: {self.output_path}")
        
        # Calculate summary statistics
        scores = [r['score'] for r in self.results]
        avg_score = sum(scores) / len(scores) if scores else 0
        score_dist = {i: scores.count(i) for i in range(1, 6)}
        
        # Build report
        report_lines = [
            "# Task 3: RAG Pipeline Qualitative Evaluation",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Vector Store**: {self.vector_store_path}",
            f"**LLM Model**: {self.llm_model}",
            f"**Total Questions**: {len(self.results)}",
            "",
            "## Summary Statistics",
            "",
            f"- **Average Quality Score**: {avg_score:.2f}/5.0",
            f"- **Score Distribution**:",
        ]
        
        for score in range(5, 0, -1):
            count = score_dist.get(score, 0)
            percentage = (count / len(scores) * 100) if scores else 0
            report_lines.append(f"  - Score {score}: {count} questions ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## Quality Scoring Criteria",
            "",
            "- **5 = Excellent**: Clear, grounded, actionable answer that synthesizes sources",
            "- **4 = Good**: Relevant and grounded answer with minor gaps",
            "- **3 = Partial**: Addresses question but vague or incomplete",
            "- **2 = Weak**: Limited relevance or grounding",
            "- **1 = Poor**: Irrelevant or hallucinated content",
            "",
            "## Evaluation Results",
            "",
            "| # | Question | Category | Generated Answer | Retrieved Sources | Score | Analysis |",
            "|---|----------|----------|------------------|-------------------|-------|----------|"
        ])
        
        # Add each evaluation result
        for i, result in enumerate(self.results, 1):
            question = result['question']
            category = result['category']
            answer = result['answer']
            sources = result['sources']
            score = result['score']
            analysis = result['analysis']
            
            # Truncate answer if too long
            answer_display = answer if len(answer) <= 150 else answer[:147] + "..."
            
            # Format sources (top 2)
            source_parts = []
            for j, source in enumerate(sources, 1):
                product = source['metadata'].get('product', 'Unknown')
                issue = source['metadata'].get('issue', 'Unknown')
                text = source['text']
                text_display = text if len(text) <= 100 else text[:97] + "..."
                source_parts.append(f"**[{j}]** *{product} - {issue}*: {text_display}")
            
            sources_display = "<br><br>".join(source_parts)
            
            # Add row
            report_lines.append(
                f"| {i} | {question} | {category} | {answer_display} | {sources_display} | **{score}**/5 | {analysis} |"
            )
        
        # Add conclusion
        report_lines.extend([
            "",
            "## Key Findings",
            "",
            f"The RAG pipeline achieved an average quality score of **{avg_score:.2f}/5.0** across {len(self.results)} evaluation questions.",
            "",
        ])
        
        if avg_score >= 4.0:
            report_lines.append("**Overall Assessment**: The RAG system performs well, providing grounded and actionable answers for most queries. Retrieved sources are generally relevant and the LLM effectively synthesizes complaint data.")
        elif avg_score >= 3.0:
            report_lines.append("**Overall Assessment**: The RAG system provides partially useful answers but has room for improvement. Some answers lack specificity or fail to fully leverage retrieved sources. Consider tuning retrieval parameters or using a more powerful LLM.")
        else:
            report_lines.append("**Overall Assessment**: The RAG system shows significant limitations. Answers often lack relevance or grounding. Recommend reviewing prompt engineering, retrieval quality, and LLM selection.")
        
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            "1. **Retrieval Optimization**: Experiment with different k values and similarity thresholds",
            "2. **Prompt Engineering**: Refine system prompt to improve answer structure and clarity",
            "3. **LLM Upgrade**: Consider using a larger model (e.g., Mistral-7B) for better synthesis",
            "4. **Metadata Filtering**: Add product/issue filters to improve retrieval precision",
            "5. **Answer Post-Processing**: Implement answer validation and formatting rules",
            "",
            "---",
            "",
            "*This evaluation was conducted programmatically using the RAG pipeline defined in `src/rag_pipeline.py`. Quality scores are based on relevance, groundedness, clarity, and completeness criteria.*"
        ])
        
        # Write report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✓ Report saved to: {self.output_path}")
        logger.info(f"  Average Score: {avg_score:.2f}/5.0")
    
    def run(self):
        """Execute full evaluation and reporting pipeline."""
        self.run_evaluation()
        self.generate_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Report: {self.output_path.absolute()}")


def main():
    """Main execution function."""
    evaluator = RAGEvaluator(
        vector_store_path="vector_store/",
        llm_model="google/flan-t5-base",
        output_path="reports/task3_evaluation.md"
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()
