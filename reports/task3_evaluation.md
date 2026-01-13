# Task 3: RAG Pipeline Qualitative Evaluation

**Date**: 2026-01-09 23:59:44
**Vector Store**: vector_store/
**LLM Model**: google/flan-t5-base
**Total Questions**: 10

## Summary Statistics

- **Average Quality Score**: 3.70/5.0
- **Score Distribution**:
  - Score 5: 0 questions (0.0%)
  - Score 4: 7 questions (70.0%)
  - Score 3: 3 questions (30.0%)
  - Score 2: 0 questions (0.0%)
  - Score 1: 0 questions (0.0%)

## Quality Scoring Criteria

- **5 = Excellent**: Clear, grounded, actionable answer that synthesizes sources
- **4 = Good**: Relevant and grounded answer with minor gaps
- **3 = Partial**: Addresses question but vague or incomplete
- **2 = Weak**: Limited relevance or grounding
- **1 = Poor**: Irrelevant or hallucinated content

## Evaluation Results

| # | Question | Category | Generated Answer | Retrieved Sources | Score | Analysis |
|---|----------|----------|------------------|-------------------|-------|----------|
| 1 | What are the most common complaints about poor customer service? | General Dissatisfaction | [Document 1] Product: Money transfer, virtual currency, or money service | **[1]** *Money transfer, virtual currency, or money service - Other transaction problem*: . In one instance, their customer service representative hung up on me after waiting and being on...<br><br>**[2]** *Credit reporting, credit repair services, or other personal consumer reports - Incorrect information on your report*: are unfair and wrongful due to the poor service received. | **4**/5 | Answer addresses key question terms; Answer shows some grounding in data; Answer has appropriate length and detail; Retrieved sources may not fully match question |
| 2 | What issues do customers report about being unable to reach their bank or get responses? | General Dissatisfaction | Lack of received services | **[1]** *Checking or savings account - Problem with a lender or other company charging your account*: . Their customer service employees in the branches and by telephone refuse to provide telephone n...<br><br>**[2]** *Money transfer, virtual currency, or money service - Other transaction problem*: . In one instance, their customer service representative hung up on me after waiting and being on... | **3**/5 | Answer addresses key question terms; Answer shows some grounding in data; Retrieved sources may not fully match question |
| 3 | What are the main complaints about credit card billing errors? | Product-Specific: Credit Card | inaccurate account listings, multiple unauthorized inquiries, inaccurate credit reporting, inconsistent billing, and accounts that do not belong to me | **[1]** *Credit card - Other features, terms, or problems*: We appreciate your business.

Sincerely, Card Services If you haven't already done so, you must n...<br><br>**[2]** *Credit reporting, credit repair services, or other personal consumer reports - Incorrect information on your report*: a billing error solely because of the obligors failure to pay the amount indicated to be in error | **4**/5 | Answer addresses key question terms; Answer appears grounded in complaint data; Answer has appropriate length and detail; Retrieved sources are relevant to question |
| 4 | What problems do customers face with unauthorized credit card charges? | Product-Specific: Credit Card | Liability Limitation | **[1]** *Credit card or prepaid card - Other features, terms, or problems*: . When I worked in the financial industry, one of the things we saw that indicated possible fraud...<br><br>**[2]** *Credit reporting or other personal consumer reports - Improper use of your report*: Liability Limitation : Cardholders are generally liable for up to {$50.00} for unauthorized use o... | **4**/5 | Answer may lack relevance to question; Answer may lack grounding in sources; Retrieved sources are relevant to question |
| 5 | What complaints do customers have about mortgage loan servicing? | Product-Specific: Mortgage | a bad reputation for swindling people out of their money | **[1]** *Student loan - Dealing with your lender or servicer*: Seems like they have some incentive to make problems for their customers? What could this be? 

C...<br><br>**[2]** *Mortgage - Trouble during payment process*: I am being unfairly treated by my mortgage lender and have been a 1 1/2 years challenge with cust... | **4**/5 | Answer may lack relevance to question; Answer may lack grounding in sources; Answer has appropriate length and detail; Retrieved sources are relevant to question |
| 6 | What issues are reported about checking or savings account fees? | Product-Specific: Bank Account | $10.00 for taking money from savings to checking per transaction on at one m onth XXXX in fees and two to my savings account overdrafting my bank a... | **[1]** *Checking or savings account - Managing an account*: Had issues with deposits accurate accounting and fees<br><br>**[2]** *Checking or savings account - Problem caused by your funds being low*: . In the past week since complaining about the fees I have been charged XXXX  non sufficient fund... | **4**/5 | Answer addresses key question terms; Answer shows some grounding in data; Answer has appropriate length and detail; Retrieved sources are relevant to question |
| 7 | How do credit card complaints compare to checking account complaints? | Cross-Product Comparison | Credit card complaints are more common. | **[1]** *Credit reporting or other personal consumer reports - Problem with a company's investigation into an existing problem*: I am filing a complaint about multiple inaccurate or unverified accounts on my credit reports. I ...<br><br>**[2]** *Credit reporting or other personal consumer reports - Problem with a company's investigation into an existing problem*: I submitted disputes to the credit bureau over XXXX  days ago and I have not heard anything back ... | **3**/5 | Answer addresses key question terms; Answer shows some grounding in data; Retrieved sources may not fully match question |
| 8 | What complaints mention fraud, identity theft, or unauthorized account access? | Compliance/Fraud | "Based on the available complaint data, there is not enough information to answer this question." | **[1]** *Credit reporting, credit repair services, or other personal consumer reports - Incorrect information on your report*: I am XXXX XXXX XXXX and I am submitting this complaint and there is no third party involved. I am...<br><br>**[2]** *Credit reporting, credit repair services, or other personal consumer reports - Incorrect information on your report*: This is not a duplicate nor is this complaint filed by a third party. I am filing this complaint ... | **4**/5 | System correctly identified insufficient context; Answer may lack relevance to question; Answer shows some grounding in data; Answer has appropriate length and detail; Retrieved sources may not fully match question |
| 9 | What problems do customers report with debt collection practices? | Compliance/Fraud | failure to provide notification and/or validation of debt, and failure to provide notice of reporting to consumer reporting agencies | **[1]** *Debt collection - Written notification about debt*: I believe all of the above to be unfair and unethical collections practices - such as failure to ...<br><br>**[2]** *Debt collection - Took or threatened to take negative or legal action*: . XXXX, XXXX XXXX XXXX ( XXXX. XXXX XXXX. XXXX ), that debt collectors engaging in collection act... | **4**/5 | Answer addresses key question terms; Answer shows some grounding in data; Answer has appropriate length and detail; Retrieved sources are relevant to question |
| 10 | What are common complaints about account closures or frozen accounts? | Account Management | Account closures or frozen accounts | **[1]** *Credit reporting or other personal consumer reports - Incorrect information on your report*: These accounts were not opened or approved by me, and it is unfair for me to be blamed for them. ...<br><br>**[2]** *Credit card - Trouble using your card*: Unlawfully froze consumer accounts and mispresented fee waivers : The bank froze more than XXXX X... | **3**/5 | Answer addresses key question terms; Answer shows some grounding in data; Retrieved sources may not fully match question |

## Key Findings

The RAG pipeline achieved an average quality score of **3.70/5.0** across 10 evaluation questions.

**Overall Assessment**: The RAG system provides partially useful answers but has room for improvement. Some answers lack specificity or fail to fully leverage retrieved sources. Consider tuning retrieval parameters or using a more powerful LLM.

## Recommendations

1. **Retrieval Optimization**: Experiment with different k values and similarity thresholds
2. **Prompt Engineering**: Refine system prompt to improve answer structure and clarity
3. **LLM Upgrade**: Consider using a larger model (e.g., Mistral-7B) for better synthesis
4. **Metadata Filtering**: Add product/issue filters to improve retrieval precision
5. **Answer Post-Processing**: Implement answer validation and formatting rules

---

*This evaluation was conducted programmatically using the RAG pipeline defined in `src/rag_pipeline.py`. Quality scores are based on relevance, groundedness, clarity, and completeness criteria.*