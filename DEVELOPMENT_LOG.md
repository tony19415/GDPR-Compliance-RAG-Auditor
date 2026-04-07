# GDPR RAG Optimization & Debugging Log
**Project Goal:** Build a high fidelity RAG pipeline to audit legal contracts against GDPR regulations using local LLM stack (Llama 3.1B & Mistral 12B)

## 1. Silent Crash (JSON Parsing Failure)
- **Issue:** Initial evaluation scores for Faithfulness and Answer Relevancy was 0.00
- **Cause:** Auditor (Llama 8B) failed to output perfect JSON. Code hig except block, returning generic fallback string: *"No reasoning provided."* Since this string contained no legal facts, judge couldn't verify it against the source text.
- **Solution:** Regex Sniffer. Impelmented ```re.search(r'\{.*\}', content, re.DOTALL)``` to extract JSON format from model's conversational chat.
    - Key Normalization: Added dictionary comprehension ```{k.lower(): v for k, v in raw_data.items()}``` to handle inconsistent capitalization ("Reasoning" vs "reasoning").
    - One-Shot Prompting: Added concrete JSON example in the prompt to guide the 8B model's output format

## 2. Precision Paradox (Context Noise)
- **Issue:** Context Precision stuck at 0.00
- **Cause:** Retriever pulling in the Vendor X contract snippet for every query, regardless of relevance. Regas penalizes score when top-ranked retrieved chunk is irrelevant noise.
- **Solution:** Context isolation, refined the system prompt to explictly separate <gdpr_content> from <contracT_context>

## 3. Recall Gap (Chunking & Search Depth)
- **Issue:** Context Recall plateaued at 0.70
- **Cause:** 30% of required legal articles were missing from context. Happened due to legal definitions being cut off by page breaks during chunking.
- **Solution:** Increased chunk_overlap from 200 to 500 in get_clean_chunks process to ensure logical bridge between PDF pages. Increased EnsembleRetriever search depth from k=5 to k=8.

## 4. Type Safety Mismatch
- **Issue:** ```TypeError``` and ```AttributeError```
- **Cause:** Mismatch return types between audit_engine.py and run_inference.py. Audit fallback was returning a string, but inference loop was trying to call .page_content on it (expecting a LangChain Document Object).
- **Solution:** Hardened the run_compliance_audit function to always return a list of Documents, even in the except block. This allowed datat lineage to remain unbroken throughout the loop.

## 5. Library Dependency
- **Issue:** ```TypeError: Chroma.__init__() got an unexpected keyword argument 'embeddings_function'```
- **Cause:** Version mismatch in ```langchain-chroma``` library
- **Solution:** Standardized parameters to ```embedding=embeddings``` (singular) for the from_documents class methods

## 6. Dependency Management & Library Modularization
- **Issue:** `ModuleNotFoundError: No module named 'langchain.retrievers'`
- **Cause:** Evolution of the LangChain ecosystem (v0.2+) into modular packages. High-level retriever logic (Contextual Compression) was decoupled from specific vector store integrations to reduce package bloat.
- **Solution:** Explicitly installed the base `langchain` and `langchain-community` packages to restore the retrieval orchestration layer. 
- **Engineering Insight:** In modern AI development, "Core" logic and "Third-party" integrations are often separate. Managing a specific dependency tree is as critical as the model logic itself.

## 7. Vector Dimension Mismatch & Schema Rigidity
- **Issue:** `chromadb.errors.InvalidArgumentError: Collection expecting dimension of 2048, got 1024`.
- **Cause:** Attempted to swap embedding models (`llama3.2:1b` at 2048 dims to `mxbai-embed-large` at 1024 dims) without clearing the persistent vector store. Vector databases require a fixed dimensionality for all vectors within a single collection.
- **Solution:** Performed a "Schema Refresh" by deleting the `chroma_db/` directory and allowing the ETL pipeline to re-index the regulations with the new 1024-dim embeddings.
- **Engineering Insight:** Embedding models are not "plug-and-play" once data is persisted. Versioning your vector indices or implementing an automated "drop and recreate" logic during model updates is essential for RAG maintenance.

## 8. Pydantic Validation & Parameter Strictness
- **Issue:** `pydantic_core._pydantic_core.ValidationError: Extra inputs are not permitted [input_value='ms-marco-MiniLM-L-12-v2']`.
- **Cause:** Attempted to initialize the `FlashrankRerank` compressor using `model_name` instead of the expected `model` parameter. Modern LangChain components use Pydantic V2, which defaults to strict validation, forbidding any undeclared keyword arguments.
- **Solution:** Updated the class initialization to use the correct `model` parameter name.
- **Engineering Insight:** When working with Pydantic-based libraries, always check the exact attribute names in the class definition. "Close enough" names (like `model_name` vs `model`) will trigger validation failures rather than being ignored.

## 9. Version-Specific Class Naming & Parameters
- **Issue:** `ImportError: cannot import name 'FlashrankReranker'`.
- **Cause:** Discrepancy between documentation and specific library versioning. In this environment, the class is named `FlashrankRerank` (extractive) rather than `FlashrankReranker` (agentic).
- **Secondary Issue:** `ValidationError` regarding `model_name`.
- **Solution:** 1. Reverted class name to `FlashrankRerank`.
    2. Switched initialization argument from `model_name` to `model` to comply with Pydantic V2 schema requirements.
- **Engineering Insight:** When working with rapidly evolving frameworks like LangChain, the `__init__.py` file of the package is the "Source of Truth" over third-party tutorials. Reading the traceback's "Did you mean..." suggestions is often the fastest path to resolution.

## 10. Pydantic Strictness & Parameter Aliasing
- **Issue:** `pydantic_core._pydantic_core.ValidationError: Extra inputs are not permitted`.
- **Cause:** The `FlashrankRerank` class (provided by `langchain-community`) expects the parameter `model` rather than `model_name`. Because the class is a Pydantic model with `extra='forbid'`, any deviation in naming—even a standard alias like `model_name`—results in a hard crash.
- **Solution:** Aligned the initialization call with the library's internal schema by renaming the keyword argument to `model`.
- **Engineering Insight:** In production RAG systems, library updates frequently change parameter names to standardize schemas. Always verify the constructor's expected arguments in the source code or local `site-packages` when encountering Pydantic validation errors.

## 11. Bridging the "Recall Gap" (0.60 -> 0.85+)
- **Issue:** Avg Context Recall plateaued at 0.60. The system was "Faithful" (0.89) but missed specific nuances required by the Ground Truth.
- **Root Cause:** Initial retrieval ($k=10$) was too narrow. In complex EU Regulations (Data Act/GDPR), relevant answers often span multiple non-consecutive articles.
- **Solution:** 1. **Wide-Funnel Retrieval:** Doubled the base retrieval depth ($k=20$) to capture a broader candidate set.
    2. **Semantic Weighting:** Adjusted Ensemble weights to 0.5/0.5 to give equal priority to semantic context (Vector) and exact terminology (BM25).
    3. **Reranker Compression:** Maintained `top_n=5` at the reranker level to prevent "Long Context Fatigue" in the Auditor LLM.
- **Engineering Insight:** High Faithfulness with low Recall indicates a "shy" retriever. Widening the initial search while keeping the reranker's output tight is the standard pattern for resolving this.

## 12. Achieving Zero-Hallucination (Faithfulness 1.00)
- **Status:** Avg Faithfulness reached 1.00 across all test samples.
- **Intervention:** Combined One-Shot Prompting with a "Strictly Extractive" role definition.
- **Result:** The Auditor (Llama 3.1 8B) now refuses to speculate, citing only the provided context. This is the baseline requirement for a legal compliance tool.

## 13. Identifying the "Recall Ceiling"
- **Issue:** Avg Context Recall plateaued at 0.60.
- **Diagnosis:** Semantic search was identifying the "Correct Topic" (e.g., Supervisory Authorities) but failing to capture "Specific Sub-Articles" (e.g., Enforcing Judgments) required by the Ground Truth.
- **Solution:** Implemented a **"Wide-Funnel" Retrieval** strategy ($k=20$ candidates) followed by **Cross-Encoder Reranking** ($top\_n=5$). 
- **Analytics Insight:** Increasing the search depth ($k$) provides the necessary "surface area" to find obscure legal clauses, while the Reranker prevents the LLM from becoming overwhelmed by irrelevant noise.

## 14. Implementing Privacy by Design (PII Sanitization)
- **Problem:** Storing raw PII (Names/Emails) in a Vector Database creates a data leakage risk and violates the principle of "Data Minimization" (GDPR Art. 5).
- **Solution:** Integrated a **Pre-embedding Sanitization Layer** using Regex pattern matching.
    - **Emails:** Replaced with `[PRIVATE_EMAIL]` to prevent identification.
    - **Names:** Implemented a detection pattern for document headers to mask specific representatives.
- **Architectural Choice:** Sanitization occurs *before* the `RecursiveCharacterTextSplitter`. This ensures that no PII ever reaches the `OllamaEmbeddings` model or the `ChromaDB` persistent storage.
- **Engineering Insight:** Embedding models can be vulnerable to "Inversion Attacks." Redacting at the source is the only way to ensure the Vector DB remains GDPR-compliant by default.

## Metric Evolution Summary
| Metric            | Baseline(initial) | Intermediate | Final(Target) |
|-------------------|-------------------|--------------|---------------|
| Faithfulness      | 0.00              | 0.30         | 0.72-0.90     |
| Answer Relevancy  | 0.00              | 0.28         | 0.79-0.85     |
| Context Recall    | 0.50              | 0.70         | 0.75-1.00     |
| Context Precision | 0.00              | 0.00         | 0.85+         |

## Engineering Takeaways
1. **Fail-Safe fallbacks:** Never return generic error string, return a structured object that maintains the expected data type to prevent cascading crashes
2. **Judge Model Selection:** Using larger model (Mistral 12B) to judge a smaller model (Llama 8B) is essential for catching subtle legal hallucinations that an 8B model might overlook
3. **Prose the Judges:** Evaluators like Ragas work best when "Answer" is clean not tagged metatdata like ("STATUS:FAIL"). Removing tags significantly improves **Answer Relevancy** scores.