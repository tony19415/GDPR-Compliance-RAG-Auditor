import json
import os
import argparse
import sys
import asyncio
import pandas as pd
from openai import OpenAI
from datasets import Dataset
from datetime import datetime
from ragas import evaluate
# from ragas.metrics.collections import (
#     Faithfulness, 
#     AnswerRelevancy, 
#     ContextPrecision, 
#     ContextRecall
# )
from ragas.metrics import (
    Faithfulness, 
    AnswerRelevancy, 
    ContextPrecision, 
    ContextRecall
)

from ragas.run_config import RunConfig

# try:
#     from ragas.llms import LangchainLLMWrapper
#     from ragas.embeddings import LangchainEmbeddingsWrapper
# except ImportError:
#     try:
#         from ragas.integrations.langchain import LangchainLLMWrapper, LangchainEmbeddingsWrapper
#     except ImportError:
#         print("❌ Critical Error: Could not find Langchain Wrappers. Please run: pip install ragas --upgrade")
#         sys.exit(1)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Windows Stability Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def run_ragas_evaluation(data_list, threshold=0.7):
    """
    Runs a stable Ragas evaluation using local Ollama models.
    """

    # 2. DATA MAPPING
    # Ensuring we handle keys from both manual audits and synthetic testsets
    data_sample = {
        "question": [item.get('user_input', '') for item in data_list],
        "answer": [item.get('answer', 'MISSING_RAG_ANSWER') for item in data_list], 
        "contexts": [item.get('reference_contexts', []) for item in data_list],
        "ground_truth": [item.get('reference', '') for item in data_list]
    }

    dataset = Dataset.from_dict(data_sample)

    # 3. SETUP LOCAL JUDGE
    print("Connecting to Local Judge (Llama 3.1) and Embedding Model...")
    
    raw_llm = ChatOllama(model="mistral-nemo:12b-instruct-2407-q8_0", temperature=0)
    raw_embeddings = OllamaEmbeddings(model="llama3.2:1b")

    eval_llm = LangchainLLMWrapper(raw_llm)
    eval_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    # 5. INITIALIZE SCORER OBJECTS
    # We pass the wrapped models directly to satisfy the Scorer interface
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm)
    ]

    # 6. STABILITY CONFIG
    # Sequential processing (max_workers=1) is vital for local hardware stability.
    config = RunConfig(max_workers=1, timeout=900)

    print(f"🚀 Running Quality Audit on {len(data_list)} GDPR samples...")
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=config
        )
        df = result.to_pandas()
    except Exception as e:
        print(f"❌ Evaluation Loop failed: {e}")
        return pd.DataFrame(), False
    
    # 7. ANALYZE RESULTS
    df = df.fillna(0.0)
    avg_faithfulness = df['faithfulness'].mean()
    
    print("\n" + "="*40)
    print("        GDPR AUDIT PERFORMANCE")
    print("="*40)
    print(f"Avg Faithfulness:     {avg_faithfulness:.2f}")
    print(f"Avg Answer Relevancy: {df['answer_relevancy'].mean():.2f}")
    print(f"Avg Context Recall:   {df['context_recall'].mean():.2f}")
    print("="*40)

    return df, (avg_faithfulness >= threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDPR RAG Evaluator (CI/CD Edition)")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum Faithfulness score")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results_df, is_pass = run_ragas_evaluation(data, args.threshold)
        
        if not results_df.empty:
            # Save results with dynamic timestamp
            date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
            output_name = f"audit_quality_scores_{date_str}.csv" 
            results_df.to_csv(output_name, index=False)
            print(f"✅ Report saved to {output_name}")
            sys.exit(0 if is_pass else 1)