import json
import os
from audit_engine import setup_dual_indices, run_compliance_audit

def run_inference(input_file="synthetic_golden_dataset.json", output_file="evaluation_input.json"):
    # 1. Load the Golden Dataset
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run generate_testset.py first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 2. Setup the RAG Indices (GDPR Regs + Contract)
    print("📂 Initializing RAG indices and loading models...")
    reg_retriever, target_vdb = setup_dual_indices()
    evaluation_ready_data = []

    print(f"🕵️ Starting Inference on {len(golden_data)} samples...")

    for i, item in enumerate(golden_data):
        print(f"[{i+1}/{len(golden_data)}] Processing: {item['user_input'][:40]}...")

        # 3. Map Golden Data to the 'Checkpoint' format your engine expects
        checkpoint = {"query": item['user_input'], "reference": item['reference'], "name": f"Audit_{i}"}

        # 4. Run the Audit Engine (using your new XML prompt)
        try:
            analysis, reg_chunks = run_compliance_audit(reg_retriever, target_vdb, checkpoint)
            
            res = {k.lower(): v for k, v in analysis.items()}
            # Provide only content for judge
            reasoning = res.get('reasoning', "No reasoning found.")
            remedy = res.get('rememdy', res.get('remedy', "Check documentation."))
            
            # This combined string is what the Faithfulness metric will judge
            verifiable_answer = f"{reasoning} Suggested fix: {remedy}".strip()
            
            # Ensure reg_chunks is a list before attempting comprehension
            if isinstance(reg_chunks, list):
                context_strings = [doc.page_content for doc in reg_chunks]
            else:
                # If reg_chunks is somehow still a string, wrap it in a list
                context_strings = [str(reg_chunks)]

            # 5. Build the final object for the Evaluator
            eval_item = {
                "user_input": item['user_input'],           # The Question
                "answer": verifiable_answer,                     # The Bot's response
                "reference_contexts": context_strings,   # The retrieved context
                "reference": item['reference']             # The Ground Truth
            }
            evaluation_ready_data.append(eval_item)
            
        except Exception as e:
            print(f"⚠️ Error processing sample {i}: {e}")

    # 6. Save the results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_ready_data, f, indent=4)

    print(f"\n✅ Inference Complete! File saved as: {output_file}")
    print(f"Next step: python evaluator.py --file {output_file}")

if __name__ == "__main__":
    run_inference()