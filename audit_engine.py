import os
import yaml
import re
import json
import hashlib
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import JsonOutputParser
# from langchain_classic.retrievers import ContextualCompressionRetriever
# from langchain_community.document_compressors import FlashrankRerank
try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

from langchain_community.document_compressors import FlashrankRerank # No 'er' at the end

# Schema Definition
class AuditResult(BaseModel):
    status: str = Field(description="Either 'PASS', 'FAIL', or 'UNCLEAR'")
    violation_found: bool = Field(description="True if the contract contradicts GDPR")
    reasoning: str = Field(description="Brief legal explanation for the status")
    rememdy: str = Field(description="Specific advice on how to fix the clause")

def redact_pii(text, mode="redact"):
    """
    Sanitizes text by removing or hashing emails and potential names.
    mode: 'redact' (replaces with [EMAIL]) or 'hash' (replaces with a unique ID)
    """
    # 1. Handle Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    
    if mode == "hash":
        # Hashing allows you to see if the same person is mentioned multiple times 
        # without knowing who they are.
        def hash_match(match):
            return hashlib.md5(match.group(0).encode()).hexdigest()[:8]
        text = re.sub(email_pattern, hash_match, text)
    else:
        text = re.sub(email_pattern, "[PRIVATE_EMAIL]", text)

    # 2. Handle Names (Pattern-based for common "Contact: Name" headers)
    # Note: For full names in prose, you'd usually use spaCy or Microsoft Presidio.
    # This is a lightweight regex for "John Doe" style patterns in headers.
    name_header_pattern = r'(?i)(name|contact|representative):\s*([A-Z][a-z]+ [A-Z][a-z]+)'
    text = re.sub(name_header_pattern, r'\1: [REDACTED_NAME]', text)

    return text
def clean_legal_text(text: str) -> str:
    """ETL: Cleaning raw PDF noise for high-fidelity RAG."""
    # Remove 'Page X' or 'Page X of Y'
    text = re.sub(r'Page \d+( of \d+)?', '', text, flags=re.IGNORECASE)
    # Remove 'Adopted' headers and annex numbering
    text = re.sub(r'Adopted\s+\d+', '', text)
    text = re.sub(r'ANNEX\s+-\s+QUESTIONS.*', '', text)
    # Clean multiple newlines and bullet symbols
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\uf0d8|\uf0b7', ' ', text) 
    return text.strip()

def get_clean_chunks(docs, chunk_size=1500, chunk_overlap=500):
    """ETL: Pre-processes and chunks documents."""
    for doc in docs:
        doc.page_content = clean_legal_text(doc.page_content)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

# Dual Index Setup
def setup_dual_indices():
    # 0. Ensure directory structure exists
    os.makedirs("data/regulations/", exist_ok=True)
    os.makedirs("data/contracts/", exist_ok=True)
    os.makedirs("chroma_db/", exist_ok=True)
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large") # or llama3.1:8b if your hardware allows

    # --- 1. Source of Truth (GDPR Regulations) ---
    reg_loader = DirectoryLoader(
        "data/regulations/", 
        glob="./*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    reg_docs = reg_loader.load()
    
    if not reg_docs:
        print("Warning: No regulation PDFs found in data/regulations/")
    
    reg_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    reg_chunks = reg_splitter.split_documents(reg_docs)

    # We use a static directory for regs because they don't change often
    reg_vdb = Chroma.from_documents(
        documents=reg_chunks, 
        embedding=embeddings,
        collection_name="gdpr_regs",
        persist_directory="chroma_db/regs",
    )

    bm25_reg = BM25Retriever.from_documents(reg_chunks)
    # Base Ensemble Retriever (The Wide Net)
    base_reg_retriever = EnsembleRetriever(
        retrievers=[reg_vdb.as_retriever(search_kwargs={"k": 20}), bm25_reg],
        weights=[0.5, 0.5] # Weight keywords slightly higher for legal citations
    )

    # Lightweight model to re-score the top 10 results
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)

    # Wrap the ensemble retriever with the reranker
    reg_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_reg_retriever
    )

    # --- 2. Audit Target (The Contract) ---
    # Generate a unique timestamp for specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    contract_persist_dir = f"chroma_db/contract_{timestamp}"

    target_loader = DirectoryLoader(
        "data/contracts/", 
        glob="./*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    target_docs = target_loader.load()
    
    for doc in target_docs:
        doc.page_content = redact_pii(doc.page_content, mode="redact")

    if not target_docs:
        print("Error: No contract found in data/contracts/")
        # Return dummy or empty VDB to avoid crash
        return reg_retriever, None

    target_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    target_chunks = target_splitter.split_documents(target_docs)

    target_vdb = Chroma.from_documents(
        target_chunks, 
        embeddings,
        collection_name=f"audit{timestamp}",
        persist_directory=contract_persist_dir
    )

    return reg_retriever, target_vdb

def run_compliance_audit(reg_retriever, target_vdb, checkpoint):
    # Retrieve from Regulations
    reg_context = reg_retriever.invoke(checkpoint['query'])
    reg_text = "\n".join([r.page_content for r in reg_context])
    contract_text = extract_contract_clause(target_vdb, checkpoint['query'])

    # Setup Structured Output
    parser = JsonOutputParser(pydantic_object=AuditResult)
    model = ChatOllama(model="llama3.1:8b", temperature=0, top_p=0.1, num_predict=512)

    prompt = f"""
    <role>You are a Precise GDPR Information Extractor and Auditor.</role>
    <task>Answer the <user_query> using ONLY the <gdpr_context> provided.</task>
    
    <strict_rules>
        1. Every claim in "reasoning" MUST begin with a direct citation (e.g., "According to Art. 15...").
        2. Respond ONLY with a JSON object. No conversational filler.
        3. If no information is found, reasoning should state: "The provided articles do not specify this."
    </strict_rules>

    <context>
        <gdpr_context>{reg_text}</gdpr_context>
        <contract_context>{contract_text}</contract_context>
    </context>

    <user_query>{checkpoint['query']}</user_query>

    <example>
    {{
        "status": "PASS",
        "violation_found": false,
        "reasoning": "According to Article 12(1) GDPR, information must be provided in writing or by electronic means.",
        "rememdy": "N/A"
    }}
    </example>
    
    <user_query>{checkpoint['query']}</user_query>

    {parser.get_format_instructions()}
    """

    try:
        response = model.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            raw_data = json.loads(match.group(0))
            res = {k.lower(): v for k, v in raw_data.items()}
            
            analysis = {
                "status": res.get("status", "UNCLEAR"),
                "reasoning": res.get("reasoning", "Extraction failed."),
                "violation_found": res.get("violation_found", False),
                "rememdy": res.get("rememdy", res.get("remedy", "N/A"))
            }
            # SUCCESS: Returns list of Document objects
            return analysis, reg_context 
        else:
            raise ValueError("LLM returned text but no JSON block.")
            
    except Exception as e:
        # 3. CRITICAL: Always return the LIST (reg_context), never the string
        print(f"⚠️ Auditor Logic Error: {e}")
        fallback = {
            "status": "ERROR", 
            "reasoning": f"Formatting failure. Context found: {reg_text[:100]}...", 
            "rememdy": "Manual check required."
        }
        return fallback, reg_context

def run_full_system_audit():
    # Load configurations
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup databases
    reg_retriever, target_vdb = setup_dual_indices()
    full_report = []

    # Iterate through each checkpoint defined in YAML
    for cp in config['checkpoints']:
        print(f"Auditing: {cp['name']}...")

        # Capture both variables from updated function
        analysis, combined_context = run_compliance_audit(reg_retriever, target_vdb, cp)

        full_report.append({
            "id": cp['id'],
            "name": cp['name'],
            "query": cp["query"],
            "standard": cp['expected_standard'],
            "analysis": analysis,
            "context": combined_context
        })
    
    return full_report

def extract_contract_clause(target_vdb, query):
    # Search the Contract for specific requirement (e.g. "Data Retention")
    results = target_vdb.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

