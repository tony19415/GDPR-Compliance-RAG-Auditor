import streamlit as st
import os
import shutil
import yaml
import tempfile
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from ollama import chat

from datetime import datetime

from audit_engine import get_clean_chunks, run_full_system_audit, run_compliance_audit, setup_dual_indices
from report_generator import generate_html_report
import streamlit.components.v1 as components

# 1. Security Layer (Guardrailes)
def is_prompt_safe(query: str) -> bool:
    """Passes the user query to Llama Guard to check for prompt injection or malicious intent."""
    response = chat(
        model="llama-guard3:8b",
        messages=[{"role": "user", "content": query}]
    )

    classification = response["message"]["content"].strip().lower()

    # Llama Guard returns "safe" or "unsafe \n [category]"
    if classification.startswith("safe"):
        return True
    return False

# 2. Data Ingestion & Hybrid Retriever
@st.cache_resource
def initialize_hybrid_retriever():
    # Load all pdf from the 'data' directory
    loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    if not docs:
        st.error("No PDFs found in the 'data/' directory. Please add documents.")
        return None
    
    # Refined Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = get_clean_chunks(docs, chunk_size=1500, chunk_overlap=300)


    # Dense Retriever (ChromaDB - Semantic Search)
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Sparse Retriever (BM25 - Exact Keyword Match)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # Ensemble (50/50 weighting)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever

hybrid_retriever = initialize_hybrid_retriever()

# 3. Retrieval Execution
def retrieve_context(query: str):
    if not hybrid_retriever:
        return "", []

    # Ensemble retriever handle querying both systems and de-duplicating
    results = hybrid_retriever.invoke(query)

    context_parts = []
    sources = []

    for doc in results:
        context_parts.append(doc.page_content)

        # Metadata extraction
        source_name = doc.metadata.get('source', 'Unknown Document')

        # Clean up the file path for display
        clean_source = os.path.basename(source_name)
        page_num = doc.metadata.get('page', 'Uknown Page')
        source_info = f"{clean_source} (Page {page_num})"

        if source_info not in sources:
            sources.append(source_info)

    return "\n\n---\n\n".join(context_parts), sources

# 4. Generation Logic
def generate_answer(query: str, context: str) -> str:
    system_prompt = """
    <prompt>
        <role>
            You are a Senior GDPR Compliance Auditor with expertise in European data protection law and contract analysis.
        </role>

        <task>
            Audit the provided contract snippets against specific GDPR standards to identify compliance gaps or alignments.
        </task>

        <instructions>
            <instruction>Analyze the "Contract Evidence" against the "Legal Basis" found in the provided context.</instruction>
            <instruction>Maintain a neutral, objective, and strictly legal tone.</instruction>
            <instruction>If the provided context does not contain the necessary GDPR Article, stop the audit immediately.</instruction>
        </instructions>

        <constraints>
            <rule>STRICT: Use ONLY the provided Context. Do not rely on internal training data or external legal knowledge.</rule>
            <rule>MANDATORY: You must explicitly quote the GDPR Article text from the Context to justify your audit status.</rule>
        </constraints>

        <output_format>
            Your response must follow this exact schema:
            
            STATUS: [Compliant / Non-Compliant / Partial]
            CONTRACT EVIDENCE: [Direct quote from the contract snippet]
            LEGAL BASIS: [Direct quote of the specific GDPR Article text from the context]
            REASONING: [Clear explanation of the gap or alignment between the contract and the law]
        </output_format>

        <fallback_responses>
            <insufficient_context>Insufficient legal context provided.</insufficient_context>
        </fallback_responses>
    </prompt>
    """

    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"<input_data>\n<context>{context}</context>\n<query>{query}</query>\n</input_data>"
            }
        ]
    )
    return response["message"]["content"]

# 5. Streamlit UI
st.title("GDPR Auditor")
st.markdown("Automated RAG system featuring Hybrid Search and Llama Guard Prompt Security.")
st.sidebar.title("Admin tools")
st.sidebar.header("Automated Contract Auditor")

uploaded_file = st.sidebar.file_uploader("Upload Vendor Contract (PDF)", type="pdf")

user_query = st.chat_input("Enter your compliance query.")

if user_query:
    # 1. Security Check
    with st.spinner("Running security guardrails..."):
        is_safe = is_prompt_safe(user_query)

    if not is_safe:
        st.error("Security Alert: This prompt was flagged by Llama Guard as unsafe or an injection attempt. Request denied.")
    else:
        # 2. Retrieval
        with st.spinner("Querying Hybrid Retriever (Semantic + BM25)..."):
            context, sources = retrieve_context(user_query)
        
        # 3. Generation
        with st.spinner("Auditing against retrieved context..."):
            answer = generate_answer(user_query, context)

        st.write("### Audit Response")
        st.write(answer)

        with st.expander("View Traceability & Extracted Sources"):
            st.write("**Referenced Documents:**")
            for source in sources:
                st.markdown(f"- `{source}`")

            st.write("**Raw Context Provided to LLM:**")
            st.text(context)

if st.sidebar.button("Run Full System Audit"):
    with st.spinner("Analyzing all contracts against GDPR standards..."):
        # 1. Execute engine
        results = run_full_system_audit()

        # 2. Prepare dynamic filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"gdpr_audit_{date_str}.json"
        
        # 3. Save to server
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        
        # 4. Download button for user
        json_string = json.dumps(results, indent=4)
        st.sidebar.download_button(
            label="Download JSON Report",
            data=json_string,
            file_name=filename,
            mime="application/json"
        )


        # 5. Generate static HTML file
        generate_html_report(results)


        st.sidebar.success("Audit Report Generated!")

        # 3. Option to view the report inside Streamlit
        with open("gdpr_audit_report.html", "r", encoding='utf-8') as f:
            html_content = f.read()
            st.markdown("---")
            st.subheader("Executive Audit Preview")
            components.html(html_content, height=600, scrolling=True)

if uploaded_file and st.sidebar.button("Run Compliance Audit"):
    with st.spinner("Analyzing contract against GDPR Knowledge Base..."):
        # 1. Ensure the directory exists
        os.makedirs("data/contracts/", exist_ok=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. REPLACE THE 'cp' COMMAND with shutil.copy2
            dest_path = os.path.join("data/contracts/", uploaded_file.name)
            shutil.copy2(temp_filepath, dest_path)
            
            reg_retriever, target_vdb = setup_dual_indices()

            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

            st.sidebar.markdown("### Audit Results")
            for cp in config['checkpoints']:
                result, _ = run_compliance_audit(reg_retriever, target_vdb, cp)

                # 3. DEFENSIVE CHECK FOR KEYERROR
                # Use .get() to avoid crashing if 'status' is missing
                status = result.get('status', 'FAIL') 
                reasoning = result.get('reasoning', 'No reasoning provided by LLM.')

                if status == 'PASS':
                    st.sidebar.success(f"✅ {cp['name']}")
                else:
                    st.sidebar.error(f"❌ {cp['name']}")
                    st.sidebar.caption(f"**Issue:** {reasoning}")

            # 4. REPLACE THE 'rm' COMMAND with os.remove
            if os.path.exists(dest_path):
                os.remove(dest_path)