# Enterprise GDPR Auditor: Advanced RAG & GraphRAG
An enterprise grade, local-first compliance platform that automates the auditing of internal documents (HR policies, vendor DPAs) against the GDPR, Dutch AP, and German BfDI frameworks.

By combining Hybrid Vector Search with Knowledge Graph Relational Mapping (GraphRAG), this system identifies both textual discrepancies and structural organizational risks that traditional RAG systems miss.

## System Architecture
The system is built on a **Decoupled Microservices Architecture** to ensure scalability and data sovereignty:
- **Logic Engine (FastAPI):** Secure REST API gateway with X-API-Key protection and Pydantic-enforced deterministic JSON outputs.

- **Vector Brain (ChromaDB + BM25):** Dual-index retrieval for high-precision semantic and keyword matching of legal text.

- **Relational Brain (Neo4j):** Maps complex organizational chains (Controller → Processor → Sub-processor) to detect structural violations.

- **Local Inference (Ollama):** Powered by **Llama 3.1 8B**, ensuring 100% of PII remains on-premise (No external API calls to US servers).

- **Data Governance (DVC):** Version control for vector and graph databases, ensuring reproducible and explainable AI audits.

<img width="1041" height="679" alt="Image" src="https://github.com/user-attachments/assets/e0ee40e6-d7b2-4bfd-bc8e-55de34bab3cc" />


## Key Features
- **Hybrid Retrieval Pipeline:** Ensembles BM25 (keyword) and ChromaDB (semantic) search to achieve high recall on specific legal articles (e.g., Article 28, Article 33).

- **Automated Entity Extraction:** Utilizes XML-tagged prompt engineering to transform unstructured PDFs into structured Neo4j triples.

- **Jurisdictional Routing:** Specialized retrieval namespaces for Dutch (AP) and German (BfDI) guidance, prioritizing local authority interpretations.

- **Deterministic Reporting:** Generates executive-level HTML Audit Reports with visual compliance scoring and automated remediation suggestions.

- **CI/CD Integration:** Automated testing suite via GitHub Actions verifying retrieval precision and schema compliance on every push.

## Tech Stack
| Category          | Tools                                           |
|-------------------|-------------------------------------------------|
| LLM Orchestration | LangChain, Ollama (Llama 3.1 8B)                |
| Databases         | ChromaDB (Vector), Neo4j (Graph), Redis (Cache) |
| Backend/API       | FastAPI, Uvicorn, Pydantic                      |
| Frontend/UI       | Streamlit, Streamlit-Agraph (Visualization)     |
| Data Engineering  | DVC (Data Version Control), Docker Compose      |
| MLOps/DevOps      | GitHub Actions, Pytest, Jinja2                  |

## Getting Started
1. Prerequisites
- Docker & Docker Compose

- Python 3.10+

- Ollama (Local)

2. Infrastructure Setup
Spin up the multi-service stack:
```
docker-compose up -d
```

3. Data Ingestion & Versioning
```
dvc pull
python graph_ingest.py  # Map organizational relationships to Neo4j
```

4. Running the Platform
Launch the Backend API and Frontend Client:
```
# Terminal 1: Start the Brain
python server.py

# Terminal 2: Start the Body
streamlit run app.py
```