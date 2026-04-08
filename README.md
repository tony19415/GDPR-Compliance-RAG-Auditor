# GDPR Compliance RAG Auditor
A Privacy-First, Multi-Stage RAG Pipeline for Automated Legal Auditing

Built with a focus on zero-hallucination and data minimization, this tool allows users to audit legal contracts against GDPR and the EU Data Act regulations using a fully local LLM stack.

## Key Features
- **Privacy by Design:** Local-only execution using **Ollama**. No legal data ever leaves the local environment.
- **PII Sanitization:** Automated **Regex-based redaction** of names and emails before documents are embedded or stored.
- **Multi-Stage Retrieval:** A hybrid search architecture combining **BM25 (Keyword)** and **Vector (Semantic) search**, optimized by a **Flashrank Cross-Encoder Reranker**.
- **High-Fidelity Inference:** Powered by **Llama 3.1 8B** with a strictly extractive prompt to ensure $100\%$ faithfulness to the legal text.
- **Quantitative Evaluation:** Integrated **Ragas framework** using **Mistral 12B** as a "Judge" to benchmark performance against a synthetic "Golden Dataset."

## System Architecture
The system utilizes a complex "Wide-Funnel" retrieval strategy to maximize recall without sacrificing precision.
### 1. The Data Pipeline (ETL)
- **Loading:** ```PyMuPDF``` with automated metadata extraction (Source, Page Number).
- **Sanitization:** PII masking layer to ensure GDPR compliance within the database.
- **Chunking:** ```RecursiveCharacterTextSplitter``` (1500 tokens / 500 overlap) to maintain legal context across page breaks.
- **Embedding:** ```mxbai-embed-large``` (1024-dim) for high-density semantic representation.
### 2. The Retrieval Stack
- **Layer 1 (Ensemble):** Combines Vector Similarity ($k=20$) and BM25 Keyword matching ($w=0.5/0.5$).
- **Layer 2 (Reranking):** ```ms-marco-MiniLM-L-12-v2``` re-scores the top 20 candidates to find the most relevant legal articles.
- **Layer 3 (Compression):** Filters the set down to the Top 5 most relevant chunks to prevent LLM "context fatigue."

<img width="1338" height="805" alt="Image" src="https://github.com/user-attachments/assets/69d14112-8dbf-4ea1-9a5b-8547893560c5" />
Architecture Diagram

<img width="1446" height="800" alt="Image" src="https://github.com/user-attachments/assets/01227c19-a7d5-4abc-9d72-f164c46a40aa" />
RAGAS Framework + Test Set Generation

## Performance & Evaluation
The system is continuously benchmarked using Ragas. Our final optimization iteration achieved the following "Gold Standard" results:

| Metric            | Score | Insight                                                    |
|-------------------|-------|------------------------------------------------------------|
| Faithfulness      | 1.00  | Zero hallucinations; the auditor only cites provided text. |
| Answer Relevancy  | 0.76  | High alignment between user queries and legal reasoning.   |
| Context Recall    | 0.80  | Successfully identifies complex, non-consecutive articles. |
| Context Precision | 0.85  | Reranker successfully places "Gold Chunks" at the #1 rank. |

## Tech Stack
| Category          | Tools                                            |
|-------------------|--------------------------------------------------|
| LLM Orchestration | LangChain                                        |
| LLMs              | Llama 3.1 8B (Auditor), Mistral Nemo 12B (Judge) |
| Vector Store      | ChromaDB                                         |
| Embeddings        | Ollama (mxbai-embed-large)                       |
| UI                | Streamlit                                        |
| Reranker          | Flashrank                                        |
| Evaluation        | Ragas                                            |

## 🐳 Docker Deployment (Recommended)
To ensure environment parity and simplify dependency management, you can run the entire auditor suite using Docker.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- Ollama installed on your host machine with the necessary models already pulled (`llama3.1:8b`, `mxbai-embed-large`).

### Deployment Steps
1. **Build and Start:**
   ```bash
   docker-compose up --build

## Installation & Setup
1. Clone Repository
```
git clone https://github.com/yourusername/gdpr-rag-auditor.git
cd gdpr-rag-auditor
```
2. Install Dependencies
```
pip install -r requirements.txt
```

3. Pull Local Models
```
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull mxbai-embed-large
```

4. Run Streamlit App
```
streamlit run app.py
```