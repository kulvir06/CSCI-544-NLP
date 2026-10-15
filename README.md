# CSCI-544-NLP

**AI-Powered Immigration Assistant for F-1 Students**

## Run the verified prototype

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn api.main:app --reload
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) to use the local
dashboard. It visualizes passage retrieval, temporal applicability, atomic fact
preservation, and the final release/block decision. The API documentation remains
available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The dashboard only searches the approved local retrieval index. A `no_evidence`
result is expected until corpus metadata has been reviewed and the index rebuilt.

---

### **Project Description**

Engineered an AI-driven immigration assistant for F-1 students using a retrieval-augmented generation (RAG) pipeline. Implemented document chunking and vector-based semantic search to retrieve top-K relevant immigration policy content, followed by response generation using a local LLM. Integrated a secondary LLM-based evaluation agent to review, score, and refine responses before presenting them to users, improving accuracy and relevance.

### Corpus Safety Policy

The corpus builder is deliberately **upload-only and offline**. It processes
documents explicitly supplied by the project team and does not crawl, scrape,
or automatically download content from government, university, or other
websites. A source URL may be recorded as provenance, but the application never
opens that URL.

Uploaded primary rules, federal guidance, and university advisories are assigned
different authority tiers. University material may support explanation or serve
as a retrieval distractor, but it cannot override primary federal material.

---

### **Tech Stack**

**Languages & Frameworks:**

- Python

**AI / NLP:**

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Large Language Models (Mistral via Ollama)
- Multi-Agent LLM Systems

**Vector Database & Storage:**

- ChromaDB
- Local document storage

**Search & Ranking:**

- Semantic Search
- Cosine Similarity
- Top-K Retrieval

**Systems & Tools:**

- Ollama
- Prompt Engineering

---

### **Skills Demonstrated**

- Generative AI Application Development
- Vector Databases & Embeddings
- LLM Orchestration & Evaluation
- AI Agent Design (Generation + Judge Agent)
- Information Retrieval Systems
- Natural Language Processing (NLP)
- Backend AI System Design
