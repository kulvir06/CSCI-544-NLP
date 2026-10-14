# CSCI-544-NLP

**AI-Powered Immigration Assistant for F-1 Students**

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
