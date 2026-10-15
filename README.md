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

This research prototype tests whether an F-1 employment explanation is safe to
release. It retrieves only reviewed passages, checks whether the source version
applies to the relevant date, and verifies that legally important facts survive
simplification. Unsafe drafts are blocked and uncertain cases cause abstention.

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

- Rule-aware document ingestion and passage selection
- BM25 passage retrieval
- Temporal applicability verification
- Atomic-fact and meaning-preservation verification
- Targeted negative-case evaluation

**Storage:**

- Reviewed local JSONL passage index
- Auditable document metadata and review log

**Search & Ranking:**

- BM25 lexical ranking
- Top-K Retrieval

**Systems & Tools:**

- FastAPI
- Local HTML dashboard
- Offline, upload-only corpus workflow

---

### **Skills Demonstrated**

- Safety-oriented NLP evaluation
- Temporal information retrieval
- Meaning-preservation analysis
- Information Retrieval Systems
- Natural Language Processing (NLP)
- Backend AI System Design
