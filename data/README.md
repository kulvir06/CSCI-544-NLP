# Local corpus data

The application processes only files explicitly supplied by the team. It never
crawls, scrapes, or downloads the source URL stored in provenance metadata.

- `raw/`: uploaded source files
- `normalized/`: extracted text
- `passages/`: automatically selected F-1 rule passages
- `temporal_cache/`: cached Ollama claim extractions
- `index/temporal_claims.jsonl`: automatic claim-level temporal and semantic index
- `manifests/`: ingestion and provenance records
- `reports/`: ingestion and evaluation reports
- `reviews/`: legacy document-review records; not consulted by the new temporal index

Build from the active local corpus:

```bash
python3 -m scripts.build_corpus \
  --input-dir corpus_documents/f1-employment-and-extensions

python3 -m scripts.build_temporal_index \
  --input-dir corpus_documents/f1-employment-and-extensions
```

The temporal index is produced automatically from passage text and surrounding
context. Each record contains an atomic claim, exact evidence quote, temporal
role, supported interval, confidence, model provenance, and embedding.

The review files are retained for reproducibility of the earlier document-level
baseline and for benchmark curation. They no longer control which records enter
the main retrieval index.
