# Local source documents

This directory contains documents explicitly supplied or approved by the project
team. The ingestion pipeline reads these files locally and never crawls or
downloads their source URLs.

The current F-1 corpus lives in `f1-employment-and-extensions/`. Downloaded HTML
files are intentionally ignored by Git; provenance and review decisions are
stored separately by the corpus review workflow.

Older PDF samples retained from the removed ChromaDB prototype live in
`legacy-sample-pdfs/`. They are not part of the new eight-document ingestion run.

Build the local corpus with:

```bash
python3 -B -m scripts.build_corpus \
  --input-dir corpus_documents/f1-employment-and-extensions
```
