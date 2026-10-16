# Local source documents

This directory contains documents explicitly supplied or approved by the project
team. The ingestion pipeline reads these files locally and never crawls or
downloads their source URLs.

The current F-1 corpus lives in `f1-employment-and-extensions/`. Downloaded HTML
files are intentionally ignored by Git; provenance and review decisions are
stored separately by the corpus review workflow.

The tracked `source-catalog.csv` records provenance for the 21 listed source
documents. Two malformed filename cells from the supplied spreadsheet were
normalized to local filenames for rows 15 and 16.

Older PDF samples retained from the removed ChromaDB prototype live in
`legacy-sample-pdfs/`. They are not part of the current ingestion run.

Build the local corpus with:

```bash
python3 -B -m scripts.build_corpus \
  --input-dir corpus_documents/f1-employment-and-extensions
```
