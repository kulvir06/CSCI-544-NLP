# Local corpus data

This project uses an upload-only corpus workflow. It does not crawl, scrape, or
automatically download government or university websites.

- `raw/`: user-supplied source files (ignored by Git)
- `normalized/`: extracted text (ignored by Git)
- `passages/`: automatically selected rule-bearing passages (ignored by Git)
- `reports/`: batch summaries and review flags (ignored by Git)
- `manifests/`: document provenance records (local manifest ignored by Git)

Source URLs are stored only as provenance metadata and are never fetched by the
corpus pipeline.

Build a corpus from an existing local folder:

```bash
python -m scripts.build_corpus --input-dir "/path/to/document-corpus"
```
