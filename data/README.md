# Local corpus data

This project uses an upload-only corpus workflow. It does not crawl, scrape, or
automatically download government or university websites.

- `raw/`: user-supplied source files (ignored by Git)
- `normalized/`: extracted text (ignored by Git)
- `passages/`: automatically selected rule-bearing passages (ignored by Git)
- `reports/`: batch summaries and review flags (ignored by Git)
- `reviews/`: append-only human metadata decisions (ignored by Git)
- `manifests/`: document provenance records (local manifest ignored by Git)

Source URLs are stored only as provenance metadata and are never fetched by the
corpus pipeline.

Build a corpus from an existing local folder:

```bash
python -m scripts.build_corpus --input-dir "/path/to/document-corpus"
```

Export the pending metadata-review queue for the team:

```bash
python -m scripts.export_review_queue
```

Fill the `decision` (`approved` or `rejected`), `corpus_use`, `reviewer`, and
known metadata columns, then import the reviewed file:

```bash
python -m scripts.import_review_queue --input data/reports/metadata-review-queue.csv
```

A document can be approved as `meaning_only` without an effective date. A
`temporal_and_meaning` document needs either an explicit effective start date or
a confirmed `valid_as_of` snapshot date, plus a short `date_evidence` note.
Unusable documents are marked `excluded`. Review decisions are append-only so
corrections remain auditable. The application records URLs as provenance but
never opens them.

After importing review decisions, build the approved-passage index:

```bash
python -m scripts.build_retrieval_index
```

The `/retrieve` API accepts a question, optional `relevant_date`, and `top_k`.
Without a date it retrieves from all approved documents. With a date it returns
only passages whose reviewed temporal metadata supports that date, or reports
`insufficient_temporal_evidence` instead of guessing. The current reproducible
baseline ranker is BM25 over the automatically selected passages.
