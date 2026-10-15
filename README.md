# CSCI-544 NLP: Temporally Grounded F-1 Explanations

This research prototype asks whether automatic claim-level temporal indexing and
meaning-preservation verification can reduce unsafe plain-language explanations
of changing F-1 employment rules.

It is a research system, not legal advice.

## Updated architecture

The retrieval index no longer depends on manually approved document-level
effective dates. Instead, a local Ollama model processes the uploaded passages
before retrieval:

1. Local PDF, HTML, and text documents are parsed into candidate passages.
2. The temporal model splits every passage into atomic regulatory claims.
3. Each claim receives an inferred role (`operative`, `historical`,
   `transitional`, `proposed`, `rejected`, `public_comment`, `example`, or
   `unknown`), an evidence quote, a supported date interval, and confidence.
4. A local embedding model creates a semantic vector for each claim.
5. For a dated question, claims are filtered by their inferred interval before
   hybrid semantic/BM25 ranking.
6. The selected claim is passed to the existing meaning-preservation and safety
   gates.

The model is not allowed to invent missing temporal boundaries. Claims with an
unknown interval remain usable for undated exploration, but a dated answer based
only on them causes abstention.

## One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama, then make sure the configured local models exist:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

`llama3.1:8b` was already present on the development machine and
`nomic-embed-text` was installed and verified. Model names can be changed in
environment variables; see `.env.example`.

## Build the corpus and automatic temporal index

The application never crawls or downloads source documents. Put team-supplied
files in `corpus_documents/f1-employment-and-extensions`, then run:

```bash
python3 -m scripts.build_corpus \
  --input-dir corpus_documents/f1-employment-and-extensions

python3 -m scripts.build_temporal_index \
  --input-dir corpus_documents/f1-employment-and-extensions
```

The second command can take time on its first run because each passage is
analyzed locally. Valid model responses are cached in `data/temporal_cache`, so
unchanged passages are not reprocessed.

For a small development run:

```bash
python3 -m scripts.build_temporal_index \
  --input-dir corpus_documents/f1-employment-and-extensions \
  --max-passages 20
```

Do not use a partial index for final evaluation.

## Run the prototype

Keep Ollama running because query embeddings are generated at search time. In a
second terminal:

```bash
source .venv/bin/activate
python3 -m uvicorn api.main:app --reload
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app).

The date picker represents the date for which the rule should be applicable—not
necessarily today's date. If no date is selected, temporal verification is not
requested.

## Retrieval decision

For dated questions, the system handles claims as follows:

- A supported interval containing the date is `applicable`.
- A supported interval outside the date is `not_applicable`.
- Proposed, rejected, public-comment, and example claims are not operative law.
- A missing interval, unknown role, or confidence below the threshold is
  `insufficient_evidence`.

If no applicable claim exists but a relevant unknown claim does, the system
abstains. If Ollama is temporarily unavailable at query time, retrieval visibly
falls back to temporally filtered BM25 instead of silently treating lexical
ranking as semantic ranking.

## Storage

- `data/passages`: passages selected during ingestion
- `data/temporal_cache`: cached structured model extractions
- `data/index/temporal_claims.jsonl`: atomic claims, temporal fields, provenance,
  and embeddings
- `data/reports`: ingestion reports

ChromaDB is not used. The prototype uses an auditable local JSONL index because
the current corpus is small.

## Evaluation

Automatic ingestion does not require manual legal metadata. Human review is
still required for a small gold evaluation set; without independent labels, the
project cannot measure whether the model inferred temporal scope correctly.

The planned evaluation compares:

1. BM25 without temporal filtering
2. Document-level temporal metadata
3. Automatic claim-level temporal filtering
4. Claim-level temporal filtering plus meaning verification

Primary measurements include claim-role accuracy, interval accuracy,
applicable-claim Recall@K, meaning-error detection, abstention coverage, and
false-safe rate.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite includes a mixed passage containing both the historical 17-month
and operative 24-month STEM OPT claims and verifies that the selected date
changes which atomic claim can be retrieved.
