# CSCI-544 NLP: Temporally Grounded F-1 Explanations

This research prototype asks whether automatic claim-level temporal indexing and
meaning-preservation verification can reduce unsafe plain-language explanations
of changing F-1 employment rules.

It is a research system, not legal advice.

## Updated architecture

The retrieval index no longer depends on manually approved document-level
effective dates. It uses a dual-index pipeline:

1. Local PDF, HTML, and text documents are parsed into candidate passages.
2. Every passage receives a semantic embedding plus previous/next passage links.
3. A high-recall automatic router tags passages containing temporal expressions,
   quantities, exceptions, or substantive rule-version/change language.
4. Routed passages first go to a fast 3B model. Invalid, low-confidence,
   unknown-role, or unresolved historical results escalate to an 8B fallback.
   The selected model splits the passage into atomic regulatory claims.
5. Each claim receives an inferred role (`operative`, `historical`,
   `transitional`, `proposed`, `rejected`, `public_comment`, `example`, or
   `unknown`), an evidence quote, a supported date interval, and confidence.
6. Retrieval searches every passage, returns sequence links for neighboring
   context traversal, and consults linked temporal claims for dated questions.
7. The selected evidence is passed to the existing meaning-preservation and safety
   gates.

The model is not allowed to invent missing temporal boundaries. Unrouted,
failed, or temporally unknown passages remain searchable, but a dated answer
based only on them causes abstention rather than a false verification.

## One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama, then make sure the configured local models exist:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

`qwen2.5:3b` is the fast primary extractor, `llama3.1:8b` is the reliability
fallback, and `nomic-embed-text` builds semantic vectors. Model names and output
limits can be changed in environment variables; see `.env.example`.

## Build the corpus and automatic temporal index

The application never crawls or downloads source documents. Put team-supplied
files in `corpus_documents/f1-employment-and-extensions`, then run:

```bash
python3 -m scripts.build_corpus \
  --input-dir corpus_documents/f1-employment-and-extensions

python3 -m scripts.build_temporal_index \
  --input-dir corpus_documents/f1-employment-and-extensions \
  --workers 2
```

The second command first embeds every passage in inexpensive batches, then sends
only automatically routed passages through the generative temporal model. Both
embeddings and valid model responses are cached, so pressing `Ctrl+C` is safe:
rerun the same command later and a startup resume scan skips completed passages
before any workers are launched. Progress therefore begins at the first passage
that still needs model work instead of replaying counters from 1. The default is
two concurrent requests. Ollama must also be configured
with `OLLAMA_NUM_PARALLEL=2` to execute them concurrently rather than queue them.
On a 16 GB machine, start with two workers; additional workers can exhaust memory
and reduce performance.

The primary model is capped at 600 output tokens. The 8B fallback is capped at
900 and runs only when the primary result is unusable or uncertain. Cache keys
include the model name, so existing `llama3.1:8b` results remain reusable as
fallback results after upgrading to the cascade.

For a small development run:

```bash
python3 -m scripts.build_temporal_index \
  --input-dir corpus_documents/f1-employment-and-extensions \
  --max-passages 20 \
  --workers 2
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
- `data/embedding_cache`: reusable semantic vectors keyed by text and model
- `data/temporal_cache`: cached structured model extractions
- `data/index/semantic_passages.jsonl`: every passage, routing decision,
  sequence links, provenance, and semantic embedding
- `data/index/temporal_claims.jsonl`: temporal claims linked to source passages
- `data/index/linked_temporal_claims.jsonl`: conservative cache-only temporal
  links; retrieval uses this file only while its recorded base-index hash matches
- `data/reports`: ingestion reports

ChromaDB is not used. The prototype uses an auditable local JSONL index because
the current corpus is small.

Federal Register effective dates often occur in a `DATES` header outside the
selected rule paragraph. The temporal builder therefore extracts exact date
anchors from the full normalized upload and links them to eligible atomic
claims. It also uses explicit supersession language (for example, a statement
that the 17-month rules remain in force through a named date) rather than
inventing an end date.

Cached claims that express the same legal proposition in different sources can
also share a supported start date. Build this local enhancement without Ollama:

```bash
python3 -m scripts.build_linked_temporal_index
```

The linker requires high embedding similarity plus compatible lexical, numeric,
form, and CFR-reference signatures. It never overwrites a claim's original
interval, records the donor claim and confidence, and leaves uncertain claims
unscoped. If the base index changes, retrieval automatically ignores a stale
linked index until it is rebuilt.

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

## Curating the 24-case evaluation

Build the fixed candidate set and two independent review sheets:

```bash
python3 -m scripts.build_selected_benchmark \
  --output-dir data/reports/benchmark
```

Follow [`evaluation/ANNOTATION_GUIDE.md`](evaluation/ANNOTATION_GUIDE.md). The
finalizer freezes only cases approved by two different reviewers with matching
labels and exports disagreements for adjudication. Until that review and the
planned DSO/legal audit are complete, all generated metrics are diagnostics—not
final test results.

The development-only diagnostic is run with:

```bash
python3 -m scripts.run_evaluation \
  --cases data/reports/benchmark/development-candidates.jsonl \
  --output data/reports/benchmark/development-diagnostic-report.json

python3 -m scripts.summarize_evaluation \
  --report data/reports/benchmark/development-diagnostic-report.json \
  --output data/reports/benchmark/development-diagnostic-summary.md
```

The current diagnostic deliberately abstains when a passage-level claim lacks
an independently supported effective interval. This is a measured system
limitation, not a date filled from hand-written metadata.
