# Literature Review Matrix: Automatic Temporal Claim Retrieval and Safe Legal Simplification

## Revised project focus

Given locally uploaded F-1 employment material, semantically index every ordered
passage, selectively decompose temporal-risk passages into atomic legal claims,
infer the temporal scope and rhetorical role of those claims before retrieval,
retrieve evidence applicable to a user-selected date, and verify that a
plain-language explanation preserves legally important meaning.

The revised target failure is more precise than document-version error. A
document can be applicable while an individual sentence inside it reports an
old rule, rejected alternative, proposal, example, or public comment. The unit
of temporal indexing is therefore the **claim**, not the document.

This review was updated on 15 October 2026. A blank or **No** means that the
capability is outside a paper's primary evaluated contribution, not that it is
never mentioned.

## Literature matrix

| Study | Main task/domain | Automatic temporal extraction | Claim/passage-level temporal representation | Time-aware retrieval | Meaning-preservation verification | Gap relative to this project |
|---|---|:---:|:---:|:---:|:---:|---|
| [Mathur et al. (2021), *TIMERS: Document-level Temporal Relation Extraction*](https://aclanthology.org/2021.acl-short.67/) | Predicts temporal relations using discourse, syntax, and semantic-role information. | Yes | Partial | No | No | Extracts event relations but does not decide which regulatory claim is operative for a user's date. |
| [Mathur et al. (2022), *DocTime: A Document-level Temporal Dependency Graph Parser*](https://aclanthology.org/2022.naacl-main.73/) | Creates temporal dependency graphs, including evaluation on contractual documents. | Yes | Partial | No | No | Represents event chronology rather than pre-retrieval legal applicability or simplification safety. |
| [Garimella et al. (2022), *Text Simplification for Legal Domain*](https://aclanthology.org/2022.nllp-1.28/) | Studies legal simplification data, methods, metrics, and human evaluation. | No | No | No | Partial | Assumes the supplied source is the correct legal source. |
| [Agrawal & Carpuat (2024), *Do Text Simplification Systems Preserve Meaning?*](https://aclanthology.org/2024.tacl-1.24/) | Evaluates meaning preservation through reading-comprehension questions. | No | No | No | Yes | Does not address evolving legal claims or source applicability. |
| [Trautmann et al. (2024), *Measuring the Groundedness of Legal QA Systems*](https://aclanthology.org/2024.nllp-1.14/) | Evaluates whether a legal answer is grounded in supplied context. | No | No | No | Yes | A response can be grounded in context that reports an inapplicable rule. |
| [Barale et al. (2025), *LexTime*](https://aclanthology.org/2025.findings-emnlp.280/) | Benchmarks temporal ordering of events in U.S. legal complaints. | Yes | Yes | No | No | Evaluates event ordering, not regulatory effect intervals or retrieval decisions. |
| [Beauchemin et al. (2025), *JUDGEBERT*](https://aclanthology.org/2025.emnlp-main.5/) | Measures legal meaning preservation between French legal sentences. | No | No | No | Yes | Does not test whether the source claim applies at the requested time. |
| [Rabbani et al. (2026), *Decode the Law / SIMPLE-LAW*](https://aclanthology.org/2026.lrec-1.45/) | Provides legal simplification pairs and evaluates readability, semantics, and hallucination. | No | No | No | Yes | Rewrites a given passage without resolving changing rule versions. |
| [Yousuf et al. (2026), *Utilizing Metadata for Better RAG*](https://arxiv.org/abs/2601.11863) | Compares metadata-as-text, unified embeddings, late fusion, and metadata-aware queries. | No | Partial | Yes | No | Shows metadata can improve retrieval but does not automatically infer legal temporal applicability. |
| [Cymbler et al. (2026), *Temporal Misgrounding in Legal RAG*](https://arxiv.org/abs/2608.09393) | Introduces a versioned French-tax corpus and demonstrates wrong-version retrieval. | No | Partial | Yes | Partial | Uses versioned legal sources; it does not automatically atomize mixed passages or test simplification loss. |
| [Huang et al. (2026), *When Do LLMs Apply the Wrong Law?*](https://arxiv.org/abs/2608.14610) | Benchmarks LLM temporal applicable-law determination and diagnoses current-law bias. | Partial | Partial | No | No | Demonstrates that LLM temporal reasoning is unreliable, motivating confidence, abstention, and a gold evaluation set. |
| [Nam et al. (2026), *TimelyRAG*](https://arxiv.org/abs/2609.11572) | Combines semantic and temporal distance for overlapping evolving documents. | Partial | Partial | Yes | No | Improves time-aware ranking but does not jointly verify meaning preservation after simplification. |
| **This project** | F-1 employment and extension rules | **Yes** | **Yes: atomic regulatory claims** | **Yes: temporal filter before hybrid ranking** | **Yes** | Evaluates the interaction between automatic temporal inference, retrieval, and simplification safety. |

## Revised architecture relative to prior work

The system now separates three concepts that ordinary document-level RAG often
conflates:

| Layer | Question answered | Implementation |
|---|---|---|
| Full semantic passage index | What passages are relevant, and where do they occur in the source sequence? | Embeddings for every passage plus previous/next passage links |
| Temporal routing | Which passages need expensive temporal analysis? | High-recall automatic tags for temporal expressions, quantities, exceptions, and rule-version language |
| Claim atomization | What independent regulatory propositions occur in a routed passage? | Local instruction model with exact evidence spans |
| Temporal semantics | Is each proposition operative, historical, transitional, proposed, rejected, commentary, an example, or unclear—and what interval is supported? | Structured model output stored before retrieval |
| Applicability-constrained retrieval | Which temporally eligible claim best matches the question? | Date filtering followed by dense semantic and BM25 ranking |
| Meaning verification | Did the explanation retain the selected claim's conditions, exceptions, negation, dates, forms, and quantities? | Existing atomic-fact verifier and release gate |

This design does not treat an embedding as a date-comparison engine. Dates and
roles are structured fields used for deterministic filtering; embeddings are
used only to rank semantically relevant claims that survive the filter.

## Defensible novelty claim

> Prior research separately studies temporal relation extraction, metadata-aware
> or version-aware retrieval, and legal meaning preservation. We investigate a
> narrower interaction: whether automatically converting mixed-temporal F-1
> regulatory passages into temporally scoped atomic claims **before retrieval**,
> then verifying meaning after retrieval, reduces false-safe plain-language
> explanations. We evaluate cases where a document is valid but an individual
> claim is historical, proposed, rejected, or otherwise non-operative.

This is a **system, benchmark, and evaluation contribution**, not a claim that
temporal extraction, embeddings, RAG, or legal simplification was independently
invented here. Use “to the best of our knowledge” and avoid “the first ever.”

## Novel failure cases enabled by the revised design

1. A current document accurately describes a superseded requirement.
2. One passage contains both an old and a new rule.
3. A public comment proposes the same number or deadline used in the question.
4. A final rule explains an alternative that the agency rejected.
5. A transition provision applies only during a bounded window.
6. The model assigns a confident but unsupported temporal interval.
7. Temporal inference is correct but semantic retrieval selects the wrong topic.
8. Retrieval is correct but simplification removes a condition or exception.

## Evaluation needed to support the contribution

| Research question | Metric |
|---|---|
| Does the router retain passages that require temporal analysis? | Temporal-routing recall, false-negative rate, and model-call reduction |
| Does the model split routed passages into the correct atomic claims? | Claim boundary precision/recall and exact-evidence validity |
| Does it recognize operative versus non-operative language? | Macro-F1 over temporal-role labels |
| Are inferred temporal scopes correct? | Start-date accuracy, end-date accuracy, interval overlap, and unsupported-date rate |
| Does filter-first retrieval find the applicable claim? | Applicable-claim Recall@K, MRR, and wrong-period retrieval rate |
| Does uncertainty handling improve safety? | Risk-coverage curve and abstention rate |
| Does the full system prevent unsafe release? | False-safe rate by error category |

The gold set should be small and carefully audited. Removing manual metadata from
the runtime pipeline does not remove the need for independent human labels in
the experiment; without them, automatic temporal extraction cannot be evaluated.

## Required ablations

1. BM25 without temporal information
2. Original document-level date filtering
3. Automatic claim-level temporal filtering with BM25
4. Automatic claim-level filtering with hybrid semantic retrieval
5. Full system with meaning-preservation verification
6. Selective temporal routing versus temporal analysis of every passage

The main empirical novelty claim is supported only if claim-level filtering
reduces subtle wrong-period retrieval and the full system further reduces
false-safe explanations.

## Core references

1. Mathur, P., et al. (2021). “TIMERS: Document-level Temporal Relation Extraction.” ACL-IJCNLP. https://doi.org/10.18653/v1/2021.acl-short.67
2. Mathur, P., et al. (2022). “DocTime: A Document-level Temporal Dependency Graph Parser.” NAACL. https://doi.org/10.18653/v1/2022.naacl-main.73
3. Garimella, A., et al. (2022). “Text Simplification for Legal Domain: Insights and Challenges.” NLLP. https://doi.org/10.18653/v1/2022.nllp-1.28
4. Agrawal, S., & Carpuat, M. (2024). “Do Text Simplification Systems Preserve Meaning?” TACL. https://doi.org/10.1162/tacl_a_00653
5. Trautmann, D., et al. (2024). “Measuring the Groundedness of Legal Question-Answering Systems.” NLLP. https://aclanthology.org/2024.nllp-1.14/
6. Barale, C., et al. (2025). “LexTime: A Benchmark for Temporal Ordering of Legal Events.” Findings of EMNLP. https://doi.org/10.18653/v1/2025.findings-emnlp.280
7. Beauchemin, D., et al. (2025). “JUDGEBERT.” EMNLP. https://doi.org/10.18653/v1/2025.emnlp-main.5
8. Rabbani, M. D., et al. (2026). “Decode the Law.” LREC. https://doi.org/10.63317/2i7en3xycmhi
9. Yousuf, R. B., et al. (2026). “Utilizing Metadata for Better Retrieval-Augmented Generation.” https://arxiv.org/abs/2601.11863
10. Cymbler, R., Guez, D., & Fabre, L. (2026). “Temporal Misgrounding in Legal RAG.” https://arxiv.org/abs/2608.09393
11. Huang, Y., et al. (2026). “When Do LLMs Apply the Wrong Law?” https://arxiv.org/abs/2608.14610
12. Nam, Y., et al. (2026). “TimelyRAG.” https://arxiv.org/abs/2609.11572
