# Literature Review Matrix: Applicability-Preserving Legal Simplification

## Project focus

Given a user question about an evolving F-1 rule, retrieve the legally applicable version for the relevant date, produce a plain-language explanation, and verify that the explanation preserves legally important meaning. The target failure is a response that appears faithful and well supported but is grounded in an inapplicable version of the rule.

This matrix compares closely related legal-NLP and text-simplification research available through 13 October 2026. A blank or **No** entry means the capability is outside the paper's primary evaluated contribution; it does not imply that the subject is never mentioned.

Legend: **Yes** = directly implemented or evaluated; **Partial** = adjacent or indirect treatment; **No** = outside the reported task.

## Literature matrix

| Study | Main contribution / domain | Date-aware retrieval | Plain-language simplification | Meaning/factuality verification | Evaluates faithful output from an inapplicable source | Limitation relative to our project |
|---|---|:---:|:---:|:---:|:---:|---|
| [Garimella et al. (2022), *Text Simplification for Legal Domain: Insights and Challenges*](https://aclanthology.org/2022.nllp-1.28/) | Studies legal simplification datasets, methods, metrics, and challenges; includes human evaluation. | No | Yes | Partial | No | Assumes a supplied source and does not select among historical versions. |
| [Agrawal & Carpuat (2024), *Do Text Simplification Systems Preserve Meaning?*](https://aclanthology.org/2024.tacl-1.24/) | Evaluates meaning preservation through reading-comprehension questions in general-domain simplification. | No | Yes | Yes | No | Does not address legal-domain version applicability. |
| [Trautmann et al. (2024), *Measuring the Groundedness of Legal Question-Answering Systems*](https://aclanthology.org/2024.nllp-1.14/) | Benchmarks similarity, NLI, and LLM methods for detecting whether legal QA responses are grounded in provided context. | No | No | Yes | No | Tests faithfulness to the supplied context, but not whether that context is the applicable version. |
| [Hu et al. (2025), *Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering*](https://aclanthology.org/2025.coling-main.298/) | Introduces LegalHalBench and metrics for statute and claim hallucinations; proposes fine-tuning to reduce them. | No | No | Yes | No | Addresses fabricated or false claims, not temporal source selection or simplification loss. |
| [Beauchemin et al. (2025), *JUDGEBERT*](https://aclanthology.org/2025.emnlp-main.5/) | Introduces FrJUDGE and a metric for legal meaning preservation between French legal sentences. | No | Partial | Yes | No | Compares texts without testing whether the source applies at the queried time. |
| [Rabbani et al. (2026), *Decode the Law / SIMPLE-LAW*](https://aclanthology.org/2026.lrec-1.45/) | Provides 6,000+ original-simplified legal passage pairs and evaluates LLM simplification using readability, semantic, and hallucination metrics. | No | Yes | Yes | No | Evaluates rewriting of a given passage, but not retrieval from evolving versions. |
| [Cymbler et al. (2026), *Temporal Misgrounding in Legal RAG*](https://arxiv.org/abs/2608.09393) | Introduces FiscalQA Pro and version-aware retrieval for French tax law; shows static RAG can cite real but inapplicable law. | Yes | No | Partial | No | Verifies date-applicable retrieval and answer facts, but not meaning loss introduced by plain-language simplification. |
| [Nam et al. (2026), *TimelyRAG*](https://arxiv.org/abs/2609.11572) | Adds temporal distance to ranking for overlapping, evolving laws, policies, and regulations; introduces TimelyQABench. | Yes | No | No | No | Improves version-appropriate retrieval without simplifying legal text or checking preservation of operative details. |
| [Zheng et al. (2026), *Pub-LawBench*](https://aclanthology.org/2026.acl-long.1680/) | Evaluates public-facing legal QA and generation for legal normativity, relevance, and format usability. | No | Partial | Partial | No | Motivates accessible public-facing answers, but excludes RAG and does not isolate the interaction studied here. |
| **Proposed project: Applicability-Preserving Simplification for Evolving F-1 Regulations** | Version-aware F-1 retrieval, plain-language generation, structured meaning checks, and selective clarification or abstention. | **Yes** | **Yes** | **Yes** | **Yes** | Tests the unstudied intersection and introduces a controlled benchmark and safety-oriented metric. |

## Synthesis of the research gap

The literature forms two well-developed but mostly separate strands. The first studies **temporal grounding**: Cymbler et al. and Nam et al. show that systems must retrieve the version of a rule applicable to the relevant date. The second studies **accessible yet faithful generation**: Garimella et al., Rabbani et al., Beauchemin et al., and Agrawal and Carpuat examine simplification and meaning preservation. Legal-groundedness and hallucination research further checks whether an answer agrees with the context it was given.

This leaves an important blind spot. A simplified answer can be faithful to its retrieved passage and still be legally wrong because the passage was not applicable at the relevant time. Conversely, a system can retrieve the correct version and then omit an exception, reverse a negation, or alter a deadline while simplifying it. The reviewed work does not jointly evaluate these hazards in an end-to-end public-facing legal explanation system.

## Defensible novelty claim

> To the best of our knowledge, prior work evaluates either temporal applicability in legal retrieval or meaning preservation in legal simplification, but not their interaction. We study **applicability-preserving legal simplification**: an explanation is considered safe only when it is grounded in the rule version applicable to the user's date and preserves that rule's legally operative content. We operationalize this interaction with controlled error types and a **false-safe rate** measuring how often a system accepts an explanation that is fluent and source-faithful yet legally unsafe because of wrong-version grounding or meaning loss.

Use **“to the best of our knowledge”**, rather than “the first ever,” unless a broader systematic review confirms absolute priority.

## What makes this more than a standard combination

The contribution should not be described merely as “temporal RAG plus simplification.” It should be framed as a new evaluation problem with an interaction-specific failure class:

1. **New failure class:** faithful-to-the-wrong-version explanations.
2. **New benchmark design:** each legal fact is paired with controlled variants—correct and faithful, wrong version but faithful, omitted condition, omitted exception, reversed negation, and changed number or deadline.
3. **New decision objective:** accept, request a missing date, revise, or abstain—not simply generate an answer.
4. **New primary metric:** **false-safe rate**, the proportion of unsafe explanations incorrectly approved by the verifier.
5. **Interaction test:** compare standard RAG, temporal-only, meaning-only, and combined systems. The combined model must reduce false-safe errors beyond either single component.

## Evidence required in the final report

The matrix supports a **novelty hypothesis**. The following experiments would provide empirical evidence:

| Research question | Comparison | Expected evidence |
|---|---|---|
| Does temporal grounding matter? | Standard RAG vs. temporal-only retrieval | Better applicable-version retrieval and fewer wrong-version answers. |
| Does meaning verification matter? | Unverified simplification vs. meaning-only verifier | Better detection of omitted conditions and exceptions, negation reversals, and altered dates or quantities. |
| Is the joint formulation necessary? | Temporal-only and meaning-only vs. combined system | The combined system achieves the lowest false-safe rate across both error families. |
| Does clarification help when time is underspecified? | Forced answering vs. date clarification or abstention | Lower false-safe rate at an explicitly reported coverage level. |
| Are results robust beyond memorized wording? | Random split vs. split by legal provision | Performance remains meaningful when test provisions are unseen during development. |

## Recommended proposal positioning

Do not claim that assembling known components is itself novel. Claim that the project:

- identifies and formalizes a previously under-evaluated interaction between **source applicability** and **output fidelity**;
- creates a small, auditable benchmark for that interaction in a narrowly scoped F-1 setting;
- introduces a safety-focused evaluation protocol and false-safe metric; and
- tests through ablations whether solving both gates is necessary.

The novelty therefore lies in the problem formulation, benchmark, failure taxonomy, and evaluation—not in presenting standard RAG or NLI components as new algorithms.

## Core references

1. Garimella, A., et al. (2022). “Text Simplification for Legal Domain: Insights and Challenges.” *NLLP 2022*. https://doi.org/10.18653/v1/2022.nllp-1.28
2. Agrawal, S., & Carpuat, M. (2024). “Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Reading Comprehension.” *TACL, 12*, 432–448. https://doi.org/10.1162/tacl_a_00653
3. Trautmann, D., et al. (2024). “Measuring the Groundedness of Legal Question-Answering Systems.” *NLLP 2024*. https://aclanthology.org/2024.nllp-1.14/
4. Hu, Y., et al. (2025). “Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering.” *COLING 2025*. https://aclanthology.org/2025.coling-main.298/
5. Beauchemin, D., et al. (2025). “JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences.” *EMNLP 2025*. https://doi.org/10.18653/v1/2025.emnlp-main.5
6. Rabbani, M. D., et al. (2026). “Decode the Law: Towards Legal Text Simplification with Large Language Models.” *LREC 2026*. https://doi.org/10.63317/2i7en3xycmhi
7. Cymbler, R., Guez, D., & Fabre, L. (2026). “Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law.” *ICML 2026 AI4Law Workshop*. https://arxiv.org/abs/2608.09393
8. Nam, Y., et al. (2026). “TimelyRAG: Semantic-Temporal Hybrid Retrieval for Time-Critical Question Answering in Overlapping-Evolving Documents.” https://arxiv.org/abs/2609.11572
9. Zheng, Q., et al. (2026). “Pub-LawBench: Public-Oriented Benchmarking for LegalAI.” *ACL 2026*. https://aclanthology.org/2026.acl-long.1680/
