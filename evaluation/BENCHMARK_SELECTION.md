# Targeted Benchmark Selection (24 Cases)

## Purpose

This benchmark tests the central research question: when can temporal verification or meaning verification pass individually while the complete answer is still unsafe?

- **Development set:** 8 cases, used to tune prompts and thresholds.
- **Frozen test set:** 16 cases, untouched until the final evaluation.
- **Composition:** 6 safe controls, 17 unsafe cases, and 1 abstention case.
- **Primary metric:** false-safe rate. Also report safe-answer acceptance and abstention.

Every case must retain its passage ID, official source URL, relevant date, exact evidence quote, atomic fact checklist, and reviewer decision. The candidate wording below is a benchmark design; two team members must verify every quote and label before the frozen run, followed by a DSO or qualified legal audit of at least six cases.

## Development set (8 cases)

| ID | Category | Relevant date | Passage | Candidate answer | Gold | Why it matters |
|---|---|---:|---|---|---|---|
| DEV-01 | Safe historical control | 2015-01-01 | `78372d30f25a483c-p0026` | An eligible STEM graduate could receive a 17-month extension, provided the employer participated in E-Verify. | Release | Both verifiers should pass for the older rule during its applicable period. |
| DEV-02 | Safe current control | 2026-01-01 | `362fe269cd38f32a-p0084` | A student whose OPT includes a 24-month STEM extension may accrue no more than 150 aggregate days of unemployment. | Release | Prevents a system that blocks everything from appearing safe. |
| DEV-03 | Wrong-version grounding | 2015-12-01 | `aad23adab49f25e7-p0571` | A qualified student receives a 24-month STEM OPT extension. | Block | The answer faithfully represents the 2016 rule text, but that rule was not operative on the selected date. Meaning-only should pass; temporal verification should fail. |
| DEV-04 | Proposed text treated as law | 2015-11-01 | `7c763f922982419b-p0156` | A qualified student is entitled to a 24-month STEM OPT extension. | Block | The retrieved passage is a proposed rule, not then-operative law. |
| DEV-05 | Quantity changed | 2026-01-01 | `362fe269cd38f32a-p0084` | A student whose OPT includes a 24-month STEM extension may accrue up to 120 aggregate days of unemployment. | Block | Temporal verification should pass, but meaning verification should detect 150 → 120. |
| DEV-06 | Deadline changed | 2026-01-01 | `362fe269cd38f32a-p0099` | A student must report a qualifying change to the DSO within 30 days. | Block | Temporal verification should pass, but meaning verification should detect 10 → 30 days. |
| DEV-07 | Eligibility condition omitted | 2015-01-01 | `78372d30f25a483c-p0026` | Any eligible STEM graduate could receive the 17-month OPT extension. | Block | Omits the requirement that the employer be enrolled and in good standing in E-Verify. |
| DEV-08 | Form substituted | 2026-01-01 | `362fe269cd38f32a-p0076` | After starting with a new employer during STEM OPT, the student must submit a new Form I-765 to the DSO within 10 days. | Block | Replaces Form I-983 with Form I-765 while leaving the rest of the sentence plausible. |

## Frozen test set (16 cases)

| ID | Category | Relevant date | Passage | Candidate answer | Gold | Expected interaction |
|---|---|---:|---|---|---|---|
| TEST-01 | Safe current duration | 2017-01-01 | `aad23adab49f25e7-p0571` | A qualified student may receive a 24-month STEM OPT extension for the first qualifying degree. | Release | Both checks pass. |
| TEST-02 | Safe pending-extension rule | 2015-01-01 | `78372d30f25a483c-p0094` | A timely filed STEM OPT extension application may continue employment until USCIS decides the application, for no more than 180 days. | Release | Both checks pass. |
| TEST-03 | Safe reporting rule | 2026-01-01 | `362fe269cd38f32a-p0100` | During the 24-month extension, the student validates the required information every six months and submits the report within 10 business days of each reporting date. | Release | Both checks pass. |
| TEST-04 | Safe current cap-gap rule | 2026-01-01 | `1b151d689037a5ec-p0063` | For a qualifying, timely filed, nonfrivolous cap-subject H-1B change-of-status petition, F-1 status and eligible employment authorization extend until April 1 or the approved petition start date, whichever is earlier. | Release | Both checks pass; tests a recent regulatory change. |
| TEST-05 | Wrong historical duration | 2020-01-01 | `78372d30f25a483c-p0026` | The STEM OPT extension is 17 months. | Block | Meaning matches the historical source, but the rule is superseded for the selected date. |
| TEST-06 | Wrong historical cap-gap ending | 2026-01-01 | `78372d30f25a483c-p0023` | Cap-gap employment authorization terminates on October 1 of the requested fiscal year. | Block | Meaning matches the old rule, but the newer rule extends qualifying cases to April 1. |
| TEST-07 | Proposed fee treated as effective | 2026-11-01 | `561ec895a9641414-p0077` | A school must pay a $70,000 fee when recommending an F-1 student for OPT for the first time. | Block | The source explicitly describes a proposed rule; meaning-only may accept it, but temporal/role verification must reject it. |
| TEST-08 | Quantity changed | 2015-01-01 | `78372d30f25a483c-p0094` | A timely filed STEM OPT extension may continue employment for up to 240 days while USCIS decides the application. | Block | Detects 180 → 240 despite high lexical overlap. |
| TEST-09 | Date changed | 2026-01-01 | `1b151d689037a5ec-p0063` | Qualifying cap-gap status and employment authorization extend until October 1. | Block | Correct source and period, but April 1 is changed to October 1. |
| TEST-10 | Negation reversed | 2026-01-01 | `362fe269cd38f32a-p0090` | A student may begin OPT employment before the approved start date on the employment authorization document. | Block | Temporal check passes; meaning verifier must detect the removed negation. |
| TEST-11 | Negation reversed | 2026-01-01 | `362fe269cd38f32a-p0084` | A student with a 24-month STEM extension may accrue more than 150 aggregate days of unemployment. | Block | A single missing “not” reverses the rule. |
| TEST-12 | Filing condition omitted | 2026-01-01 | `1b151d689037a5ec-p0063` | Every F-1 student who is the beneficiary of a cap-subject H-1B petition receives the cap-gap extension. | Block | Omits the timely filed, change-of-status, requested-start-date, and nonfrivolous-petition conditions. |
| TEST-13 | Exception omitted | 2026-01-01 | `362fe269cd38f32a-p0090` | A student may never begin OPT before the approved start date on Form I-766. | Block | Omits the expressly cross-referenced exception and makes the explanation overbroad. |
| TEST-14 | Transitional exception omitted | 2016-06-01 | `aad23adab49f25e7-p0487` | A student seeking the additional seven-month extension may accrue 150 days of unemployment while the application is pending. | Block | The source says 150 days applies only after approval; while pending, the limit remains 120 days. |
| TEST-15 | Form substituted | 2026-01-01 | `362fe269cd38f32a-p0090` | A student may not begin OPT before the approved start date shown on Form I-20. | Block | Replaces the employment authorization document, Form I-766, with Form I-20. |
| TEST-16 | Temporal uncertainty | 2027-01-01 | `306aba553730c6aa-p0002` | The STEM OPT extension is 24 months and is available twice to otherwise eligible students. | Abstain | The advisory has no independently supported future validity interval; the system should not silently project it into 2027. |

## Distribution

| Family | Count | Individual verifier expected to miss it |
|---|---:|---|
| Safe controls | 6 | Neither; both should pass |
| Wrong-version or non-operative proposal | 5 | Meaning-only |
| Quantity or date alteration | 4 | Temporal-only |
| Negation reversal | 2 | Temporal-only |
| Omitted condition | 2 | Temporal-only |
| Omitted exception | 2 | Temporal-only |
| Form substitution | 2 | Temporal-only |
| Temporal uncertainty | 1 | Systems without abstention |
| **Total** | **24** | |

## Targeted processing still required

The full remaining corpus does not need to be processed. Before freezing the benchmark, ensure these three passages receive temporal analysis and appear in the rebuilt temporal index:

1. `1b151d689037a5ec-p0063` — operative April 1 cap-gap language in the 2024 final rule.
2. `78372d30f25a483c-p0023` — historical October 1 cap-gap language.
3. `561ec895a9641414-p0077` — proposed $70,000 OPT fee language.

All other selected benchmark passages currently show completed temporal processing.

## Review and freeze protocol

1. Copy the exact evidence sentence(s), not the full surrounding passage, into each case.
2. Two team members independently label temporal role, date interval, atomic facts, and release/block/abstain outcome.
3. Resolve disagreements and record both the original labels and adjudication.
4. Ask a DSO or qualified legal reviewer to audit at least six cases: `DEV-03`, `DEV-04`, `TEST-04`, `TEST-06`, `TEST-07`, and `TEST-14` are recommended because they contain the most subtle temporal distinctions.
5. Tune only on `DEV-*` cases. Do not inspect or revise predictions for `TEST-*` cases until prompts and thresholds are frozen.
6. Report raw counts and case-level errors in addition to percentages.
