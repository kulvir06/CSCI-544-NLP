# Evaluation Summary



- Cases: 10
- Temporal-verdict accuracy: 90.0%
- Mean critical-fact preservation: 55.8%

| Configuration | Exact accuracy | False-safe (count/rate, 95% CI) | Safe accepted (count/rate, 95% CI) | Abstention |
|---|---:|---:|---:|---:|
| Static Baseline | 20.0% | 7/7 (100.0%; 64.6%–100.0%) | 2/2 (100.0%; 34.2%–100.0%) | 0.0% |
| Temporal Only | 40.0% | 5/7 (71.4%; 35.9%–91.8%) | 1/2 (50.0%; 9.4%–90.5%) | 20.0% |
| Meaning Only | 80.0% | 1/7 (14.3%; 2.6%–51.3%) | 2/2 (100.0%; 34.2%–100.0%) | 0.0% |
| Combined | 90.0% | 0/7 (0.0%; 0.0%–35.4%) | 1/2 (50.0%; 9.4%–90.5%) | 20.0% |

## Diagnostic takeaway

Temporal-only verification falsely approved 5 unsafe case(s), and meaning-only verification falsely approved 1. The combined verifier falsely approved 0, while abstaining on 20.0% of cases.

## Case-level diagnostic

| Case | Gold | Temporal observed / expected | Meaning |
|---|---|---|---|
| TEST-03 | safe | insufficient_temporal_evidence / verified_applicable | pass |
| TEST-04 | safe | verified_applicable / verified_applicable | pass |
| TEST-05 | unsafe | verified_not_applicable / verified_not_applicable | pass |
| TEST-07 | unsafe | verified_not_applicable / verified_not_applicable | fail |
| TEST-10 | unsafe | verified_applicable / verified_applicable | fail |
| TEST-12 | unsafe | verified_applicable / verified_applicable | fail |
| TEST-13 | unsafe | verified_applicable / verified_applicable | fail |
| TEST-14 | unsafe | verified_applicable / verified_applicable | fail |
| TEST-15 | unsafe | verified_applicable / verified_applicable | fail |
| TEST-16 | abstain | insufficient_temporal_evidence / insufficient_temporal_evidence | fail |
