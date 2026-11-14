import unittest

from scripts.summarize_evaluation import build_markdown


class EvaluationSummaryTests(unittest.TestCase):
    def test_candidate_report_is_clearly_marked_diagnostic(self):
        markdown = build_markdown(
            {
                "dataset_status": "benchmark_candidates",
                "total_cases": 1,
                "temporal_verdict_accuracy": 0.0,
                "mean_critical_fact_preservation": 1.0,
                "systems": {
                    "combined": {
                        "exact_accuracy": 0.0,
                        "false_safe_count": 0,
                        "unsafe_cases": 1,
                        "false_safe_rate": 0.0,
                        "false_safe_wilson_95": [0.0, 0.7935],
                        "safe_accepted_count": 0,
                        "safe_cases": 0,
                        "safe_acceptance_rate": 0.0,
                        "safe_acceptance_wilson_95": [0.0, 0.0],
                        "abstention_rate": 1.0,
                    }
                },
                "cases": [
                    {
                        "case_id": "DEV-01",
                        "gold_label": "unsafe",
                        "observed_temporal_verdict": "insufficient_temporal_evidence",
                        "expected_temporal_verdict": "verified_applicable",
                        "meaning_verdict": "fail",
                    }
                ],
            }
        )
        self.assertIn("Diagnostic only", markdown)
        self.assertIn("0/1", markdown)
        self.assertIn("DEV-01", markdown)


if __name__ == "__main__":
    unittest.main()
