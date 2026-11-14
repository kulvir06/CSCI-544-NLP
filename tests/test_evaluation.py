from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.mutations import (
    change_first_date,
    change_first_form,
    change_first_quantity,
    remove_clause_from_cue,
    reverse_first_negation,
)
from evaluation.runner import load_cases, run_evaluation, save_report
from evaluation.schema import ErrorType, EvaluationCase, GoldLabel, SystemConfiguration
from retrieval.schema import TemporalVerdict


CASE_FILE = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "cases"
    / "development_cases.jsonl"
)


class EvaluationTests(unittest.TestCase):
    def test_four_configuration_report_exposes_false_safe_reduction(self):
        report = run_evaluation(load_cases(CASE_FILE))

        self.assertEqual(report.dataset_status, "development_fixtures")
        self.assertEqual(report.total_cases, 9)
        self.assertEqual(report.temporal_verdict_accuracy, 1.0)
        self.assertEqual(
            report.systems[SystemConfiguration.STATIC_BASELINE].false_safe_rate,
            1.0,
        )
        self.assertGreater(
            report.systems[SystemConfiguration.TEMPORAL_ONLY].false_safe_rate,
            report.systems[SystemConfiguration.MEANING_ONLY].false_safe_rate,
        )
        self.assertEqual(
            report.systems[SystemConfiguration.COMBINED].false_safe_rate,
            0.0,
        )
        self.assertEqual(
            report.systems[SystemConfiguration.COMBINED].exact_accuracy,
            1.0,
        )
        self.assertEqual(
            report.systems[SystemConfiguration.STATIC_BASELINE].false_safe_count,
            7,
        )
        self.assertEqual(
            len(report.systems[SystemConfiguration.COMBINED].false_safe_wilson_95),
            2,
        )

    def test_report_is_serializable(self):
        report = run_evaluation(load_cases(CASE_FILE))
        with TemporaryDirectory() as temporary_directory:
            output = save_report(
                report,
                Path(temporary_directory) / "evaluation-report.json",
            )
            self.assertTrue(output.exists())

    def test_negative_mutation_helpers_make_subtle_changes(self):
        self.assertIn(
            "21 hours",
            change_first_quantity("Work is limited to 20 hours."),
        )
        self.assertIn(
            "I-765",
            change_first_form("The student must submit Form I-983."),
        )
        self.assertIn(
            "may begin",
            reverse_first_negation("The student may not begin employment."),
        )
        self.assertIn(
            "March 16, 2025",
            change_first_date("The deadline is March 15, 2025."),
        )
        self.assertEqual(
            remove_clause_from_cue(
                "Employment may continue unless authorization ends.",
                "unless",
            ),
            "Employment may continue.",
        )

    def test_automatic_temporal_verification_uses_supplied_source_passage(self):
        case = EvaluationCase(
            case_id="automatic-wrong-version",
            question="How long is the STEM OPT extension?",
            source_text="The STEM OPT extension is 17 months.",
            explanation="The STEM OPT extension is 17 months.",
            error_type=ErrorType.WRONG_VERSION,
            gold_label=GoldLabel.UNSAFE,
            temporal_verification_required=True,
            expected_temporal_verdict=TemporalVerdict.VERIFIED_NOT_APPLICABLE,
            relevant_date="2026-01-01",
            passage_id="historical-passage",
            automatic_temporal_verification=True,
        )
        with patch(
            "evaluation.runner.verify_source_temporality",
            return_value=(
                TemporalVerdict.VERIFIED_NOT_APPLICABLE,
                "historical-claim",
            ),
        ) as verification:
            report = run_evaluation([case])

        verification.assert_called_once_with(
            "historical-passage",
            "The STEM OPT extension is 17 months.",
            case.relevant_date,
        )
        result = report.cases[0]
        self.assertEqual(
            result.observed_temporal_verdict,
            TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        )
        self.assertEqual(result.matched_temporal_claim_id, "historical-claim")
        self.assertEqual(
            result.predictions[SystemConfiguration.TEMPORAL_ONLY].value,
            "unsafe",
        )


if __name__ == "__main__":
    unittest.main()
