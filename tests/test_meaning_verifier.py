import unittest

from retrieval.schema import TemporalVerdict
from verification.meaning import verify_meaning
from verification.safety import verify_safety
from verification.schema import (
    FactStatus,
    MeaningVerificationRequest,
    MeaningVerdict,
    SafetyVerificationRequest,
    SafetyVerdict,
)


class MeaningVerifierTests(unittest.TestCase):
    def test_preserves_numeric_limit_and_negative_meaning(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text=(
                    "An F-1 student may work no more than 20 hours per week "
                    "while school is in session."
                ),
                explanation=(
                    "During the school session, an F-1 student may work up to "
                    "20 hours each week."
                ),
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.PASS)
        self.assertEqual(response.critical_fact_preservation, 1.0)

    def test_detects_changed_quantity(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text="An F-1 student may work no more than 20 hours per week.",
                explanation="An F-1 student may work 40 hours per week.",
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.FAIL)
        self.assertTrue(
            any(check.status == FactStatus.CONTRADICTED for check in response.checks)
        )

    def test_detects_substituted_form(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text="The student must submit Form I-983 to the DSO.",
                explanation="The student must submit Form I-765 to the DSO.",
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.FAIL)
        self.assertGreater(response.contradicted_fact_count, 0)

    def test_detects_altered_calendar_date(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text="The filing window closes on March 15, 2025.",
                explanation="The filing window closes on March 30, 2025.",
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.FAIL)
        self.assertGreater(response.contradicted_fact_count, 0)

    def test_detects_omitted_exception(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text=(
                    "An F-1 student may continue employment unless the work "
                    "authorization has ended."
                ),
                explanation="An F-1 student may continue employment.",
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.FAIL)
        self.assertGreater(response.missing_fact_count, 0)

    def test_detects_omitted_condition(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text=(
                    "A student is eligible for STEM OPT only if the employer "
                    "participates in E-Verify."
                ),
                explanation=(
                    "A student is eligible for STEM OPT and the employer "
                    "participates in E-Verify."
                ),
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.FAIL)

    def test_no_atomic_facts_requires_review(self):
        response = verify_meaning(
            MeaningVerificationRequest(
                source_text="This chapter provides general background information.",
                explanation="This is general background information.",
            )
        )

        self.assertEqual(response.verdict, MeaningVerdict.REVIEW)

    def test_combined_verifier_passes_only_when_both_stages_pass(self):
        response = verify_safety(
            SafetyVerificationRequest(
                source_text="The student must submit Form I-983.",
                explanation="The student must submit Form I-983.",
                temporal_verdict=TemporalVerdict.VERIFIED_APPLICABLE,
                temporal_verification_required=True,
            )
        )

        self.assertEqual(response.verdict, SafetyVerdict.SAFE)

    def test_combined_verifier_abstains_on_temporal_uncertainty(self):
        response = verify_safety(
            SafetyVerificationRequest(
                source_text="The student must submit Form I-983.",
                explanation="The student must submit Form I-983.",
                temporal_verdict=TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
                temporal_verification_required=True,
            )
        )

        self.assertEqual(response.verdict, SafetyVerdict.ABSTAIN)

    def test_combined_verifier_rejects_wrong_version(self):
        response = verify_safety(
            SafetyVerificationRequest(
                source_text="The student must submit Form I-983.",
                explanation="The student must submit Form I-983.",
                temporal_verdict=TemporalVerdict.VERIFIED_NOT_APPLICABLE,
                temporal_verification_required=True,
            )
        )

        self.assertEqual(response.verdict, SafetyVerdict.UNSAFE)


if __name__ == "__main__":
    unittest.main()
