from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from api import config
from benchmark.draft import draft_cases
from benchmark.review import finalize_reviewed_cases, write_drafts
from corpus.schema import CorpusUse, DocumentStatus, SourceTier
from evaluation.runner import load_cases
from evaluation.schema import ErrorType, EvaluationCase, GoldLabel
from retrieval.schema import IndexedPassage, TemporalVerdict


class BenchmarkCurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.patch_stack = ExitStack()
        self.index_path = self.base / "index" / "approved_passages.jsonl"
        self.patch_stack.enter_context(
            patch.object(config, "RETRIEVAL_INDEX_PATH", self.index_path)
        )
        self.index_path.parent.mkdir(parents=True)
        passage = IndexedPassage(
            passage_id="doc-1-p0001",
            document_id="doc-1",
            text=(
                "An F-1 student may work no more than 20 hours per week only if "
                "authorized, unless authorization ends, and must submit Form I-983 "
                "by March 15, 2025."
            ),
            page_number=7,
            passage_relevance_score=8.0,
            title="Reviewed F-1 employment rule",
            issuer="USCIS",
            source_url="https://www.uscis.gov/example",
            source_tier=SourceTier.FEDERAL_GUIDANCE,
            corpus_use=CorpusUse.TEMPORAL_AND_MEANING,
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 12, 31),
            date_evidence="Effective interval confirmed by reviewer.",
            document_status=DocumentStatus.CURRENT,
        )
        self.index_path.write_text(passage.model_dump_json() + "\n", encoding="utf-8")

    def tearDown(self):
        self.patch_stack.close()
        self.temporary_directory.cleanup()

    def test_drafts_are_generated_from_passage_facts_with_provenance(self):
        cases = draft_cases(max_per_error=1, safe_controls=1)
        error_types = {case.error_type for case in cases}

        self.assertIn(ErrorType.SAFE, error_types)
        self.assertIn(ErrorType.QUANTITY_CHANGED, error_types)
        self.assertIn(ErrorType.DATE_CHANGED, error_types)
        self.assertIn(ErrorType.FORM_CHANGED, error_types)
        self.assertIn(ErrorType.NEGATION_REVERSED, error_types)
        self.assertIn(ErrorType.CONDITION_OMITTED, error_types)
        self.assertIn(ErrorType.EXCEPTION_OMITTED, error_types)
        self.assertIn(ErrorType.WRONG_VERSION, error_types)
        self.assertTrue(all(case.passage_id == "doc-1-p0001" for case in cases))
        self.assertTrue(all(case.page_number == 7 for case in cases))
        self.assertTrue(all(case.development_fixture for case in cases))

    def test_only_signed_off_rows_become_reviewed_benchmark_cases(self):
        cases = draft_cases(max_per_error=1, safe_controls=1)
        draft_path, review_path = write_drafts(
            cases,
            self.base / "draft.jsonl",
            self.base / "review.csv",
        )
        with review_path.open(encoding="utf-8", newline="") as review_file:
            reader = csv.DictReader(review_file)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["approve"] = "yes"
        rows[0]["reviewer"] = "Reviewer One"
        rows[0]["review_notes"] = "Checked against the cited uploaded passage."
        with review_path.open("w", encoding="utf-8", newline="") as review_file:
            writer = csv.DictWriter(review_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        output_path = self.base / "reviewed.jsonl"
        report = finalize_reviewed_cases(draft_path, review_path, output_path)
        finalized = load_cases(output_path)

        self.assertEqual(report["finalized_cases"], 1)
        self.assertEqual(report["failed_rows"], 0)
        self.assertFalse(finalized[0].development_fixture)
        self.assertEqual(finalized[0].reviewer, "Reviewer One")

    def test_reviewed_case_without_provenance_is_rejected(self):
        with self.assertRaises(ValidationError):
            EvaluationCase(
                case_id="invalid",
                source_text="The student must submit a form.",
                explanation="The student must submit a form.",
                error_type=ErrorType.SAFE,
                gold_label=GoldLabel.SAFE,
                observed_temporal_verdict=TemporalVerdict.NOT_REQUESTED,
                expected_temporal_verdict=TemporalVerdict.NOT_REQUESTED,
                development_fixture=False,
            )


if __name__ == "__main__":
    unittest.main()
