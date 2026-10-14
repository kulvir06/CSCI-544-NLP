from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest
from unittest.mock import patch

from api import config
from corpus.pipeline import ingest_upload
from corpus.review import (
    build_review_queue,
    load_retrieval_ready_documents,
    submit_review,
)
from corpus.schema import (
    DocumentMetadata,
    DocumentReviewRequest,
    DocumentStatus,
    MetadataCorrections,
    ReviewDecision,
)
from scripts.export_review_queue import export_review_queue


class MetadataReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.patch_stack = ExitStack()
        for name, value in (
            ("RAW_DOCUMENT_DIR", self.base / "raw"),
            ("NORMALIZED_DOCUMENT_DIR", self.base / "normalized"),
            ("PASSAGE_DIR", self.base / "passages"),
            ("REPORT_DIR", self.base / "reports"),
            ("REVIEW_DIR", self.base / "reviews"),
            ("REVIEW_LOG_PATH", self.base / "reviews" / "document_reviews.jsonl"),
            ("MANIFEST_DIR", self.base / "manifests"),
            ("MANIFEST_PATH", self.base / "manifests" / "documents.jsonl"),
        ):
            self.patch_stack.enter_context(patch.object(config, name, value))

        result = ingest_upload(
            "uscis-stem.txt",
            b"F-1 students on STEM OPT must submit Form I-983 for a 24 month extension.",
            DocumentMetadata(original_filename="uscis-stem.txt"),
        )
        self.document_id = result.document.document_id

    def tearDown(self):
        self.patch_stack.close()
        self.temporary_directory.cleanup()

    def test_unreviewed_document_appears_in_queue(self):
        queue = build_review_queue()

        self.assertEqual(len(queue), 1)
        self.assertIn("missing_source_url", queue[0].blocking_issues)
        self.assertIn("missing_effective_date", queue[0].blocking_issues)

    def test_approval_requires_complete_temporal_provenance(self):
        request = DocumentReviewRequest(
            decision=ReviewDecision.APPROVED,
            reviewer="Team Member",
        )

        with self.assertRaisesRegex(ValueError, "Cannot approve"):
            submit_review(self.document_id, request)

    def test_complete_approval_is_audited_and_retrieval_ready(self):
        request = DocumentReviewRequest(
            decision=ReviewDecision.APPROVED,
            reviewer="Team Member",
            corrections=MetadataCorrections(
                issuer="USCIS",
                source_url="https://www.uscis.gov/example",
                effective_from=date(2025, 1, 1),
                status=DocumentStatus.CURRENT,
                rule_family="stem_opt",
            ),
            notes="Checked against the uploaded document.",
        )
        review = submit_review(self.document_id, request)

        self.assertEqual(review.decision, ReviewDecision.APPROVED)
        self.assertEqual(build_review_queue(), [])
        self.assertEqual(len(load_retrieval_ready_documents()), 1)
        self.assertEqual(
            len(config.REVIEW_LOG_PATH.read_text(encoding="utf-8").splitlines()),
            1,
        )

    def test_queue_can_be_exported_for_team_review(self):
        output_path = export_review_queue(self.base / "queue.csv")
        with output_path.open(encoding="utf-8", newline="") as queue_file:
            rows = list(csv.DictReader(queue_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document_id"], self.document_id)
        self.assertIn("missing_source_url", rows[0]["blocking_issues"])


if __name__ == "__main__":
    unittest.main()
