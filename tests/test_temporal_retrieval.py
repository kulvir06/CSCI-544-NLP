from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from api import config
from corpus.pipeline import ingest_upload
from corpus.review import submit_review
from corpus.schema import (
    CorpusUse,
    DocumentMetadata,
    DocumentReviewRequest,
    DocumentStatus,
    MetadataCorrections,
    ReviewDecision,
)
from retrieval.index import build_index, load_index
from retrieval.schema import (
    PassageTemporalStatus,
    RetrievalRequest,
    RetrievalStatus,
    TemporalVerdict,
)
from retrieval.search import retrieve


class TemporalRetrievalTests(unittest.TestCase):
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
            ("INDEX_DIR", self.base / "index"),
            ("RETRIEVAL_INDEX_PATH", self.base / "index" / "approved_passages.jsonl"),
            ("MANIFEST_DIR", self.base / "manifests"),
            ("MANIFEST_PATH", self.base / "manifests" / "documents.jsonl"),
        ):
            self.patch_stack.enter_context(patch.object(config, name, value))

        temporal = self._ingest(
            "uscis-stem.txt",
            "F-1 students on STEM OPT must submit Form I-983 for a 24 month extension.",
        )
        meaning_only = self._ingest(
            "uscis-cap-gap.txt",
            "An F-1 cap-gap employment extension may continue work authorization.",
        )
        snapshot = self._ingest(
            "uscis-cpt.txt",
            "F-1 curricular practical training CPT requires authorization from the DSO.",
        )
        self._ingest(
            "unreviewed-campus.txt",
            "F-1 students may work in on-campus employment for 20 hours per week.",
        )

        submit_review(
            temporal,
            DocumentReviewRequest(
                decision=ReviewDecision.APPROVED,
                corpus_use=CorpusUse.TEMPORAL_AND_MEANING,
                reviewer="Test Reviewer",
                corrections=MetadataCorrections(
                    issuer="USCIS",
                    source_url="https://www.uscis.gov/stem",
                    effective_from=date(2025, 1, 1),
                    effective_to=date(2025, 12, 31),
                    date_evidence="The uploaded notice states this effective interval.",
                    status=DocumentStatus.HISTORICAL,
                ),
            ),
        )
        submit_review(
            meaning_only,
            DocumentReviewRequest(
                decision=ReviewDecision.APPROVED,
                corpus_use=CorpusUse.MEANING_ONLY,
                reviewer="Test Reviewer",
                corrections=MetadataCorrections(
                    issuer="USCIS",
                    source_url="https://www.uscis.gov/cap-gap",
                ),
            ),
        )
        submit_review(
            snapshot,
            DocumentReviewRequest(
                decision=ReviewDecision.APPROVED,
                corpus_use=CorpusUse.TEMPORAL_AND_MEANING,
                reviewer="Test Reviewer",
                corrections=MetadataCorrections(
                    issuer="USCIS",
                    source_url="https://www.uscis.gov/cpt",
                    valid_as_of=date(2025, 7, 1),
                    date_evidence="The uploaded snapshot says current as of July 1, 2025.",
                    status=DocumentStatus.CURRENT,
                ),
            ),
        )
        self.report = build_index()

    def tearDown(self):
        self.patch_stack.close()
        self.temporary_directory.cleanup()

    def _ingest(self, filename: str, text: str) -> str:
        result = ingest_upload(
            filename,
            text.encode("utf-8"),
            DocumentMetadata(original_filename=filename),
        )
        return result.document.document_id

    def test_index_contains_only_approved_documents(self):
        self.assertEqual(self.report.approved_documents, 3)
        self.assertEqual(self.report.indexed_documents, 3)
        self.assertEqual(len(load_index()), 3)

    def test_temporal_query_returns_applicable_version(self):
        response = retrieve(
            RetrievalRequest(
                query="What is required for a STEM OPT extension?",
                relevant_date=date(2025, 6, 1),
            )
        )

        self.assertEqual(response.temporal_verdict, TemporalVerdict.VERIFIED_APPLICABLE)
        self.assertEqual(response.hits[0].temporal_status, PassageTemporalStatus.APPLICABLE)

    def test_out_of_range_version_is_not_applicable(self):
        response = retrieve(
            RetrievalRequest(
                query="What is required for a STEM OPT extension?",
                relevant_date=date(2024, 6, 1),
            )
        )

        self.assertEqual(
            response.temporal_verdict,
            TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        )

    def test_meaning_only_document_causes_temporal_abstention(self):
        response = retrieve(
            RetrievalRequest(
                query="How does the cap-gap employment extension work?",
                relevant_date=date(2025, 6, 1),
            )
        )

        self.assertEqual(
            response.temporal_verdict,
            TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
        )
        self.assertEqual(
            response.hits[0].temporal_status,
            PassageTemporalStatus.INSUFFICIENT_EVIDENCE,
        )

    def test_query_without_date_can_use_meaning_only_document(self):
        response = retrieve(
            RetrievalRequest(query="Explain the cap-gap employment extension.")
        )

        self.assertEqual(response.retrieval_status, RetrievalStatus.FOUND)
        self.assertEqual(response.temporal_verdict, TemporalVerdict.NOT_REQUESTED)

    def test_snapshot_only_supports_its_confirmed_as_of_date(self):
        exact = retrieve(
            RetrievalRequest(
                query="Does CPT require DSO authorization?",
                relevant_date=date(2025, 7, 1),
            )
        )
        other_date = retrieve(
            RetrievalRequest(
                query="Does CPT require DSO authorization?",
                relevant_date=date(2025, 8, 1),
            )
        )

        self.assertEqual(exact.temporal_verdict, TemporalVerdict.VERIFIED_APPLICABLE)
        self.assertEqual(
            other_date.temporal_verdict,
            TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
        )

    def test_unreviewed_document_is_not_retrieved(self):
        response = retrieve(RetrievalRequest(query="on-campus 20 hours"))

        self.assertEqual(
            response.retrieval_status,
            RetrievalStatus.NO_RELEVANT_PASSAGES,
        )


if __name__ == "__main__":
    unittest.main()
