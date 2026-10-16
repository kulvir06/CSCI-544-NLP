from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from api import config
from corpus.schema import CorpusUse, DocumentStatus, SourceTier
from prototype.pipeline import run_prototype
from prototype.schema import GeneratorMode, PrototypeRequest, PrototypeStatus
from retrieval.schema import IndexedPassage
from verification.schema import SafetyVerdict


class EndToEndPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.patch_stack = ExitStack()
        self.index_path = self.base / "index" / "approved_passages.jsonl"
        self.patch_stack.enter_context(
            patch.object(config, "RETRIEVAL_INDEX_PATH", self.index_path)
        )
        self.patch_stack.enter_context(
            patch.object(
                config,
                "SEMANTIC_PASSAGE_INDEX_PATH",
                self.base / "index" / "semantic_passages.jsonl",
            )
        )
        self.index_path.parent.mkdir(parents=True)

        temporal = IndexedPassage(
            passage_id="stem-p001",
            document_id="stem-doc",
            text="The F-1 student must submit Form I-983 for the STEM OPT extension.",
            page_number=4,
            passage_relevance_score=7.0,
            title="STEM OPT guidance",
            issuer="USCIS",
            source_url="https://www.uscis.gov/stem",
            source_tier=SourceTier.FEDERAL_GUIDANCE,
            corpus_use=CorpusUse.TEMPORAL_AND_MEANING,
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 12, 31),
            date_evidence="Effective interval confirmed.",
            document_status=DocumentStatus.HISTORICAL,
        )
        meaning_only = IndexedPassage(
            passage_id="cap-gap-p001",
            document_id="cap-gap-doc",
            text="An F-1 cap-gap employment extension may continue work authorization.",
            page_number=2,
            passage_relevance_score=6.0,
            title="Cap-gap guidance",
            issuer="USCIS",
            source_url="https://www.uscis.gov/cap-gap",
            source_tier=SourceTier.FEDERAL_GUIDANCE,
            corpus_use=CorpusUse.MEANING_ONLY,
            document_status=DocumentStatus.UNKNOWN,
        )
        self.index_path.write_text(
            temporal.model_dump_json() + "\n" + meaning_only.model_dump_json() + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.patch_stack.close()
        self.temporary_directory.cleanup()

    def test_applicable_faithful_answer_is_released_with_citation(self):
        response = run_prototype(
            PrototypeRequest(
                question="What form is required for the STEM OPT extension?",
                relevant_date=date(2025, 6, 1),
            )
        )

        self.assertEqual(response.status, PrototypeStatus.ANSWERED)
        self.assertIsNotNone(response.answer)
        self.assertEqual(response.verification.verdict, SafetyVerdict.SAFE)
        self.assertEqual(response.citations[0].passage_id, "stem-p001")
        self.assertEqual(response.citations[0].page_number, 4)

    def test_wrong_version_answer_is_blocked(self):
        response = run_prototype(
            PrototypeRequest(
                question="What form is required for the STEM OPT extension?",
                relevant_date=date(2024, 6, 1),
            )
        )

        self.assertEqual(response.status, PrototypeStatus.UNSAFE_BLOCKED)
        self.assertIsNone(response.answer)
        self.assertIsNotNone(response.draft_explanation)

    def test_meaning_corruption_is_blocked(self):
        response = run_prototype(
            PrototypeRequest(
                question="What form is required for the STEM OPT extension?",
                relevant_date=date(2025, 6, 1),
                generator_mode=GeneratorMode.PROVIDED_CANDIDATE,
                candidate_explanation="The F-1 student must submit Form I-765.",
            )
        )

        self.assertEqual(response.status, PrototypeStatus.UNSAFE_BLOCKED)
        self.assertEqual(response.verification.verdict, SafetyVerdict.UNSAFE)

    def test_temporally_unknown_evidence_causes_abstention(self):
        response = run_prototype(
            PrototypeRequest(
                question="How does the cap-gap employment extension work?",
                relevant_date=date(2025, 6, 1),
            )
        )

        self.assertEqual(response.status, PrototypeStatus.ABSTAINED)
        self.assertIsNone(response.answer)

    def test_no_relevant_passage_returns_no_evidence(self):
        response = run_prototype(
            PrototypeRequest(question="What are the rules for underwater basket weaving?")
        )

        self.assertEqual(response.status, PrototypeStatus.NO_EVIDENCE)
        self.assertIsNone(response.answer)
        self.assertIsNone(response.verification)

    def test_provided_mode_requires_candidate(self):
        with self.assertRaises(ValidationError):
            PrototypeRequest(
                question="Question",
                generator_mode=GeneratorMode.PROVIDED_CANDIDATE,
            )


if __name__ == "__main__":
    unittest.main()
