from contextlib import ExitStack
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from api import config
from corpus.pipeline import ingest_upload
from corpus.schema import DocumentMetadata, DocumentStatus, Passage, SourceTier
from retrieval.schema import RetrievalRequest, TemporalVerdict
from retrieval.search import retrieve
from temporal.enrichment import (
    _apply_source_anchors,
    _source_temporal_anchors,
    build_temporal_index,
    load_semantic_passages,
    load_temporal_claims,
)
from temporal.schema import (
    ClaimExtraction,
    ExtractedClaim,
    TemporalClaim,
    TemporalRole,
)
from temporal.ollama import OllamaClient


class FakeTemporalModel:
    temporal_model = "fake-temporal-model"
    embedding_model = "fake-embedding-model"

    def extract_claims(self, prompt: str) -> ClaimExtraction:
        return ClaimExtraction(
            claims=[
                ExtractedClaim(
                    claim_text="The historical STEM OPT extension lasted 17 months.",
                    evidence_quote="The previous STEM OPT rule provided a 17 month extension.",
                    temporal_role=TemporalRole.HISTORICAL,
                    valid_from=date(2008, 4, 8),
                    valid_to=date(2016, 5, 9),
                    date_basis="The context identifies this as the previous rule.",
                    confidence=0.96,
                ),
                ExtractedClaim(
                    claim_text="The operative STEM OPT extension lasts 24 months.",
                    evidence_quote="the final rule provides a 24 month STEM OPT extension.",
                    temporal_role=TemporalRole.OPERATIVE,
                    valid_from=date(2016, 5, 10),
                    date_basis="The passage states the final rule's effective date.",
                    confidence=0.98,
                ),
            ]
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


class LowConfidenceTemporalModel(FakeTemporalModel):
    temporal_model = "fast-primary-model"

    def extract_claims(self, prompt: str) -> ClaimExtraction:
        return ClaimExtraction(
            claims=[
                ExtractedClaim(
                    claim_text="The STEM OPT extension duration is unclear.",
                    evidence_quote=(
                        "The previous STEM OPT rule provided a 17 month extension."
                    ),
                    temporal_role=TemporalRole.UNKNOWN,
                    date_basis="The primary model could not resolve the role.",
                    confidence=0.4,
                )
            ]
        )


class TemporalEnrichmentTests(unittest.TestCase):
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
            ("INDEX_DIR", self.base / "index"),
            ("RETRIEVAL_INDEX_PATH", self.base / "index" / "temporal_claims.jsonl"),
            ("SEMANTIC_PASSAGE_INDEX_PATH", self.base / "index" / "semantic_passages.jsonl"),
            ("TEMPORAL_CACHE_DIR", self.base / "temporal_cache"),
            ("EMBEDDING_CACHE_DIR", self.base / "embedding_cache"),
            ("MANIFEST_DIR", self.base / "manifests"),
            ("MANIFEST_PATH", self.base / "manifests" / "documents.jsonl"),
        ):
            self.patch_stack.enter_context(patch.object(config, name, value))

    def tearDown(self):
        self.patch_stack.close()
        self.temporary_directory.cleanup()

    def test_one_passage_becomes_two_temporally_distinct_claims(self):
        result = ingest_upload(
            "stem-rule.txt",
            (
                "The previous STEM OPT rule provided a 17 month extension. "
                "Effective May 10, 2016, the final rule provides a 24 month "
                "STEM OPT extension."
            ).encode("utf-8"),
            DocumentMetadata(original_filename="stem-rule.txt"),
        )

        report = build_temporal_index(
            FakeTemporalModel(), document_ids={result.document.document_id}
        )
        claims = load_temporal_claims()

        self.assertEqual(report.indexed_claims, 2)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].source_passage_id, claims[1].source_passage_id)
        self.assertEqual(claims[0].temporal_role, TemporalRole.HISTORICAL)
        self.assertEqual(claims[1].temporal_role, TemporalRole.OPERATIVE)

    def test_full_source_effective_header_is_linked_to_passage_claim(self):
        result = ingest_upload(
            "final-rule.txt",
            (
                "DATES: This final rule is effective May 10, 2016.\n\n"
                "The final rule provides a 24 month STEM OPT extension."
            ).encode("utf-8"),
            DocumentMetadata(
                original_filename="final-rule.txt",
                title="STEM OPT Final Rule",
                status=DocumentStatus.HISTORICAL,
            ),
        )
        anchors, legacy_end = _source_temporal_anchors(
            {result.document.document_id: result.document}
        )
        extracted = ExtractedClaim(
            claim_text="The rule provides a 24 month STEM OPT extension.",
            evidence_quote="The final rule provides a 24 month STEM OPT extension.",
            temporal_role=TemporalRole.OPERATIVE,
            date_basis="No date in the selected passage.",
            confidence=0.9,
        )

        role, valid_from, valid_to, basis = _apply_source_anchors(
            result.document, extracted, anchors, legacy_end
        )

        self.assertEqual(role, TemporalRole.OPERATIVE)
        self.assertEqual(valid_from, date(2016, 5, 10))
        self.assertIsNone(valid_to)
        self.assertIn("Source effective-date anchor", basis)

    def test_proposed_document_cannot_be_promoted_to_operative(self):
        result = ingest_upload(
            "proposal.txt",
            b"The proposal provides a 24 month STEM OPT extension.",
            DocumentMetadata(
                original_filename="proposal.txt",
                title="STEM OPT Proposed Rule",
                status=DocumentStatus.PROPOSED,
            ),
        )
        extracted = ExtractedClaim(
            claim_text="The rule provides a 24 month STEM OPT extension.",
            evidence_quote="The proposal provides a 24 month STEM OPT extension.",
            temporal_role=TemporalRole.OPERATIVE,
            date_basis="No date.",
            confidence=0.9,
        )
        role, valid_from, valid_to, basis = _apply_source_anchors(
            result.document, extracted, {}, None
        )
        self.assertEqual(role, TemporalRole.PROPOSED)
        self.assertIsNone(valid_from)
        self.assertIsNone(valid_to)
        self.assertIn("Proposed Rule", basis)

    def test_build_reports_live_progress_and_cache_reuse(self):
        result = ingest_upload(
            "rule.txt",
            b"The final rule provides a 24 month STEM OPT extension.",
            DocumentMetadata(original_filename="rule.txt"),
        )
        messages: list[str] = []

        build_temporal_index(
            FakeTemporalModel(),
            document_ids={result.document.document_id},
            progress=messages.append,
        )
        first_run = "\n".join(messages)
        self.assertIn("Analyzing", first_run)
        self.assertIn("Done via model", first_run)
        self.assertIn("Index written", first_run)

        messages.clear()
        build_temporal_index(
            FakeTemporalModel(),
            document_ids={result.document.document_id},
            progress=messages.append,
        )
        second_run = "\n".join(messages)
        self.assertIn("Resume scan: 1/1 passages already complete", second_run)
        self.assertIn("No model requests needed", second_run)
        self.assertNotIn("Analyzing", second_run)

    def test_resume_counter_starts_after_completed_cache_entries(self):
        result = ingest_upload(
            "rule.txt",
            b"The final rule provides a 24 month STEM OPT extension.",
            DocumentMetadata(original_filename="rule.txt"),
        )
        passages = [
            Passage(
                passage_id=f"{result.document.document_id}-p{number:05d}",
                document_id=result.document.document_id,
                text=(
                    "The previous STEM OPT rule provided a 17 month extension. "
                    "The final rule provides a 24 month STEM OPT extension."
                ),
                block_index=number,
                relevance_score=5.0,
                selection_reasons=["rule:quantity"],
            )
            for number in (1, 2)
        ]

        with patch("temporal.enrichment._load_passages", return_value=passages):
            build_temporal_index(
                FakeTemporalModel(),
                document_ids={result.document.document_id},
                max_passages=1,
            )

        messages: list[str] = []
        model = FakeTemporalModel()
        with (
            patch("temporal.enrichment._load_passages", return_value=passages),
            patch.object(model, "extract_claims", wraps=model.extract_claims) as extraction,
        ):
            report = build_temporal_index(
                model,
                document_ids={result.document.document_id},
                progress=messages.append,
            )

        output = "\n".join(messages)
        extraction.assert_called_once()
        self.assertEqual(report.source_passages, 2)
        self.assertIn("Resume scan: 1/2 passages already complete", output)
        self.assertIn(f"[2/2] Analyzing {passages[1].passage_id}", output)
        self.assertNotIn(f"Analyzing {passages[0].passage_id}", output)

    def test_parallel_workers_preserve_passage_order(self):
        result = ingest_upload(
            "rule.txt",
            b"The final rule provides a 24 month STEM OPT extension.",
            DocumentMetadata(original_filename="rule.txt"),
        )
        passages = [
            Passage(
                passage_id=f"{result.document.document_id}-p{number:05d}",
                document_id=result.document.document_id,
                text=(
                    "The previous STEM OPT rule provided a 17 month extension. "
                    "The final rule provides a 24 month STEM OPT extension."
                ),
                block_index=number,
                relevance_score=5.0,
                selection_reasons=["rule:quantity"],
            )
            for number in (1, 2)
        ]
        messages: list[str] = []

        with patch("temporal.enrichment._load_passages", return_value=passages):
            report = build_temporal_index(
                FakeTemporalModel(),
                document_ids={result.document.document_id},
                workers=2,
                progress=messages.append,
            )
        claims = load_temporal_claims()

        self.assertEqual(report.source_passages, 2)
        self.assertEqual(report.indexed_claims, 4)
        self.assertIn("Using 2 parallel workers", "\n".join(messages))
        self.assertEqual(claims[0].source_passage_id, passages[0].passage_id)
        self.assertEqual(claims[2].source_passage_id, passages[1].passage_id)
        semantic = load_semantic_passages()
        self.assertEqual(semantic[0].next_passage_id, passages[1].passage_id)
        self.assertEqual(semantic[1].previous_passage_id, passages[0].passage_id)

    def test_targeted_run_forces_selected_passage_and_defers_other_work(self):
        result = ingest_upload(
            "rule.txt",
            b"The final rule provides a 24 month STEM OPT extension.",
            DocumentMetadata(original_filename="rule.txt"),
        )
        text = (
            "The previous STEM OPT rule provided a 17 month extension. "
            "The final rule provides a 24 month STEM OPT extension."
        )
        passages = [
            Passage(
                passage_id=f"{result.document.document_id}-p00001",
                document_id=result.document.document_id,
                text=text,
                block_index=1,
                relevance_score=5.0,
                selection_reasons=["rule:quantity"],
            ),
            Passage(
                passage_id=f"{result.document.document_id}-p00002",
                document_id=result.document.document_id,
                text=text,
                block_index=2,
                relevance_score=5.0,
                selection_reasons=[],
            ),
        ]
        model = FakeTemporalModel()
        messages: list[str] = []
        with (
            patch("temporal.enrichment._load_passages", return_value=passages),
            patch.object(model, "extract_claims", wraps=model.extract_claims) as extraction,
        ):
            report = build_temporal_index(
                model,
                document_ids={result.document.document_id},
                passage_ids={passages[1].passage_id},
                progress=messages.append,
            )

        extraction.assert_called_once()
        self.assertEqual(report.temporal_candidates, 2)
        self.assertEqual(report.indexed_claims, 2)
        semantic = {item.passage_id: item for item in load_semantic_passages()}
        self.assertEqual(
            semantic[passages[0].passage_id].temporal_scan_status, "pending"
        )
        self.assertEqual(
            semantic[passages[1].passage_id].temporal_scan_status, "completed"
        )
        self.assertIn(
            "explicit_target",
            semantic[passages[1].passage_id].temporal_routing_reasons,
        )
        self.assertIn("1 unrelated uncached routed passage(s) deferred", "\n".join(messages))

        with patch("temporal.enrichment._load_passages", return_value=passages):
            second_report = build_temporal_index(
                model,
                document_ids={result.document.document_id},
                passage_ids={passages[0].passage_id},
            )
        rebuilt_semantic = {
            item.passage_id: item for item in load_semantic_passages()
        }
        self.assertEqual(second_report.indexed_claims, 4)
        self.assertEqual(
            rebuilt_semantic[passages[1].passage_id].temporal_scan_status,
            "completed",
        )
        self.assertIn(
            "cached_temporal_analysis",
            rebuilt_semantic[passages[1].passage_id].temporal_routing_reasons,
        )

    def test_targeted_run_rejects_unknown_passage_id(self):
        result = ingest_upload(
            "rule.txt",
            b"The final rule provides a 24 month STEM OPT extension.",
            DocumentMetadata(original_filename="rule.txt"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown targeted passage ID"):
            build_temporal_index(
                FakeTemporalModel(),
                document_ids={result.document.document_id},
                passage_ids={"missing-passage"},
            )

    def test_ollama_repairs_percentage_confidence_and_end_only_interval(self):
        client = OllamaClient()
        response = {
            "message": {
                "content": (
                    '{"claims":[{"claim_text":"Cap-gap protection extends '
                    'until April 1.","evidence_quote":"Cap-gap protection '
                    'extends until April 1.","temporal_role":"operative",'
                    '"valid_from":null,"valid_to":"2025-04-01",'
                    '"date_basis":"annual cutoff","confidence":90}]}'
                )
            }
        }
        with patch.object(client, "_post", return_value=response):
            extraction = client.extract_claims("prompt")

        self.assertEqual(extraction.claims[0].confidence, 0.9)
        self.assertIsNone(extraction.claims[0].valid_to)

    def test_low_confidence_primary_result_uses_fallback_model(self):
        result = ingest_upload(
            "stem-rule.txt",
            (
                "The previous STEM OPT rule provided a 17 month extension. "
                "Effective May 10, 2016, the final rule provides a 24 month "
                "STEM OPT extension."
            ).encode("utf-8"),
            DocumentMetadata(original_filename="stem-rule.txt"),
        )
        fallback = FakeTemporalModel()
        with patch.object(
            fallback, "extract_claims", wraps=fallback.extract_claims
        ) as fallback_extraction:
            report = build_temporal_index(
                LowConfidenceTemporalModel(),
                fallback_model=fallback,
                document_ids={result.document.document_id},
            )

        claims = load_temporal_claims()
        fallback_extraction.assert_called_once()
        self.assertEqual(report.indexed_claims, 2)
        self.assertTrue(
            all(claim.extraction_model == fallback.temporal_model for claim in claims)
        )

    def test_existing_fallback_cache_skips_new_primary_model(self):
        result = ingest_upload(
            "stem-rule.txt",
            (
                "The previous STEM OPT rule provided a 17 month extension. "
                "Effective May 10, 2016, the final rule provides a 24 month "
                "STEM OPT extension."
            ).encode("utf-8"),
            DocumentMetadata(original_filename="stem-rule.txt"),
        )
        fallback = FakeTemporalModel()
        build_temporal_index(
            fallback,
            document_ids={result.document.document_id},
        )

        primary = LowConfidenceTemporalModel()
        with patch.object(
            primary, "extract_claims", wraps=primary.extract_claims
        ) as primary_extraction:
            build_temporal_index(
                primary,
                fallback_model=fallback,
                document_ids={result.document.document_id},
            )

        primary_extraction.assert_not_called()
        claims = load_temporal_claims()
        self.assertTrue(
            all(claim.extraction_model == fallback.temporal_model for claim in claims)
        )

    def test_unflagged_passage_is_searchable_but_temporally_unverified(self):
        result = ingest_upload(
            "employment.txt",
            b"F-1 students require authorization for off-campus employment.",
            DocumentMetadata(original_filename="employment.txt"),
        )
        model = FakeTemporalModel()
        with patch.object(
            model, "extract_claims", wraps=model.extract_claims
        ) as extraction:
            report = build_temporal_index(
                model, document_ids={result.document.document_id}
            )

        semantic = load_semantic_passages()
        self.assertEqual(report.semantic_passages, 1)
        self.assertEqual(report.temporal_candidates, 0)
        self.assertEqual(report.indexed_claims, 0)
        extraction.assert_not_called()
        self.assertFalse(semantic[0].temporal_scan_required)
        self.assertEqual(semantic[0].temporal_scan_status, "not_required")

        response = retrieve(
            RetrievalRequest(
                query="What authorization is required for F-1 off-campus employment?",
                relevant_date=date(2026, 1, 1),
            )
        )
        self.assertEqual(response.retrieval_status.value, "found")
        self.assertEqual(
            response.temporal_verdict,
            TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
        )

    def test_temporal_filter_runs_before_claim_ranking(self):
        result = ingest_upload(
            "stem-rule.txt",
            (
                "The previous STEM OPT rule provided a 17 month extension. "
                "Effective May 10, 2016, the final rule provides a 24 month "
                "STEM OPT extension."
            ).encode("utf-8"),
            DocumentMetadata(original_filename="stem-rule.txt"),
        )
        build_temporal_index(
            FakeTemporalModel(), document_ids={result.document.document_id}
        )

        old = retrieve(
            RetrievalRequest(
                query="How long is the STEM OPT extension?",
                relevant_date=date(2010, 6, 1),
            )
        )
        current = retrieve(
            RetrievalRequest(
                query="How long is the STEM OPT extension?",
                relevant_date=date(2020, 6, 1),
            )
        )

        self.assertEqual(old.temporal_verdict, TemporalVerdict.VERIFIED_APPLICABLE)
        self.assertIn("17 months", old.hits[0].passage.text)
        self.assertEqual(current.temporal_verdict, TemporalVerdict.VERIFIED_APPLICABLE)
        self.assertIn("24 months", current.hits[0].passage.text)

    def test_weak_applicable_claim_cannot_displace_direct_unknown_claim(self):
        direct_unknown = TemporalClaim(
            claim_id="duration-unknown",
            document_id="doc",
            source_passage_id="p1",
            text="The STEM OPT extension lasts 17 months.",
            evidence_quote="The STEM OPT extension lasts 17 months.",
            context="The STEM OPT extension lasts 17 months.",
            passage_relevance_score=5.0,
            source_tier=SourceTier.PRIMARY_LAW,
            document_status=DocumentStatus.HISTORICAL,
            temporal_role=TemporalRole.OPERATIVE,
            date_basis="No supported interval in this excerpt.",
            temporal_confidence=0.9,
            extraction_model="fake",
            embedding_model="none",
        )
        weak_applicable = TemporalClaim(
            claim_id="reporting-applicable",
            document_id="doc",
            source_passage_id="p2",
            text="An employer must report a STEM OPT extension change within 48 hours.",
            evidence_quote="An employer must report a STEM OPT extension change within 48 hours.",
            context="An employer must report a STEM OPT extension change within 48 hours.",
            passage_relevance_score=5.0,
            source_tier=SourceTier.PRIMARY_LAW,
            document_status=DocumentStatus.HISTORICAL,
            temporal_role=TemporalRole.OPERATIVE,
            valid_from=date(2008, 4, 8),
            valid_to=date(2016, 5, 9),
            date_basis="Explicit effective interval.",
            temporal_confidence=0.9,
            extraction_model="fake",
            embedding_model="none",
        )
        self.index_path = config.RETRIEVAL_INDEX_PATH
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            direct_unknown.model_dump_json() + "\n" + weak_applicable.model_dump_json() + "\n",
            encoding="utf-8",
        )

        response = retrieve(
            RetrievalRequest(
                query="How long is the STEM OPT extension?",
                relevant_date=date(2010, 6, 1),
            )
        )

        self.assertEqual(
            response.temporal_verdict,
            TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
        )
        self.assertEqual(response.hits[0].passage.passage_id, "duration-unknown")


if __name__ == "__main__":
    unittest.main()
