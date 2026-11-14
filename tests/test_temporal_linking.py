from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Optional

from corpus.schema import DocumentStatus, SourceTier
from temporal.linking import (
    build_linked_index,
    link_cached_claims,
    linked_index_is_current,
)
from temporal.schema import TemporalClaim, TemporalRole


def claim(
    claim_id: str,
    text: str,
    valid_from: Optional[date],
    embedding: list[float],
) -> TemporalClaim:
    return TemporalClaim(
        claim_id=claim_id,
        document_id=f"doc-{claim_id}",
        source_passage_id=f"passage-{claim_id}",
        text=text,
        evidence_quote=text,
        context=text,
        passage_relevance_score=5.0,
        title="Rule",
        issuer="DHS",
        source_url="https://example.gov/rule",
        source_tier=SourceTier.PRIMARY_LAW,
        document_status=DocumentStatus.CURRENT,
        temporal_role=TemporalRole.OPERATIVE,
        valid_from=valid_from,
        date_basis="Explicit effective-date anchor." if valid_from else "Unknown.",
        temporal_confidence=0.95,
        extraction_model="cached-model",
        embedding_model="cached-embedding",
        embedding=embedding,
    )


class TemporalLinkingTests(unittest.TestCase):
    def test_equivalent_cached_claim_inherits_supported_start_date(self):
        donor = claim(
            "donor",
            "A student must submit Form I-983 within 10 days for a new employer.",
            date(2016, 5, 10),
            [1.0, 0.0],
        )
        target = claim(
            "target",
            "A student must submit Form I-983 within 10 days for a new employer.",
            None,
            [1.0, 0.0],
        )

        linked, report = link_cached_claims([donor, target])
        result = linked[1]

        self.assertEqual(result.valid_from, date(2016, 5, 10))
        self.assertIsNone(result.base_valid_from)
        self.assertEqual(result.linked_valid_from, date(2016, 5, 10))
        self.assertEqual(result.temporal_link_sources, ["donor"])
        self.assertEqual(report["newly_linked_claims"], 1)

    def test_different_quantity_is_not_linked(self):
        donor = claim(
            "donor",
            "The STEM OPT extension lasts 24 months.",
            date(2016, 5, 10),
            [1.0, 0.0],
        )
        target = claim(
            "target",
            "The STEM OPT extension lasts 17 months.",
            None,
            [1.0, 0.0],
        )

        linked, report = link_cached_claims([donor, target])

        self.assertIsNone(linked[1].valid_from)
        self.assertEqual(report["newly_linked_claims"], 0)

    def test_form_number_is_not_treated_as_a_conflicting_quantity(self):
        donor = claim(
            "donor",
            "A student must submit the new plan within 10 days for a new employer.",
            date(2016, 5, 10),
            [1.0, 0.0],
        )
        target = claim(
            "target",
            "A student must submit Form I-983 within 10 days for a new employer.",
            None,
            [1.0, 0.0],
        )

        linked, report = link_cached_claims([donor, target])

        self.assertEqual(linked[1].valid_from, date(2016, 5, 10))
        self.assertEqual(report["newly_linked_claims"], 1)

    def test_linked_index_is_used_only_while_base_hash_matches(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            source = base / "temporal.jsonl"
            linked = base / "linked.jsonl"
            metadata = base / "linked.meta.json"
            source.write_text(
                claim("donor", "The rule applies.", date(2025, 1, 1), [1.0])
                .model_dump_json()
                + "\n",
                encoding="utf-8",
            )
            build_linked_index(source, linked, metadata)
            self.assertTrue(linked_index_is_current(source, linked, metadata))

            source.write_text(source.read_text() + "\n", encoding="utf-8")
            self.assertFalse(linked_index_is_current(source, linked, metadata))


if __name__ == "__main__":
    unittest.main()
