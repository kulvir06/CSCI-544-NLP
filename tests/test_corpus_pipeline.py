from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from api import config
from corpus.batch import ingest_directory
from corpus.metadata_extraction import enrich_metadata
from corpus.parser import parse_text
from corpus.pipeline import ingest_upload
from corpus.schema import DocumentMetadata, ParsedBlock, SourceTier
from corpus.selector import select_passages
from corpus.source_classification import classify_source


class CorpusPipelineTests(unittest.TestCase):
    def test_federal_register_is_primary_source(self):
        metadata = DocumentMetadata(
            original_filename="rule.txt",
            issuer="Department of Homeland Security",
            source_url="https://www.federalregister.gov/example",
        )
        self.assertEqual(classify_source(metadata), SourceTier.PRIMARY_LAW)

    def test_8_cfr_filename_is_primary_source(self):
        metadata = DocumentMetadata(
            original_filename="8 CFR 214.2.pdf",
            title="8 CFR 214.2",
            issuer="U.S. Federal Regulation",
        )
        self.assertEqual(classify_source(metadata), SourceTier.PRIMARY_LAW)

    def test_unrelated_visa_rule_is_not_selected(self):
        blocks = parse_text(
            Path(__file__).parent / "fixtures" / "unrelated-rule.txt"
        )
        self.assertEqual(select_passages("doc", blocks), [])

    def test_university_advisory_is_secondary(self):
        metadata = DocumentMetadata(
            original_filename="advisory.txt",
            issuer="Example University International Students Office",
            source_url="https://example.edu/advisory",
        )
        self.assertEqual(classify_source(metadata), SourceTier.UNIVERSITY_ADVISORY)

    def test_invalid_effective_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            DocumentMetadata(
                original_filename="rule.txt",
                effective_from=date(2025, 1, 2),
                effective_to=date(2025, 1, 1),
            )

    def test_as_of_date_is_only_an_unconfirmed_candidate(self):
        metadata, candidates = enrich_metadata(
            DocumentMetadata(original_filename="8 CFR 214.2.pdf"),
            [
                ParsedBlock(
                    text="This edition is up to date as of 12/05/2025.",
                    block_index=0,
                )
            ],
        )

        self.assertIsNone(metadata.valid_as_of)
        self.assertEqual(candidates.valid_as_of_candidates, ["12/05/2025"])

    def test_rule_passages_are_selected(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.txt"
            path.write_text(
                "General introduction without a rule.\n\n"
                "F-1 students on STEM OPT must submit Form I-983 and the employer must participate in E-Verify for a 24 month extension.",
                encoding="utf-8",
            )
            blocks = parse_text(path)
            passages = select_passages("doc123", blocks)

        self.assertEqual(len(passages), 1)
        self.assertIn("topic:opt", passages[0].selection_reasons)
        self.assertIn("rule:obligation", passages[0].selection_reasons)

    def test_network_ingestion_is_disabled(self):
        self.assertFalse(config.NETWORK_INGESTION_ENABLED)

    def test_upload_is_parsed_selected_and_persisted_locally(self):
        content = (
            b"F-1 students seeking a STEM OPT extension must submit Form I-983. "
            b"The employer must participate in E-Verify for the 24 month extension."
        )
        metadata = DocumentMetadata(
            original_filename="stem-notice.txt",
            issuer="Example University International Students Office",
            source_url="https://example.edu/stem-notice",
            effective_from=date(2025, 1, 1),
            rule_family="stem_opt",
        )

        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            with (
                patch.object(config, "RAW_DOCUMENT_DIR", base / "raw"),
                patch.object(config, "NORMALIZED_DOCUMENT_DIR", base / "normalized"),
                patch.object(config, "PASSAGE_DIR", base / "passages"),
                patch.object(config, "REPORT_DIR", base / "reports"),
                patch.object(config, "REVIEW_DIR", base / "reviews"),
                patch.object(config, "REVIEW_LOG_PATH", base / "reviews" / "document_reviews.jsonl"),
                patch.object(config, "MANIFEST_DIR", base / "manifests"),
                patch.object(config, "MANIFEST_PATH", base / "manifests" / "documents.jsonl"),
            ):
                result = ingest_upload("stem-notice.txt", content, metadata)
                duplicate = ingest_upload("stem-notice.txt", content, metadata)

                self.assertEqual(result.selected_passages, 1)
                self.assertEqual(result.document.source_tier, SourceTier.UNIVERSITY_ADVISORY)
                self.assertTrue(Path(result.normalized_text_file).exists())
                self.assertTrue(Path(result.passage_file).exists())
                self.assertTrue(duplicate.duplicate_upload)
                self.assertEqual(
                    len(config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()),
                    1,
                )

    def test_batch_ingestion_writes_report(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            inputs = base / "inputs"
            inputs.mkdir()
            (inputs / "uscis-stem.txt").write_text(
                "USCIS guidance states that F-1 students on STEM OPT must submit "
                "Form I-983 for a 24 month extension.",
                encoding="utf-8",
            )
            with (
                patch.object(config, "RAW_DOCUMENT_DIR", base / "raw"),
                patch.object(config, "NORMALIZED_DOCUMENT_DIR", base / "normalized"),
                patch.object(config, "PASSAGE_DIR", base / "passages"),
                patch.object(config, "REPORT_DIR", base / "reports"),
                patch.object(config, "REVIEW_DIR", base / "reviews"),
                patch.object(config, "REVIEW_LOG_PATH", base / "reviews" / "document_reviews.jsonl"),
                patch.object(config, "MANIFEST_DIR", base / "manifests"),
                patch.object(config, "MANIFEST_PATH", base / "manifests" / "documents.jsonl"),
            ):
                report = ingest_directory(inputs)

            self.assertEqual(report.processed_files, 1)
            self.assertEqual(report.failed_files, 0)
            self.assertEqual(report.total_selected_passages, 1)
            self.assertTrue(Path(report.report_file).exists())


if __name__ == "__main__":
    unittest.main()
