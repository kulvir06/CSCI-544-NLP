"""Build the fixed 24-case benchmark candidate set from cited corpus passages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import shutil

from benchmark.review import write_drafts
from evaluation.schema import (
    BenchmarkSplit,
    ErrorType,
    EvaluationCase,
    GoldLabel,
)
from retrieval.schema import TemporalVerdict
from temporal.enrichment import load_semantic_passages
from temporal.schema import SemanticPassage


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    split: BenchmarkSplit
    question: str
    relevant_date: date
    passage_id: str
    evidence_start: str
    evidence_end: str
    explanation: str
    error_type: ErrorType
    gold_label: GoldLabel
    expected_temporal_verdict: TemporalVerdict
    mutation_description: str
    atomic_facts: tuple[str, ...]


SPECS = (
    CaseSpec(
        "DEV-01", BenchmarkSplit.DEVELOPMENT,
        "How long was the STEM OPT extension in 2015 and what employer condition applied?",
        date(2015, 1, 1), "78372d30f25a483c-p0026",
        "This interim rule will allow F-1 students", "17-month extension.",
        "An eligible STEM graduate could receive a 17-month extension, provided the employer participated in E-Verify.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful historical control.",
        ("17-month extension", "employer enrolled and in good standing in E-Verify"),
    ),
    CaseSpec(
        "DEV-02", BenchmarkSplit.DEVELOPMENT,
        "What is the unemployment limit for a student whose OPT includes a 24-month STEM extension?",
        date(2026, 1, 1), "362fe269cd38f32a-p0084",
        "Students granted a 24-month OPT extension", "24-month extension period.",
        "A student whose OPT includes a 24-month STEM extension may accrue no more than 150 aggregate days of unemployment.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful current-rule control.",
        ("24-month STEM extension", "no more than 150 aggregate days of unemployment"),
    ),
    CaseSpec(
        "DEV-03", BenchmarkSplit.DEVELOPMENT,
        "How long was the STEM OPT extension on December 1, 2015?",
        date(2015, 12, 1), "aad23adab49f25e7-p0571",
        "An extension will be for 24 months", "must have been conferred.",
        "A student may receive a 24-month STEM OPT extension based on the first qualifying STEM degree, which must already have been conferred.",
        ErrorType.WRONG_VERSION, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        "Kept the 2016 rule meaning but queried a pre-effective date.",
        ("24-month extension", "first qualifying degree", "degree must have been conferred"),
    ),
    CaseSpec(
        "DEV-04", BenchmarkSplit.DEVELOPMENT,
        "Was a 24-month STEM OPT extension operative on November 1, 2015?",
        date(2015, 11, 1), "7c763f922982419b-p0156",
        "An extension will be for 24 months", "previously obtained degree that qualifies.",
        "Yes. The proposed rule text would provide a qualified student with a 24-month STEM OPT extension.",
        ErrorType.WRONG_VERSION, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        "Treated proposed regulatory text as operative law.",
        ("proposed rule text", "24-month extension"),
    ),
    CaseSpec(
        "DEV-05", BenchmarkSplit.DEVELOPMENT,
        "What is the unemployment limit during a 24-month STEM OPT extension?",
        date(2026, 1, 1), "362fe269cd38f32a-p0084",
        "Students granted a 24-month OPT extension", "24-month extension period.",
        "A student whose OPT includes a 24-month STEM extension may accrue no more than 120 aggregate days of unemployment.",
        ErrorType.QUANTITY_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Changed the unemployment limit from 150 to 120 days.",
        ("24-month STEM extension", "150 aggregate days of unemployment"),
    ),
    CaseSpec(
        "DEV-06", BenchmarkSplit.DEVELOPMENT,
        "How quickly must a STEM OPT student report a qualifying change to the DSO?",
        date(2026, 1, 1), "362fe269cd38f32a-p0099",
        "Within 10 days of the change", "loss of employment.",
        "A student must report a qualifying change to the DSO within 30 days.",
        ErrorType.DATE_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Changed the reporting deadline from 10 to 30 days.",
        ("report to the DSO", "within 10 days", "listed qualifying changes"),
    ),
    CaseSpec(
        "DEV-07", BenchmarkSplit.DEVELOPMENT,
        "Who was eligible for the 17-month STEM OPT extension in 2015?",
        date(2015, 1, 1), "78372d30f25a483c-p0026",
        "This interim rule will allow F-1 students", "17-month extension.",
        "Any eligible STEM graduate could receive the 17-month OPT extension.",
        ErrorType.CONDITION_OMITTED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Omitted the E-Verify employer condition.",
        ("17-month extension", "employer enrolled and in good standing in E-Verify"),
    ),
    CaseSpec(
        "DEV-08", BenchmarkSplit.DEVELOPMENT,
        "What must a STEM OPT student submit after beginning work with a new employer?",
        date(2026, 1, 1), "362fe269cd38f32a-p0076",
        "If a student initiates a new practical training opportunity", "new DSO recommendation.",
        "After starting with a new employer during STEM OPT, the student must submit a new Form I-765 to the DSO within 10 days.",
        ErrorType.FORM_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Substituted Form I-765 for Form I-983.",
        ("new employer during the 24-month extension", "within 10 days", "new Form I-983", "new DSO recommendation"),
    ),
    CaseSpec(
        "TEST-01", BenchmarkSplit.TEST,
        "How long was the STEM OPT extension in 2017?",
        date(2017, 1, 1), "aad23adab49f25e7-p0571",
        "An extension will be for 24 months", "must have been conferred.",
        "A qualified student may receive a 24-month STEM OPT extension for the first qualifying degree, which must have been conferred.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful post-2016 control.",
        ("24-month extension", "first qualifying degree", "degree conferred"),
    ),
    CaseSpec(
        "TEST-02", BenchmarkSplit.TEST,
        "How long could employment continue while a timely filed STEM OPT extension was pending in 2015?",
        date(2015, 1, 1), "78372d30f25a483c-p0094",
        "Employment is authorized beginning on the expiration date", "not to exceed 180 days",
        "With a timely filed Form I-765, employment begins from the expiration of the prior Form I-766 and continues until USCIS decides the application, but for no more than 180 days.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful pending-application control.",
        ("timely filed Form I-765", "prior Form I-766 expired", "until USCIS decision", "not more than 180 days"),
    ),
    CaseSpec(
        "TEST-03", BenchmarkSplit.TEST,
        "What validation reporting is required during a 24-month STEM OPT extension?",
        date(2026, 1, 1), "362fe269cd38f32a-p0100",
        "The student must complete a validation report", "within 10 business days of each reporting date.",
        "During the 24-month extension, the student must validate the required information every six months and submit the report to the DSO within 10 business days of each reporting date.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful reporting control.",
        ("validation every six months", "during the 24-month extension", "DSO", "within 10 business days"),
    ),
    CaseSpec(
        "TEST-04", BenchmarkSplit.TEST,
        "How long does qualifying cap-gap protection extend in 2026?",
        date(2026, 1, 1), "1b151d689037a5ec-p0063",
        "The duration of status", "Is nonfrivolous.",
        "For a qualifying, timely filed, nonfrivolous cap-subject H-1B change-of-status petition, F-1 status and eligible employment authorization extend until April 1 or the approved petition start date, whichever is earlier.",
        ErrorType.SAFE, GoldLabel.SAFE, TemporalVerdict.VERIFIED_APPLICABLE,
        "Faithful current cap-gap control.",
        ("cap-subject H-1B petition", "change of status", "until April 1 or earlier approved start date", "timely filed", "nonfrivolous"),
    ),
    CaseSpec(
        "TEST-05", BenchmarkSplit.TEST,
        "How long was the STEM OPT extension in 2020?",
        date(2020, 1, 1), "78372d30f25a483c-p0026",
        "This interim rule will allow F-1 students", "17-month extension.",
        "The STEM OPT extension is 17 months for a student working for an E-Verify employer.",
        ErrorType.WRONG_VERSION, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        "Used the superseded 17-month rule after the 24-month rule took effect.",
        ("17-month extension", "E-Verify condition"),
    ),
    CaseSpec(
        "TEST-06", BenchmarkSplit.TEST,
        "When does cap-gap employment authorization terminate for a qualifying case in 2026?",
        date(2026, 1, 1), "78372d30f25a483c-p0023",
        "this rule extends the authorized period of stay", "H-1B visa has been requested.",
        "Cap-gap employment authorization terminates on October 1 of the requested fiscal year.",
        ErrorType.WRONG_VERSION, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        "Used the historical October 1 cap-gap ending after the 2025 change.",
        ("timely filed H-1B petition", "historical October 1 termination"),
    ),
    CaseSpec(
        "TEST-07", BenchmarkSplit.TEST,
        "Must a school pay a $70,000 initial OPT recommendation fee in November 2026?",
        date(2026, 11, 1), "561ec895a9641414-p0077",
        "The proposed rule would establish", "214.13(i)(1).",
        "A school must pay a $70,000 fee when recommending an F-1 student for OPT for the first time.",
        ErrorType.WRONG_VERSION, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_NOT_APPLICABLE,
        "Converted a proposed fee into an operative obligation.",
        ("proposed rule", "$70,000", "first OPT recommendation", "SEVP-certified school"),
    ),
    CaseSpec(
        "TEST-08", BenchmarkSplit.TEST,
        "How long could employment continue while a timely filed STEM OPT extension was pending in 2015?",
        date(2015, 1, 1), "78372d30f25a483c-p0094",
        "Employment is authorized beginning on the expiration date", "not to exceed 180 days",
        "A timely filed STEM OPT extension may continue employment for up to 240 days while USCIS decides the application.",
        ErrorType.QUANTITY_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Changed 180 days to 240 days.",
        ("until USCIS decision", "not more than 180 days"),
    ),
    CaseSpec(
        "TEST-09", BenchmarkSplit.TEST,
        "When does qualifying cap-gap protection end in 2026?",
        date(2026, 1, 1), "1b151d689037a5ec-p0063",
        "The duration of status", "whichever is earlier",
        "Qualifying cap-gap status and employment authorization extend until October 1 or the approved petition start date, whichever is earlier.",
        ErrorType.DATE_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Changed April 1 to October 1.",
        ("status and employment authorization", "April 1", "earlier approved start date"),
    ),
    CaseSpec(
        "TEST-10", BenchmarkSplit.TEST,
        "May a student begin OPT before the approved start date?",
        date(2026, 1, 1), "362fe269cd38f32a-p0090",
        "A student may not begin OPT employment", "paragraph (f)(11)(i)(C) of this section.",
        "A student may begin OPT employment before the approved start date on the employment authorization document, except as otherwise provided.",
        ErrorType.NEGATION_REVERSED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Removed the negative restriction on starting employment.",
        ("may not begin before approved start date", "Form I-766", "cross-referenced exception"),
    ),
    CaseSpec(
        "TEST-11", BenchmarkSplit.TEST,
        "May a student exceed 150 unemployment days during a 24-month STEM extension?",
        date(2026, 1, 1), "362fe269cd38f32a-p0084",
        "Students granted a 24-month OPT extension", "24-month extension period.",
        "A student with a 24-month STEM extension may accrue more than 150 aggregate days of unemployment.",
        ErrorType.NEGATION_REVERSED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Removed the 'may not' limitation.",
        ("may not accrue more than 150 days", "total OPT period"),
    ),
    CaseSpec(
        "TEST-12", BenchmarkSplit.TEST,
        "Who qualifies for cap-gap protection in 2026?",
        date(2026, 1, 1), "1b151d689037a5ec-p0063",
        "The duration of status", "Is nonfrivolous.",
        "Every F-1 student who is the beneficiary of a cap-subject H-1B petition receives the cap-gap extension.",
        ErrorType.CONDITION_OMITTED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Omitted the change-of-status, timely filing, requested-start-date, and nonfrivolous conditions.",
        ("change-of-status request", "timely filed", "qualifying start date", "nonfrivolous"),
    ),
    CaseSpec(
        "TEST-13", BenchmarkSplit.TEST,
        "May a student ever begin OPT before the approved EAD start date?",
        date(2026, 1, 1), "362fe269cd38f32a-p0090",
        "A student may not begin OPT employment", "paragraph (f)(11)(i)(C) of this section.",
        "A student may never begin OPT before the approved start date on Form I-766.",
        ErrorType.EXCEPTION_OMITTED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Omitted the explicit cross-referenced exception.",
        ("general prohibition", "exception in paragraph (f)(11)(i)(C)", "Form I-766"),
    ),
    CaseSpec(
        "TEST-14", BenchmarkSplit.TEST,
        "What unemployment limit applied while the transitional seven-month extension was pending?",
        date(2016, 6, 1), "aad23adab49f25e7-p0487",
        "The only exception concerns", "entire OPT period.",
        "A student seeking the additional seven-month extension may accrue 150 days of unemployment while the application is pending.",
        ErrorType.EXCEPTION_OMITTED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Dropped the exception that 150 days applies only after approval; the pending limit is 120 days.",
        ("150 days only upon approval", "120 days while pending", "seven-month transitional extension"),
    ),
    CaseSpec(
        "TEST-15", BenchmarkSplit.TEST,
        "Which document contains the approved OPT employment start date?",
        date(2026, 1, 1), "362fe269cd38f32a-p0090",
        "A student may not begin OPT employment", "paragraph (f)(11)(i)(C) of this section.",
        "A student may not begin OPT before the approved start date shown on Form I-20, except as otherwise provided.",
        ErrorType.FORM_CHANGED, GoldLabel.UNSAFE,
        TemporalVerdict.VERIFIED_APPLICABLE,
        "Substituted Form I-20 for the employment authorization document, Form I-766.",
        ("approved start date", "Form I-766", "cross-referenced exception"),
    ),
    CaseSpec(
        "TEST-16", BenchmarkSplit.TEST,
        "Will the same STEM OPT duration and lifetime limit still apply on January 1, 2027?",
        date(2027, 1, 1), "306aba553730c6aa-p0002",
        "The STEM OPT extension is", "accredited U.S. institution.",
        "The STEM OPT extension is 24 months and is available twice to otherwise eligible students.",
        ErrorType.TEMPORAL_UNCERTAINTY, GoldLabel.ABSTAIN,
        TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE,
        "Queried beyond the advisory's independently supported temporal scope.",
        ("24-month extension", "available twice", "STEM degree within 10 years", "accredited U.S. institution"),
    ),
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _evidence_excerpt(passage: SemanticPassage, spec: CaseSpec) -> str:
    text = _normalized(passage.text)
    try:
        start = text.index(spec.evidence_start)
        end = text.index(spec.evidence_end, start) + len(spec.evidence_end)
    except ValueError as error:
        raise ValueError(
            f"{spec.case_id}: evidence boundary not found in {spec.passage_id}"
        ) from error
    return text[start:end]


def build_selected_cases() -> list[EvaluationCase]:
    passages = {item.passage_id: item for item in load_semantic_passages()}
    missing = sorted({spec.passage_id for spec in SPECS} - passages.keys())
    if missing:
        raise ValueError("Benchmark passages missing from semantic index: " + ", ".join(missing))

    cases: list[EvaluationCase] = []
    for spec in SPECS:
        passage = passages[spec.passage_id]
        cases.append(
            EvaluationCase(
                case_id=spec.case_id,
                benchmark_split=spec.split,
                question=spec.question,
                source_text=_evidence_excerpt(passage, spec),
                explanation=spec.explanation,
                error_type=spec.error_type,
                gold_label=spec.gold_label,
                temporal_verification_required=True,
                expected_temporal_verdict=spec.expected_temporal_verdict,
                relevant_date=spec.relevant_date,
                document_id=passage.document_id,
                passage_id=passage.passage_id,
                page_number=passage.page_number,
                source_url=passage.source_url,
                mutation_description=spec.mutation_description,
                atomic_fact_checklist=list(spec.atomic_facts),
                automatic_temporal_verification=True,
                development_fixture=True,
                review_notes=(
                    "Candidate only: requires two-person annotation and external audit "
                    "where designated before it can enter the frozen benchmark."
                ),
            )
        )
    return cases


def _write_jsonl(cases: list[EvaluationCase], path: Path) -> None:
    body = "\n".join(case.model_dump_json() for case in cases)
    path.write_text(f"{body}\n" if body else "", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the selected 24-case benchmark packet.")
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_selected_cases()
    draft_path, review_path = write_drafts(
        cases,
        output_dir / "benchmark-candidates.jsonl",
        output_dir / "benchmark-review.csv",
    )
    reviewer_a_path = output_dir / "reviewer-a.csv"
    reviewer_b_path = output_dir / "reviewer-b.csv"
    shutil.copyfile(review_path, reviewer_a_path)
    shutil.copyfile(review_path, reviewer_b_path)
    development = [case for case in cases if case.benchmark_split == BenchmarkSplit.DEVELOPMENT]
    test = [case for case in cases if case.benchmark_split == BenchmarkSplit.TEST]
    _write_jsonl(development, output_dir / "development-candidates.jsonl")
    _write_jsonl(test, output_dir / "test-candidates.jsonl")
    summary = {
        "total": len(cases),
        "development": len(development),
        "test": len(test),
        "safe": sum(case.gold_label == GoldLabel.SAFE for case in cases),
        "unsafe": sum(case.gold_label == GoldLabel.UNSAFE for case in cases),
        "abstain": sum(case.gold_label == GoldLabel.ABSTAIN for case in cases),
        "draft_jsonl": str(draft_path),
        "review_csv": str(review_path),
        "reviewer_a_csv": str(reviewer_a_path),
        "reviewer_b_csv": str(reviewer_b_path),
        "status": "candidate_only_pending_human_review",
    }
    (output_dir / "benchmark-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
