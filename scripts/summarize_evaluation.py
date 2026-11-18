"""Create a concise, report-ready Markdown summary of an evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_markdown(payload: dict) -> str:
    candidate_warning = (
        "> **Diagnostic only.** These cases are benchmark candidates and have not "
        "yet completed independent annotation and external audit. Do not present "
        "these numbers as final test results.\n\n"
        if payload.get("dataset_status") == "benchmark_candidates"
        else ""
    )
    lines = [
        "# Evaluation Summary",
        "",
        candidate_warning.rstrip(),
        "" if candidate_warning else "",
        f"- Cases: {payload['total_cases']}",
        f"- Temporal-verdict accuracy: {_percent(payload['temporal_verdict_accuracy'])}",
        f"- Mean critical-fact preservation: {_percent(payload['mean_critical_fact_preservation'])}",
        "",
        "| Configuration | Exact accuracy | False-safe (count/rate, 95% CI) | Safe accepted (count/rate, 95% CI) | Abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["systems"].items():
        false_ci = metrics["false_safe_wilson_95"]
        safe_ci = metrics["safe_acceptance_wilson_95"]
        lines.append(
            "| {name} | {accuracy} | {false_count}/{unsafe} ({false_rate}; {false_low}–{false_high}) "
            "| {safe_count}/{safe} ({safe_rate}; {safe_low}–{safe_high}) | {abstain} |".format(
                name=name.replace("_", " ").title(),
                accuracy=_percent(metrics["exact_accuracy"]),
                false_count=metrics["false_safe_count"],
                unsafe=metrics["unsafe_cases"],
                false_rate=_percent(metrics["false_safe_rate"]),
                false_low=_percent(false_ci[0]),
                false_high=_percent(false_ci[1]),
                safe_count=metrics["safe_accepted_count"],
                safe=metrics["safe_cases"],
                safe_rate=_percent(metrics["safe_acceptance_rate"]),
                safe_low=_percent(safe_ci[0]),
                safe_high=_percent(safe_ci[1]),
                abstain=_percent(metrics["abstention_rate"]),
            )
        )
    if all(
        name in payload["systems"]
        for name in ("temporal_only", "meaning_only", "combined")
    ):
        temporal = payload["systems"]["temporal_only"]
        meaning = payload["systems"]["meaning_only"]
        combined = payload["systems"]["combined"]
        lines.extend(
            [
                "",
                "## Diagnostic takeaway",
                "",
                (
                    f"Temporal-only verification falsely approved "
                    f"{temporal['false_safe_count']} unsafe case(s), and meaning-only "
                    f"verification falsely approved {meaning['false_safe_count']}. "
                    f"The combined verifier falsely approved "
                    f"{combined['false_safe_count']}, while abstaining on "
                    f"{_percent(combined['abstention_rate'])} of cases."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Case-level diagnostic",
            "",
            "| Case | Gold | Temporal observed / expected | Meaning |",
            "|---|---|---|---|",
        ]
    )
    for case in payload["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['gold_label']} | "
            f"{case['observed_temporal_verdict']} / {case['expected_temporal_verdict']} | "
            f"{case['meaning_verdict']} |"
        )
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(payload), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
