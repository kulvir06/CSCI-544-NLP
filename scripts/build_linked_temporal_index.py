"""Build a better temporal index locally from cached claim data."""

from __future__ import annotations

import argparse
import json

from api import config
from temporal.linking import build_linked_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-confidence", type=float, default=0.78)
    args = parser.parse_args()
    output = config.INDEX_DIR / "linked_temporal_claims.jsonl"
    metadata = config.INDEX_DIR / "linked_temporal_claims.meta.json"
    report = build_linked_index(
        config.RETRIEVAL_INDEX_PATH,
        output,
        metadata,
        args.minimum_confidence,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
