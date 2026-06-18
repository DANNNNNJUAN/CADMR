"""Score CADMR STALE-compatible answer files with local rule-based metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cadmr.metrics import (
    CADMRMetricScorer,
    StaleDimensionScorer,
    summarize_cadmr_metrics,
    summarize_stale_scores,
)


SCORER_TYPE = "rule_based_stale_compatible_not_official"


def score_answers_file(answers_path: str | Path, output_path: str | Path) -> dict:
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    if isinstance(answers, dict) and isinstance(answers.get("data"), list):
        answers = answers["data"]
    if not isinstance(answers, list):
        raise ValueError("answers file must be a JSON list or an object with a data list.")

    stale_scorer = StaleDimensionScorer()
    cadmr_scorer = CADMRMetricScorer()
    stale_details = [stale_scorer.score_answer_item(item) for item in answers]
    cadmr_details = [cadmr_scorer.score_answer_item(item) for item in answers]
    result = {
        "scorer_type": SCORER_TYPE,
        "stale_style_scores": summarize_stale_scores(stale_details),
        "cadmr_internal_metrics": summarize_cadmr_metrics(cadmr_details),
        "details": {
            "stale_style": stale_details,
            "cadmr_internal": cadmr_details,
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score STALE-compatible CADMR answer files.")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = score_answers_file(args.answers, args.output)
    if args.verbose:
        for item in result["details"]["stale_style"]:
            print(
                item["uid"],
                "SR=", item["dim1"]["passed"],
                "PR=", item["dim2"]["passed"],
                "IPA=", item["dim3"]["passed"],
            )
    print(json.dumps(result["stale_style_scores"], ensure_ascii=False, indent=2))
    print(f"Wrote scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
