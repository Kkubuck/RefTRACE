#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reftrace.metrics import evaluate_reference_faithful_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSQ, RRQ, and RFQ")
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--tau-image", type=float, default=0.05)
    parser.add_argument("--tau-object", type=float, default=0.05)
    parser.add_argument("--tau-positive-object", type=float, default=0.10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate_reference_faithful_predictions(
        args.pred_root,
        args.gt_root,
        args.triplets,
        tau_image=args.tau_image,
        tau_object=args.tau_object,
        tau_positive_object=args.tau_positive_object,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
