from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd


def anonymize(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    records = []
    for method_spec in args.method:
        method, directory = method_spec.split("=", maxsplit=1)
        paths = sorted(
            p
            for p in Path(directory).rglob("*")
            if p.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        )

        if args.num_per_method:
            paths = random.sample(
                paths,
                min(args.num_per_method, len(paths)),
            )

        for path in paths:
            anonymous_id = hashlib.sha256(
                f"{method}:{path}:{args.seed}".encode()
            ).hexdigest()[:12]

            records.append(
                {
                    "anonymous_id": anonymous_id,
                    "image_path": str(path),
                    "method": method,
                }
            )

    random.shuffle(records)
    key_df = pd.DataFrame(records)
    key_df.to_csv(args.key_output, index=False)

    form_df = key_df[
        ["anonymous_id", "image_path"]
    ].copy()
    form_df["evaluator_id"] = ""
    form_df["spirit_resonance"] = ""
    form_df["composition"] = ""
    form_df["brushwork"] = ""
    form_df.to_csv(args.form_output, index=False)


def summarize(args: argparse.Namespace) -> None:
    key = pd.read_csv(args.key)
    ratings = pd.read_csv(args.ratings)

    required = {
        "anonymous_id",
        "evaluator_id",
        "spirit_resonance",
        "composition",
        "brushwork",
    }
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing rating columns: {missing}")

    merged = ratings.merge(
        key[["anonymous_id", "method"]],
        on="anonymous_id",
        how="left",
    )

    criteria = [
        "spirit_resonance",
        "composition",
        "brushwork",
    ]

    summary = {}
    for criterion in criteria:
        grouped = merged.groupby("method")[criterion]
        summary[criterion] = {
            method: {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "count": int(values.count()),
            }
            for method, values in grouped
        }

    agreement = {}

    for criterion in criteria:
        matrix = merged.pivot_table(
            index="evaluator_id",
            columns="anonymous_id",
            values=criterion,
            aggfunc="mean",
        )

        agreement[criterion] = float(
            krippendorff.alpha(
                reliability_data=matrix.to_numpy(),
                level_of_measurement="ordinal",
            )
        )

    result = {
        "summary": summary,
        "krippendorff_alpha": agreement,
    }

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    blind = subparsers.add_parser("anonymize")
    blind.add_argument(
        "--method",
        action="append",
        required=True,
        help="Format: method_name=image_directory",
    )
    blind.add_argument("--num-per-method", type=int, default=50)
    blind.add_argument("--seed", type=int, default=42)
    blind.add_argument(
        "--key-output",
        default="human_eval_key.csv",
    )
    blind.add_argument(
        "--form-output",
        default="human_eval_form.csv",
    )
    blind.set_defaults(func=anonymize)

    stats = subparsers.add_parser("summarize")
    stats.add_argument("--key", required=True)
    stats.add_argument("--ratings", required=True)
    stats.add_argument(
        "--output",
        default="human_eval_results.json",
    )
    stats.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()