from __future__ import annotations

import argparse
import json

from .config import load_config
from .retrieval import KnowledgeRetriever
from .workflow import MACGuohuaWorkflow


def build_index(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    retrieval_cfg = cfg["retrieval"]

    retriever = KnowledgeRetriever(
        knowledge_dir=retrieval_cfg["knowledge_dir"],
        index_dir=retrieval_cfg["index_dir"],
        embedding_model=retrieval_cfg["embedding_model"],
    )
    retriever.build()

    print(
        f"Knowledge index built at "
        f"{retrieval_cfg['index_dir']}"
    )


def generate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    if args.no_regional:
        cfg["generation"]["regional_prompting"] = False

    workflow = MACGuohuaWorkflow(cfg)
    record = workflow.generate(
        prompt=args.prompt,
        output_dir=args.output,
        seed=args.seed,
    )

    print(
        json.dumps(
            record.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAC-Guohua"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser("build-index")
    index_parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    index_parser.set_defaults(func=build_index)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    generate_parser.add_argument(
        "--prompt",
        required=True,
    )
    generate_parser.add_argument(
        "--output",
        default="outputs/example",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    generate_parser.add_argument(
        "--no-regional",
        action="store_true",
        help="Use one global prompt instead of regional denoising.",
    )
    generate_parser.set_defaults(func=generate)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)