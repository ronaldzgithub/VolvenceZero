# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""``companion-encoder`` CLI: train / evaluate / baseline.

torch-dependent code paths import lazily so ``baseline`` (torch-free)
works without the ``[train]`` extra installed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="companion-trajgen output dir (contains train/ and val/)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion-encoder",
        description="Open-weights relationship encoder: train / evaluate / baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train an encoder checkpoint")
    _add_data_dir(train_parser)
    train_parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    train_parser.add_argument("--backbone", default="tiny", help="'tiny' or 'hf:<model_id>'")
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--embedding-dim", type=int, default=64)
    train_parser.add_argument("--max-input-bytes", type=int, default=4096)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score a checkpoint + baselines into a G2 report"
    )
    _add_data_dir(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    evaluate_parser.add_argument("--report", type=pathlib.Path, required=True)
    evaluate_parser.add_argument("--device", default="cpu")
    _add_zero_shot_arguments(evaluate_parser)

    baseline_parser = subparsers.add_parser(
        "baseline", help="score baselines only (no checkpoint, torch-free)"
    )
    _add_data_dir(baseline_parser)
    baseline_parser.add_argument("--report", type=pathlib.Path, default=None)
    _add_zero_shot_arguments(baseline_parser)

    return parser


def _add_zero_shot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible endpoint for the zero-shot baseline column "
        "(omit to skip that column)",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key (never passed as a flag)",
    )


def _zero_shot_predictions(args: argparse.Namespace, val_examples: tuple) -> tuple | None:
    if args.llm_base_url is None:
        return None
    if args.llm_model is None:
        raise SystemExit("--llm-model is required when --llm-base-url is set")
    api_key = os.environ.get(args.llm_api_key_env)
    if not api_key:
        raise SystemExit(f"env var {args.llm_api_key_env} is not set")
    from companion_encoder.baselines import OpenAICompatibleZeroShotLabeler

    labeler = OpenAICompatibleZeroShotLabeler(
        base_url=args.llm_base_url, api_key=api_key, model=args.llm_model
    )
    return labeler.predict(val_examples)


def _command_train(args: argparse.Namespace) -> int:
    from companion_encoder.model import EncoderConfig
    from companion_encoder.train import TrainConfig, train

    result = train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        encoder_config=EncoderConfig(
            backbone=args.backbone,
            embedding_dim=args.embedding_dim,
            max_input_bytes=args.max_input_bytes,
        ),
        train_config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "train_examples": result.train_example_count,
                "val_examples": result.val_example_count,
                "final_train_loss": result.history[-1]["train_loss"],
                "final_val_loss": result.history[-1]["val_loss"],
            },
            indent=2,
        )
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    from companion_encoder.dataset import load_dataset
    from companion_encoder.evaluate import build_g2_report, write_report

    splits = load_dataset(args.data_dir)
    report = build_g2_report(
        splits=splits,
        checkpoint_path=args.checkpoint,
        zero_shot_predictions=_zero_shot_predictions(args, splits.val),
        device=args.device,
    )
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _command_baseline(args: argparse.Namespace) -> int:
    from companion_encoder.baselines import MajorityBaseline
    from companion_encoder.dataset import load_dataset
    from companion_encoder.evaluate import score_predictions, write_report

    splits = load_dataset(args.data_dir)
    majority = MajorityBaseline.fit(splits.train)
    report: dict = {
        "val_example_count": len(splits.val),
        "structured_prediction": {
            "majority_baseline": score_predictions(
                majority.predict(splits.val), splits.val
            ),
        },
    }
    zero_shot = _zero_shot_predictions(args, splits.val)
    if zero_shot is not None:
        report["structured_prediction"]["llm_zero_shot_baseline"] = score_predictions(
            zero_shot, splits.val
        )
    if args.report is not None:
        write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _command_train(args)
    if args.command == "evaluate":
        return _command_evaluate(args)
    if args.command == "baseline":
        return _command_baseline(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
