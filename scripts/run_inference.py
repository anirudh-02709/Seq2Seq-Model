"""
CLI script for standalone inference on the trained NL2SQL Seq2Seq model.

Usage:
    python scripts/run_inference.py --question "Show students with CGPA above 8"
    python scripts/run_inference.py --question "List all courses" --schema "courses(course_id, course_name, credits)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import NL2SQLInferencePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone inference with the trained Seq2Seq NL2SQL model."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="Natural language question to convert into SQL",
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default=None,
        help="Database schema string. If omitted, uses default student DB schema.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="checkpoints/candidates/phase2_clean_best.pt",
        help="Path to trained checkpoint file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=60,
        help="Maximum generated sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--decoding_mode",
        type=str,
        choices=["schema_aware", "greedy"],
        default="schema_aware",
        help="Decoding mode ('schema_aware' constrained beam or 'greedy' baseline)",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=5,
        help="Beam width / candidate limit for schema-aware decoding (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print diagnostic metadata along with predicted SQL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = NL2SQLInferencePipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        decoding_mode=args.decoding_mode,
        beam_width=args.beam_width,
    )

    if args.verbose:
        result = pipeline.predict(
            question=args.question,
            schema=args.schema,
            max_length=args.max_length,
        )
        print("=" * 60)
        print("NL2SQL Standalone Inference")
        print("=" * 60)
        print(f"Question:       {result.question}")
        print(f"Schema:         {result.schema}")
        print(f"Decoding Mode:  {result.decoding_mode} (beam={result.beam_width})")
        print(f"Predicted SQL:  {result.sql}")
        print(f"Valid MySQL:    {result.is_valid_sql}")
        print(f"Schema Valid:   {result.is_schema_valid}")
        if result.syntax_error:
            print(f"Syntax Error:   {result.syntax_error}")
        if result.validation_error:
            print(f"Schema Error:   {result.validation_error}")
        print(f"Candidates:     examined={result.candidates_examined}, rejected={result.candidates_rejected}")
        print(f"Latency (ms):   {result.decoding_latency_ms}")
        print(f"Source UNK:     {result.source_unk_count}")
        print(f"Target UNK:     {result.target_unk_count}")
        print("=" * 60)
    else:
        sql = pipeline.generate_sql(
            question=args.question,
            schema=args.schema,
            max_length=args.max_length,
        )
        print(sql)


if __name__ == "__main__":
    main()
