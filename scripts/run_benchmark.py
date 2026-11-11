"""
NL2SQL Independent Evaluation Benchmark Runner (Phase 1, Step 2).

Executes the independent benchmark against the existing trained Seq2Seq model checkpoint
without altering model weights, hyperparameters, inference pipelines, or database records.

Computes:
1. Overall Metrics:
   - Strict Exact Match Accuracy
   - AST-Normalized Exact Match Accuracy
   - SQL Syntactic Validity Rate
   - Live MySQL Execution Accuracy
   - Clause-Level Accuracy Breakdown (SELECT, WHERE, ORDER BY, LIMIT)
2. Diagnostic Breakdowns:
   - Performance across 10 benchmark categories
   - In-vocabulary vs. Out-of-vocabulary (OOV) literal performance
   - Error categorization (redundant conditions, literal mismatch, operator mismatch, etc.)
   - Table routing accuracy
3. Outputs:
   - Structured JSON results to benchmark_results.json
   - Comprehensive Markdown report to BENCHMARK_EVALUATION_REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import DEFAULT_STUDENT_SCHEMA
from src.database.config import DatabaseConfig
from src.database.connection import DatabaseConnection
from src.database.executor import SQLExecutor
from src.evaluation.comparator import (
    ASTComparator,
    ASTComparisonResult,
    ExecutionComparator,
    ExecutionComparisonResult,
)
from src.inference.pipeline import NL2SQLInferencePipeline
from src.validation.validator import SQLValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmark")


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    if not file_path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent NL2SQL evaluation benchmark against trained model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/candidates/phase2_clean_best.pt",
        help="Path to trained PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--benchmark-file",
        "--test-file",
        "--test_file",
        dest="benchmark_file",
        type=str,
        default="data/benchmark_eval.jsonl",
        help="Path to benchmark or split evaluation dataset (.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path to output JSON results.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="BENCHMARK_EVALUATION_REPORT.md",
        help="Path to output markdown report.",
    )
    parser.add_argument(
        "--max-length",
        "--max_length",
        dest="max_length",
        type=int,
        default=60,
        help="Maximum generation sequence length.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum rows to fetch during execution comparison.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--decoding-mode",
        "--decoding_mode",
        dest="decoding_mode",
        type=str,
        choices=["schema_aware", "greedy"],
        default="schema_aware",
        help="Decoding mode ('schema_aware' constrained beam or 'greedy' baseline).",
    )
    parser.add_argument(
        "--beam-width",
        "--beam_width",
        dest="beam_width",
        type=int,
        default=5,
        help="Beam width / candidate count for schema-aware decoding (default: 5).",
    )
    parser.add_argument(
        "--length-penalty-alpha",
        "--length_penalty_alpha",
        dest="length_penalty_alpha",
        type=float,
        default=0.0,
        help="Optional beam length-normalization exponent alpha (default: 0.0).",
    )
    return parser.parse_args()


def load_benchmark_records(benchmark_path: Path) -> list[dict[str, Any]]:
    """Loads benchmark JSONL records."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at: {benchmark_path.resolve()}")
    records: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def identify_oov_literals(ref_sql: str, tgt_vocab: Any, tokenizer: Any) -> tuple[bool, list[str]]:
    """
    Identifies whether the reference SQL requires generating literals outside the target vocabulary.
    """
    tokens = tokenizer.tokenize_sql(ref_sql)
    oov_tokens = [t for t in tokens if t not in tgt_vocab]
    return (len(oov_tokens) > 0, oov_tokens)


def run_benchmark(
    checkpoint_path: str = "checkpoints/candidates/phase2_clean_best.pt",
    benchmark_file: str = "data/benchmark_eval.jsonl",
    output_path: str = "benchmark_results.json",
    report_path: str = "BENCHMARK_EVALUATION_REPORT.md",
    max_length: int = 60,
    max_rows: int = 100,
    device: Optional[str] = None,
    decoding_mode: str = "schema_aware",
    beam_width: int = 5,
    length_penalty_alpha: float = 0.0,
) -> dict[str, Any]:
    """
    Executes the benchmark evaluation workflow and generates report artifacts.
    """
    start_time = time.time()
    chk_file = Path(checkpoint_path)
    bench_file = Path(benchmark_file)
    out_file = Path(output_path)
    rep_file = Path(report_path)

    print("=" * 78)
    print("  NL2SQL Independent Evaluation Benchmark Runner (Phase 1, Step 2)")
    print("=" * 78)

    # 1. Hashes & Metadata
    chk_hash = compute_file_sha256(chk_file)
    bench_hash = compute_file_sha256(bench_file)
    print(f"Model Checkpoint:    {chk_file} (SHA256: {chk_hash[:16]}...)")
    print(f"Benchmark Dataset:   {bench_file} (SHA256: {bench_hash[:16]}...)")

    # 2. Load Checkpoint and Inference Pipeline
    print("\n[1/4] Restoring trained Seq2Seq model from checkpoint...")
    inference_pipeline = NL2SQLInferencePipeline.from_checkpoint(
        checkpoint_path=chk_file,
        device=device,
        default_schema=DEFAULT_STUDENT_SCHEMA,
        decoding_mode=decoding_mode,
        beam_width=beam_width,
        length_penalty_alpha=length_penalty_alpha,
    )
    src_vocab = inference_pipeline.src_vocab
    tgt_vocab = inference_pipeline.tgt_vocab
    tokenizer = inference_pipeline.tokenizer
    exec_device = str(inference_pipeline.device)
    print(f"  Architecture: 1-layer GRU Encoder + Bahdanau Attention + 1-layer GRU Decoder")
    print(f"  Checkpoint Epoch: {inference_pipeline.checkpoint_info.get('epoch')}")
    print(f"  Vocabularies: Source={len(src_vocab)} tokens, Target={len(tgt_vocab)} tokens")
    print(f"  Inference Device: {exec_device}")
    print(f"  Decoding Mode:    {decoding_mode} (beam_width={beam_width if decoding_mode == 'schema_aware' else 1})")

    # 3. Load Benchmark Records
    print(f"\n[2/4] Loading benchmark dataset from: {bench_file}...")
    records = load_benchmark_records(bench_file)
    total_evaluable = len(records)
    print(f"  Loaded {total_evaluable} independent benchmark evaluation queries.")

    # 4. Initialize Evaluation Components
    print("\n[3/4] Initializing AST and Database Execution Comparators...")
    ast_comparator = ASTComparator(dialect="mysql")
    sql_validator = SQLValidator(schema=DEFAULT_STUDENT_SCHEMA, enforce_model_scope=True)

    # Configure MySQL SQLExecutor
    db_available = False
    db_user = None
    db_name = None
    db_error = None
    executor: Optional[SQLExecutor] = None

    try:
        db_cfg = DatabaseConfig.from_exec_env()
        db_user = db_cfg.user
        db_name = db_cfg.database
        executor = SQLExecutor(config=db_cfg, default_max_rows=max_rows)
        # Test connection
        conn = DatabaseConnection(config=db_cfg)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        db_available = True
        print(f"  MySQL Database: CONNECTED (User: {db_user}, DB: {db_name}, Max Rows: {max_rows})")
    except Exception as err:
        db_error = str(err)
        print(f"  MySQL Database: UNAVAILABLE ({db_error})")

    exec_comparator = ExecutionComparator(
        executor=executor,
        validator=sql_validator,
        schema=DEFAULT_STUDENT_SCHEMA,
        max_rows=max_rows,
    )

    # 5. Process Benchmark Queries
    print(f"\n[4/4] Executing inference, AST validation, and database verification...")

    per_example_results: list[dict[str, Any]] = []

    # Aggregators
    strict_em_count = 0
    norm_em_count = 0
    valid_sql_count = 0
    ast_parse_count = 0
    exec_acc_count = 0
    select_match_count = 0
    where_match_count = 0
    order_match_count = 0
    limit_match_count = 0
    table_match_count = 0
    total_candidates_examined = 0
    total_candidates_rejected = 0
    no_valid_candidate_count = 0
    fallback_used_count = 0
    total_decoding_ms = 0.0

    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "strict_em": 0, "norm_em": 0, "valid": 0, "exec_match": 0}
    )
    oov_stats: dict[str, dict[str, int]] = {
        "in_vocab": {"total": 0, "strict_em": 0, "norm_em": 0, "valid": 0, "exec_match": 0},
        "has_oov": {"total": 0, "strict_em": 0, "norm_em": 0, "valid": 0, "exec_match": 0},
    }
    failure_mode_counts: Counter[str] = Counter()

    for rec in records:
        rec_id = rec["id"]
        category = rec["category"]
        table = rec.get("table", "unknown")
        question = rec["question"]
        ref_sql = rec["sql"]
        schema = rec.get("schema", DEFAULT_STUDENT_SCHEMA)

        # Identify OOV in reference SQL
        has_oov, ref_oov_tokens = identify_oov_literals(ref_sql, tgt_vocab, tokenizer)
        oov_bucket = "has_oov" if has_oov else "in_vocab"

        # A. Standalone Inference
        inf_res = inference_pipeline.predict(
            question=question,
            schema=schema,
            max_length=max_length,
            decoding_mode=decoding_mode,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
        )
        pred_sql = inf_res.sql
        total_candidates_examined += inf_res.candidates_examined
        total_candidates_rejected += inf_res.candidates_rejected
        if inf_res.no_valid_candidate:
            no_valid_candidate_count += 1
        if inf_res.fallback_used:
            fallback_used_count += 1
        if inf_res.decoding_latency_ms is not None:
            total_decoding_ms += inf_res.decoding_latency_ms

        # B. SQL Validation
        val_res = sql_validator.validate(pred_sql)
        is_valid = val_res.is_valid
        if is_valid:
            valid_sql_count += 1

        # C. AST Comparison
        ast_res: ASTComparisonResult = ast_comparator.compare(
            ref_sql=ref_sql,
            pred_sql=pred_sql,
        )
        if ast_res.is_valid_sql:
            ast_parse_count += 1

        if ast_res.strict_exact_match:
            strict_em_count += 1
        if ast_res.normalized_exact_match:
            norm_em_count += 1

        if ast_res.select_match:
            select_match_count += 1
        if ast_res.where_match:
            where_match_count += 1
        if ast_res.order_match:
            order_match_count += 1
        if ast_res.limit_match:
            limit_match_count += 1
        if ast_res.table_match:
            table_match_count += 1

        failure_mode_counts[ast_res.failure_mode] += 1

        # D. Execution Comparison
        has_order = "order by" in ref_sql.lower()
        exec_res: ExecutionComparisonResult = exec_comparator.execute_and_compare(
            ref_sql=ref_sql,
            pred_sql=pred_sql,
            is_pred_valid=is_valid,
            has_order_by=has_order,
        )

        if exec_res.results_match:
            exec_acc_count += 1

        # Tally category & OOV stats
        category_stats[category]["total"] += 1
        if ast_res.strict_exact_match:
            category_stats[category]["strict_em"] += 1
        if ast_res.normalized_exact_match:
            category_stats[category]["norm_em"] += 1
        if is_valid:
            category_stats[category]["valid"] += 1
        if exec_res.results_match:
            category_stats[category]["exec_match"] += 1

        oov_stats[oov_bucket]["total"] += 1
        if ast_res.strict_exact_match:
            oov_stats[oov_bucket]["strict_em"] += 1
        if ast_res.normalized_exact_match:
            oov_stats[oov_bucket]["norm_em"] += 1
        if is_valid:
            oov_stats[oov_bucket]["valid"] += 1
        if exec_res.results_match:
            oov_stats[oov_bucket]["exec_match"] += 1

        per_example_results.append({
            "id": rec_id,
            "category": category,
            "table": table,
            "question": question,
            "reference_sql": ref_sql,
            "predicted_sql": pred_sql,
            "has_oov_literals": has_oov,
            "ref_oov_tokens": ref_oov_tokens,
            "decoding_stats": {
                "decoding_mode": inf_res.decoding_mode,
                "beam_width": inf_res.beam_width,
                "candidates_examined": inf_res.candidates_examined,
                "candidates_rejected": inf_res.candidates_rejected,
                "no_valid_candidate": inf_res.no_valid_candidate,
                "fallback_used": inf_res.fallback_used,
                "is_schema_valid": inf_res.is_schema_valid,
                "validation_error": val_res.error_message or inf_res.validation_error,
                "decoding_latency_ms": inf_res.decoding_latency_ms,
            },
            "ast_comparison": ast_res.to_dict(),
            "execution_comparison": exec_res.to_dict(),
        })

    elapsed_time = time.time() - start_time

    # Calculate Rates
    strict_em_rate = strict_em_count / total_evaluable if total_evaluable else 0.0
    norm_em_rate = norm_em_count / total_evaluable if total_evaluable else 0.0
    valid_rate = valid_sql_count / total_evaluable if total_evaluable else 0.0
    ast_parse_rate = ast_parse_count / total_evaluable if total_evaluable else 0.0
    exec_rate = exec_acc_count / total_evaluable if total_evaluable else 0.0

    select_rate = select_match_count / total_evaluable if total_evaluable else 0.0
    where_rate = where_match_count / total_evaluable if total_evaluable else 0.0
    order_rate = order_match_count / total_evaluable if total_evaluable else 0.0
    limit_rate = limit_match_count / total_evaluable if total_evaluable else 0.0
    table_rate = table_match_count / total_evaluable if total_evaluable else 0.0

    summary = {
        "benchmark_metadata": {
            "checkpoint_path": str(chk_file),
            "checkpoint_sha256": chk_hash,
            "benchmark_file": str(bench_file),
            "benchmark_sha256": bench_hash,
            "total_examples": total_evaluable,
            "device": exec_device,
            "decoding_mode": decoding_mode,
            "beam_width": beam_width if decoding_mode == "schema_aware" else 1,
            "length_penalty_alpha": length_penalty_alpha if decoding_mode == "schema_aware" else 0.0,
            "elapsed_seconds": round(elapsed_time, 2),
            "database_status": {
                "available": db_available,
                "user": db_user,
                "database": db_name,
                "error": db_error,
            },
        },
        "overall_metrics": {
            "strict_exact_match": {"count": strict_em_count, "rate": round(strict_em_rate, 4)},
            "ast_normalized_exact_match": {"count": norm_em_count, "rate": round(norm_em_rate, 4)},
            "sql_ast_parse_rate": {"count": ast_parse_count, "rate": round(ast_parse_rate, 4)},
            "sql_syntactic_validity": {"count": valid_sql_count, "rate": round(valid_rate, 4)},
            "execution_accuracy": {
                "status": "evaluated" if db_available else "unavailable",
                "count": exec_acc_count,
                "rate": round(exec_rate, 4) if db_available else None,
            },
        },
        "decoding_diagnostics": {
            "decoding_mode": decoding_mode,
            "beam_width": beam_width if decoding_mode == "schema_aware" else 1,
            "length_penalty_alpha": length_penalty_alpha if decoding_mode == "schema_aware" else 0.0,
            "total_candidates_examined": total_candidates_examined,
            "avg_candidates_examined": round(total_candidates_examined / total_evaluable, 4) if total_evaluable else 0.0,
            "total_candidates_rejected": total_candidates_rejected,
            "no_valid_candidate_count": no_valid_candidate_count,
            "fallback_used_count": fallback_used_count,
            "total_decoding_seconds": round(total_decoding_ms / 1000.0, 3),
            "avg_decoding_latency_ms": round(total_decoding_ms / total_evaluable, 3) if total_evaluable else 0.0,
        },
        "clause_level_metrics": {
            "select_clause_accuracy": {"count": select_match_count, "rate": round(select_rate, 4)},
            "where_clause_accuracy": {"count": where_match_count, "rate": round(where_rate, 4)},
            "order_by_clause_accuracy": {"count": order_match_count, "rate": round(order_rate, 4)},
            "limit_clause_accuracy": {"count": limit_match_count, "rate": round(limit_rate, 4)},
            "table_routing_accuracy": {"count": table_match_count, "rate": round(table_rate, 4)},
        },
        "category_breakdown": {
            cat: {
                "total": stats["total"],
                "strict_exact_match_count": stats["strict_em"],
                "strict_exact_match_rate": round(stats["strict_em"] / stats["total"], 4) if stats["total"] else 0.0,
                "normalized_exact_match_count": stats["norm_em"],
                "normalized_exact_match_rate": round(stats["norm_em"] / stats["total"], 4) if stats["total"] else 0.0,
                "valid_sql_count": stats["valid"],
                "valid_sql_rate": round(stats["valid"] / stats["total"], 4) if stats["total"] else 0.0,
                "execution_accuracy_count": stats["exec_match"],
                "execution_accuracy_rate": round(stats["exec_match"] / stats["total"], 4) if stats["total"] else 0.0,
            }
            for cat, stats in category_stats.items()
        },
        "oov_analysis": {
            bucket: {
                "total": b_stats["total"],
                "strict_exact_match_count": b_stats["strict_em"],
                "strict_exact_match_rate": round(b_stats["strict_em"] / b_stats["total"], 4) if b_stats["total"] else 0.0,
                "normalized_exact_match_count": b_stats["norm_em"],
                "normalized_exact_match_rate": round(b_stats["norm_em"] / b_stats["total"], 4) if b_stats["total"] else 0.0,
                "valid_sql_count": b_stats["valid"],
                "valid_sql_rate": round(b_stats["valid"] / b_stats["total"], 4) if b_stats["total"] else 0.0,
                "execution_accuracy_count": b_stats["exec_match"],
                "execution_accuracy_rate": round(b_stats["exec_match"] / b_stats["total"], 4) if b_stats["total"] else 0.0,
            }
            for bucket, b_stats in oov_stats.items()
        },
        "failure_modes": dict(failure_mode_counts),
    }

    # Save Results JSON
    output_data = {"summary": summary, "results": per_example_results}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved structured benchmark results to: {out_file.resolve()}")

    # Generate Markdown Report
    generate_markdown_report(summary, per_example_results, rep_file)
    print(f"Generated comprehensive report at: {rep_file.resolve()}")

    # Print summary to console
    print_console_summary(summary)

    return output_data


def print_console_summary(summary: dict[str, Any]) -> None:
    om = summary["overall_metrics"]
    cl = summary["clause_level_metrics"]
    print("\n" + "=" * 78)
    print("                     BENCHMARK EVALUATION SUMMARY")
    print("=" * 78)
    print(f"  Total Evaluable Queries:       {summary['benchmark_metadata']['total_examples']}")
    print(f"  Strict Exact Match Accuracy:    {om['strict_exact_match']['count']}/{summary['benchmark_metadata']['total_examples']} ({om['strict_exact_match']['rate'] * 100:.2f}%)")
    print(f"  AST-Normalized Exact Match:    {om['ast_normalized_exact_match']['count']}/{summary['benchmark_metadata']['total_examples']} ({om['ast_normalized_exact_match']['rate'] * 100:.2f}%)")
    print(f"  SQL Syntactic Validity Rate:   {om['sql_syntactic_validity']['count']}/{summary['benchmark_metadata']['total_examples']} ({om['sql_syntactic_validity']['rate'] * 100:.2f}%)")
    exec_rate_str = (
        f"{om['execution_accuracy']['rate'] * 100:.2f}%"
        if om['execution_accuracy']['rate'] is not None
        else "N/A"
    )
    print(f"  Execution Accuracy on MySQL:   {om['execution_accuracy']['count']}/{summary['benchmark_metadata']['total_examples']} ({exec_rate_str})")
    print("-" * 78)
    print("  Clause-Level Accuracy:")
    print(f"    - Table Routing:             {cl['table_routing_accuracy']['rate'] * 100:.2f}%")
    print(f"    - SELECT / Projection:       {cl['select_clause_accuracy']['rate'] * 100:.2f}%")
    print(f"    - WHERE / Filtering:         {cl['where_clause_accuracy']['rate'] * 100:.2f}%")
    print(f"    - ORDER BY Clause:           {cl['order_by_clause_accuracy']['rate'] * 100:.2f}%")
    print(f"    - LIMIT Clause:              {cl['limit_clause_accuracy']['rate'] * 100:.2f}%")
    if "decoding_diagnostics" in summary:
        dd = summary["decoding_diagnostics"]
        print("-" * 78)
        print("  Decoding Diagnostics:")
        print(f"    - Mode / Beam Width:         {dd['decoding_mode']} (beam={dd['beam_width']})")
        print(f"    - Candidates Examined:       {dd['total_candidates_examined']} (avg {dd['avg_candidates_examined']:.2f}/q)")
        print(f"    - Candidates Rejected:       {dd['total_candidates_rejected']}")
        print(f"    - No-Valid-Candidate Queries:{dd['no_valid_candidate_count']}")
        print(f"    - Avg Decoding Latency:      {dd['avg_decoding_latency_ms']:.2f} ms/query")
    print("=" * 78 + "\n")


def generate_markdown_report(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    report_file: Path,
) -> None:
    meta = summary["benchmark_metadata"]
    om = summary["overall_metrics"]
    cl = summary["clause_level_metrics"]
    cats = summary["category_breakdown"]
    oov = summary["oov_analysis"]
    fmodes = summary["failure_modes"]

    md: list[str] = []
    md.append("# NL2SQL Independent Evaluation Benchmark Report (Phase 1, Step 2)\n")
    md.append("**Evaluation Date:** November 9, 2026  ")
    md.append(f"**Model Checkpoint:** `{meta['checkpoint_path']}` (SHA256: `{meta['checkpoint_sha256'][:16]}...`)  ")
    md.append(f"**Benchmark Dataset:** `{meta['benchmark_file']}` (SHA256: `{meta['benchmark_sha256'][:16]}...`)  ")
    md.append(f"**Evaluated Examples:** {meta['total_examples']} (10 categories, 8 examples per category)  ")
    md.append(f"**Execution Environment:** Device `{meta['device']}`, MySQL Database: `{meta['database_status']['database']}` under `{meta['database_status']['user']}`  \n")
    md.append("---\n")

    md.append("## 1. Executive Summary\n")
    md.append("This document reports the baseline evaluation of our trained Sequence-to-Sequence model with Bahdanau attention on the completely independent 80-example evaluation benchmark. The benchmark was designed to probe SQL capabilities beyond the training distribution without modifying model architecture, parameters, or database records.\n")

    md.append("### Key Performance Highlights:\n")
    md.append(f"- **Strict Exact Match:** `{om['strict_exact_match']['count']}/{meta['total_examples']}` (**`{om['strict_exact_match']['rate'] * 100:.2f}%`**)")
    md.append(f"- **AST-Normalized Exact Match:** `{om['ast_normalized_exact_match']['count']}/{meta['total_examples']}` (**`{om['ast_normalized_exact_match']['rate'] * 100:.2f}%`**)")
    md.append(f"- **SQL Syntactic Validity:** `{om['sql_syntactic_validity']['count']}/{meta['total_examples']}` (**`{om['sql_syntactic_validity']['rate'] * 100:.2f}%`**)")
    exec_rate_md = (
        f"{om['execution_accuracy']['rate'] * 100:.2f}%"
        if om['execution_accuracy']['rate'] is not None
        else "N/A"
    )
    md.append(f"- **Execution Accuracy on MySQL:** `{om['execution_accuracy']['count']}/{meta['total_examples']}` (**`{exec_rate_md}`**)\n")

    md.append("> [!IMPORTANT]\n")
    md.append(f"> **Execution Accuracy vs. Exact Match:** Execution Accuracy ({exec_rate_md}) demonstrates the actual semantic utility of generated queries against MySQL. Queries that contain redundant boolean filters (e.g. `cgpa > 8.0 AND cgpa > 8.0`) or equivalent conjunction ordering fail syntactic exact match but execute successfully to return the exact ground-truth relational rows.\n\n")

    md.append("## 2. Overall Performance & Clause-Level Accuracy\n")
    md.append("| Evaluation Dimension | Match Count / Total | Accuracy Rate | Description |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Strict Exact Match** | `{om['strict_exact_match']['count']}/{meta['total_examples']}` | **`{om['strict_exact_match']['rate'] * 100:.2f}%`** | Literal character-for-character string match |")
    md.append(f"| **AST-Normalized Exact Match** | `{om['ast_normalized_exact_match']['count']}/{meta['total_examples']}` | **`{om['ast_normalized_exact_match']['rate'] * 100:.2f}%`** | Conjunction & disjunction order-invariant AST match |")
    md.append(f"| **SQL Syntactic Validity** | `{om['sql_syntactic_validity']['count']}/{meta['total_examples']}` | **`{om['sql_syntactic_validity']['rate'] * 100:.2f}%`** | Valid parseable MySQL Select AST via SQLGlot |")
    md.append(f"| **Execution Accuracy** | `{om['execution_accuracy']['count']}/{meta['total_examples']}` | **`{om['execution_accuracy']['rate'] * 100:.2f}%`** | Returned rows match reference query result set |")
    md.append(f"| **Table Routing Match** | `{cl['table_routing_accuracy']['count']}/{meta['total_examples']}` | **`{cl['table_routing_accuracy']['rate'] * 100:.2f}%`** | Correct target table in FROM clause |")
    md.append(f"| **SELECT / Projection Match** | `{cl['select_clause_accuracy']['count']}/{meta['total_examples']}` | **`{cl['select_clause_accuracy']['rate'] * 100:.2f}%`** | Exact matching projection columns |")
    md.append(f"| **WHERE / Filter Match** | `{cl['where_clause_accuracy']['count']}/{meta['total_examples']}` | **`{cl['where_clause_accuracy']['rate'] * 100:.2f}%`** | Semantically matching filtering predicates |")
    md.append(f"| **ORDER BY Clause Match** | `{cl['order_by_clause_accuracy']['count']}/{meta['total_examples']}` | **`{cl['order_by_clause_accuracy']['rate'] * 100:.2f}%`** | Matching sort columns and directions |")
    md.append(f"| **LIMIT Clause Match** | `{cl['limit_clause_accuracy']['count']}/{meta['total_examples']}` | **`{cl['limit_clause_accuracy']['rate'] * 100:.2f}%`** | Matching row count limit |\n")

    md.append("## 3. Category-Wise Performance Breakdown\n")
    md.append("| Benchmark Category | Total | Strict EM | Norm EM | Valid SQL | Execution Accuracy |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for cat_name, stats in cats.items():
        md.append(
            f"| **{cat_name}** | {stats['total']} | "
            f"{stats['strict_exact_match_count']} ({stats['strict_exact_match_rate'] * 100:.1f}%) | "
            f"{stats['normalized_exact_match_count']} ({stats['normalized_exact_match_rate'] * 100:.1f}%) | "
            f"{stats['valid_sql_count']} ({stats['valid_sql_rate'] * 100:.1f}%) | "
            f"**{stats['execution_accuracy_count']} ({stats['execution_accuracy_rate'] * 100:.1f}%)** |"
        )
    md.append("\n")

    md.append("## 4. In-Vocabulary vs. Out-of-Vocabulary (OOV) Literal Analysis\n")
    md.append("| Literal Subset | Total Queries | Strict EM | Norm EM | Valid SQL | Execution Accuracy |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for bucket_key, b_stats in oov.items():
        label = "In-Vocabulary Literals" if bucket_key == "in_vocab" else "Contains OOV Literals"
        md.append(
            f"| **{label}** | {b_stats['total']} | "
            f"{b_stats['strict_exact_match_count']} ({b_stats['strict_exact_match_rate'] * 100:.1f}%) | "
            f"{b_stats['normalized_exact_match_count']} ({b_stats['normalized_exact_match_rate'] * 100:.1f}%) | "
            f"{b_stats['valid_sql_count']} ({b_stats['valid_sql_rate'] * 100:.1f}%) | "
            f"**{b_stats['execution_accuracy_count']} ({b_stats['execution_accuracy_rate'] * 100:.1f}%)** |"
        )
    md.append("\n")

    md.append("## 5. Failure Mode Distribution\n")
    md.append("| Failure Classification | Occurrences | Percentage | Root Mechanism |")
    md.append("| :--- | :--- | :--- | :--- |")
    for mode, count in fmodes.items():
        pct = (count / meta['total_examples']) * 100
        md.append(f"| `{mode}` | {count} | {pct:.1f}% | See section 6 below |")
    md.append("\n")

    md.append("## 6. Representative Successes and Failures\n")

    # Pick 1 success and 3 diverse failures
    successes = [r for r in results if r["execution_comparison"]["results_match"]]
    failures = [r for r in results if not r["execution_comparison"]["results_match"]]

    if successes:
        s = successes[0]
        md.append("### Success Case: Execution and Semantic Match")
        md.append(f"- **ID:** {s['id']} ({s['category']})")
        md.append(f"- **Question:** *\"{s['question']}\"*")
        md.append(f"- **Reference SQL:** `{s['reference_sql']}`")
        md.append(f"- **Predicted SQL:** `{s['predicted_sql']}`")
        md.append(f"- **Execution Match:** `True` (Rows: {s['execution_comparison']['ref_row_count']})\n")

    if failures:
        md.append("### Failure Case 1: Domain Mapping Disconnect")
        dom_fails = [r for r in failures if r["category"] == "Column-value and domain terminology mapping"]
        f1 = dom_fails[0] if dom_fails else failures[0]
        md.append(f"- **ID:** {f1['id']} ({f1['category']})")
        md.append(f"- **Question:** *\"{f1['question']}\"*")
        md.append(f"- **Reference SQL:** `{f1['reference_sql']}`")
        md.append(f"- **Predicted SQL:** `{f1['predicted_sql']}`")
        md.append(f"- **Execution Match:** `False` (Expected {f1['execution_comparison']['ref_row_count']} rows, got {f1['execution_comparison']['pred_row_count']} rows)")
        md.append(f"- **Failure Reason:** {f1['execution_comparison']['failure_reason'] or f1['ast_comparison']['failure_reason']}\n")

        md.append("### Failure Case 2: Out-of-Vocabulary Literal Bottleneck")
        oov_fails = [r for r in failures if r["has_oov_literals"]]
        if oov_fails:
            f2 = oov_fails[0]
            md.append(f"- **ID:** {f2['id']} ({f2['category']})")
            md.append(f"- **Question:** *\"{f2['question']}\"*")
            md.append(f"- **Reference SQL:** `{f2['reference_sql']}`")
            md.append(f"- **Predicted SQL:** `{f2['predicted_sql']}`")
            md.append(f"- **OOV Tokens:** `{f2['ref_oov_tokens']}`")
            md.append(f"- **Failure Reason:** Literal value was out-of-vocabulary for decoder classification head.\n")

    md.append("## 7. Limitations & Reproducibility\n")
    md.append("1. **Single-Table Scope:** The benchmark tests single-table MySQL queries in alignment with the model capability boundary. Multi-table JOIN queries and aggregations are intentionally excluded to measure what the current architecture was designed to learn.\n")
    md.append("2. **Reproducibility Command:**\n")
    md.append("```powershell\npython scripts/run_benchmark.py\n```\n")
    md.append("All inputs, configurations, and outputs are deterministically reproducible without modifying model parameters, training data, or database tables.\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        checkpoint_path=args.checkpoint,
        benchmark_file=args.benchmark_file,
        output_path=args.output,
        report_path=args.report,
        max_length=args.max_length,
        max_rows=args.max_rows,
        device=args.device,
        decoding_mode=args.decoding_mode,
        beam_width=args.beam_width,
        length_penalty_alpha=args.length_penalty_alpha,
    )
