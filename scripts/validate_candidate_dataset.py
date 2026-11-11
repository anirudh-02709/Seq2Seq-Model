"""
CLI entry point for the Step 7B Automated Anti-Leakage, Provenance, and Acceptance Gate.

Validates a candidate training JSONL file against Population A (immutable data/train.jsonl),
read-only held-out evaluation splits (validation, test, benchmark_eval), a machine-verifiable
literal provenance manifest, projection-policy rules, and mandatory live MySQL verification
via SQLValidator and SQLExecutor. Fails closed if the database is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.anti_leakage import AntiLeakageConfig, AntiLeakageGate, load_heldout_splits, normalize_numeric_str
from src.data.dataset import load_jsonl_records
from src.database.config import DatabaseConfig
from src.database.connection import DatabaseConnection
from src.database.executor import SQLExecutor
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA


def fetch_live_db_catalogs() -> Tuple[
    Dict[str, Set[str]],
    Dict[Tuple[str, str], Set[int]],
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], Tuple[float, float]],
    Dict[str, int],
]:
    """
    Reads live column string domains, PK/FK identifier sets, observed numeric value sets,
    numeric min/max bounds, and table row counts from MySQL nl2sql_student_db.
    """
    cfg = DatabaseConfig.from_exec_env()
    conn = DatabaseConnection(config=cfg)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT department FROM students WHERE department IS NOT NULL")
            depts = {str(r["department"]) for r in cur.fetchall()}
            cur.execute("SELECT DISTINCT course_name FROM courses WHERE course_name IS NOT NULL")
            courses = {str(r["course_name"]) for r in cur.fetchall()}
            cur.execute("SELECT DISTINCT semester FROM enrollments WHERE semester IS NOT NULL")
            sems = {str(r["semester"]) for r in cur.fetchall()}

            cur.execute("SELECT student_id FROM students")
            student_ids = {int(r["student_id"]) for r in cur.fetchall()}
            cur.execute("SELECT course_id FROM courses")
            course_ids = {int(r["course_id"]) for r in cur.fetchall()}
            cur.execute("SELECT enrollment_id FROM enrollments")
            enrollment_ids = {int(r["enrollment_id"]) for r in cur.fetchall()}

            table_row_counts: Dict[str, int] = {}
            for tbl in ("students", "courses", "enrollments"):
                cur.execute(f"SELECT COUNT(*) AS cnt FROM {tbl}")
                table_row_counts[tbl] = int(cur.fetchone()["cnt"])

            observed_numeric: Dict[Tuple[str, str], Set[str]] = {}
            live_bounds: Dict[Tuple[str, str], Tuple[float, float]] = {}
            for tbl, col in (
                ("students", "age"),
                ("students", "cgpa"),
                ("courses", "credits"),
                ("enrollments", "marks"),
            ):
                cur.execute(f"SELECT DISTINCT {col} AS v FROM {tbl} WHERE {col} IS NOT NULL")
                raw_vals = [r["v"] for r in cur.fetchall()]
                observed_numeric[(tbl, col)] = {normalize_numeric_str(str(v)) for v in raw_vals}
                if raw_vals:
                    fvals = [float(v) for v in raw_vals]
                    live_bounds[(tbl, col)] = (min(fvals), max(fvals))
    finally:
        conn.close()

    string_catalog = {
        "students.department": depts,
        "courses.course_name": courses,
        "enrollments.semester": sems,
    }
    id_catalog = {
        ("students", "student_id"): student_ids,
        ("courses", "course_id"): course_ids,
        ("enrollments", "enrollment_id"): enrollment_ids,
        ("enrollments", "student_id"): student_ids,
        ("enrollments", "course_id"): course_ids,
    }
    return string_catalog, id_catalog, observed_numeric, live_bounds, table_row_counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate candidate NL2SQL dataset via fail-closed anti-leakage acceptance gate."
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate training JSONL file (e.g., data/candidates/phase2_clean_train.jsonl).",
    )
    parser.add_argument(
        "--original-train",
        type=str,
        default="data/train.jsonl",
        help="Path to immutable Population A original training JSONL file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory holding validation.jsonl, test.jsonl, and benchmark_eval.jsonl.",
    )
    parser.add_argument(
        "--provenance-manifest",
        type=str,
        default=None,
        help="Path to JSON literal provenance & reviewed-flag manifest.",
    )
    parser.add_argument(
        "--no-require-manifest",
        action="store_true",
        help="Disable mandatory provenance manifest check (strictly for auditing legacy corpora).",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional path to write full structured JSON gate report.",
    )
    args = parser.parse_args(argv)
    cand_recs = load_jsonl_records(args.candidate)
    orig_recs = load_jsonl_records(args.original_train)
    heldout = load_heldout_splits(args.data_dir)

    manifest: Optional[Dict[str, Any]] = None
    reviewed_flags: Optional[Dict[str, Any]] = None
    reviewed_proj_flags: Optional[Dict[str, Any]] = None
    reviewed_zero_flags: Optional[Dict[str, Any]] = None
    if args.provenance_manifest:
        with open(args.provenance_manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        reviewed_flags = manifest.get("reviewed_soft_flags", {})
        reviewed_proj_flags = manifest.get("reviewed_projection_flags", {})
        reviewed_zero_flags = manifest.get("reviewed_zero_row_queries", {})

    executor: Optional[SQLExecutor] = None
    catalog_strings: Optional[Dict[str, Set[str]]] = None
    id_catalog: Optional[Dict[Tuple[str, str], Set[int]]] = None
    observed_numeric: Optional[Dict[Tuple[str, str], Set[str]]] = None
    live_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None
    table_row_counts: Optional[Dict[str, int]] = None
    db_unavailable_error: Optional[str] = None

    try:
        cfg = DatabaseConfig.from_exec_env()
        executor = SQLExecutor(config=cfg, default_max_rows=500)
        catalog_strings, id_catalog, observed_numeric, live_bounds, table_row_counts = fetch_live_db_catalogs()
    except Exception as err:
        db_unavailable_error = f"Database unavailable for mandatory live execution verification: {err}"

    gate_cfg = AntiLeakageConfig(
        schema=DEFAULT_STUDENT_SCHEMA,
        require_literal_manifest=not args.no_require_manifest,
        require_manifest_v2=not args.no_require_manifest,
        require_catalog_verification=True,
        require_live_execution=True,
    )
    gate = AntiLeakageGate(gate_cfg)
    report = gate.evaluate(
        candidate_records=cand_recs,
        original_train_records=orig_recs,
        heldout_splits=heldout,
        literal_manifest=manifest,
        reviewed_flag_ids=reviewed_flags,
        reviewed_projection_flags=reviewed_proj_flags,
        reviewed_zero_row_queries=reviewed_zero_flags,
        catalog_string_values=catalog_strings,
        identifier_catalog=id_catalog,
        observed_numeric_values=observed_numeric,
        live_metric_bounds=live_bounds,
        table_row_counts=table_row_counts,
        executor=executor,
        db_unavailable_error=db_unavailable_error,
    )

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    print("=" * 72)
    print(f"ANTI-LEAKAGE GATE VERDICT: {report.summary['gate_verdict']}")
    print("=" * 72)
    print(json.dumps(report.summary, indent=2))
    print(
        "\nTemplate overlaps:",
        json.dumps(
            {
                k: {
                    sk: sv
                    for sk, sv in v.items()
                    if sk != "population_b_new"
                }
                | {
                    "population_b_new": {
                        pk: pv
                        for pk, pv in v["population_b_new"].items()
                        if pk not in {"matching_new_record_ids", "new_only_details"}
                    }
                }
                for k, v in report.template_overlaps.items()
            },
            indent=2,
        ),
    )
    print("\nLiteral provenance audit:", json.dumps(report.literal_provenance_audit, indent=2))
    print("\nProjection policy audit:", json.dumps(report.projection_policy_audit, indent=2))
    print(
        "\nValidation & live DB execution audit:",
        json.dumps(
            {
                k: v
                for k, v in report.validation_audit.items()
                if k not in {"invalid_records", "valid_zero_rows_queries", "execution_infrastructure_failures"}
            },
            indent=2,
        ),
    )
    if report.hard_failures:
        print(f"\nHARD FAILURES ({len(report.hard_failures)}):")
        for hf in report.hard_failures:
            print(" ", json.dumps(hf)[:300])
    if report.review_flags:
        print(f"\nSOFT REVIEW FLAGS ({len(report.review_flags)}):")
        for rf in report.review_flags:
            print(" ", rf["key"], "| sims=", rf["sims"], "| approved=", rf["approved"])
    proj_flags = report.projection_policy_audit.get("review_flags", [])
    if proj_flags:
        print(f"\nPROJECTION REVIEW FLAGS ({len(proj_flags)}):")
        for pf in proj_flags:
            print(" ", pf["key"], "| violation=", pf["violation"], "| approved=", pf["approved"])

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
