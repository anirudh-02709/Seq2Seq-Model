"""
End-to-End NL2SQL Execution Pipeline Verification CLI Tool (Milestone 4, Task 4.4).

Verifies:
1. Checkpoint loading and inference (Question -> SQL).
2. AST validation and model-scope enforcement.
3. Execution approval gating (stops at execution_pending when approved=False).
4. MySQL execution with read-only account when approved=True.
5. Least-privilege account checks (prohibits root).

Usage:
    python scripts/verify_execution.py
    python scripts/verify_execution.py --question "Show students with CGPA above 8"
    python scripts/verify_execution.py --question "Show students with CGPA above 8" --approve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.connection import DatabaseConnection
from src.database.executor import SQLExecutor
from src.pipeline.orchestrator import NL2SQLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify end-to-end NL2SQL pipeline from natural language to MySQL execution."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom natural language question to process.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Optional database schema string override.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicitly grant execution approval to execute SQL against MySQL.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Maximum rows to retrieve (default: 10).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional path to custom .env file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results.",
    )
    return parser.parse_args()


def check_exec_user_configuration(env_file: str | None = None) -> tuple[bool, str | None, DatabaseConfig | None]:
    """Checks whether dedicated read-only execution credentials are configured."""
    try:
        cfg = DatabaseConfig.from_exec_env(env_file=env_file)
        return True, None, cfg
    except DatabaseConfigurationError as err:
        return False, str(err), None
    except Exception as err:
        return False, str(err), None


def print_user_guidance() -> None:
    """Prints instructions for configuring a dedicated read-only database user."""
    print("\n" + "=" * 76)
    print("  NOTICE: DEDICATED READ-ONLY EXECUTION ACCOUNT REQUIRED (Task 4.4)")
    print("=" * 76)
    print("The execution layer enforces least-privilege security and prohibits running")
    print("queries with the 'root' MySQL account.")
    print("\nTo enable live execution against your local MySQL instance, execute:")
    print("-" * 76)
    print("  CREATE USER 'nl2sql_exec'@'localhost' IDENTIFIED BY 'your_chosen_password';")
    print("  GRANT SELECT ON nl2sql_student_db.* TO 'nl2sql_exec'@'localhost';")
    print("  FLUSH PRIVILEGES;")
    print("-" * 76)
    print("Then add the credentials to your project root .env file:")
    print("  MYSQL_EXEC_USER=nl2sql_exec")
    print("  MYSQL_EXEC_PASSWORD=your_chosen_password")
    print("=" * 76 + "\n")


def run_pipeline_demo() -> int:
    args = parse_args()

    # Step 1: Check execution account configuration
    has_exec_account, exec_err, exec_cfg = check_exec_user_configuration(env_file=args.env_file)

    if not args.json:
        print("=" * 76)
        print("  NL2SQL End-to-End Pipeline Verification (Milestone 4, Task 4.4)")
        print("=" * 76)
        print(f"Model Checkpoint:    checkpoints/best_model.pt")
        print(f"Approval Granted:    {args.approve}")
        print(f"Max Rows Limit:      {args.max_rows}")
        if has_exec_account and exec_cfg:
            print(f"Execution User:      {exec_cfg.user} (Host: {exec_cfg.host}:{exec_cfg.port})")
            print(f"Execution DB:        {exec_cfg.database}")
        else:
            print(f"Execution Account:   Not configured ({exec_err})")
        print("-" * 76)

    # Instantiate pipeline from checkpoint
    try:
        executor = SQLExecutor(config=exec_cfg) if has_exec_account else None
        pipeline = NL2SQLPipeline.from_checkpoint(
            checkpoint_path=PROJECT_ROOT / "checkpoints" / "best_model.pt",
            executor=executor,
            default_schema=args.schema,
        )
    except Exception as err:
        print(f"Error initializing NL2SQLPipeline: {err}", file=sys.stderr)
        return 1

    questions = [args.question] if args.question else [
        "Show students with CGPA above 8",
        "Find courses with credits equal to 4",
        "List students whose department is Computer Science",
    ]

    all_results = []

    for q in questions:
        if not args.json:
            print(f"\n[Question] \"{q}\"")

        res = pipeline.process(
            question=q,
            schema=args.schema,
            approved=args.approve,
            max_rows=args.max_rows,
        )
        all_results.append(res.to_dict())

        if not args.json:
            print(f"  Stage Reached:   {res.stage_reached}")
            print(f"  Generated SQL:   {res.generated_sql}")
            if res.validation_result:
                print(f"  Validation:      {'PASSED' if res.validation_result.is_valid else 'FAILED'}")
                print(f"    - Syntax Valid:        {res.validation_result.is_syntax_valid}")
                print(f"    - Read-Only Permitted: {res.validation_result.is_permitted_read_only}")
                print(f"    - Schema Compatible:   {res.validation_result.is_schema_compatible}")
                print(f"    - In Model Scope:      {res.validation_result.is_in_model_scope}")

            if res.stage_reached == "execution_pending":
                print(f"  Execution Gate:  HALTED (Query was NOT sent to MySQL because approved=False)")
                print(f"                   To execute, pass --approve flag.")

            elif res.stage_reached == "executed" and res.execution_result:
                exec_r = res.execution_result
                print(f"  Execution:       {'SUCCESS' if exec_r.success else 'FAILED'}")
                print(f"    - Time:        {exec_r.execution_time_ms} ms")
                print(f"    - Rows:        {exec_r.row_count} rows retrieved")
                print(f"    - Truncated:   {exec_r.is_truncated}")
                if exec_r.columns:
                    print(f"    - Columns:     {', '.join(exec_r.columns)}")
                if exec_r.rows:
                    print("    - Preview:")
                    for row in exec_r.rows[:3]:
                        print(f"        {row}")
                    if exec_r.row_count > 3:
                        print(f"        ... ({exec_r.row_count - 3} more rows)")
                if exec_r.error:
                    print(f"    - Error:       {exec_r.error}")

    if args.json:
        print(json.dumps(all_results, indent=2))
    elif not has_exec_account:
        print_user_guidance()

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline_demo())
