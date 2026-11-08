"""
MySQL Verification and Schema Inspection CLI Tool (Milestone 4, Task 4.2).

Usage:
    python scripts/verify_mysql.py
    python scripts/verify_mysql.py --inspect-schema
    python scripts/verify_mysql.py --env-file .env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseConnection, DatabaseError
from src.database.schema import DatabaseSchemaInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MySQL connectivity and inspect database schema."
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional path to custom .env file",
    )
    parser.add_argument(
        "--inspect-schema",
        action="store_true",
        help="Perform comprehensive schema inspection via INFORMATION_SCHEMA",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    return parser.parse_args()


def run_verification(env_file: str | None = None, inspect_schema: bool = False) -> int:
    config = DatabaseConfig.from_env(env_file=env_file)

    print("=" * 70)
    print("MySQL Database Connectivity Verification (Milestone 4, Task 4.2)")
    print("=" * 70)
    print(f"Target Host:     {config.host}")
    print(f"Target Port:     {config.port}")
    print(f"Target User:     {config.user}")
    print(f"Target Database: {config.database}")
    print(f"Password Set:    {'Yes (Masked: ' + config.masked_password + ')' if config.password else 'No (Empty)'}")
    print("-" * 70)

    conn = DatabaseConnection(config=config)
    print("\n[1/2] Testing MySQL connection and database selection...")
    status = conn.verify_connection()

    if not status["connected"]:
        print(f"FAILED: Connection could not be established.")
        print(f"Error Details: {status['error']}")
        print("\nPlease check:")
        print("  1. Is MySQL server running?")
        print("  2. Are the host, port, user, and password in .env correct?")
        print("  3. Does the target database exist?")
        print("=" * 70)
        return 1

    print("SUCCESS: Connection established and verified.")
    print(f"  MySQL Version:     {status['mysql_version']}")
    print(f"  Active Database:   {status['database']}")
    print(f"  Ping Status:       Alive ({status['alive']})")

    if inspect_schema:
        print("\n[2/2] Inspecting Database Schema via INFORMATION_SCHEMA...")
        inspector = DatabaseSchemaInspector(connection=conn)
        schema_meta = inspector.inspect_database()

        print(f"  Total Tables Found: {schema_meta.total_tables}")
        print(f"  Total Foreign Keys: {schema_meta.total_foreign_keys}")
        print("-" * 70)

        for table_name in schema_meta.table_names:
            t_meta = schema_meta.tables[table_name]
            print(f"\nTable: `{table_name}`")
            print(f"  Signature:    {t_meta.format_signature()}")
            print(f"  Primary Keys: {t_meta.primary_keys or 'None'}")

            print("  Columns:")
            for col in t_meta.columns:
                pk_flag = " [PK]" if col.is_primary_key else ""
                null_flag = "NULL" if col.is_nullable else "NOT NULL"
                print(f"    - {col.name:<18} {col.column_type:<15} {null_flag:<8}{pk_flag}")

            if t_meta.foreign_keys:
                print("  Foreign Keys:")
                for fk in t_meta.foreign_keys:
                    print(
                        f"    - {fk.constraint_name}: `{fk.column_name}` -> "
                        f"`{fk.referenced_table}`(`{fk.referenced_column}`) "
                        f"[ON UPDATE {fk.update_rule}, ON DELETE {fk.delete_rule}]"
                    )

        print("\nCanonical Schema Summary:")
        print(f"  {schema_meta.format_schema_summary()}")
    else:
        print("\n[2/2] Schema inspection skipped (use --inspect-schema to inspect tables).")

    print("=" * 70)
    print("Verification Completed Successfully.")
    print("=" * 70)
    return 0


def main() -> None:
    args = parse_args()
    exit_code = run_verification(env_file=args.env_file, inspect_schema=args.inspect_schema)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
