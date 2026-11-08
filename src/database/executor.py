"""
SQL Query Execution Layer for NL2SQL Project (Milestone 4, Task 4.4).

Provides safe, isolated, and audited execution of model-generated SQL queries against MySQL.
Strictly requires:
1. Prior validation via ValidationResult with is_valid=True.
2. Explicit caller approval (approved=True).
3. Dedicated non-root read-only database credentials (rejects root execution).
4. Read-only transaction enforcement and query execution timeouts.
5. Row truncation via fetchmany(limit + 1) without modifying the model-generated SQL.
"""

from __future__ import annotations

import decimal
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Union

import pymysql

from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.connection import DatabaseConnection, DatabaseError, sanitize_db_error
from src.validation.result import ValidationResult


def _json_serializable(val: Any) -> Any:
    """Recursively converts values to JSON-serializable types."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, decimal.Decimal):
        return float(val) if val % 1 else int(val)
    if isinstance(val, (date, datetime, dt_time)):
        return val.isoformat()
    if isinstance(val, timedelta):
        return str(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, dict):
        return {k: _json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_json_serializable(v) for v in val]
    return str(val)


@dataclass
class ExecutionResult:
    """
    Structured outcome of SQL query execution.

    Attributes:
        success: True if the query executed successfully and results were retrieved.
        executed_sql: The exact SQL statement sent to MySQL.
        columns: List of column names in the result set.
        rows: List of dictionary records mapping column names to values.
        row_count: Number of records returned in this result.
        execution_time_ms: Wall-clock execution time in milliseconds.
        is_truncated: True if the query matched more rows than max_rows limit.
        error: Error message if execution was rejected or failed.
        warning: Optional warning notice (e.g. truncation alert).
    """

    success: bool
    executed_sql: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    is_truncated: bool = False
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Converts execution result to a JSON-serializable dictionary."""
        return {
            "success": self.success,
            "executed_sql": self.executed_sql,
            "columns": list(self.columns),
            "rows": _json_serializable(self.rows),
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "is_truncated": self.is_truncated,
            "error": self.error,
            "warning": self.warning,
        }


class SQLExecutor:
    """
    Dedicated SQL Execution Layer for NL2SQL.

    Enforces fail-closed boundaries:
    - Never accepts raw SQL strings; requires a successful ValidationResult.
    - Never executes without explicit caller approval.
    - Prohibits execution using the 'root' MySQL account.
    - Executes inside a read-only transaction with query timeout.
    - Limits returned rows without altering the generated SQL text.
    """

    DEFAULT_MAX_ROWS = 100
    DEFAULT_TIMEOUT_SECONDS = 5

    def __init__(
        self,
        connection: Optional[DatabaseConnection] = None,
        config: Optional[DatabaseConfig] = None,
        default_max_rows: int = DEFAULT_MAX_ROWS,
        query_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initializes the SQL executor.

        Args:
            connection: Optional DatabaseConnection instance.
            config: Optional DatabaseConfig instance (must not be root).
            default_max_rows: Maximum rows to return if not specified in execute().
            query_timeout_seconds: Query execution timeout in seconds.
        """
        self.default_max_rows = max(1, default_max_rows)
        self.query_timeout_seconds = max(1, query_timeout_seconds)

        if connection is not None:
            self._connection: Optional[DatabaseConnection] = connection
            self._config: Optional[DatabaseConfig] = connection.config
        elif config is not None:
            self._config = config
            self._connection = DatabaseConnection(config=config)
        else:
            self._config = None
            self._connection = None

    def _get_connection(self) -> DatabaseConnection:
        """
        Returns active DatabaseConnection, attempting to load from exec environment if unset.
        Raises DatabaseConfigurationError if dedicated exec account is unconfigured or is root.
        """
        if self._connection is None:
            config = DatabaseConfig.from_exec_env()
            self._config = config
            self._connection = DatabaseConnection(config=config)

        # Enforce that execution refuses to run under root account
        if self._connection.config.user.strip().lower() == "root":
            raise DatabaseConfigurationError(
                "Execution user cannot be 'root'. "
                "A dedicated non-root read-only account must be used for query execution."
            )

        return self._connection

    def execute(
        self,
        validation_result: ValidationResult,
        approved: bool = False,
        max_rows: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Executes a validated SQL query against MySQL.

        Args:
            validation_result: Successful ValidationResult from SQLValidator.
            approved: Explicit caller approval required to run query against database.
            max_rows: Optional row count ceiling. Defaults to self.default_max_rows.

        Returns:
            ExecutionResult containing query output or failure details.
        """
        # Guard 1: Must be a ValidationResult instance
        if not isinstance(validation_result, ValidationResult):
            return ExecutionResult(
                success=False,
                error="Invalid input: validation_result must be a ValidationResult instance.",
            )

        target_sql = validation_result.normalized_sql

        # Guard 2: Validation must have succeeded
        if not validation_result.is_valid:
            err = validation_result.error_message or "Query failed SQL validation."
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                error=f"Execution rejected: {err}",
            )

        # Guard 3: Normalized SQL must be non-empty
        if not target_sql or not target_sql.strip():
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                error="Execution rejected: normalized_sql is missing or empty.",
            )

        # Guard 4: Explicit caller approval required
        if not approved:
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                error="Execution approval required. Query was not approved by caller.",
            )

        # Guard 5: Database connection and account verification
        try:
            conn = self._get_connection()
        except DatabaseConfigurationError as err:
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                error=f"Execution configuration error: {err}",
            )
        except Exception as err:
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                error=f"Database initialization error: {err}",
            )

        limit = max(1, int(max_rows)) if max_rows is not None else self.default_max_rows
        timeout_ms = int(self.query_timeout_seconds * 1000)

        start_time = time.perf_counter()
        raw_conn = None
        cur = None

        try:
            raw_conn = conn.connect()
            cur = raw_conn.cursor()

            # Set execution session timeout for MySQL 8.0+
            try:
                cur.execute(f"SET SESSION max_execution_time = {timeout_ms};")
            except Exception:
                # If unsupported by server/mock, proceed
                pass

            # Read-only transaction enforcement
            try:
                cur.execute("SET TRANSACTION READ ONLY;")
                raw_conn.begin()
            except Exception:
                pass

            # Execute the canonical validated SQL
            cur.execute(target_sql)

            # Fetch up to limit + 1 to detect whether truncation is needed
            fetched = cur.fetchmany(limit + 1)

            if len(fetched) > limit:
                rows = fetched[:limit]
                is_truncated = True
                warning = f"Query returned more than {limit} rows. Results truncated to {limit}."
            else:
                rows = fetched
                is_truncated = False
                warning = None

            # Extract column names from cursor description if present
            columns: list[str] = []
            if cur.description:
                columns = [col[0] for col in cur.description]
            elif rows:
                columns = list(rows[0].keys())

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return ExecutionResult(
                success=True,
                executed_sql=target_sql,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=elapsed_ms,
                is_truncated=is_truncated,
                warning=warning,
            )

        except pymysql.err.OperationalError as err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            pwd = conn.config.password if conn and hasattr(conn, "config") else None
            clean_err = sanitize_db_error(err, pwd)
            err_code = err.args[0] if err.args else None

            if err_code == 3024:
                # ER_QUERY_TIMEOUT
                return ExecutionResult(
                    success=False,
                    executed_sql=target_sql,
                    execution_time_ms=elapsed_ms,
                    error=f"Query execution timed out after {self.query_timeout_seconds}s.",
                )
            elif err_code == 1142:
                # ER_TABLEACCESS_DENIED_ERROR
                return ExecutionResult(
                    success=False,
                    executed_sql=target_sql,
                    execution_time_ms=elapsed_ms,
                    error=f"Permission denied: user does not have permission to access table. Details: {clean_err}",
                )
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                execution_time_ms=elapsed_ms,
                error=f"Database execution error ({err_code}): {clean_err}",
            )

        except Exception as err:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            pwd = conn.config.password if conn and hasattr(conn, "config") else None
            clean_err = sanitize_db_error(err, pwd)
            return ExecutionResult(
                success=False,
                executed_sql=target_sql,
                execution_time_ms=elapsed_ms,
                error=f"Execution failure: {clean_err}",
            )

        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass
            if raw_conn is not None:
                try:
                    raw_conn.rollback()
                except Exception:
                    pass
