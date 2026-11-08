"""
Unit and Integration Test Suite for End-to-End SQL Execution & Orchestration (Milestone 4, Task 4.4).

Tests:
- Category A: Validation Boundary Enforcement (valid result accepted, failed result rejected, raw SQL rejected, missing/empty SQL rejected).
- Category B: Execution Approval Gate (approved=False rejected, approved=True executes).
- Category C: Result Handling & Serialization (multiple rows, empty results, NULLs, Decimals, truncation, query preservation).
- Category D: Safety & Database Isolation (prohibits root, requires dedicated exec user, timeouts, cleanup, sanitized errors).
- Category E: Orchestration Layer (staged execution, unapproved stopping, validation failure halting, end-to-end execution, serialization).
"""

from __future__ import annotations

import decimal
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymysql

from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.connection import DatabaseConnection
from src.database.executor import ExecutionResult, SQLExecutor
from src.inference.pipeline import InferenceResult, NL2SQLInferencePipeline
from src.pipeline.orchestrator import NL2SQLPipeline, PipelineResult
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator


def make_valid_result(sql: str = "SELECT name, cgpa FROM students WHERE cgpa > 8.0;") -> ValidationResult:
    """Creates a mock successful ValidationResult."""
    return ValidationResult(
        is_valid=True,
        is_syntax_valid=True,
        is_permitted_read_only=True,
        is_schema_compatible=True,
        is_in_supported_scope=True,
        is_in_model_scope=True,
        normalized_sql=sql,
        raw_sql=sql,
        errors=[],
        warnings=[],
        ast_details={"tables": ["students"], "columns": ["name", "cgpa"]},
    )


def make_invalid_result(
    sql: str = "DELETE FROM students WHERE student_id = 1;",
    error: str = "Only SELECT queries are permitted; got 'DELETE'.",
) -> ValidationResult:
    """Creates a mock failed ValidationResult."""
    return ValidationResult(
        is_valid=False,
        is_syntax_valid=True,
        is_permitted_read_only=False,
        is_schema_compatible=True,
        is_in_supported_scope=False,
        is_in_model_scope=False,
        normalized_sql=sql,
        raw_sql=sql,
        errors=[error],
        warnings=[],
        ast_details={},
    )


def create_mock_connection(user: str = "nl2sql_exec", rows: list = None, description: tuple = None):
    """Creates a mock DatabaseConnection with cursor and execution tracking."""
    conn = MagicMock(spec=DatabaseConnection)
    config = DatabaseConfig(user=user, password="secret_exec_password", database="nl2sql_student_db")
    conn.config = config

    raw_conn = MagicMock()
    cur = MagicMock()

    cur.fetchall.return_value = rows if rows is not None else []
    cur.fetchmany.return_value = rows if rows is not None else []
    cur.description = description if description is not None else (("name",), ("cgpa",))

    raw_conn.cursor.return_value = cur
    conn.connect.return_value = raw_conn

    return conn, raw_conn, cur


class TestCategoryAValidationBoundary(unittest.TestCase):
    """Category A: Enforces that executor strictly gates on valid ValidationResult."""

    def setUp(self) -> None:
        self.conn, self.raw_conn, self.cur = create_mock_connection()
        self.executor = SQLExecutor(connection=self.conn)

    def test_valid_result_accepted(self) -> None:
        """Valid ValidationResult with approved=True should execute query."""
        val_res = make_valid_result()
        res = self.executor.execute(val_res, approved=True)
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        self.assertEqual(res.executed_sql, val_res.normalized_sql)
        self.cur.execute.assert_any_call(val_res.normalized_sql)

    def test_invalid_result_rejected(self) -> None:
        """Failed ValidationResult must be rejected immediately."""
        val_res = make_invalid_result()
        res = self.executor.execute(val_res, approved=True)
        self.assertFalse(res.success)
        self.assertIn("Execution rejected", res.error)
        self.assertIn("DELETE", res.error)
        self.raw_conn.cursor.assert_not_called()

    def test_raw_sql_rejected(self) -> None:
        """Passing raw SQL string directly instead of ValidationResult must fail."""
        res = self.executor.execute("SELECT * FROM students;", approved=True)  # type: ignore
        self.assertFalse(res.success)
        self.assertIn("ValidationResult instance", res.error)
        self.raw_conn.cursor.assert_not_called()

    def test_none_validation_result_rejected(self) -> None:
        """Passing None must fail."""
        res = self.executor.execute(None, approved=True)  # type: ignore
        self.assertFalse(res.success)
        self.assertIn("ValidationResult instance", res.error)

    def test_empty_normalized_sql_rejected(self) -> None:
        """ValidationResult with empty normalized_sql must be rejected."""
        val_res = make_valid_result(sql="")
        res = self.executor.execute(val_res, approved=True)
        self.assertFalse(res.success)
        self.assertIn("normalized_sql is missing or empty", res.error)
        self.raw_conn.cursor.assert_not_called()


class TestCategoryBApprovalGate(unittest.TestCase):
    """Category B: Enforces that explicit caller approval is mandatory."""

    def setUp(self) -> None:
        self.conn, self.raw_conn, self.cur = create_mock_connection()
        self.executor = SQLExecutor(connection=self.conn)

    def test_approved_false_rejected(self) -> None:
        """approved=False must halt execution and return error without executing query."""
        val_res = make_valid_result()
        res = self.executor.execute(val_res, approved=False)
        self.assertFalse(res.success)
        self.assertIn("Execution approval required", res.error)
        self.assertEqual(res.executed_sql, val_res.normalized_sql)
        self.raw_conn.cursor.assert_not_called()

    def test_approved_none_rejected(self) -> None:
        """Falsy approval (None, 0, '') must also be rejected."""
        val_res = make_valid_result()
        res = self.executor.execute(val_res, approved=None)  # type: ignore
        self.assertFalse(res.success)
        self.assertIn("Execution approval required", res.error)
        self.raw_conn.cursor.assert_not_called()

    def test_approved_true_executes(self) -> None:
        """approved=True proceeds to database execution."""
        val_res = make_valid_result()
        res = self.executor.execute(val_res, approved=True)
        self.assertTrue(res.success)
        self.cur.execute.assert_any_call(val_res.normalized_sql)


class TestCategoryCResultHandling(unittest.TestCase):
    """Category C: Verifies result parsing, serialization, and row truncation."""

    def test_multiple_rows_and_column_names(self) -> None:
        """Verifies multiple rows and column headers are retrieved correctly."""
        sample_rows = [
            {"name": "Alice", "cgpa": 9.2},
            {"name": "Bob", "cgpa": 8.5},
        ]
        conn, _, _ = create_mock_connection(rows=sample_rows, description=(("name",), ("cgpa",)))
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertTrue(res.success)
        self.assertEqual(res.row_count, 2)
        self.assertEqual(res.columns, ["name", "cgpa"])
        self.assertEqual(res.rows, sample_rows)
        self.assertFalse(res.is_truncated)
        self.assertIsNone(res.warning)

    def test_empty_result_set(self) -> None:
        """Verifies empty result set produces row_count=0 and empty rows list."""
        conn, _, _ = create_mock_connection(rows=[], description=(("name",), ("cgpa",)))
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertTrue(res.success)
        self.assertEqual(res.row_count, 0)
        self.assertEqual(res.rows, [])
        self.assertEqual(res.columns, ["name", "cgpa"])

    def test_null_values_in_rows(self) -> None:
        """Verifies NULL values in rows are handled without errors."""
        sample_rows = [{"name": "Charlie", "cgpa": None}]
        conn, _, _ = create_mock_connection(rows=sample_rows)
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertTrue(res.success)
        self.assertIsNone(res.rows[0]["cgpa"])

        d = res.to_dict()
        self.assertIsNone(d["rows"][0]["cgpa"])

    def test_decimal_and_date_serialization(self) -> None:
        """Verifies Decimal, Date, and DateTime objects serialize cleanly in to_dict()."""
        sample_rows = [
            {
                "name": "Diana",
                "cgpa": decimal.Decimal("8.75"),
                "dob": date(2002, 5, 14),
                "created_at": datetime(2023, 1, 15, 10, 30, 0),
            }
        ]
        conn, _, _ = create_mock_connection(
            rows=sample_rows,
            description=(("name",), ("cgpa",), ("dob",), ("created_at",)),
        )
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertTrue(res.success)

        # Serialize to dict
        d = res.to_dict()
        self.assertIsInstance(d["rows"][0]["cgpa"], float)
        self.assertEqual(d["rows"][0]["cgpa"], 8.75)
        self.assertEqual(d["rows"][0]["dob"], "2002-05-14")
        self.assertEqual(d["rows"][0]["created_at"], "2023-01-15T10:30:00")

    def test_row_truncation_when_exceeding_max_rows(self) -> None:
        """Verifies truncation activates when fetched rows exceed max_rows."""
        # 5 returned rows when max_rows=3
        sample_rows = [{"id": i} for i in range(1, 6)]
        conn, _, cur = create_mock_connection(rows=sample_rows, description=(("id",),))
        executor = SQLExecutor(connection=conn, default_max_rows=3)

        res = executor.execute(make_valid_result(), approved=True, max_rows=3)
        self.assertTrue(res.success)
        self.assertEqual(res.row_count, 3)
        self.assertTrue(res.is_truncated)
        self.assertIsNotNone(res.warning)
        self.assertIn("truncated to 3", res.warning)
        # Verify fetchmany called with max_rows + 1
        cur.fetchmany.assert_called_with(4)

    def test_truncation_does_not_modify_sql_query(self) -> None:
        """Verifies that truncation enforces limit in Python via fetchmany, NOT by modifying SQL."""
        original_sql = "SELECT name, cgpa FROM students WHERE cgpa > 8.0;"
        val_res = make_valid_result(sql=original_sql)
        conn, _, cur = create_mock_connection(rows=[{"name": "Alice"}])
        executor = SQLExecutor(connection=conn)

        res = executor.execute(val_res, approved=True, max_rows=5)
        self.assertTrue(res.success)
        self.assertEqual(res.executed_sql, original_sql)
        # Assert the query passed to cursor.execute was the exact unmodified string
        cur.execute.assert_any_call(original_sql)


class TestCategoryDSafetyAndIsolation(unittest.TestCase):
    """Category D: Enforces security constraints and operational safety."""

    def test_root_account_rejected_by_executor(self) -> None:
        """SQLExecutor must strictly reject running queries with the 'root' account."""
        root_conn, _, _ = create_mock_connection(user="root")
        executor = SQLExecutor(connection=root_conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertFalse(res.success)
        self.assertIn("Execution user cannot be 'root'", res.error)

    def test_missing_exec_user_in_from_exec_env(self) -> None:
        """DatabaseConfig.from_exec_env must raise error if user is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.database.config.Path.is_file", return_value=False):
                with self.assertRaises(DatabaseConfigurationError) as ctx:
                    DatabaseConfig.from_exec_env(env_file="nonexistent.env")
                self.assertIn("MYSQL_EXEC_USER", str(ctx.exception))

    def test_root_rejected_by_from_exec_env(self) -> None:
        """DatabaseConfig.from_exec_env must raise error if user is configured as root."""
        with patch.dict("os.environ", {"MYSQL_EXEC_USER": "root", "MYSQL_EXEC_PASSWORD": "pwd"}):
            with patch("src.database.config.Path.is_file", return_value=False):
                with self.assertRaises(DatabaseConfigurationError) as ctx:
                    DatabaseConfig.from_exec_env()
                self.assertIn("cannot be 'root'", str(ctx.exception))

    def test_query_timeout_applied_in_session(self) -> None:
        """Verifies that SET SESSION max_execution_time is called with timeout in ms."""
        conn, _, cur = create_mock_connection()
        executor = SQLExecutor(connection=conn, query_timeout_seconds=7)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertTrue(res.success)
        cur.execute.assert_any_call("SET SESSION max_execution_time = 7000;")

    def test_timeout_error_handling(self) -> None:
        """Verifies that MySQL operational error 3024 (ER_QUERY_TIMEOUT) produces clean timeout error."""
        conn, _, cur = create_mock_connection()
        cur.execute.side_effect = pymysql.err.OperationalError(3024, "Query execution was interrupted, max_statement_time exceeded")
        executor = SQLExecutor(connection=conn, query_timeout_seconds=3)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertFalse(res.success)
        self.assertIn("timed out after 3s", res.error)

    def test_connection_error_handling(self) -> None:
        """Verifies that connection failure returns a clean error result without crashing."""
        conn = MagicMock(spec=DatabaseConnection)
        conn.config = DatabaseConfig(user="nl2sql_exec", password="pwd")
        conn.connect.side_effect = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertFalse(res.success)
        self.assertIn("2003", res.error)

    def test_cursor_and_transaction_cleanup_in_finally(self) -> None:
        """Verifies that cursor is closed and rollback is performed even on failure."""
        conn, raw_conn, cur = create_mock_connection()
        cur.execute.side_effect = RuntimeError("Simulated query execution crash")
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertFalse(res.success)
        cur.close.assert_called_once()
        raw_conn.rollback.assert_called_once()

    def test_sensitive_password_scrubbed_from_error(self) -> None:
        """Verifies that raw password is not exposed in error messages."""
        conn = MagicMock(spec=DatabaseConnection)
        secret_pwd = "super_secret_exec_pwd_987"
        conn.config = DatabaseConfig(user="nl2sql_exec", password=secret_pwd)
        conn.connect.side_effect = Exception(f"Failed using password {secret_pwd} on host localhost")
        executor = SQLExecutor(connection=conn)

        res = executor.execute(make_valid_result(), approved=True)
        self.assertFalse(res.success)
        self.assertNotIn(secret_pwd, res.error)
        self.assertIn("********", res.error)


class TestCategoryEOrchestrationLayer(unittest.TestCase):
    """Category E: Verifies end-to-end pipeline coordination and stage gating."""

    def setUp(self) -> None:
        # Mock inference pipeline
        self.mock_inf = MagicMock(spec=NL2SQLInferencePipeline)
        self.mock_inf.default_schema = (
            "students(student_id, name, age, department, cgpa) | "
            "courses(course_id, course_name, credits) | "
            "enrollments(enrollment_id, student_id, course_id, semester, marks)"
        )
        self.mock_inf_result = InferenceResult(
            sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0;",
            tokens=["SELECT", "name", ",", "cgpa", "FROM", "students", "WHERE", "cgpa", ">", "8.0", ";"],
            source_tokens=["students", "(", "name", ")"],
            source_unk_count=0,
            target_unk_count=0,
            is_valid_sql=True,
            syntax_error=None,
            generated_token_ids=[1, 2, 3],
            question="Show students with CGPA above 8",
            schema=self.mock_inf.default_schema,
        )
        self.mock_inf.predict.return_value = self.mock_inf_result

        # Mock executor with non-root account
        self.conn, self.raw_conn, self.cur = create_mock_connection(
            rows=[{"name": "Alice", "cgpa": 9.1}],
            description=(("name",), ("cgpa",)),
        )
        self.executor = SQLExecutor(connection=self.conn)

        # Real validator
        self.validator = SQLValidator(schema=self.mock_inf.default_schema, enforce_model_scope=True)

        self.pipeline = NL2SQLPipeline(
            inference_pipeline=self.mock_inf,
            validator=self.validator,
            executor=self.executor,
            default_schema=self.mock_inf.default_schema,
        )

    def test_generate_sql_only_infers(self) -> None:
        """generate_sql() should only call inference, never validation or execution."""
        inf_res = self.pipeline.generate_sql("Show students with CGPA above 8")
        self.assertEqual(inf_res.sql, self.mock_inf_result.sql)
        self.mock_inf.predict.assert_called_once()
        self.raw_conn.cursor.assert_not_called()

    def test_validate_sql_enforces_model_scope_by_default(self) -> None:
        """validate_sql() defaults to strict model scope enforcement."""
        # Query with LIKE which is outside current Seq2Seq vocabulary
        out_of_scope_sql = "SELECT name FROM students WHERE name LIKE 'A%';"
        val_res = self.pipeline.validate_sql(out_of_scope_sql)
        self.assertFalse(val_res.is_valid)
        self.assertFalse(val_res.is_in_model_scope)

    def test_pipeline_halts_on_validation_failure(self) -> None:
        """When inference produces invalid or out-of-scope SQL, pipeline stops at validation stage."""
        self.mock_inf.predict.return_value = InferenceResult(
            sql="DROP TABLE students;",
            tokens=["DROP", "TABLE", "students", ";"],
            source_tokens=[],
            source_unk_count=0,
            target_unk_count=0,
            is_valid_sql=False,
            syntax_error="Mutations disallowed",
            generated_token_ids=[],
            question="Drop table",
            schema=self.pipeline.default_schema,
        )

        res = self.pipeline.process(question="Drop table", approved=True)
        self.assertEqual(res.stage_reached, "validation")
        self.assertFalse(res.validation_result.is_valid)
        self.assertIsNone(res.execution_result)
        self.assertIn("Validation stage failed", res.error)
        self.raw_conn.cursor.assert_not_called()

    def test_pipeline_halts_at_execution_pending_when_unapproved(self) -> None:
        """When approved=False, pipeline validates SQL and pauses at execution_pending."""
        res = self.pipeline.process(question="Show students with CGPA above 8", approved=False)
        self.assertEqual(res.stage_reached, "execution_pending")
        self.assertFalse(res.approved)
        self.assertIsNotNone(res.validation_result)
        self.assertTrue(res.validation_result.is_valid)
        self.assertIsNone(res.execution_result)
        self.assertIsNone(res.error)
        self.raw_conn.cursor.assert_not_called()

    def test_pipeline_executes_to_completion_when_approved(self) -> None:
        """When approved=True and SQL is valid, pipeline executes and returns records."""
        res = self.pipeline.process(question="Show students with CGPA above 8", approved=True)
        self.assertEqual(res.stage_reached, "executed")
        self.assertTrue(res.approved)
        self.assertIsNotNone(res.execution_result)
        self.assertTrue(res.execution_result.success)
        self.assertEqual(res.execution_result.row_count, 1)
        self.assertEqual(res.execution_result.rows[0]["name"], "Alice")
        self.assertIsNone(res.error)

    def test_pipeline_result_to_dict_serialization(self) -> None:
        """PipelineResult to_dict() must serialize all stages cleanly."""
        res = self.pipeline.process(question="Show students with CGPA above 8", approved=True)
        d = res.to_dict()
        self.assertEqual(d["stage_reached"], "executed")
        self.assertTrue(d["approved"])
        self.assertIn("inference_result", d)
        self.assertIn("validation_result", d)
        self.assertIn("execution_result", d)
        self.assertEqual(d["execution_result"]["rows"][0]["name"], "Alice")


if __name__ == "__main__":
    unittest.main()
