"""Consolidated tests for database configuration, connection, schema inspection, execution safety, and orchestration."""

from __future__ import annotations

import decimal
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymysql

from src.database.config import DatabaseConfig, DatabaseConfigurationError, parse_dotenv_file
from src.database.connection import (
    DatabaseAuthenticationError,
    DatabaseConnection,
    DatabaseConnectionError,
    DatabaseNotFoundError,
)
from src.database.executor import SQLExecutor
from src.database.schema import (
    ColumnMetadata,
    DatabaseSchemaInspector,
    DatabaseSchemaMetadata,
    ForeignKeyMetadata,
    TableMetadata,
)
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA, InferenceResult, NL2SQLInferencePipeline
from src.pipeline.orchestrator import NL2SQLPipeline
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator


def make_validation_result(
    sql: str = "SELECT name, cgpa FROM students WHERE cgpa > 8.0;",
    is_valid: bool = True,
    error: str | None = None,
) -> ValidationResult:
    return ValidationResult(
        is_valid=is_valid,
        is_syntax_valid=True,
        is_permitted_read_only=is_valid,
        is_schema_compatible=True,
        is_in_supported_scope=is_valid,
        is_in_model_scope=is_valid,
        normalized_sql=sql,
        raw_sql=sql,
        errors=[] if error is None else [error],
        warnings=[],
        ast_details={"tables": ["students"], "columns": ["name", "cgpa"]} if is_valid else {},
    )


def create_mock_connection(user: str = "nl2sql_exec", rows: list | None = None, description: tuple | None = None):
    conn = MagicMock(spec=DatabaseConnection)
    conn.config = DatabaseConfig(user=user, password="secret_exec_password", database="nl2sql_student_db")
    raw_conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = rows if rows is not None else []
    cur.fetchmany.return_value = rows if rows is not None else []
    cur.description = description if description is not None else (("name",), ("cgpa",))
    raw_conn.cursor.return_value = cur
    conn.connect.return_value = raw_conn
    return conn, raw_conn, cur


class TestDatabaseAndExecution(unittest.TestCase):
    """Verifies DB config/connection, schema inspection, safe read-only execution, and orchestration."""

    def test_database_config_validation_masking_and_env_loading(self) -> None:
        cfg = DatabaseConfig(password="SuperSecretPassword123!")
        self.assertEqual((cfg.host, cfg.port, cfg.user, cfg.database), ("localhost", 3306, "root", "nl2sql_student_db"))
        self.assertNotIn("SuperSecretPassword123!", repr(cfg))
        self.assertIn("********", repr(cfg))
        self.assertEqual(cfg.to_dict(mask_password=True)["password"], "********")
        self.assertEqual(cfg.to_dict(mask_password=False)["password"], "SuperSecretPassword123!")

        for bad_kwargs in ({"port": 0}, {"port": 70000}, {"host": ""}, {"database": ""}):
            with self.assertRaises(ValueError):
                DatabaseConfig(**bad_kwargs)

        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "MYSQL_HOST=db.example.com\nMYSQL_PORT=3307\nMYSQL_USER='u'\nMYSQL_PASSWORD=\"p\"\nMYSQL_DATABASE=d\n",
                encoding="utf-8",
            )
            parsed = parse_dotenv_file(env_path)
            self.assertEqual(parsed["MYSQL_HOST"], "db.example.com")
            loaded = DatabaseConfig.from_env(env_file=env_path)
            self.assertEqual((loaded.port, loaded.user, loaded.password, loaded.database), (3307, "u", "p", "d"))

        with patch.dict("os.environ", {}, clear=True), patch("src.database.config.Path.is_file", return_value=False):
            with self.assertRaises(DatabaseConfigurationError):
                DatabaseConfig.from_exec_env(env_file="nonexistent.env")

        with patch.dict("os.environ", {"MYSQL_EXEC_USER": "root", "MYSQL_EXEC_PASSWORD": "pwd"}), patch("src.database.config.Path.is_file", return_value=False):
            with self.assertRaises(DatabaseConfigurationError):
                DatabaseConfig.from_exec_env()

    @patch("pymysql.connect")
    def test_connection_lifecycle_verify_and_error_mapping(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = {"alive": 1, "current_db": "nl2sql_student_db", "version": "8.0.43"}
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        mgr = DatabaseConnection(config=DatabaseConfig(password="dummy"))
        with mgr:
            self.assertTrue(mgr.is_connected())
        mock_conn.close.assert_called_once()
        self.assertFalse(mgr.is_connected())

        status = DatabaseConnection().verify_connection()
        self.assertEqual((status["status"], status["alive"], status["mysql_version"]), ("success", True, "8.0.43"))

        secret_pwd = "MySecretPassword123"
        for err_code, msg, exc_cls, expected_frag in [
            (1045, f"Access denied using password {secret_pwd}", DatabaseAuthenticationError, "Authentication failed"),
            (1049, "Unknown database 'missing_db'", DatabaseNotFoundError, "missing_db"),
            (2003, "Can't connect to MySQL server", DatabaseConnectionError, "Could not connect to MySQL server"),
        ]:
            mock_connect.side_effect = pymysql.err.OperationalError(err_code, msg)
            with self.assertRaises(exc_cls) as ctx:
                DatabaseConnection(config=DatabaseConfig(password=secret_pwd, database="missing_db")).connect()
            self.assertNotIn(secret_pwd, str(ctx.exception))
            self.assertIn(expected_frag, str(ctx.exception))

    def test_live_socket_error_scrubbing_and_cleanup(self) -> None:
        conn_port = DatabaseConnection(config=DatabaseConfig(host="127.0.0.1", port=59999, connect_timeout=1, password="SecretPasswordXYZ"))
        with self.assertRaises(DatabaseConnectionError) as ctx:
            conn_port.connect()
        self.assertNotIn("SecretPasswordXYZ", str(ctx.exception))
        self.assertFalse(conn_port.is_connected())

        conn_auth = DatabaseConnection(config=DatabaseConfig(host="127.0.0.1", port=3306, user="nonexistent_user_xyz", password="wrong_password_999", connect_timeout=2))
        with self.assertRaises(DatabaseAuthenticationError) as ctx_auth:
            conn_auth.connect()
        self.assertNotIn("wrong_password_999", str(ctx_auth.exception))

    @patch.object(DatabaseConnection, "execute_read_query")
    def test_schema_metadata_and_inspector_queries(self, mock_query: MagicMock) -> None:
        s_meta = TableMetadata(
            name="students",
            columns=[ColumnMetadata(name="student_id", data_type="int", column_type="int", is_nullable=False, is_primary_key=True)],
            primary_keys=["student_id"],
        )
        e_meta = TableMetadata(
            name="enrollments",
            columns=[ColumnMetadata(name="enrollment_id", data_type="int", column_type="int", is_nullable=False, is_primary_key=True)],
            primary_keys=["enrollment_id"],
            foreign_keys=[ForeignKeyMetadata("fk_s", "student_id", "students", "student_id", "CASCADE", "CASCADE")],
        )
        db_meta = DatabaseSchemaMetadata("nl2sql_student_db", tables={"students": s_meta, "enrollments": e_meta})
        self.assertEqual((db_meta.total_tables, db_meta.total_foreign_keys), (2, 1))
        self.assertEqual(db_meta.format_schema_summary(), "enrollments(enrollment_id) | students(student_id)")

        mock_query.side_effect = [
            [{"TABLE_NAME": "courses"}, {"TABLE_NAME": "students"}],
            [{"COLUMN_NAME": "student_id"}],
            [{"COLUMN_NAME": "student_id", "DATA_TYPE": "int", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_DEFAULT": None, "EXTRA": "auto_increment"}],
        ]
        inspector = DatabaseSchemaInspector(connection=DatabaseConnection())
        self.assertEqual(inspector.get_table_names(), ["courses", "students"])
        cols = inspector.get_columns("students")
        self.assertTrue(cols[0].is_primary_key)

    def test_executor_validation_approval_and_root_guards(self) -> None:
        conn, raw_conn, cur = create_mock_connection()
        executor = SQLExecutor(connection=conn, query_timeout_seconds=7)

        res_ok = executor.execute(make_validation_result(), approved=True)
        self.assertTrue(res_ok.success)
        cur.execute.assert_any_call("SET SESSION max_execution_time = 7000;")

        for val_arg, approved_flag in [
            (make_validation_result(sql="DELETE FROM students", is_valid=False, error="DELETE forbidden"), True),
            ("SELECT * FROM students;", True),
            (None, True),
            (make_validation_result(sql=""), True),
            (make_validation_result(), False),
            (make_validation_result(), None),
        ]:
            conn_i, raw_i, _ = create_mock_connection()
            res = SQLExecutor(connection=conn_i).execute(val_arg, approved=approved_flag)  # type: ignore[arg-type]
            self.assertFalse(res.success)
            raw_i.cursor.assert_not_called()

        root_conn, _, _ = create_mock_connection(user="root")
        res_root = SQLExecutor(connection=root_conn).execute(make_validation_result(), approved=True)
        self.assertFalse(res_root.success)
        self.assertIn("cannot be 'root'", res_root.error)

    def test_executor_result_handling_truncation_and_error_cleanup(self) -> None:
        # Multi-row, NULL, Decimal/date serialization
        rows = [
            {"name": "Diana", "cgpa": decimal.Decimal("8.75"), "dob": date(2002, 5, 14), "ts": datetime(2023, 1, 15, 10, 30, 0)},
            {"name": "Bob", "cgpa": None, "dob": None, "ts": None},
        ]
        conn, _, _ = create_mock_connection(rows=rows, description=(("name",), ("cgpa",), ("dob",), ("ts",)))
        res = SQLExecutor(connection=conn).execute(make_validation_result(), approved=True)
        d = res.to_dict()
        self.assertEqual((res.row_count, res.columns, res.is_truncated), (2, ["name", "cgpa", "dob", "ts"], False))
        self.assertEqual((d["rows"][0]["cgpa"], d["rows"][0]["dob"], d["rows"][0]["ts"]), (8.75, "2002-05-14", "2023-01-15T10:30:00"))
        self.assertIsNone(d["rows"][1]["cgpa"])

        # Empty result set and Python-side fetchmany truncation preserving exact SQL
        conn_e, _, _ = create_mock_connection(rows=[], description=(("name",),))
        res_empty = SQLExecutor(connection=conn_e).execute(make_validation_result(), approved=True)
        self.assertEqual((res_empty.success, res_empty.row_count, res_empty.rows), (True, 0, []))

        orig_sql = "SELECT id FROM students;"
        conn_t, _, cur_t = create_mock_connection(rows=[{"id": i} for i in range(5)], description=(("id",),))
        res_trunc = SQLExecutor(connection=conn_t).execute(make_validation_result(sql=orig_sql), approved=True, max_rows=3)
        self.assertTrue(res_trunc.is_truncated)
        self.assertEqual(res_trunc.row_count, 3)
        self.assertEqual(res_trunc.executed_sql, orig_sql)
        cur_t.fetchmany.assert_called_with(4)
        cur_t.execute.assert_any_call(orig_sql)

        # Timeout, cleanup rollback, and password scrubbing
        conn_err, raw_err, cur_err = create_mock_connection()
        cur_err.execute.side_effect = pymysql.err.OperationalError(3024, "max_statement_time exceeded")
        res_to = SQLExecutor(connection=conn_err, query_timeout_seconds=3).execute(make_validation_result(), approved=True)
        self.assertFalse(res_to.success)
        self.assertIn("timed out after 3s", res_to.error)
        cur_err.close.assert_called_once()
        raw_err.rollback.assert_called_once()

        conn_pwd = MagicMock(spec=DatabaseConnection)
        conn_pwd.config = DatabaseConfig(user="nl2sql_exec", password="super_secret_exec_pwd")
        conn_pwd.connect.side_effect = Exception("Failed using password super_secret_exec_pwd")
        res_pwd = SQLExecutor(connection=conn_pwd).execute(make_validation_result(), approved=True)
        self.assertNotIn("super_secret_exec_pwd", res_pwd.error)
        self.assertIn("********", res_pwd.error)

    def test_orchestrator_staged_gating_and_serialization(self) -> None:
        mock_inf = MagicMock(spec=NL2SQLInferencePipeline)
        mock_inf.default_schema = DEFAULT_STUDENT_SCHEMA
        valid_inf = InferenceResult(
            sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0;",
            tokens=["SELECT", "name", ",", "cgpa", "FROM", "students", "WHERE", "cgpa", ">", "8.0", ";"],
            source_tokens=["students"],
            source_unk_count=0,
            target_unk_count=0,
            is_valid_sql=True,
            syntax_error=None,
            generated_token_ids=[1, 2],
            question="Show students with CGPA above 8",
            schema=DEFAULT_STUDENT_SCHEMA,
        )
        conn, raw_conn, _ = create_mock_connection(rows=[{"name": "Alice", "cgpa": 9.1}], description=(("name",), ("cgpa",)))
        pipeline = NL2SQLPipeline(
            inference_pipeline=mock_inf,
            validator=SQLValidator(schema=DEFAULT_STUDENT_SCHEMA, enforce_model_scope=True),
            executor=SQLExecutor(connection=conn),
            default_schema=DEFAULT_STUDENT_SCHEMA,
        )

        # Validation halts unsafe SQL even when approved=True
        mock_inf.predict.return_value = InferenceResult(
            sql="DROP TABLE students;", tokens=[], source_tokens=[], source_unk_count=0, target_unk_count=0,
            is_valid_sql=False, syntax_error="Mutations disallowed", generated_token_ids=[], question="Drop", schema=DEFAULT_STUDENT_SCHEMA,
        )
        halted_val = pipeline.process("Drop", approved=True)
        self.assertEqual(halted_val.stage_reached, "validation")
        self.assertIsNone(halted_val.execution_result)
        raw_conn.cursor.assert_not_called()

        # Unapproved pauses at execution_pending without querying DB
        mock_inf.predict.return_value = valid_inf
        self.assertEqual(pipeline.generate_sql("Show students").sql, valid_inf.sql)
        self.assertFalse(pipeline.validate_sql("SELECT name FROM students WHERE name LIKE 'A%'").is_in_model_scope)
        pending = pipeline.process("Show students", approved=False)
        self.assertEqual(pending.stage_reached, "execution_pending")
        self.assertIsNone(pending.execution_result)
        raw_conn.cursor.assert_not_called()

        # Approved completes execution and serializes cleanly
        done = pipeline.process("Show students", approved=True)
        self.assertEqual(done.stage_reached, "executed")
        self.assertTrue(done.execution_result and done.execution_result.success)
        serialized = done.to_dict()
        self.assertEqual(serialized["execution_result"]["rows"][0]["name"], "Alice")


if __name__ == "__main__":
    unittest.main()
