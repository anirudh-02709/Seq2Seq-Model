"""
Unit and Integration Test Suite for NL2SQL FastAPI Backend (Milestone 5.2).

Tests:
1. Health endpoint (/api/health).
2. Schema endpoint (/api/schema).
3. Valid natural-language generation (/api/generate).
4. Invalid or empty requests (/api/generate).
5. Invalid generated SQL handling.
6. Execution of valid SELECT statements (/api/execute).
7. Rejection of non-SELECT statements (INSERT, UPDATE, DELETE, DROP).
8. Rejection of multi-statement SQL.
9. Server-side revalidation of submitted SQL.
10. Database execution errors.
11. Empty query results.
12. Result truncation behavior.
13. Model initialization and lifecycle behavior.
14. Failure handling when required components cannot initialize.
15. CORS configuration and headers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.app import create_app, resolve_cors_origins
from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.executor import ExecutionResult
from src.database.schema import ColumnMetadata, DatabaseSchemaMetadata, ForeignKeyMetadata, TableMetadata
from src.inference.pipeline import InferenceResult
from src.pipeline.orchestrator import NL2SQLPipeline
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator


def build_mock_pipeline() -> MagicMock:
    """Creates a mock NL2SQLPipeline instance for testing without loading model weights."""
    mock_pipeline = MagicMock(spec=NL2SQLPipeline)
    mock_pipeline.default_schema = (
        "students(student_id, name, age, department, cgpa) | "
        "courses(course_id, course_name, credits) | "
        "enrollments(enrollment_id, student_id, course_id, semester, marks)"
    )

    # Mock real SQLValidator for accurate AST validation testing
    real_validator = SQLValidator(schema=mock_pipeline.default_schema, enforce_model_scope=True)
    mock_pipeline.validator = real_validator
    mock_pipeline.validate_sql.side_effect = real_validator.validate

    # Mock vocabulary lengths and device
    mock_src_vocab = MagicMock()
    mock_src_vocab.__len__.return_value = 198
    mock_tgt_vocab = MagicMock()
    mock_tgt_vocab.__len__.return_value = 70

    mock_inf_pipe = MagicMock()
    mock_inf_pipe.src_vocab = mock_src_vocab
    mock_inf_pipe.tgt_vocab = mock_tgt_vocab
    mock_inf_pipe.device = "cpu"
    mock_pipeline.inference_pipeline = mock_inf_pipe

    # Default generate_sql behavior
    mock_pipeline.generate_sql.return_value = InferenceResult(
        sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0",
        tokens=["SELECT", "name", ",", "cgpa", "FROM", "students", "WHERE", "cgpa", ">", "8.0"],
        source_tokens=["students", "(", "name", ")", "<SEP>", "show", "students"],
        source_unk_count=0,
        target_unk_count=0,
        is_valid_sql=True,
        syntax_error=None,
        generated_token_ids=[5, 10, 15],
        question="Show students with CGPA above 8",
        schema=mock_pipeline.default_schema,
    )

    # Default execute_sql behavior
    mock_pipeline.execute_sql.return_value = ExecutionResult(
        success=True,
        executed_sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0",
        columns=["name", "cgpa"],
        rows=[
            {"name": "Aarav Sharma", "cgpa": 8.92},
            {"name": "Priya Patel", "cgpa": 9.15},
        ],
        row_count=2,
        execution_time_ms=2.45,
        is_truncated=False,
        warning=None,
        error=None,
    )

    return mock_pipeline


class TestNL2SQLAPI(unittest.TestCase):
    """Test suite for NL2SQL FastAPI endpoints."""

    def setUp(self) -> None:
        self.mock_pipeline = build_mock_pipeline()
        self.app = create_app(pipeline=self.mock_pipeline)
        self.client = TestClient(self.app)

    # ==========================================================================
    # 1. Health Endpoint Tests
    # ==========================================================================

    @patch("src.api.routes.DatabaseConnection")
    @patch("src.api.routes.DatabaseConfig")
    def test_health_endpoint_healthy(self, mock_db_config: MagicMock, mock_db_conn_cls: MagicMock) -> None:
        """Test GET /api/health returns healthy when model and DB are online."""
        mock_cfg = MagicMock()
        mock_cfg.database = "nl2sql_student_db"
        mock_cfg.host = "localhost"
        mock_cfg.port = 3306
        mock_cfg.user = "nl2sql_exec"
        mock_db_config.from_exec_env.return_value = mock_cfg

        mock_conn = MagicMock()
        mock_conn.verify_connection.return_value = {
            "status": "success",
            "connected": True,
            "mysql_version": "8.0.43",
            "database": "nl2sql_student_db",
        }
        mock_db_conn_cls.return_value = mock_conn

        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model"]["loaded"])
        self.assertEqual(data["model"]["source_vocab_size"], 198)
        self.assertEqual(data["model"]["target_vocab_size"], 70)
        self.assertTrue(data["database"]["connected"])
        self.assertEqual(data["database"]["database"], "nl2sql_student_db")
        self.assertEqual(data["database"]["exec_user"], "nl2sql_exec")

        # Crucial security check: Ensure no password fields are exposed
        raw_text = response.text
        self.assertNotIn("password", raw_text.lower())
        self.assertNotIn("secret", raw_text.lower())

    @patch("src.api.routes.DatabaseConnection")
    @patch("src.api.routes.DatabaseConfig")
    def test_health_endpoint_degraded_when_db_down(self, mock_db_config: MagicMock, mock_db_conn_cls: MagicMock) -> None:
        """Test GET /api/health reports degraded if DB is down but model is loaded."""
        mock_cfg = MagicMock()
        mock_cfg.database = "nl2sql_student_db"
        mock_cfg.host = "localhost"
        mock_cfg.port = 3306
        mock_cfg.user = "nl2sql_exec"
        mock_db_config.from_exec_env.return_value = mock_cfg

        mock_conn = MagicMock()
        mock_conn.verify_connection.return_value = {
            "status": "failure",
            "connected": False,
            "error": "Could not connect to MySQL server at localhost:3306",
        }
        mock_db_conn_cls.return_value = mock_conn

        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "degraded")
        self.assertTrue(data["model"]["loaded"])
        self.assertFalse(data["database"]["connected"])
        self.assertIn("Could not connect", data["database"]["error"])

    # ==========================================================================
    # 2. Schema Endpoint Tests
    # ==========================================================================

    @patch("src.api.routes.DatabaseSchemaInspector")
    def test_schema_endpoint_success(self, mock_inspector_cls: MagicMock) -> None:
        """Test GET /api/schema returns structured metadata for tables and columns."""
        mock_inspector = MagicMock()
        mock_metadata = DatabaseSchemaMetadata(
            database_name="nl2sql_student_db",
            tables={
                "students": TableMetadata(
                    name="students",
                    columns=[
                        ColumnMetadata(
                            name="student_id",
                            data_type="int",
                            column_type="int",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="name",
                            data_type="varchar",
                            column_type="varchar(100)",
                            is_nullable=False,
                            is_primary_key=False,
                        ),
                    ],
                    primary_keys=["student_id"],
                    foreign_keys=[],
                )
            },
        )
        mock_inspector.inspect_database.return_value = mock_metadata
        mock_inspector_cls.return_value = mock_inspector

        response = self.client.get("/api/schema")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["database_name"], "nl2sql_student_db")
        self.assertEqual(data["table_count"], 1)
        self.assertIn("students", data["tables"])
        students_tbl = data["tables"]["students"]
        self.assertEqual(students_tbl["primary_keys"], ["student_id"])
        self.assertEqual(len(students_tbl["columns"]), 2)
        self.assertEqual(students_tbl["columns"][0]["name"], "student_id")
        self.assertTrue(students_tbl["columns"][0]["is_primary_key"])

    @patch("src.api.routes.DatabaseSchemaInspector")
    def test_schema_endpoint_handles_failure(self, mock_inspector_cls: MagicMock) -> None:
        """Test GET /api/schema returns 503 if database schema inspection raises an error."""
        mock_inspector = MagicMock()
        mock_inspector.inspect_database.side_effect = RuntimeError("Access denied for table inspection")
        mock_inspector_cls.return_value = mock_inspector

        response = self.client.get("/api/schema")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Database schema inspection failed", response.json()["detail"])

    # ==========================================================================
    # 3. SQL Generation Endpoint Tests (/api/generate)
    # ==========================================================================

    def test_generate_sql_success(self) -> None:
        """Test POST /api/generate successfully generates and validates SQL."""
        payload = {"question": "Find all students with CGPA above 8.5"}
        response = self.client.post("/api/generate", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["question"], "Show students with CGPA above 8")
        self.assertEqual(data["generated_sql"], "SELECT name, cgpa FROM students WHERE cgpa > 8.0")
        self.assertEqual(data["stage_reached"], "execution_pending")
        self.assertTrue(data["can_execute"])
        self.assertIsNone(data["error"])

        # Check validation breakdown
        val = data["validation"]
        self.assertTrue(val["is_valid"])
        self.assertTrue(val["is_syntax_valid"])
        self.assertTrue(val["is_permitted_read_only"])
        self.assertTrue(val["is_schema_compatible"])
        self.assertTrue(val["is_in_model_scope"])
        self.assertEqual(val["ast_details"]["statement_type"], "SELECT")

    def test_generate_sql_empty_question_rejected(self) -> None:
        """Test POST /api/generate rejects empty or whitespace-only questions."""
        response = self.client.post("/api/generate", json={"question": ""})
        self.assertEqual(response.status_code, 422)

        response_whitespace = self.client.post("/api/generate", json={"question": "   "})
        self.assertEqual(response_whitespace.status_code, 422)

    def test_generate_sql_missing_question_field(self) -> None:
        """Test POST /api/generate rejects missing 'question' key."""
        response = self.client.post("/api/generate", json={})
        self.assertEqual(response.status_code, 422)

    def test_generate_sql_when_generated_sql_is_invalid(self) -> None:
        """Test POST /api/generate handles model generating an invalid SQL string."""
        self.mock_pipeline.generate_sql.return_value = InferenceResult(
            sql="SELECT FROM WHERE",
            tokens=["SELECT", "FROM", "WHERE"],
            source_tokens=["show"],
            source_unk_count=0,
            target_unk_count=0,
            is_valid_sql=False,
            syntax_error="Syntax error",
            generated_token_ids=[1, 2],
            question="Invalid question",
            schema=self.mock_pipeline.default_schema,
        )

        response = self.client.post("/api/generate", json={"question": "Invalid query"})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data["can_execute"])
        self.assertFalse(data["validation"]["is_valid"])
        self.assertEqual(data["stage_reached"], "validation")
        self.assertIsNotNone(data["error"])

    # ==========================================================================
    # 4. SQL Execution Endpoint Tests (/api/execute)
    # ==========================================================================

    def test_execute_sql_success(self) -> None:
        """Test POST /api/execute successfully re-validates and executes query."""
        payload = {
            "sql": "SELECT name, cgpa FROM students WHERE cgpa > 8.0",
            "approved": True,
            "max_rows": 50,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["row_count"], 2)
        self.assertEqual(len(data["rows"]), 2)
        self.assertEqual(data["columns"], ["name", "cgpa"])
        self.assertEqual(data["executed_sql"], "SELECT name, cgpa FROM students WHERE cgpa > 8.0")
        self.assertFalse(data["is_truncated"])
        self.assertIsNone(data["error"])

    def test_execute_sql_unapproved_rejected(self) -> None:
        """Test POST /api/execute rejects execution when approved is False."""
        payload = {
            "sql": "SELECT * FROM students",
            "approved": False,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Execution approval required", response.json()["detail"])

    def test_execute_sql_rejects_non_select_drop(self) -> None:
        """Test POST /api/execute server-side revalidation rejects DROP TABLE."""
        payload = {
            "sql": "DROP TABLE students",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Execution rejected", response.json()["detail"])

    def test_execute_sql_rejects_non_select_insert(self) -> None:
        """Test POST /api/execute server-side revalidation rejects INSERT INTO."""
        payload = {
            "sql": "INSERT INTO students (name, department) VALUES ('Hacker', 'CSE')",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Execution rejected", response.json()["detail"])

    def test_execute_sql_rejects_non_select_update(self) -> None:
        """Test POST /api/execute server-side revalidation rejects UPDATE."""
        payload = {
            "sql": "UPDATE students SET cgpa = 10.0 WHERE student_id = 1",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Execution rejected", response.json()["detail"])

    def test_execute_sql_rejects_non_select_delete(self) -> None:
        """Test POST /api/execute server-side revalidation rejects DELETE."""
        payload = {
            "sql": "DELETE FROM students WHERE student_id = 1",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Execution rejected", response.json()["detail"])

    def test_execute_sql_rejects_multi_statement(self) -> None:
        """Test POST /api/execute rejects multi-statement SQL injection attempts."""
        payload = {
            "sql": "SELECT * FROM students; DROP TABLE students;",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Multiple SQL statements detected", response.json()["detail"])

    def test_execute_sql_rejects_out_of_model_scope(self) -> None:
        """Test POST /api/execute rejects out-of-model-scope features like JOIN and LIKE."""
        # 1. Multi-table JOIN (outside general supported scope)
        payload_join = {
            "sql": "SELECT * FROM students s JOIN enrollments e ON s.student_id = e.student_id",
            "approved": True,
        }
        response_join = self.client.post("/api/execute", json=payload_join)
        self.assertEqual(response_join.status_code, 400)
        self.assertIn("JOIN / Multi-table query", response_join.json()["detail"])

        # 2. LIKE operator (outside trained model scope)
        payload_like = {
            "sql": "SELECT * FROM students WHERE name LIKE '%A%'",
            "approved": True,
        }
        response_like = self.client.post("/api/execute", json=payload_like)
        self.assertEqual(response_like.status_code, 400)
        self.assertIn("outside the trained Seq2Seq model vocabulary", response_like.json()["detail"])

    def test_execute_sql_rejects_incompatible_schema_table(self) -> None:
        """Test POST /api/execute rejects references to non-existent tables."""
        payload = {
            "sql": "SELECT * FROM non_existent_table",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("does not exist in the database schema", response.json()["detail"])

    def test_execute_sql_handles_database_execution_error(self) -> None:
        """Test POST /api/execute returns success=False when database execution encounters a runtime error."""
        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=False,
            executed_sql="SELECT name FROM students WHERE cgpa > 8.0",
            error="Query execution timed out after 5s.",
        )

        payload = {
            "sql": "SELECT name FROM students WHERE cgpa > 8.0",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("timed out", data["error"])

    def test_execute_sql_handles_empty_result_set(self) -> None:
        """Test POST /api/execute properly handles queries that match 0 rows."""
        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=True,
            executed_sql="SELECT name FROM students WHERE cgpa > 9.99",
            columns=["name"],
            rows=[],
            row_count=0,
            execution_time_ms=1.12,
            is_truncated=False,
        )

        payload = {
            "sql": "SELECT name FROM students WHERE cgpa > 9.99",
            "approved": True,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["row_count"], 0)
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["columns"], ["name"])

    def test_execute_sql_handles_truncation(self) -> None:
        """Test POST /api/execute returns truncation warning when rows exceed limit."""
        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=True,
            executed_sql="SELECT * FROM students",
            columns=["student_id", "name"],
            rows=[{"student_id": 1, "name": "Aarav"}],
            row_count=1,
            execution_time_ms=2.01,
            is_truncated=True,
            warning="Query returned more than 1 rows. Results truncated to 1.",
        )

        payload = {
            "sql": "SELECT * FROM students",
            "approved": True,
            "max_rows": 1,
        }
        response = self.client.post("/api/execute", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["is_truncated"])
        self.assertIn("truncated", data["warning"])

    # ==========================================================================
    # 5. Lifecycle and Uninitialized Pipeline Tests
    # ==========================================================================

    def test_uninitialized_pipeline_endpoints(self) -> None:
        """Test endpoints return 503 if the pipeline is uninitialized in app.state."""
        app_empty = create_app(pipeline=None, checkpoint_path=Path("non_existent_checkpoint.pt"))
        client_empty = TestClient(app_empty)

        # Health reports unhealthy
        resp_health = client_empty.get("/api/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertFalse(resp_health.json()["model"]["loaded"])
        self.assertEqual(resp_health.json()["status"], "unhealthy")

        # Generate returns 503
        resp_gen = client_empty.post("/api/generate", json={"question": "Test question"})
        self.assertEqual(resp_gen.status_code, 503)

        # Execute returns 503
        resp_exec = client_empty.post("/api/execute", json={"sql": "SELECT * FROM students", "approved": True})
        self.assertEqual(resp_exec.status_code, 503)

    # ==========================================================================
    # 6. CORS Configuration Tests
    # ==========================================================================

    def test_cors_origins_resolution(self) -> None:
        """Test resolution of CORS origins from argument and environment."""
        resolved = resolve_cors_origins("http://example.com, https://app.example.com")
        self.assertEqual(resolved, ["http://example.com", "https://app.example.com"])

        resolved_default = resolve_cors_origins(None)
        self.assertIn("http://localhost:5173", resolved_default)

    def test_cors_preflight_headers(self) -> None:
        """Test CORS OPTIONS preflight returns configured origin headers."""
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = self.client.options("/api/generate", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")


if __name__ == "__main__":
    unittest.main()
