"""Consolidated unit and integration tests for the NL2SQL FastAPI backend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.app import DEFAULT_CHECKPOINT_PATH, create_app, resolve_cors_origins
from src.database.executor import ExecutionResult
from src.database.schema import ColumnMetadata, DatabaseSchemaMetadata, TableMetadata
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA, InferenceResult
from src.pipeline.orchestrator import NL2SQLPipeline
from src.validation.validator import SQLValidator


def build_mock_pipeline() -> MagicMock:
    mock_pipeline = MagicMock(spec=NL2SQLPipeline)
    mock_pipeline.default_schema = DEFAULT_STUDENT_SCHEMA
    real_validator = SQLValidator(schema=DEFAULT_STUDENT_SCHEMA, enforce_model_scope=True)
    mock_pipeline.validator = real_validator
    mock_pipeline.validate_sql.side_effect = real_validator.validate

    mock_src_vocab = MagicMock()
    mock_src_vocab.__len__.return_value = 290
    mock_tgt_vocab = MagicMock()
    mock_tgt_vocab.__len__.return_value = 96

    mock_inf_pipe = MagicMock()
    mock_inf_pipe.src_vocab = mock_src_vocab
    mock_inf_pipe.tgt_vocab = mock_tgt_vocab
    mock_inf_pipe.device = "cpu"
    mock_inf_pipe.checkpoint_info = {
        "checkpoint_path": str((PROJECT_ROOT / "checkpoints" / "candidates" / "phase2_clean_best.pt").resolve())
    }
    mock_pipeline.inference_pipeline = mock_inf_pipe

    mock_pipeline.generate_sql.return_value = InferenceResult(
        sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0",
        tokens=["SELECT", "name", ",", "cgpa", "FROM", "students", "WHERE", "cgpa", ">", "8.0"],
        source_tokens=["students", "<SEP>", "show", "students"],
        source_unk_count=0,
        target_unk_count=0,
        is_valid_sql=True,
        syntax_error=None,
        generated_token_ids=[5, 10, 15],
        question="Show students with CGPA above 8",
        schema=DEFAULT_STUDENT_SCHEMA,
    )
    mock_pipeline.execute_sql.return_value = ExecutionResult(
        success=True,
        executed_sql="SELECT name, cgpa FROM students WHERE cgpa > 8.0",
        columns=["name", "cgpa"],
        rows=[{"name": "Aarav Sharma", "cgpa": 8.92}, {"name": "Priya Patel", "cgpa": 9.15}],
        row_count=2,
        execution_time_ms=2.45,
        is_truncated=False,
    )
    return mock_pipeline


class TestNL2SQLAPI(unittest.TestCase):
    """Covers /api/health, /api/schema, /api/generate, /api/execute, lifecycle, and CORS."""

    def setUp(self) -> None:
        self.mock_pipeline = build_mock_pipeline()
        self.app = create_app(pipeline=self.mock_pipeline)
        self.client = TestClient(self.app)

    def test_default_checkpoint_points_to_official_phase2_model(self) -> None:
        self.assertEqual(
            DEFAULT_CHECKPOINT_PATH.resolve(),
            (PROJECT_ROOT / "checkpoints" / "candidates" / "phase2_clean_best.pt").resolve(),
        )
        self.assertTrue(DEFAULT_CHECKPOINT_PATH.is_file())

    @patch("src.api.routes.DatabaseConnection")
    @patch("src.api.routes.DatabaseConfig")
    def test_health_endpoint_healthy_and_degraded(self, mock_db_config: MagicMock, mock_db_conn_cls: MagicMock) -> None:
        mock_cfg = MagicMock(database="nl2sql_student_db", host="localhost", port=3306, user="nl2sql_exec")
        mock_db_config.from_exec_env.return_value = mock_cfg

        mock_conn = MagicMock()
        mock_conn.verify_connection.return_value = {"status": "success", "connected": True, "mysql_version": "8.0.43"}
        mock_db_conn_cls.return_value = mock_conn

        resp_ok = self.client.get("/api/health")
        self.assertEqual(resp_ok.status_code, 200)
        data_ok = resp_ok.json()
        self.assertEqual(data_ok["status"], "healthy")
        self.assertEqual(data_ok["model"]["checkpoint"], "checkpoints/candidates/phase2_clean_best.pt")
        self.assertEqual((data_ok["model"]["source_vocab_size"], data_ok["model"]["target_vocab_size"]), (290, 96))
        self.assertNotIn("password", resp_ok.text.lower())

        mock_conn.verify_connection.return_value = {"status": "failure", "connected": False, "error": "Connection refused"}
        resp_deg = self.client.get("/api/health")
        self.assertEqual(resp_deg.status_code, 200)
        self.assertEqual(resp_deg.json()["status"], "degraded")
        self.assertFalse(resp_deg.json()["database"]["connected"])

    @patch("src.api.routes.DatabaseSchemaInspector")
    def test_schema_endpoint_success_and_failure(self, mock_inspector_cls: MagicMock) -> None:
        mock_inspector = MagicMock()
        mock_inspector.inspect_database.return_value = DatabaseSchemaMetadata(
            database_name="nl2sql_student_db",
            tables={
                "students": TableMetadata(
                    name="students",
                    columns=[ColumnMetadata("student_id", "int", "int", False, is_primary_key=True)],
                    primary_keys=["student_id"],
                )
            },
        )
        mock_inspector_cls.return_value = mock_inspector

        resp_ok = self.client.get("/api/schema")
        self.assertEqual(resp_ok.status_code, 200)
        self.assertIn("students", resp_ok.json()["tables"])

        mock_inspector.inspect_database.side_effect = RuntimeError("Access denied")
        resp_err = self.client.get("/api/schema")
        self.assertEqual(resp_err.status_code, 503)

    def test_generate_endpoint_valid_invalid_sql_and_request_errors(self) -> None:
        resp_ok = self.client.post("/api/generate", json={"question": "Find students with CGPA > 8"})
        self.assertEqual(resp_ok.status_code, 200)
        data = resp_ok.json()
        self.assertTrue(data["can_execute"])
        self.assertEqual(data["stage_reached"], "execution_pending")
        self.assertTrue(data["validation"]["is_valid"])

        for bad_payload in ({"question": ""}, {"question": "   "}, {}):
            self.assertEqual(self.client.post("/api/generate", json=bad_payload).status_code, 422)

        self.mock_pipeline.generate_sql.return_value = InferenceResult(
            sql="SELECT FROM WHERE", tokens=["SELECT"], source_tokens=["show"], source_unk_count=0,
            target_unk_count=0, is_valid_sql=False, syntax_error="Syntax error",
            generated_token_ids=[1], question="bad", schema=DEFAULT_STUDENT_SCHEMA,
        )
        resp_bad_sql = self.client.post("/api/generate", json={"question": "bad"})
        self.assertEqual(resp_bad_sql.status_code, 200)
        self.assertFalse(resp_bad_sql.json()["can_execute"])
        self.assertEqual(resp_bad_sql.json()["stage_reached"], "validation")

    def test_execute_endpoint_success_empty_truncation_and_runtime_error(self) -> None:
        resp_ok = self.client.post("/api/execute", json={"sql": "SELECT name, cgpa FROM students WHERE cgpa > 8.0", "approved": True, "max_rows": 50})
        self.assertEqual(resp_ok.status_code, 200)
        self.assertEqual((resp_ok.json()["success"], resp_ok.json()["row_count"]), (True, 2))

        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=True, executed_sql="SELECT name FROM students", columns=["name"], rows=[], row_count=0, execution_time_ms=1.1, is_truncated=False
        )
        resp_empty = self.client.post("/api/execute", json={"sql": "SELECT name FROM students", "approved": True})
        self.assertEqual((resp_empty.status_code, resp_empty.json()["row_count"]), (200, 0))

        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=True, executed_sql="SELECT * FROM students", columns=["name"], rows=[{"name": "Aarav"}],
            row_count=1, execution_time_ms=1.5, is_truncated=True, warning="Results truncated to 1.",
        )
        resp_trunc = self.client.post("/api/execute", json={"sql": "SELECT * FROM students", "approved": True, "max_rows": 1})
        self.assertTrue(resp_trunc.json()["is_truncated"])

        self.mock_pipeline.execute_sql.return_value = ExecutionResult(
            success=False, executed_sql="SELECT name FROM students", error="Query execution timed out after 5s."
        )
        resp_err = self.client.post("/api/execute", json={"sql": "SELECT name FROM students", "approved": True})
        self.assertEqual((resp_err.status_code, resp_err.json()["success"]), (200, False))

    def test_execute_endpoint_rejects_unapproved_and_unsafe_sql(self) -> None:
        rejected_cases = [
            ({"sql": "SELECT * FROM students", "approved": False}, "Execution approval required"),
            ({"sql": "DROP TABLE students", "approved": True}, "Execution rejected"),
            ({"sql": "INSERT INTO students (name, department) VALUES ('X', 'CSE')", "approved": True}, "Execution rejected"),
            ({"sql": "UPDATE students SET cgpa = 10.0 WHERE student_id = 1", "approved": True}, "Execution rejected"),
            ({"sql": "DELETE FROM students WHERE student_id = 1", "approved": True}, "Execution rejected"),
            ({"sql": "SELECT * FROM students; DROP TABLE students;", "approved": True}, "Multiple SQL statements detected"),
            ({"sql": "SELECT * FROM students s JOIN enrollments e ON s.student_id = e.student_id", "approved": True}, "JOIN"),
            ({"sql": "SELECT * FROM students WHERE name LIKE '%A%'", "approved": True}, "model vocabulary"),
            ({"sql": "SELECT * FROM non_existent_table", "approved": True}, "does not exist"),
        ]
        for payload, expected_detail in rejected_cases:
            with self.subTest(payload=payload):
                resp = self.client.post("/api/execute", json=payload)
                self.assertEqual(resp.status_code, 400)
                self.assertIn(expected_detail, resp.json()["detail"])

    def test_uninitialized_pipeline_and_cors_headers(self) -> None:
        empty_client = TestClient(create_app(pipeline=None, checkpoint_path=Path("missing_ckpt.pt")))
        h = empty_client.get("/api/health").json()
        self.assertEqual((h["status"], h["model"]["loaded"]), ("unhealthy", False))
        self.assertEqual(empty_client.post("/api/generate", json={"question": "List students"}).status_code, 503)
        self.assertEqual(empty_client.post("/api/execute", json={"sql": "SELECT * FROM students", "approved": True}).status_code, 503)

        self.assertEqual(resolve_cors_origins("http://a.com, https://b.com"), ["http://a.com", "https://b.com"])
        self.assertIn("http://localhost:5173", resolve_cors_origins(None))

        preflight = self.client.options(
            "/api/generate",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Content-Type"},
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers.get("access-control-allow-origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
