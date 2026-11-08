"""
Unit and Integration Test Suite for MySQL Connectivity & Schema Inspection (Milestone 4, Task 4.2).

Tests:
- Test A: Successful connection
- Test B: Database selection
- Test C: Schema retrieval (tables, columns, types)
- Test D: Metadata inspection (primary keys and foreign keys)
- Test E: Failure scenarios (bad host, bad port, bad user/password, bad database, unavailable server)
- Test F: Resource cleanup (connections & cursors properly closed)
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymysql

from src.database.config import DatabaseConfig, parse_dotenv_file
from src.database.connection import (
    DatabaseAuthenticationError,
    DatabaseConfigurationError,
    DatabaseConnection,
    DatabaseConnectionError,
    DatabaseNotFoundError,
    sanitize_db_error,
)
from src.database.schema import (
    ColumnMetadata,
    DatabaseSchemaInspector,
    DatabaseSchemaMetadata,
    ForeignKeyMetadata,
    TableMetadata,
)


class TestDatabaseConfig(unittest.TestCase):
    """Verifies configuration loading, validation, and credential masking."""

    def test_default_config(self) -> None:
        config = DatabaseConfig()
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 3306)
        self.assertEqual(config.user, "root")
        self.assertEqual(config.database, "nl2sql_student_db")
        self.assertEqual(config.charset, "utf8mb4")

    def test_port_validation(self) -> None:
        with self.assertRaises(ValueError):
            DatabaseConfig(port=0)

        with self.assertRaises(ValueError):
            DatabaseConfig(port=70000)

        with self.assertRaises(ValueError):
            DatabaseConfig(host="")

        with self.assertRaises(ValueError):
            DatabaseConfig(database="")

    def test_credential_masking(self) -> None:
        config = DatabaseConfig(password="SuperSecretPassword123!")
        repr_str = repr(config)
        self.assertNotIn("SuperSecretPassword123!", repr_str)
        self.assertIn("********", repr_str)

        d_masked = config.to_dict(mask_password=True)
        self.assertEqual(d_masked["password"], "********")

        d_unmasked = config.to_dict(mask_password=False)
        self.assertEqual(d_unmasked["password"], "SuperSecretPassword123!")

    def test_dotenv_parser(self) -> None:
        temp_env = PROJECT_ROOT / "temp_test.env"
        temp_env.write_text(
            "# Comment line\n"
            "MYSQL_HOST=db.example.com\n"
            "MYSQL_PORT=3307\n"
            "MYSQL_USER='test_user'\n"
            'MYSQL_PASSWORD="secret_pwd"\n'
            "MYSQL_DATABASE=test_db\n",
            encoding="utf-8",
        )
        try:
            parsed = parse_dotenv_file(temp_env)
            self.assertEqual(parsed["MYSQL_HOST"], "db.example.com")
            self.assertEqual(parsed["MYSQL_PORT"], "3307")
            self.assertEqual(parsed["MYSQL_USER"], "test_user")
            self.assertEqual(parsed["MYSQL_PASSWORD"], "secret_pwd")
            self.assertEqual(parsed["MYSQL_DATABASE"], "test_db")

            cfg = DatabaseConfig.from_env(env_file=temp_env)
            self.assertEqual(cfg.host, "db.example.com")
            self.assertEqual(cfg.port, 3307)
            self.assertEqual(cfg.user, "test_user")
            self.assertEqual(cfg.password, "secret_pwd")
            self.assertEqual(cfg.database, "test_db")
        finally:
            if temp_env.exists():
                temp_env.unlink()


class TestDatabaseConnectionUnit(unittest.TestCase):
    """Unit tests for connection logic, error mapping, and resource cleanup."""

    @patch("pymysql.connect")
    def test_connection_success_and_cleanup(self, mock_connect: MagicMock) -> None:
        """Test F: Resource cleanup via context manager."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        config = DatabaseConfig(password="dummy")
        conn_mgr = DatabaseConnection(config=config)

        with conn_mgr as cm:
            self.assertTrue(conn_mgr.is_connected())

        # Confirm connection was closed upon exit
        mock_conn.close.assert_called_once()
        self.assertFalse(conn_mgr.is_connected())

    @patch("pymysql.connect")
    def test_auth_failure_error_mapping(self, mock_connect: MagicMock) -> None:
        """Test E: Authentication error mapping (error 1045)."""
        secret_pwd = "MySecretPassword123"
        mock_connect.side_effect = pymysql.err.OperationalError(
            1045, f"Access denied for user 'root'@'localhost' (using password: {secret_pwd})"
        )

        config = DatabaseConfig(password=secret_pwd)
        conn_mgr = DatabaseConnection(config=config)

        with self.assertRaises(DatabaseAuthenticationError) as ctx:
            conn_mgr.connect()

        # Confirm password is NOT leaked in exception message
        self.assertNotIn(secret_pwd, str(ctx.exception))
        self.assertIn("Authentication failed", str(ctx.exception))

    @patch("pymysql.connect")
    def test_bad_db_error_mapping(self, mock_connect: MagicMock) -> None:
        """Test E: Nonexistent database error mapping (error 1049)."""
        mock_connect.side_effect = pymysql.err.OperationalError(
            1049, "Unknown database 'missing_db'"
        )

        config = DatabaseConfig(database="missing_db")
        conn_mgr = DatabaseConnection(config=config)

        with self.assertRaises(DatabaseNotFoundError) as ctx:
            conn_mgr.connect()

        self.assertIn("missing_db", str(ctx.exception))

    @patch("pymysql.connect")
    def test_host_unreachable_error_mapping(self, mock_connect: MagicMock) -> None:
        """Test E: Host unreachable / server unavailable (error 2003)."""
        mock_connect.side_effect = pymysql.err.OperationalError(
            2003, "Can't connect to MySQL server on 'bad_host' (10061)"
        )

        config = DatabaseConfig(host="bad_host")
        conn_mgr = DatabaseConnection(config=config)

        with self.assertRaises(DatabaseConnectionError) as ctx:
            conn_mgr.connect()

        self.assertIn("Could not connect to MySQL server", str(ctx.exception))

    @patch("pymysql.connect")
    def test_verify_connection_alive(self, mock_connect: MagicMock) -> None:
        """Test A & B: Harmless verification query execution and output."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = {
            "alive": 1,
            "current_db": "nl2sql_student_db",
            "version": "8.0.43",
        }
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        conn_mgr = DatabaseConnection()
        status = conn_mgr.verify_connection()

        self.assertEqual(status["status"], "success")
        self.assertTrue(status["alive"])
        self.assertEqual(status["database"], "nl2sql_student_db")
        self.assertEqual(status["mysql_version"], "8.0.43")
        self.assertIsNone(status["error"])


class TestDatabaseSchemaInspectorUnit(unittest.TestCase):
    """Unit tests for schema metadata extraction and signature formatting."""

    def test_metadata_dataclasses_and_formatting(self) -> None:
        """Test C & D: Metadata structuring and signature formatting."""
        cols = [
            ColumnMetadata(name="student_id", data_type="int", column_type="int", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
            ColumnMetadata(name="age", data_type="int", column_type="int", is_nullable=True),
            ColumnMetadata(name="department", data_type="varchar", column_type="varchar(50)", is_nullable=False),
            ColumnMetadata(name="cgpa", data_type="decimal", column_type="decimal(4,2)", is_nullable=True),
        ]
        t_meta = TableMetadata(
            name="students",
            columns=cols,
            primary_keys=["student_id"],
            foreign_keys=[],
        )
        self.assertEqual(
            t_meta.format_signature(),
            "students(student_id, name, age, department, cgpa)",
        )
        self.assertEqual(t_meta.primary_keys, ["student_id"])

        # Course table
        c_cols = [
            ColumnMetadata(name="course_id", data_type="int", column_type="int", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="course_name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
            ColumnMetadata(name="credits", data_type="int", column_type="int", is_nullable=False),
        ]
        c_meta = TableMetadata(
            name="courses",
            columns=c_cols,
            primary_keys=["course_id"],
        )

        # Enrollments with FKs
        e_cols = [
            ColumnMetadata(name="enrollment_id", data_type="int", column_type="int", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="student_id", data_type="int", column_type="int", is_nullable=False),
            ColumnMetadata(name="course_id", data_type="int", column_type="int", is_nullable=False),
            ColumnMetadata(name="semester", data_type="varchar", column_type="varchar(20)", is_nullable=False),
            ColumnMetadata(name="marks", data_type="decimal", column_type="decimal(5,2)", is_nullable=True),
        ]
        e_fks = [
            ForeignKeyMetadata(
                constraint_name="fk_enrollments_student",
                column_name="student_id",
                referenced_table="students",
                referenced_column="student_id",
                update_rule="CASCADE",
                delete_rule="CASCADE",
            ),
            ForeignKeyMetadata(
                constraint_name="fk_enrollments_course",
                column_name="course_id",
                referenced_table="courses",
                referenced_column="course_id",
                update_rule="CASCADE",
                delete_rule="CASCADE",
            ),
        ]
        e_meta = TableMetadata(
            name="enrollments",
            columns=e_cols,
            primary_keys=["enrollment_id"],
            foreign_keys=e_fks,
        )

        db_meta = DatabaseSchemaMetadata(
            database_name="nl2sql_student_db",
            tables={"students": t_meta, "courses": c_meta, "enrollments": e_meta},
        )

        self.assertEqual(db_meta.total_tables, 3)
        self.assertEqual(db_meta.total_foreign_keys, 2)
        expected_summary = (
            "courses(course_id, course_name, credits) | "
            "enrollments(enrollment_id, student_id, course_id, semester, marks) | "
            "students(student_id, name, age, department, cgpa)"
        )
        self.assertEqual(db_meta.format_schema_summary(), expected_summary)

    @patch.object(DatabaseConnection, "execute_read_query")
    def test_inspector_queries(self, mock_query: MagicMock) -> None:
        """Test inspector method querying INFORMATION_SCHEMA."""
        inspector = DatabaseSchemaInspector(connection=DatabaseConnection())

        # Mock responses
        mock_query.side_effect = [
            # Call 1: get_table_names()
            [{"TABLE_NAME": "courses"}, {"TABLE_NAME": "enrollments"}, {"TABLE_NAME": "students"}],
            # Call 2: get_primary_keys("students")
            [{"COLUMN_NAME": "student_id"}],
            # Call 3: get_primary_keys inside get_columns("students")
            [{"COLUMN_NAME": "student_id"}],
            # Call 4: get_columns query
            [{"COLUMN_NAME": "student_id", "DATA_TYPE": "int", "COLUMN_TYPE": "int", "IS_NULLABLE": "NO", "COLUMN_DEFAULT": None, "EXTRA": "auto_increment"}],
        ]

        tables = inspector.get_table_names()
        self.assertEqual(tables, ["courses", "enrollments", "students"])

        pks = inspector.get_primary_keys("students")
        self.assertEqual(pks, ["student_id"])

        cols = inspector.get_columns("students")
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0].name, "student_id")
        self.assertTrue(cols[0].is_primary_key)


class TestLiveFailureScenarios(unittest.TestCase):
    """
    Test E: Actual live tests for failure handling against invalid destinations.
    These execute against real network sockets to verify proper error trapping and masking.
    """

    def test_e1_incorrect_hostname(self) -> None:
        """Incorrect hostname must raise DatabaseConnectionError without credential leaks."""
        cfg = DatabaseConfig(host="invalid-mysql-host.invalid", connect_timeout=1, password="SecretPasswordXYZ")
        conn = DatabaseConnection(config=cfg)
        with self.assertRaises(DatabaseConnectionError) as ctx:
            conn.connect()
        self.assertNotIn("SecretPasswordXYZ", str(ctx.exception))

    def test_e2_incorrect_port(self) -> None:
        """Unused port must raise DatabaseConnectionError."""
        # 59999 is typically unused
        cfg = DatabaseConfig(host="127.0.0.1", port=59999, connect_timeout=1, password="SecretPasswordXYZ")
        conn = DatabaseConnection(config=cfg)
        with self.assertRaises(DatabaseConnectionError) as ctx:
            conn.connect()
        self.assertNotIn("SecretPasswordXYZ", str(ctx.exception))

    def test_e3_incorrect_credentials(self) -> None:
        """Incorrect credentials against real local MySQL must raise DatabaseAuthenticationError."""
        # The local MySQL server is running on 3306. Supplying deliberately invalid credentials
        cfg = DatabaseConfig(
            host="127.0.0.1",
            port=3306,
            user="definitely_nonexistent_user_abc123",
            password="wrong_password_999",
            connect_timeout=2,
        )
        conn = DatabaseConnection(config=cfg)
        with self.assertRaises(DatabaseAuthenticationError) as ctx:
            conn.connect()
        self.assertNotIn("wrong_password_999", str(ctx.exception))
        self.assertIn("Authentication failed", str(ctx.exception))

    def test_e4_resource_cleanup_on_failure(self) -> None:
        """Test F: Connection remains closed when an exception occurs."""
        cfg = DatabaseConfig(host="invalid-mysql-host.invalid", connect_timeout=1)
        conn = DatabaseConnection(config=cfg)
        try:
            with conn:
                pass
        except DatabaseConnectionError:
            pass
        self.assertFalse(conn.is_connected())


def run_tests() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabaseConfig)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseConnectionUnit))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseSchemaInspectorUnit))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLiveFailureScenarios))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
