"""Consolidated tests for SQLValidator, read-only safety, scope enforcement, and schema checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.schema import ColumnMetadata, DatabaseSchemaMetadata, TableMetadata
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator, validate_sql

TEST_SCHEMA_DICT = {
    "students": ["student_id", "name", "age", "department", "cgpa"],
    "courses": ["course_id", "course_name", "credits", "department"],
    "enrollments": ["enrollment_id", "student_id", "course_id", "semester", "marks"],
}

CANONICAL_SCHEMA_STRING = (
    "students(student_id, name, age, department, cgpa) | "
    "courses(course_id, course_name, credits, department) | "
    "enrollments(enrollment_id, student_id, course_id, semester, marks)"
)


def create_mock_schema_metadata() -> DatabaseSchemaMetadata:
    return DatabaseSchemaMetadata(
        database_name="nl2sql_student_db",
        tables={
            "students": TableMetadata(
                name="students",
                columns=[
                    ColumnMetadata(name="student_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
                    ColumnMetadata(name="name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
                    ColumnMetadata(name="age", data_type="int", column_type="int(11)", is_nullable=False),
                    ColumnMetadata(name="department", data_type="varchar", column_type="varchar(50)", is_nullable=False),
                    ColumnMetadata(name="cgpa", data_type="decimal", column_type="decimal(3,2)", is_nullable=True),
                ],
                primary_keys=["student_id"],
            ),
            "courses": TableMetadata(
                name="courses",
                columns=[
                    ColumnMetadata(name="course_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
                    ColumnMetadata(name="course_name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
                    ColumnMetadata(name="credits", data_type="int", column_type="int(11)", is_nullable=False),
                    ColumnMetadata(name="department", data_type="varchar", column_type="varchar(50)", is_nullable=False),
                ],
                primary_keys=["course_id"],
            ),
            "enrollments": TableMetadata(
                name="enrollments",
                columns=[
                    ColumnMetadata(name="enrollment_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
                    ColumnMetadata(name="student_id", data_type="int", column_type="int(11)", is_nullable=False),
                    ColumnMetadata(name="course_id", data_type="int", column_type="int(11)", is_nullable=False),
                    ColumnMetadata(name="semester", data_type="varchar", column_type="varchar(20)", is_nullable=False),
                    ColumnMetadata(name="marks", data_type="decimal", column_type="decimal(5,2)", is_nullable=True),
                ],
                primary_keys=["enrollment_id"],
            ),
        },
    )


class TestSQLValidation(unittest.TestCase):
    """Covers read-only safety, AST syntax, model-scope boundaries, and schema checks."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_valid_in_scope_queries_and_normalization(self) -> None:
        valid_queries = [
            "SELECT * FROM students",
            "SELECT student_id, name, cgpa FROM students",
            "SELECT name FROM students WHERE age > 20",
            "SELECT name FROM students WHERE cgpa >= 8.5",
            "SELECT name FROM students WHERE age < 25 AND cgpa <= 9.0",
            "SELECT name FROM students WHERE department != 'ME' AND department <> 'ECE'",
            "SELECT name, department FROM students WHERE (department = 'CSE' OR department = 'AI&DS') AND cgpa > 8.0",
            "SELECT name, cgpa FROM students ORDER BY cgpa ASC",
            "SELECT name, age FROM students ORDER BY age DESC LIMIT 5",
            "SELECT students.name, students.age FROM students WHERE students.cgpa > 7.5",
            "sElEcT NaMe FROM sTuDeNtS;",
            "SELECT name FROM students -- fetch names",
            "SELECT /* col */ name FROM /* tbl */ students",
        ]
        for q in valid_queries:
            with self.subTest(query=q):
                res = validate_sql(q, schema=TEST_SCHEMA_DICT)
                self.assertTrue(res.is_valid, f"Unexpected error for '{q}': {res.errors}")
                self.assertTrue(res.is_syntax_valid)
                self.assertTrue(res.is_permitted_read_only)
                self.assertTrue(res.is_schema_compatible)
                self.assertTrue(res.is_in_supported_scope)
                self.assertTrue(res.is_in_model_scope)
                self.assertNotIn(";", res.normalized_sql or "")

    def test_forbidden_mutations_ddl_locks_into_and_multi_statements(self) -> None:
        forbidden = [
            ("INSERT INTO students (student_id, name) VALUES (101, 'Alice')", "Only single read-only SELECT"),
            ("UPDATE students SET cgpa = 9.5 WHERE student_id = 1", "Only single read-only SELECT"),
            ("DELETE FROM students WHERE student_id = 1", "Only single read-only SELECT"),
            ("DROP TABLE students", "Only single read-only SELECT"),
            ("ALTER TABLE students ADD COLUMN email VARCHAR(100)", "Only single read-only SELECT"),
            ("CREATE TABLE test (id INT)", "Only single read-only SELECT"),
            ("TRUNCATE TABLE students", "Only single read-only SELECT"),
            ("GRANT ALL PRIVILEGES ON *.* TO 'user'@'%'", "Only single read-only SELECT"),
            ("REVOKE SELECT ON students FROM 'user'@'%'", "Only single read-only SELECT"),
            ("SELECT * FROM students; DROP TABLE students;", "Multiple SQL statements detected"),
            ("SELECT 1; SELECT 2;", "Multiple SQL statements detected"),
            ("SELECT * FROM students INTO OUTFILE '/tmp/out.txt'", "INTO"),
            ("SELECT * INTO DUMPFILE '/tmp/dump.txt' FROM students", "INTO"),
            ("SELECT * INTO new_table FROM students", "INTO"),
            ("SELECT * FROM students FOR UPDATE", "Locking clauses"),
            ("SELECT name FROM students WHERE student_id = 1 LOCK IN SHARE MODE", "Locking clauses"),
            ("SELECT * FROM students FOR SHARE", "Locking clauses"),
        ]
        for query, expected_frag in forbidden:
            with self.subTest(query=query):
                res = self.validator.validate(query)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn(expected_frag, res.error_message)

    def test_security_functions_udfs_and_variables_rejected(self) -> None:
        dangerous = [
            ("SELECT SLEEP(5) FROM students", "prohibited or security-sensitive function"),
            ("SELECT BENCHMARK(1000, MD5('x')) FROM students", "prohibited or security-sensitive function"),
            ("SELECT LOAD_FILE('/etc/passwd') FROM students", "prohibited or security-sensitive function"),
            ("SELECT VERSION() FROM students", "prohibited or security-sensitive function"),
            ("SELECT connection_id() FROM students", "unapproved, stored, or user-defined function"),
            ("SELECT is_free_lock('audit_lock') FROM students", "unapproved, stored, or user-defined function"),
            ("SELECT custom_stored_proc() FROM students", "unapproved, stored, or user-defined function"),
            ("SELECT my_udf(student_id) FROM students", "unapproved, stored, or user-defined function"),
            ("SELECT @@version FROM students", "System variables (@@) and session parameters (@)"),
            ("SELECT @my_var FROM students", "System variables (@@) and session parameters (@)"),
        ]
        for query, expected_frag in dangerous:
            with self.subTest(query=query):
                res = self.validator.validate(query)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn(expected_frag, res.error_message)

        res_abs = self.validator.validate("SELECT ABS(age) FROM students", enforce_scope=False, enforce_model_scope=False)
        self.assertTrue(res_abs.is_valid)
        self.assertTrue(res_abs.is_permitted_read_only)

    def test_model_scope_enforcement_and_opt_out(self) -> None:
        # Strict default (enforce_model_scope=True) rejects constructs outside closed vocabulary
        out_of_scope_parser_supported = [
            ("SELECT name FROM students WHERE name LIKE 'A%'", "LIKE operator"),
            ("SELECT name FROM students WHERE department IN ('CSE', 'ECE')", "IN operator"),
            ("SELECT name FROM students WHERE age BETWEEN 18 AND 22", "BETWEEN operator"),
            ("SELECT name FROM students WHERE cgpa IS NULL", "IS NULL / IS NOT NULL"),
            ("SELECT name FROM students WHERE cgpa IS NOT NULL", "IS NULL / IS NOT NULL"),
            ("SELECT name FROM students WHERE NOT age > 20", "NOT operator"),
            ("SELECT s.name FROM students s", "Table Aliases"),
            ("SELECT name AS student_name FROM students", "Column AS Aliases"),
            ("SELECT ABS(age) FROM students", "Function Calls"),
        ]
        for query, expected_feature in out_of_scope_parser_supported:
            with self.subTest(query=query):
                res_default = validate_sql(query, schema=TEST_SCHEMA_DICT)
                self.assertFalse(res_default.is_valid)
                self.assertFalse(res_default.is_in_model_scope)
                self.assertIn(expected_feature, res_default.error_message)

                res_opt_out = validate_sql(query, schema=TEST_SCHEMA_DICT, enforce_model_scope=False)
                self.assertTrue(res_opt_out.is_valid, f"Opt-out failed for '{query}': {res_opt_out.errors}")
                self.assertFalse(res_opt_out.is_in_model_scope)
                self.assertTrue(len(res_opt_out.warnings) > 0)

        # Structural multi-table / aggregation / subquery constructs blocked by general supported scope too
        structural_oos = [
            "SELECT COUNT(*) FROM students",
            "SELECT department, COUNT(*) FROM students GROUP BY department",
            "SELECT department FROM students GROUP BY department HAVING COUNT(*) > 1",
            "SELECT s.name, e.marks FROM students s JOIN enrollments e ON s.student_id = e.student_id",
            "SELECT students.name, courses.course_name FROM students, courses",
            "SELECT name FROM (SELECT name, age FROM students) sub",
            "SELECT name FROM students UNION SELECT course_name FROM courses",
            "SELECT DISTINCT department FROM students",
        ]
        for query in structural_oos:
            with self.subTest(query=query):
                res = validate_sql(query, schema=TEST_SCHEMA_DICT)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_in_supported_scope)

    def test_schema_compatibility_and_metadata_inputs(self) -> None:
        bad_schema_queries = [
            ("SELECT name FROM professors", "Table 'professors' does not exist", {}),
            ("SELECT salary FROM students", "Column 'salary' does not exist", {}),
            ("SELECT students.credits FROM students", "Column 'credits' does not exist in table 'students'", {}),
            ("SELECT x.name FROM students s", "Table or alias 'x'", {"enforce_model_scope": False}),
            ("SELECT x.* FROM students s", "Table or alias 'x'", {"enforce_model_scope": False}),
            ("SELECT department FROM students, courses", "Ambiguous column reference", {"enforce_scope": False, "enforce_model_scope": False}),
        ]
        for query, expected_frag, kwargs in bad_schema_queries:
            with self.subTest(query=query):
                res = self.validator.validate(query, **kwargs)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_schema_compatible)
                self.assertIn(expected_frag, res.error_message)

        # Qualified column resolves multi-table ambiguity when scope checks disabled
        res_qual = self.validator.validate("SELECT students.department FROM students, courses", enforce_scope=False, enforce_model_scope=False)
        self.assertTrue(res_qual.is_valid)

        # Missing or empty schema fails closed
        self.assertFalse(SQLValidator(schema=None).validate("SELECT name FROM students").is_schema_compatible)
        self.assertFalse(self.validator.validate("SELECT name FROM students", schema={}).is_schema_compatible)

        # Canonical string and DatabaseSchemaMetadata object inputs work identically
        self.assertTrue(validate_sql("SELECT name, age FROM students", schema=CANONICAL_SCHEMA_STRING).is_valid)
        self.assertTrue(validate_sql("SELECT name, cgpa FROM students WHERE cgpa > 8.0", schema=create_mock_schema_metadata()).is_valid)

    def test_malformed_inputs_credential_scrubbing_and_result_contract(self) -> None:
        for bad_empty in ("", "   \t\r\n "):
            res = self.validator.validate(bad_empty)
            self.assertFalse(res.is_valid)
            self.assertFalse(res.is_syntax_valid)

        for bad_type in (None, 12345, ["SELECT 1"], {"sql": "SELECT 1"}):
            res = self.validator.validate(bad_type)  # type: ignore[arg-type]
            self.assertFalse(res.is_valid)
            self.assertIn("non-null string", res.error_message)

        for incomplete in ("SELECT", "SELECT FROM students", "WHERE age > 20", "SELECT name FROM"):
            res = self.validator.validate(incomplete)
            self.assertFalse(res.is_valid)
            self.assertFalse(res.is_syntax_valid)

        res_secret = self.validator.validate("SELECT * FROM students WHERE password = 'xyz' AND secret_token = 'abc'")
        self.assertTrue(all("password =" not in e and "secret_token =" not in e for e in res_secret.errors))

        vr = ValidationResult(
            is_valid=False,
            is_syntax_valid=False,
            is_permitted_read_only=False,
            is_schema_compatible=False,
            is_in_supported_scope=False,
            is_in_model_scope=False,
            errors=["First error", "Second error"],
        )
        self.assertEqual(vr.error_message, "First error")
        self.assertEqual(vr.to_dict()["error_message"], "First error")

    def test_live_mysql_schema_inspection_validation(self) -> None:
        from src.database.connection import DatabaseConnection
        from src.database.schema import DatabaseSchemaInspector

        with DatabaseConnection.from_env() as conn:
            live_metadata = DatabaseSchemaInspector(connection=conn).inspect_database()

        live_val = SQLValidator(schema=live_metadata, enforce_scope=True)
        self.assertTrue(live_val.validate("SELECT student_id, name, cgpa FROM students WHERE cgpa > 8.0").is_valid)
        self.assertFalse(live_val.validate("SELECT name FROM faculty WHERE age > 30").is_schema_compatible)
        self.assertFalse(live_val.validate("SELECT salary FROM students").is_schema_compatible)


if __name__ == "__main__":
    unittest.main()
