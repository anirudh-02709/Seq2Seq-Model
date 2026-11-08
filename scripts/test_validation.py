"""
Comprehensive Test Suite for SQL Validation and Read-Only Safety Layer (Milestone 4, Task 4.3).

Covers:
- Category A: Valid queries within current model scope
- Category B: Forbidden operations (mutations, DDL, multi-statement, INTO, locks, dangerous functions)
- Category C: Schema validation (tables, columns, qualifiers, aliases, wildcards, ambiguous cols, metadata)
- Category D: Malformed and adversarial inputs (empty, comments, casing, trailing semicolon, non-string, injections)
- Category E: Model scope enforcement (aggregations, GROUP BY, HAVING, JOIN, UNION, subqueries)
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.schema import ColumnMetadata, DatabaseSchemaMetadata, TableMetadata
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator, normalize_schema, validate_sql

# Standard Student Management schema for testing
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
    """Builds a DatabaseSchemaMetadata instance mirroring the test schema."""
    students_table = TableMetadata(
        name="students",
        columns=[
            ColumnMetadata(name="student_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
            ColumnMetadata(name="age", data_type="int", column_type="int(11)", is_nullable=False),
            ColumnMetadata(name="department", data_type="varchar", column_type="varchar(50)", is_nullable=False),
            ColumnMetadata(name="cgpa", data_type="decimal", column_type="decimal(3,2)", is_nullable=True),
        ],
        primary_keys=["student_id"],
    )
    courses_table = TableMetadata(
        name="courses",
        columns=[
            ColumnMetadata(name="course_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="course_name", data_type="varchar", column_type="varchar(100)", is_nullable=False),
            ColumnMetadata(name="credits", data_type="int", column_type="int(11)", is_nullable=False),
            ColumnMetadata(name="department", data_type="varchar", column_type="varchar(50)", is_nullable=False),
        ],
        primary_keys=["course_id"],
    )
    enrollments_table = TableMetadata(
        name="enrollments",
        columns=[
            ColumnMetadata(name="enrollment_id", data_type="int", column_type="int(11)", is_nullable=False, is_primary_key=True),
            ColumnMetadata(name="student_id", data_type="int", column_type="int(11)", is_nullable=False),
            ColumnMetadata(name="course_id", data_type="int", column_type="int(11)", is_nullable=False),
            ColumnMetadata(name="semester", data_type="varchar", column_type="varchar(20)", is_nullable=False),
            ColumnMetadata(name="marks", data_type="decimal", column_type="decimal(5,2)", is_nullable=True),
        ],
        primary_keys=["enrollment_id"],
    )
    return DatabaseSchemaMetadata(
        database_name="nl2sql_student_db",
        tables={
            "students": students_table,
            "courses": courses_table,
            "enrollments": enrollments_table,
        },
    )


class TestCategoryAValidQueries(unittest.TestCase):
    """Category A: Valid queries within current model scope."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_basic_select_single_column(self) -> None:
        result = self.validator.validate("SELECT name FROM students")
        self.assertTrue(result.is_valid)
        self.assertTrue(result.is_syntax_valid)
        self.assertTrue(result.is_permitted_read_only)
        self.assertTrue(result.is_schema_compatible)
        self.assertTrue(result.is_in_supported_scope)
        self.assertEqual(len(result.errors), 0)
        self.assertIn("name", result.ast_details["columns"])
        self.assertIn("students", result.ast_details["tables"])
        self.assertEqual(result.normalized_sql, "SELECT name FROM students")

    def test_select_multiple_columns(self) -> None:
        result = self.validator.validate("SELECT student_id, name, cgpa FROM students")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.normalized_sql, "SELECT student_id, name, cgpa FROM students")

    def test_select_with_where_comparisons(self) -> None:
        comparisons = [
            "SELECT name FROM students WHERE age > 20",
            "SELECT name FROM students WHERE cgpa >= 8.5",
            "SELECT name FROM students WHERE age < 25",
            "SELECT name FROM students WHERE cgpa <= 9.0",
            "SELECT name FROM students WHERE department = 'CSE'",
            "SELECT name FROM students WHERE department != 'ME'",
            "SELECT name FROM students WHERE department <> 'ECE'",
        ]
        for query in comparisons:
            with self.subTest(query=query):
                res = self.validator.validate(query)
                self.assertTrue(res.is_valid, f"Failed for query: {query}, errors: {res.errors}")
                self.assertTrue(res.ast_details["has_where"])
                self.assertTrue(res.is_in_model_scope)

    def test_select_compound_where_and_or(self) -> None:
        query = (
            "SELECT name, department FROM students "
            "WHERE (department = 'CSE' OR department = 'AI&DS') AND cgpa > 8.0"
        )
        res = self.validator.validate(query)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.ast_details["has_where"])
        self.assertTrue(res.is_in_model_scope)

    def test_select_order_by_asc_desc(self) -> None:
        q_asc = "SELECT name, cgpa FROM students ORDER BY cgpa ASC"
        res_asc = self.validator.validate(q_asc)
        self.assertTrue(res_asc.is_valid)
        self.assertTrue(res_asc.ast_details["has_order_by"])
        self.assertTrue(res_asc.is_in_model_scope)

        q_desc = "SELECT name, age FROM students ORDER BY age DESC"
        res_desc = self.validator.validate(q_desc)
        self.assertTrue(res_desc.is_valid)
        self.assertTrue(res_desc.ast_details["has_order_by"])
        self.assertTrue(res_desc.is_in_model_scope)

    def test_select_limit(self) -> None:
        query = "SELECT name FROM students LIMIT 5"
        res = self.validator.validate(query)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.ast_details["has_limit"])
        self.assertTrue(res.is_in_model_scope)

    def test_select_order_by_and_limit(self) -> None:
        query = "SELECT name, cgpa FROM students WHERE department = 'CSE' ORDER BY cgpa DESC LIMIT 3"
        res = self.validator.validate(query)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.ast_details["has_where"])
        self.assertTrue(res.ast_details["has_order_by"])
        self.assertTrue(res.ast_details["has_limit"])
        self.assertTrue(res.is_in_model_scope)

    def test_qualified_columns(self) -> None:
        query = "SELECT students.name, students.age FROM students WHERE students.cgpa > 7.5"
        res = self.validator.validate(query)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_in_model_scope)

    def test_table_alias_opt_out_mode(self) -> None:
        # Table aliases are outside trained model scope, but permitted in explicit manual opt-out mode
        query = "SELECT s.name, s.cgpa FROM students s WHERE s.age > 20"
        res_default = self.validator.validate(query)
        self.assertFalse(res_default.is_valid)
        self.assertFalse(res_default.is_in_model_scope)

        res_opt_out = self.validator.validate(query, enforce_model_scope=False)
        self.assertTrue(res_opt_out.is_valid)
        self.assertEqual(res_opt_out.ast_details["aliases"].get("s"), "students")

    def test_table_alias_with_as_opt_out_mode(self) -> None:
        query = "SELECT s.name, s.cgpa FROM students AS s WHERE s.age > 20"
        res_default = self.validator.validate(query)
        self.assertFalse(res_default.is_valid)
        self.assertFalse(res_default.is_in_model_scope)

        res_opt_out = self.validator.validate(query, enforce_model_scope=False)
        self.assertTrue(res_opt_out.is_valid)
        self.assertEqual(res_opt_out.ast_details["aliases"].get("s"), "students")

    def test_wildcard_selection(self) -> None:
        q1 = "SELECT * FROM students"
        res1 = self.validator.validate(q1)
        self.assertTrue(res1.is_valid)
        self.assertTrue(res1.is_in_model_scope)

        # Qualified wildcard alias requires opt-out
        q2 = "SELECT s.* FROM students s WHERE s.age > 20"
        res2_default = self.validator.validate(q2)
        self.assertFalse(res2_default.is_valid)
        res2_opt_out = self.validator.validate(q2, enforce_model_scope=False)
        self.assertTrue(res2_opt_out.is_valid)

    def test_order_by_projection_alias_opt_out_mode(self) -> None:
        query = "SELECT name AS student_name, age FROM students ORDER BY student_name"
        res_default = self.validator.validate(query)
        self.assertFalse(res_default.is_valid)
        self.assertFalse(res_default.is_in_model_scope)

        res_opt_out = self.validator.validate(query, enforce_model_scope=False)
        self.assertTrue(res_opt_out.is_valid)


class TestCategoryBForbiddenOperations(unittest.TestCase):
    """Category B: Forbidden mutation, DDL, multi-statement, and administrative queries."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_forbidden_insert(self) -> None:
        queries = [
            "INSERT INTO students (student_id, name) VALUES (101, 'Alice')",
            "INSERT INTO courses VALUES (1, 'Math', 4, 'CSE')",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("Only single read-only SELECT", res.error_message)

    def test_forbidden_update(self) -> None:
        query = "UPDATE students SET cgpa = 9.5 WHERE student_id = 1"
        res = self.validator.validate(query)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_permitted_read_only)

    def test_forbidden_delete(self) -> None:
        query = "DELETE FROM students WHERE student_id = 1"
        res = self.validator.validate(query)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_permitted_read_only)

    def test_forbidden_ddl(self) -> None:
        ddl_queries = [
            "DROP TABLE students",
            "DROP DATABASE nl2sql_student_db",
            "ALTER TABLE students ADD COLUMN email VARCHAR(100)",
            "CREATE TABLE test (id INT)",
            "TRUNCATE TABLE students",
        ]
        for q in ddl_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)

    def test_forbidden_multi_statement(self) -> None:
        multi_queries = [
            "SELECT * FROM students; DROP TABLE students;",
            "SELECT 1; SELECT 2;",
            "SELECT * FROM students; DELETE FROM students WHERE 1=1;",
            "SELECT name FROM students; SELECT course_name FROM courses;",
        ]
        for q in multi_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("Multiple SQL statements detected", res.error_message)

    def test_forbidden_select_into(self) -> None:
        into_queries = [
            "SELECT * INTO OUTFILE '/tmp/students.txt' FROM students",
            "SELECT * FROM students INTO OUTFILE '/tmp/students.txt'",
            "SELECT * INTO DUMPFILE '/tmp/dump.txt' FROM students",
            "SELECT 1 INTO @my_var",
            "SELECT * INTO new_table FROM students",
        ]
        for q in into_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)

    def test_forbidden_locking_clauses(self) -> None:
        locking_queries = [
            "SELECT * FROM students FOR UPDATE",
            "SELECT name FROM students WHERE student_id = 1 LOCK IN SHARE MODE",
            "SELECT * FROM students FOR SHARE",
        ]
        for q in locking_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("Locking clauses", res.error_message)

    def test_forbidden_security_functions(self) -> None:
        dangerous_queries = [
            "SELECT SLEEP(5) FROM students",
            "SELECT BENCHMARK(1000000, MD5('test')) FROM students",
            "SELECT LOAD_FILE('/etc/passwd') FROM students",
            "SELECT VERSION() FROM students",
            "SELECT USER() FROM students",
            "SELECT DATABASE() FROM students",
        ]
        for q in dangerous_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("prohibited or security-sensitive function", res.error_message)

    def test_forbidden_grant_and_revoke(self) -> None:
        queries = [
            "GRANT ALL PRIVILEGES ON *.* TO 'user'@'%'",
            "REVOKE SELECT ON students FROM 'user'@'%'",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)


class TestCategoryCSchemaValidation(unittest.TestCase):
    """Category C: Schema compatibility, aliases, qualifiers, and ambiguity."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_valid_tables_and_columns(self) -> None:
        res = self.validator.validate("SELECT course_id, course_name, credits FROM courses")
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_schema_compatible)

    def test_nonexistent_table(self) -> None:
        res = self.validator.validate("SELECT name FROM professors")
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Table 'professors' does not exist", res.error_message)

    def test_nonexistent_column(self) -> None:
        res = self.validator.validate("SELECT salary FROM students")
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Column 'salary' does not exist", res.error_message)

    def test_invalid_qualified_column(self) -> None:
        # 'credits' is in courses, not students
        res = self.validator.validate("SELECT students.credits FROM students")
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Column 'credits' does not exist in table 'students'", res.error_message)

    def test_unknown_alias_in_column(self) -> None:
        res = self.validator.validate("SELECT x.name FROM students s", enforce_model_scope=False)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Table or alias 'x'", res.error_message)

    def test_unknown_alias_in_wildcard(self) -> None:
        res = self.validator.validate("SELECT x.* FROM students s", enforce_model_scope=False)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Table or alias 'x'", res.error_message)

    def test_ambiguous_column_in_multi_table(self) -> None:
        # Both students and courses have 'department'
        # With enforce_scope=False and enforce_model_scope=False so scope doesn't short-circuit schema validation
        res = self.validator.validate(
            "SELECT department FROM students, courses",
            enforce_scope=False,
            enforce_model_scope=False,
        )
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Ambiguous column reference", res.error_message)

    def test_qualified_resolves_ambiguity(self) -> None:
        res = self.validator.validate(
            "SELECT students.department FROM students, courses",
            enforce_scope=False,
            enforce_model_scope=False,
        )
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_schema_compatible)

    def test_missing_schema_metadata(self) -> None:
        validator_no_schema = SQLValidator(schema=None)
        res = validator_no_schema.validate("SELECT name FROM students")
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Schema metadata was not provided or is empty", res.error_message)

    def test_empty_dict_schema_metadata(self) -> None:
        res = self.validator.validate("SELECT name FROM students", schema={})
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_schema_compatible)
        self.assertIn("Schema metadata was not provided or is empty", res.error_message)

    def test_canonical_schema_string_compatibility(self) -> None:
        res = validate_sql("SELECT name, age FROM students", schema=CANONICAL_SCHEMA_STRING)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_schema_compatible)

    def test_database_schema_metadata_object_compatibility(self) -> None:
        meta_obj = create_mock_schema_metadata()
        res = validate_sql("SELECT name, cgpa FROM students WHERE cgpa > 8.0", schema=meta_obj)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_schema_compatible)


class TestCategoryDMalformedAndAdversarial(unittest.TestCase):
    """Category D: Malformed, adversarial, injection, and edge-case inputs."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_empty_and_whitespace_inputs(self) -> None:
        bad_inputs = ["", "   ", "\t", "\n\n", " \t \r\n "]
        for inp in bad_inputs:
            with self.subTest(inp=inp):
                res = self.validator.validate(inp)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_syntax_valid)
                self.assertIn("empty or contains only whitespace", res.error_message)

    def test_non_string_inputs(self) -> None:
        bad_inputs = [None, 12345, ["SELECT 1"], {"sql": "SELECT 1"}, 3.14]
        for inp in bad_inputs:
            with self.subTest(inp=inp):
                res = self.validator.validate(inp)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_syntax_valid)
                self.assertIn("must be a non-null string", res.error_message)

    def test_trailing_semicolons_handling(self) -> None:
        valid_with_semi = "SELECT name FROM students;"
        res = self.validator.validate(valid_with_semi)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_sql, "SELECT name FROM students")

        valid_with_semi_spaces = "SELECT name FROM students;   "
        res2 = self.validator.validate(valid_with_semi_spaces)
        self.assertTrue(res2.is_valid)

    def test_sql_comments_stripped_in_normalization(self) -> None:
        q_line_comment = "SELECT name FROM students -- fetch student names"
        res1 = self.validator.validate(q_line_comment)
        self.assertTrue(res1.is_valid)
        self.assertEqual(res1.normalized_sql, "SELECT name FROM students")

        q_block_comment = "SELECT /* column */ name FROM /* table */ students"
        res2 = self.validator.validate(q_block_comment)
        self.assertTrue(res2.is_valid)
        self.assertEqual(res2.normalized_sql, "SELECT name FROM students")

    def test_case_insensitivity_handling(self) -> None:
        q_mixed = "sElEcT NaMe, AgE fRoM sTuDeNtS wHeRe DePaRtMeNt = 'CSE'"
        res = self.validator.validate(q_mixed)
        self.assertTrue(res.is_valid)
        self.assertIn("SELECT", res.normalized_sql)
        self.assertIn("FROM", res.normalized_sql)
        self.assertIn("WHERE", res.normalized_sql)
        self.assertEqual(res.ast_details["tables"], ["students"])
        self.assertEqual(sorted(res.ast_details["columns"]), ["age", "department", "name"])

    def test_incomplete_syntax_errors(self) -> None:
        malformed = [
            "SELECT",
            "SELECT FROM students",
            "SELECT * WHERE",
            "WHERE age > 20",
            "SELECT * FROM students WHERE",
            "SELECT name FROM",
        ]
        for q in malformed:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_syntax_valid)

    def test_injection_attempts(self) -> None:
        injections = [
            "SELECT * FROM students WHERE name = 'x' OR 1=1; DROP TABLE students; --",
            "SELECT * FROM students WHERE 1=1; SHUTDOWN;",
            "SELECT * FROM students UNION SELECT * FROM mysql.user",
        ]
        for inj in injections:
            with self.subTest(query=inj):
                res = self.validator.validate(inj)
                self.assertFalse(res.is_valid)

    def test_no_credential_leak_in_error(self) -> None:
        res = self.validator.validate("SELECT * FROM students WHERE password = 'xyz' AND secret_token = 'abc'")
        for err in res.errors:
            self.assertNotIn("password =", err)
            self.assertNotIn("secret_token =", err)


class TestCategoryEScopeChecking(unittest.TestCase):
    """Category E: Model scope enforcement vs general SQL."""

    def setUp(self) -> None:
        self.validator_strict = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)
        self.validator_permissive = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=False, enforce_model_scope=False)

    def test_aggregations_out_of_scope(self) -> None:
        aggs = [
            "SELECT COUNT(*) FROM students",
            "SELECT SUM(credits) FROM courses",
            "SELECT AVG(cgpa) FROM students",
            "SELECT MIN(age) FROM students",
            "SELECT MAX(marks) FROM enrollments",
        ]
        for q in aggs:
            with self.subTest(query=q):
                # Under strict enforcement: fails is_valid
                res_strict = self.validator_strict.validate(q)
                self.assertFalse(res_strict.is_valid)
                self.assertFalse(res_strict.is_in_supported_scope)
                self.assertTrue(res_strict.ast_details["has_aggregations"])
                self.assertIn("Aggregations", res_strict.error_message)

                # Under permissive enforcement: scope flagged but query passes
                res_perm = self.validator_permissive.validate(q)
                self.assertTrue(res_perm.is_valid)
                self.assertFalse(res_perm.is_in_supported_scope)
                self.assertTrue(len(res_perm.warnings) > 0)

    def test_group_by_out_of_scope(self) -> None:
        query = "SELECT department, COUNT(*) FROM students GROUP BY department"
        res = self.validator_strict.validate(query)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_supported_scope)
        self.assertTrue(res.ast_details["has_group_by"])

    def test_having_out_of_scope(self) -> None:
        query = "SELECT department FROM students GROUP BY department HAVING COUNT(*) > 1"
        res = self.validator_strict.validate(query)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_supported_scope)
        self.assertTrue(res.ast_details["has_having"])

    def test_joins_out_of_scope(self) -> None:
        explicit_join = (
            "SELECT s.name, e.marks FROM students s "
            "JOIN enrollments e ON s.student_id = e.student_id"
        )
        res_exp = self.validator_strict.validate(explicit_join)
        self.assertFalse(res_exp.is_valid)
        self.assertFalse(res_exp.is_in_supported_scope)
        self.assertTrue(res_exp.ast_details["has_joins"])

        implicit_join = "SELECT students.name, courses.course_name FROM students, courses"
        res_imp = self.validator_strict.validate(implicit_join)
        self.assertFalse(res_imp.is_valid)
        self.assertFalse(res_imp.is_in_supported_scope)
        self.assertTrue(res_imp.ast_details["has_joins"])

    def test_subqueries_out_of_scope(self) -> None:
        q_sub = "SELECT name FROM (SELECT name, age FROM students WHERE age > 20) s"
        res = self.validator_strict.validate(q_sub)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_supported_scope)
        self.assertTrue(res.ast_details["has_subqueries"])

    def test_unions_out_of_scope(self) -> None:
        q_union = "SELECT name FROM students UNION SELECT course_name FROM courses"
        res = self.validator_strict.validate(q_union)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_supported_scope)

    def test_distinct_out_of_scope(self) -> None:
        q_distinct = "SELECT DISTINCT department FROM students"
        res = self.validator_strict.validate(q_distinct)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_supported_scope)


class TestValidationResultClass(unittest.TestCase):
    """Verifies the ValidationResult dataclass serialization and convenience properties."""

    def test_validation_result_to_dict(self) -> None:
        vr = ValidationResult(
            is_valid=True,
            is_syntax_valid=True,
            is_permitted_read_only=True,
            is_schema_compatible=True,
            is_in_supported_scope=True,
            is_in_model_scope=True,
            normalized_sql="SELECT name FROM students",
            raw_sql="select name from students",
            errors=[],
            warnings=["test warning"],
            ast_details={"tables": ["students"]},
        )
        d = vr.to_dict()
        self.assertTrue(d["is_valid"])
        self.assertIsNone(d["error_message"])
        self.assertEqual(d["normalized_sql"], "SELECT name FROM students")
        self.assertEqual(d["warnings"], ["test warning"])
        self.assertTrue(d["is_in_model_scope"])

    def test_error_message_property(self) -> None:
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


class TestSecurityAuditExtensions(unittest.TestCase):
    """Targeted security audit test cases: UDFs, variables, and process inspection."""

    def setUp(self) -> None:
        self.validator = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True)

    def test_unapproved_stored_proc_and_udf_rejected(self) -> None:
        udf_queries = [
            "SELECT custom_stored_proc() FROM students",
            "SELECT my_udf(student_id) FROM students",
            "SELECT sys_get_secret() FROM students",
        ]
        for q in udf_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("unapproved, stored, or user-defined function", res.error_message)

    def test_system_and_session_variables_rejected(self) -> None:
        var_queries = [
            "SELECT @@global.version_comment FROM students",
            "SELECT @@version FROM students",
            "SELECT @my_user_variable FROM students",
            "SELECT name FROM students WHERE @my_var = 1",
        ]
        for q in var_queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)
                self.assertIn("System variables (@@) and session parameters (@)", res.error_message)

    def test_process_and_lock_inspection_functions_rejected(self) -> None:
        queries = [
            "SELECT connection_id() FROM students",
            "SELECT last_insert_id() FROM students",
            "SELECT is_free_lock('audit_lock') FROM students",
            "SELECT is_used_lock('audit_lock') FROM students",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.validator.validate(q)
                self.assertFalse(res.is_valid)
                self.assertFalse(res.is_permitted_read_only)

    def test_approved_readonly_functions_allowed_in_permissive_mode(self) -> None:
        res = self.validator.validate(
            "SELECT ABS(age) FROM students",
            enforce_scope=False,
            enforce_model_scope=False,
        )
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_permitted_read_only)


class TestModelScopeAuditExtensions(unittest.TestCase):
    """Targeted model scope audit: distinguishing model-supported vs parser-supported."""

    def setUp(self) -> None:
        self.validator_general = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True, enforce_model_scope=False)
        self.validator_strict_model = SQLValidator(schema=TEST_SCHEMA_DICT, enforce_scope=True, enforce_model_scope=True)

    def test_model_scope_flags_constructs_absent_from_vocabulary(self) -> None:
        parser_supported_queries = [
            ("SELECT name FROM students WHERE name LIKE 'A%'", "LIKE operator"),
            ("SELECT name FROM students WHERE age BETWEEN 18 AND 22", "BETWEEN operator"),
            ("SELECT name FROM students WHERE department IN ('CSE', 'ECE')", "IN operator"),
            ("SELECT name FROM students WHERE cgpa IS NULL", "IS NULL / IS NOT NULL"),
            ("SELECT s.name FROM students s", "Table Aliases"),
            ("SELECT name AS student_name FROM students", "Column AS Aliases"),
            ("SELECT ABS(age) FROM students", "Function Calls"),
        ]
        for query, expected_feature in parser_supported_queries:
            with self.subTest(query=query):
                # In general parser-supported mode: passes validity, but flagged as out-of-model-scope
                res_gen = self.validator_general.validate(query, enforce_scope=False)
                self.assertFalse(res_gen.is_in_model_scope)
                self.assertIn(expected_feature, res_gen.ast_details.get("out_of_model_scope_features", []))

                # In strict model enforcement mode: fails validity
                res_strict = self.validator_strict_model.validate(query)
                self.assertFalse(res_strict.is_valid)
                self.assertFalse(res_strict.is_in_model_scope)

    def test_pure_model_query_in_model_scope(self) -> None:
        query = (
            "SELECT student_id, name, cgpa FROM students "
            "WHERE cgpa > 8.0 AND age <= 22 ORDER BY name ASC LIMIT 5"
        )
        res = self.validator_strict_model.validate(query)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.is_in_model_scope)
        self.assertTrue(res.is_in_supported_scope)
        self.assertEqual(len(res.errors), 0)


class TestLiveSchemaAuditIntegration(unittest.TestCase):
    """Targeted live schema integration test querying live MySQL metadata."""

    def test_live_database_schema_validation(self) -> None:
        from src.database.connection import DatabaseConnection
        from src.database.schema import DatabaseSchemaInspector

        try:
            conn = DatabaseConnection.from_env()
            conn.connect()
            inspector = DatabaseSchemaInspector(connection=conn)
            live_metadata = inspector.inspect_database()
            conn.close()
        except Exception as e:
            self.skipTest(f"Live MySQL database not available locally: {e}")
            return

        validator = SQLValidator(schema=live_metadata, enforce_scope=True)

        # 1. Valid students query
        res_stud = validator.validate("SELECT student_id, name, cgpa FROM students WHERE cgpa > 8.0")
        self.assertTrue(res_stud.is_valid)
        self.assertTrue(res_stud.is_schema_compatible)

        # 2. Valid courses query
        res_course = validator.validate("SELECT course_name, credits FROM courses WHERE credits >= 3")
        self.assertTrue(res_course.is_valid)
        self.assertTrue(res_course.is_schema_compatible)

        # 3. Rejection of unknown table
        res_unknown_tbl = validator.validate("SELECT name FROM faculty WHERE age > 30")
        self.assertFalse(res_unknown_tbl.is_valid)
        self.assertFalse(res_unknown_tbl.is_schema_compatible)
        self.assertIn("Table 'faculty' does not exist", res_unknown_tbl.error_message)

        # 4. Rejection of unknown column
        res_unknown_col = validator.validate("SELECT salary FROM students")
        self.assertFalse(res_unknown_col.is_valid)
        self.assertFalse(res_unknown_col.is_schema_compatible)
        self.assertIn("Column 'salary' does not exist", res_unknown_col.error_message)

        # 5. Rejection of ambiguous column (using opt-out mode to test schema inspection directly)
        res_ambig = validator.validate("SELECT course_id FROM courses, enrollments", enforce_scope=False, enforce_model_scope=False)
        self.assertFalse(res_ambig.is_valid)
        self.assertFalse(res_ambig.is_schema_compatible)
        self.assertIn("Ambiguous column reference", res_ambig.error_message)


class TestDefaultModelScopeRejection(unittest.TestCase):
    """
    Dedicated test suite verifying that default validation (enforce_model_scope=True)
    strictly rejects constructs absent from the trained Seq2Seq model's 70-token vocabulary.
    """

    def test_default_rejection_like(self) -> None:
        res = validate_sql("SELECT name FROM students WHERE name LIKE 'A%'", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("LIKE operator", res.error_message)

    def test_default_rejection_in(self) -> None:
        res = validate_sql("SELECT name FROM students WHERE department IN ('CSE', 'ECE')", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("IN operator", res.error_message)

    def test_default_rejection_between(self) -> None:
        res = validate_sql("SELECT name FROM students WHERE age BETWEEN 18 AND 22", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("BETWEEN operator", res.error_message)

    def test_default_rejection_is_null(self) -> None:
        res_null = validate_sql("SELECT name FROM students WHERE cgpa IS NULL", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res_null.is_valid)
        self.assertFalse(res_null.is_in_model_scope)
        self.assertIn("IS NULL / IS NOT NULL", res_null.error_message)

        res_not_null = validate_sql("SELECT name FROM students WHERE cgpa IS NOT NULL", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res_not_null.is_valid)
        self.assertFalse(res_not_null.is_in_model_scope)

    def test_default_rejection_not(self) -> None:
        res = validate_sql("SELECT name FROM students WHERE NOT age > 20", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("NOT operator", res.error_message)

    def test_default_rejection_table_aliases(self) -> None:
        res1 = validate_sql("SELECT s.name FROM students s", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res1.is_valid)
        self.assertFalse(res1.is_in_model_scope)
        self.assertIn("Table Aliases", res1.error_message)

        res2 = validate_sql("SELECT s.name FROM students AS s", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res2.is_valid)
        self.assertFalse(res2.is_in_model_scope)

    def test_default_rejection_column_aliases(self) -> None:
        res = validate_sql("SELECT name AS student_name FROM students", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("Column AS Aliases", res.error_message)

    def test_default_rejection_functions_count(self) -> None:
        res = validate_sql("SELECT COUNT(*) FROM students", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertTrue(any(frag in res.error_message for frag in ["Aggregations", "Function Calls"]))

    def test_default_rejection_functions_avg(self) -> None:
        res = validate_sql("SELECT AVG(cgpa) FROM students", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)

    def test_default_rejection_functions_general(self) -> None:
        res = validate_sql("SELECT ABS(age) FROM students", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)
        self.assertIn("Function Calls", res.error_message)

    def test_default_rejection_join(self) -> None:
        res_exp = validate_sql(
            "SELECT * FROM students s JOIN enrollments e ON s.student_id = e.student_id",
            schema=TEST_SCHEMA_DICT,
        )
        self.assertFalse(res_exp.is_valid)
        self.assertFalse(res_exp.is_in_model_scope)

        res_imp = validate_sql("SELECT students.name, courses.course_name FROM students, courses", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res_imp.is_valid)
        self.assertFalse(res_imp.is_in_model_scope)

    def test_default_rejection_group_by(self) -> None:
        res = validate_sql("SELECT department FROM students GROUP BY department", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)

    def test_default_rejection_subqueries(self) -> None:
        res = validate_sql("SELECT name FROM (SELECT name FROM students) sub", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)

    def test_default_rejection_union(self) -> None:
        res = validate_sql("SELECT name FROM students UNION SELECT course_name FROM courses", schema=TEST_SCHEMA_DICT)
        self.assertFalse(res.is_valid)
        self.assertFalse(res.is_in_model_scope)

    def test_default_acceptance_in_vocabulary_queries(self) -> None:
        """Verifies that all true in-vocabulary constructs pass under default validation."""
        valid_queries = [
            "SELECT name FROM students",
            "SELECT student_id, name, cgpa FROM students",
            "SELECT * FROM students",
            "SELECT name FROM students WHERE age > 20",
            "SELECT name FROM students WHERE cgpa >= 8.5",
            "SELECT name FROM students WHERE age < 25",
            "SELECT name FROM students WHERE cgpa <= 9.0",
            "SELECT name FROM students WHERE department = 'CSE'",
            "SELECT name FROM students WHERE department != 'ME'",
            "SELECT name FROM students WHERE department <> 'ECE'",
            "SELECT name, department FROM students WHERE department = 'CSE' OR department = 'AI&DS'",
            "SELECT name, cgpa FROM students WHERE department = 'CSE' AND cgpa > 8.0",
            "SELECT name, cgpa FROM students ORDER BY cgpa ASC",
            "SELECT name, age FROM students ORDER BY age DESC",
            "SELECT name FROM students LIMIT 5",
            "SELECT student_id, name, cgpa FROM students WHERE cgpa > 8.0 ORDER BY name ASC LIMIT 5",
        ]
        for q in valid_queries:
            with self.subTest(query=q):
                res = validate_sql(q, schema=TEST_SCHEMA_DICT)
                self.assertTrue(res.is_valid, f"Failed for query: {q}, errors: {res.errors}")
                self.assertTrue(res.is_in_model_scope)
                self.assertTrue(res.is_syntax_valid)
                self.assertTrue(res.is_permitted_read_only)
                self.assertTrue(res.is_schema_compatible)
                self.assertEqual(len(res.errors), 0)

    def test_explicit_opt_out_allows_parser_supported_queries(self) -> None:
        """Verifies that enforce_model_scope=False opt-out permits parser-supported queries."""
        opt_out_queries = [
            "SELECT name FROM students WHERE name LIKE 'A%'",
            "SELECT name FROM students WHERE age BETWEEN 18 AND 22",
            "SELECT s.name FROM students s WHERE s.age > 20",
            "SELECT name AS student_name FROM students",
        ]
        for q in opt_out_queries:
            with self.subTest(query=q):
                res = validate_sql(q, schema=TEST_SCHEMA_DICT, enforce_model_scope=False)
                self.assertTrue(res.is_valid, f"Failed for query: {q}, errors: {res.errors}")
                self.assertFalse(res.is_in_model_scope)
                self.assertTrue(len(res.warnings) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
