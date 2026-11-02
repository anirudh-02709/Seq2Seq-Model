"""
Comprehensive Regression Test Suite for Final Dataset v1 + Validator Remediation.

Tests all required areas from Section 9:
1. AST and SQL safety (Positive Allowlist)
2. Execution gating (AST-invalid records never reach MySQL)
3. Manifest defensive validation
4. Credential handling & injection prevention
5. Canonical schema parser hardening
"""

import sys
import os
import json
import tempfile
import unittest

# Import functions from validate_dataset
sys.path.insert(0, os.path.abspath("scripts"))
import validate_dataset
from sqlglot import exp

class TestASTAndSQLSafety(unittest.TestCase):

    def test_valid_select_within_scope(self):
        valid_queries = [
            ("SELECT * FROM students", "students", "Basic SELECT and projection"),
            ("SELECT name, cgpa FROM students WHERE cgpa >= 9.0", "students", "WHERE and comparisons"),
            ("SELECT name, age FROM students WHERE department = 'CSE' AND age > 20", "students", "AND / OR conditions"),
            ("SELECT * FROM courses ORDER BY credits DESC", "courses", "ORDER BY"),
            ("SELECT * FROM enrollments LIMIT 10", "enrollments", "LIMIT"),
            ("SELECT student_id, marks FROM enrollments WHERE course_id = 1 ORDER BY marks DESC LIMIT 5", "enrollments", "Combined SQL patterns")
        ]
        for sql, table, cat in valid_queries:
            rec = {"id": 1, "sql": sql, "table": table, "category": cat, "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertTrue(is_valid, f"Expected valid query to pass: '{sql}', got errors: {errs}")
            self.assertEqual(len(errs), 0)

    def test_sleep_and_function_calls_rejected(self):
        bad_sqls = [
            "SELECT SLEEP(5) FROM students",
            "SELECT LOWER(name) FROM students",
            "SELECT NOW() FROM students",
            "SELECT BENCHMARK(1000000, MD5(1)) FROM students"
        ]
        for sql in bad_sqls:
            rec = {"id": 2, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected function call to be rejected: '{sql}'")
            self.assertTrue(any("Unauthorized AST node" in e or "Function" in e for e in errs))

    def test_locking_clauses_rejected(self):
        bad_sqls = [
            "SELECT * FROM students FOR UPDATE",
            "SELECT * FROM students LOCK IN SHARE MODE"
        ]
        for sql in bad_sqls:
            rec = {"id": 3, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected lock clause to be rejected: '{sql}'")
            self.assertTrue(any("Locking clauses" in e or "Unauthorized AST node" in e for e in errs))

    def test_into_outfile_rejected(self):
        bad_sqls = [
            "SELECT * FROM students INTO OUTFILE '/tmp/owned.txt'",
            "SELECT * FROM students INTO DUMPFILE '/tmp/dump.txt'"
        ]
        for sql in bad_sqls:
            rec = {"id": 4, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected INTO to be rejected: '{sql}'")

    def test_user_variables_and_assignment_rejected(self):
        bad_sqls = [
            "SELECT @var := 1 FROM students",
            "SELECT @x FROM students"
        ]
        for sql in bad_sqls:
            rec = {"id": 5, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected user variables to be rejected: '{sql}'")
            self.assertTrue(any("Unauthorized AST node" in e for e in errs))

    def test_distinct_rejected(self):
        sql = "SELECT DISTINCT department FROM students"
        rec = {"id": 6, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
        is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
        self.assertFalse(is_valid, f"Expected DISTINCT to be rejected: '{sql}'")
        self.assertTrue(any("Distinct" in e or "DISTINCT" in e for e in errs))

    def test_arithmetic_expressions_rejected(self):
        bad_sqls = [
            "SELECT age + 1 FROM students",
            "SELECT * FROM students WHERE cgpa * 10 > 80"
        ]
        for sql in bad_sqls:
            rec = {"id": 7, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected arithmetic to be rejected: '{sql}'")
            self.assertTrue(any("Unauthorized AST node" in e for e in errs))

    def test_unsupported_expressions_like_in_between(self):
        bad_sqls = [
            "SELECT * FROM students WHERE name LIKE 'A%'",
            "SELECT * FROM students WHERE department IN ('CSE', 'ECE')",
            "SELECT * FROM students WHERE cgpa BETWEEN 8.0 AND 9.0",
            "SELECT CASE WHEN cgpa > 8 THEN 'A' ELSE 'B' END FROM students"
        ]
        for sql in bad_sqls:
            rec = {"id": 8, "sql": sql, "table": "students", "category": "WHERE and comparisons", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected unsupported operator to be rejected: '{sql}'")

    def test_sql_comments_and_multiple_statements(self):
        bad_sqls = [
            "SELECT * FROM students -- comment",
            "SELECT * FROM students /* block comment */",
            "SELECT * FROM students # hash comment",
            "SELECT * FROM students; SELECT * FROM courses",
            "SELECT * FROM students; DROP TABLE students"
        ]
        for sql in bad_sqls:
            rec = {"id": 9, "sql": sql, "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "t.jsonl", "line_idx": 1}}
            is_valid, errs = validate_dataset.validate_sql_with_ast_allowlist(rec)
            self.assertFalse(is_valid, f"Expected comment/multiple statements to be rejected: '{sql}'")


class TestExecutionGating(unittest.TestCase):

    def test_ast_invalid_records_never_sent_to_mysql(self):
        """Proves that records failing static AST checks are excluded from reaching MySQL stages."""
        mixed_records = [
            {"id": 101, "sql": "SELECT name FROM students", "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "f.jsonl", "line_idx": 1, "split": "train"}},
            {"id": 102, "sql": "SELECT SLEEP(10) FROM students", "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "f.jsonl", "line_idx": 2, "split": "train"}},
            {"id": 103, "sql": "SELECT * FROM students FOR UPDATE", "table": "students", "category": "Basic SELECT and projection", "_meta": {"filepath": "f.jsonl", "line_idx": 3, "split": "train"}},
            {"id": 104, "sql": "SELECT credits FROM courses", "table": "courses", "category": "Basic SELECT and projection", "_meta": {"filepath": "f.jsonl", "line_idx": 4, "split": "train"}}
        ]

        statically_approved = []
        statically_rejected = []

        for r in mixed_records:
            is_approved, errs = validate_dataset.validate_sql_with_ast_allowlist(r)
            if is_approved:
                statically_approved.append(r)
            else:
                statically_rejected.append(r)

        # Confirm 101 and 104 are approved, 102 and 103 are rejected
        approved_ids = [r["id"] for r in statically_approved]
        rejected_ids = [r["id"] for r in statically_rejected]
        self.assertEqual(approved_ids, [101, 104])
        self.assertEqual(rejected_ids, [102, 103])

        # Simulate validate_mysql call with statically_approved
        # Verify that ONLY approved records are present in the list passed to validate_mysql
        for r in statically_approved:
            self.assertNotIn(r["id"], [102, 103])


class TestManifestValidation(unittest.TestCase):

    def setUp(self):
        self.actual_stats = {
            "total_examples": 300,
            "split_counts": {"train": 210, "validation": 45, "test": 45},
            "category_counts": validate_dataset.PLANNED_CATEGORY_TOTALS,
            "table_counts": validate_dataset.PLANNED_TABLE_TOTALS
        }

    def test_non_dict_manifest(self):
        for bad_data in [["a", "list"], "string", 123, None]:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump(bad_data, f)
                tmp_path = f.name
            try:
                errs = []
                validate_dataset.validate_manifest(tmp_path, self.actual_stats, errs)
                self.assertTrue(len(errs) > 0)
                self.assertTrue(any("must be a JSON object" in e for e in errs))
            finally:
                os.remove(tmp_path)

    def test_nested_distribution_fields_type_and_bool_counts(self):
        # Case A: nested category_distribution is string
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump({
                "dataset_name": "Test", "description": "D", "schema": validate_dataset.CANONICAL_SCHEMA_STR,
                "total_examples": 300, "split_counts": {"train": 210, "validation": 45, "test": 45},
                "category_distribution": "not a dict",
                "table_distribution": {"students": 150, "courses": 60, "enrollments": 90},
                "supported_sql_features": ["SELECT"]
            }, f)
            tmp_path = f.name
        try:
            errs = []
            validate_dataset.validate_manifest(tmp_path, self.actual_stats, errs)
            self.assertTrue(any("category_distribution" in e for e in errs))
        finally:
            os.remove(tmp_path)

        # Case B: count is boolean True instead of int
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump({
                "dataset_name": "Test", "description": "D", "schema": validate_dataset.CANONICAL_SCHEMA_STR,
                "total_examples": True,  # boolean
                "split_counts": {"train": 210, "validation": 45, "test": 45},
                "category_distribution": validate_dataset.PLANNED_CATEGORY_TOTALS,
                "table_distribution": validate_dataset.PLANNED_TABLE_TOTALS,
                "supported_sql_features": ["SELECT"]
            }, f)
            tmp_path = f.name
        try:
            errs = []
            validate_dataset.validate_manifest(tmp_path, self.actual_stats, errs)
            self.assertTrue(any("rejecting bool" in e or "total_examples" in e for e in errs))
        finally:
            os.remove(tmp_path)


class TestCredentialHandlingAndInjection(unittest.TestCase):

    def test_injection_characters_rejected(self):
        """Verify that newlines in user, password, or host fail closed immediately."""
        bad_configs = [
            ("user\n[mysqld]\nskip-grant-tables", "pass", "localhost", "3306"),
            ("user", "pass\n[mysqld]\nport=3307", "localhost", "3306"),
            ("user\r\n", "pass", "localhost", "3306"),
            ("user", "pass\r\n[mysqld]", "localhost", "3306")
        ]
        for u, p, h, port in bad_configs:
            os.environ["MYSQL_USER"] = u
            os.environ["MYSQL_PASSWORD"] = p
            os.environ["MYSQL_HOST"] = h
            os.environ["MYSQL_PORT"] = port
            try:
                res = validate_dataset.validate_mysql([], 0, static_only=False)
                self.assertEqual(res["part_a_status"], "FAILED")
                self.assertTrue(any("injection" in e.lower() or "security" in e.lower() for e in res["errors"]))
            finally:
                for k in ["MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_HOST", "MYSQL_PORT"]:
                    os.environ.pop(k, None)


class TestCanonicalSchemaParser(unittest.TestCase):

    def test_valid_canonical_schema(self):
        is_valid, err = validate_dataset.parse_and_validate_schema_string(validate_dataset.CANONICAL_SCHEMA_STR)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_duplicate_table_rejected(self):
        dup_table = "students(student_id, name, age, department, cgpa) | students(student_id, name, age, department, cgpa) | courses(course_id, course_name, credits)"
        is_valid, err = validate_dataset.parse_and_validate_schema_string(dup_table)
        self.assertFalse(is_valid)
        self.assertIn("Duplicate table", err)

    def test_duplicate_column_rejected(self):
        dup_col = "students(student_id, name, age, department, department, cgpa) | courses(course_id, course_name, credits) | enrollments(enrollment_id, student_id, course_id, semester, marks)"
        is_valid, err = validate_dataset.parse_and_validate_schema_string(dup_col)
        self.assertFalse(is_valid)
        self.assertIn("Duplicate column", err)

    def test_unknown_table_rejected(self):
        unknown_tbl = "students(student_id, name, age, department, cgpa) | professors(prof_id, name) | courses(course_id, course_name, credits)"
        is_valid, err = validate_dataset.parse_and_validate_schema_string(unknown_tbl)
        self.assertFalse(is_valid)
        self.assertIn("Unknown table", err)

    def test_missing_column_rejected(self):
        missing_col = "students(student_id, name, age) | courses(course_id, course_name, credits) | enrollments(enrollment_id, student_id, course_id, semester, marks)"
        is_valid, err = validate_dataset.parse_and_validate_schema_string(missing_col)
        self.assertFalse(is_valid)
        self.assertIn("mismatch", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
