"""Consolidated tests for AST/Execution evaluation comparators and benchmark runner."""

from __future__ import annotations

import decimal
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_benchmark
from src.database.executor import ExecutionResult, SQLExecutor
from src.evaluation.comparator import ASTComparator, ExecutionComparator
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA
from src.validation.validator import SQLValidator


class TestEvaluationAndBenchmark(unittest.TestCase):
    """Verifies AST normalization, execution multiset/sequence comparison, and benchmark contracts."""

    def setUp(self) -> None:
        self.ast_comp = ASTComparator(dialect="mysql")
        self.validator = SQLValidator(schema=DEFAULT_STUDENT_SCHEMA, enforce_model_scope=True)

    def test_ast_normalization_and_boolean_precedence(self) -> None:
        res_ws = self.ast_comp.compare(
            "SELECT student_id, name FROM students WHERE age > 20",
            "  select  STUDENT_ID ,  name FROM Students WHERE AGE > 20 ",
        )
        self.assertTrue(res_ws.is_valid_sql)
        self.assertTrue(res_ws.normalized_exact_match)

        # Commutative AND matches AST without claiming strict string match
        res_and = self.ast_comp.compare(
            "SELECT name, cgpa FROM students WHERE age > 20 AND cgpa >= 8.0",
            "SELECT name, cgpa FROM students WHERE cgpa >= 8.0 AND age > 20",
        )
        self.assertFalse(res_and.strict_exact_match)
        self.assertTrue(res_and.normalized_exact_match)
        self.assertTrue(res_and.where_match)

        # Precedence preserved: (A OR B) AND C != A OR (B AND C), but (B OR A) AND C matches
        q_grouped = "SELECT * FROM students WHERE (department = 'CSE' OR department = 'ECE') AND cgpa > 8.0"
        q_diff_prec = "SELECT * FROM students WHERE department = 'CSE' OR (department = 'ECE' AND cgpa > 8.0)"
        q_comm_or = "SELECT * FROM students WHERE (department = 'ECE' OR department = 'CSE') AND cgpa > 8.0"
        self.assertFalse(self.ast_comp.compare(q_grouped, q_diff_prec).normalized_exact_match)
        self.assertTrue(self.ast_comp.compare(q_grouped, q_comm_or).normalized_exact_match)

    def test_clause_extraction_comparison_and_syntax_errors(self) -> None:
        _, _, parsed = self.ast_comp.parse_sql("SELECT student_id, name FROM students WHERE cgpa >= 8.5 ORDER BY cgpa DESC LIMIT 5")
        clauses = self.ast_comp.extract_clauses(parsed)
        self.assertEqual(clauses["table"], "students")
        self.assertEqual(clauses["select"], ["student_id", "name"])
        self.assertEqual(clauses["order"], [("cgpa", "DESC")])
        self.assertEqual(clauses["limit"], 5)
        self.assertIsNotNone(clauses["where_canonical"])

        res_clause = self.ast_comp.compare(
            "SELECT name FROM students WHERE cgpa > 8.0 ORDER BY name ASC LIMIT 3",
            "SELECT name FROM students WHERE cgpa > 9.0 ORDER BY name ASC LIMIT 3",
        )
        self.assertTrue(all([res_clause.table_match, res_clause.select_match, res_clause.order_match, res_clause.limit_match]))
        self.assertFalse(res_clause.where_match)

        res_bad = self.ast_comp.compare("SELECT * FROM students", "SELECT FROM WHERE SYNTAX ERROR")
        self.assertFalse(res_bad.is_valid_sql)
        self.assertEqual(res_bad.failure_mode, "SYNTAX_ERROR")

    def test_execution_comparator_multiset_order_null_and_precision(self) -> None:
        exec_comp = ExecutionComparator(executor=None, validator=self.validator)

        # Unordered multiset preserves duplicate multiplicities
        ref_rows = [{"id": 1, "score": 90}, {"id": 2, "score": 85}, {"id": 1, "score": 90}]
        pred_reordered = [{"score": 85, "id": 2}, {"id": 1, "score": 90}, {"id": 1, "score": 90}]
        pred_wrong_dup = [{"score": 85, "id": 2}, {"id": 1, "score": 90}, {"id": 2, "score": 85}]
        self.assertTrue(exec_comp.compare_result_rows(ref_rows, pred_reordered, has_order_by=False)[0])
        self.assertFalse(exec_comp.compare_result_rows(ref_rows, pred_wrong_dup, has_order_by=False)[0])

        # Ordered comparison enforces sequence when ORDER BY exists
        r_ord = [{"name": "Aarav", "cgpa": 9.5}, {"name": "Priya", "cgpa": 9.0}]
        p_ord = [{"name": "Priya", "cgpa": 9.0}, {"name": "Aarav", "cgpa": 9.5}]
        self.assertFalse(exec_comp.compare_result_rows(r_ord, p_ord, has_order_by=True)[0])

        # NULL semantics and Decimal/float precision
        self.assertTrue(exec_comp.compare_result_rows([{"id": 1, "age": None}], [{"id": 1, "age": None}], has_order_by=False)[0])
        self.assertFalse(exec_comp.compare_result_rows([{"id": 1, "age": None}], [{"id": 1, "age": 0}], has_order_by=False)[0])
        self.assertFalse(exec_comp.compare_result_rows([{"id": 1, "age": None}], [{"id": 1, "age": ""}], has_order_by=False)[0])
        self.assertTrue(exec_comp.compare_result_rows([{"cgpa": decimal.Decimal("8.50"), "credits": 4}], [{"cgpa": 8.5, "credits": 4.0}], has_order_by=False)[0])

    def test_execution_comparator_failure_and_unsafe_sql_isolation(self) -> None:
        mock_executor = MagicMock(spec=SQLExecutor)
        mock_executor.execute.side_effect = [
            ExecutionResult(success=True, row_count=2, rows=[{"id": 1}]),
            ExecutionResult(success=False, error="MySQL deadlock detected"),
        ]
        exec_comp = ExecutionComparator(executor=mock_executor, validator=self.validator)
        res_fail = exec_comp.execute_and_compare("SELECT student_id FROM students", "SELECT student_id FROM students WHERE age > 20")
        self.assertTrue(res_fail.ref_success)
        self.assertFalse(res_fail.pred_success)
        self.assertFalse(res_fail.results_match)

        mock_executor2 = MagicMock(spec=SQLExecutor)
        exec_comp2 = ExecutionComparator(executor=mock_executor2, validator=self.validator)
        res_drop = exec_comp2.execute_and_compare("SELECT * FROM students", "DROP TABLE students")
        self.assertFalse(res_drop.pred_success)
        self.assertEqual(mock_executor2.execute.call_count, 1)

    def test_benchmark_dataset_integrity_and_runner_defaults(self) -> None:
        bench_path = PROJECT_ROOT / "data" / "benchmark_eval.jsonl"
        records = [json.loads(line) for line in bench_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 80)
        self.assertEqual(len({r["id"] for r in records}), 80)
        cats = Counter(r["category"] for r in records)
        self.assertEqual(len(cats), 10)
        self.assertTrue(all(cnt == 8 for cnt in cats.values()))
        for r in records:
            self.assertTrue(self.validator.validate(r["sql"]).is_valid)

        with patch.object(sys, "argv", ["run_benchmark.py"]):
            args = run_benchmark.parse_args()
        self.assertEqual(args.checkpoint, "checkpoints/candidates/phase2_clean_best.pt")
        self.assertEqual(args.decoding_mode, "schema_aware")
        self.assertEqual(args.beam_width, 5)


if __name__ == "__main__":
    unittest.main()
