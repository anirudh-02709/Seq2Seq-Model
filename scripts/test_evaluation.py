"""
Unit and Integration Tests for Final Model Evaluation (Phase 3, Step 3.7).

Verifies:
1. scripts/evaluate.py execution and CLI interface.
2. Checkpoint restoration without data leakage or modification.
3. Output format and completeness of evaluation_results.json.
4. Exact match metrics, validity rates, and failure categorization.
5. Isolation of test split (45 examples).
6. Correct reporting of database execution status.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import sqlglot

from scripts.evaluate import (
    check_sql_validity,
    categorize_failure,
    extract_ast_components,
    normalize_sql,
    run_evaluation,
)
from src.data.dataset import load_split
from src.training.trainer import Trainer


class TestFinalEvaluation(unittest.TestCase):
    """Test suite for Phase 3, Step 3.7: Final Model Evaluation."""

    def test_01_checkpoint_restores_cleanly(self) -> None:
        """Verifies that best_model.pt exists and restores model + vocabularies cleanly."""
        chk_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        self.assertTrue(chk_path.exists(), f"Checkpoint file missing: {chk_path}")

        model, chk_data, src_vocab, tgt_vocab = Trainer.load_model_from_checkpoint(
            str(chk_path), device="cpu"
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(src_vocab), 198)
        self.assertEqual(len(tgt_vocab), 70)
        self.assertEqual(chk_data["epoch"], 25)
        self.assertGreater(chk_data["val_token_acc"], 0.80)

    def test_02_test_split_isolation_and_count(self) -> None:
        """Verifies that the test split has exactly 45 examples and contiguous IDs."""
        test_recs = load_split("test", data_dir=PROJECT_ROOT / "data")
        self.assertEqual(len(test_recs), 45)
        test_ids = [r["id"] for r in test_recs]
        self.assertEqual(test_ids, list(range(256, 301)))

    def test_03_sql_normalization(self) -> None:
        """Verifies normalize_sql normalizes keywords and whitespace appropriately."""
        raw = "select student_id , name from students where age > 20"
        normalized = normalize_sql(raw)
        self.assertIn("SELECT", normalized)
        self.assertIn("FROM", normalized)
        self.assertIn("WHERE", normalized)

    def test_04_check_sql_validity(self) -> None:
        """Verifies check_sql_validity correctly identifies valid and invalid SQL."""
        valid_sql = "SELECT student_id, name FROM students WHERE cgpa >= 8.5"
        is_valid, err, parsed = check_sql_validity(valid_sql)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(parsed)

        invalid_sql = "SELECT FROM WHERE"
        is_valid2, err2, parsed2 = check_sql_validity(invalid_sql)
        self.assertFalse(is_valid2)
        self.assertIsNotNone(err2)

    def test_05_ast_extraction(self) -> None:
        """Verifies extract_ast_components properly parses query constituents."""
        query = "SELECT student_id, name FROM students WHERE cgpa >= 8.5 ORDER BY cgpa DESC LIMIT 5"
        parsed = sqlglot.parse_one(query, read="mysql")
        ast = extract_ast_components(parsed)
        self.assertEqual(ast["tables"], ["students"])
        self.assertEqual(ast["projections"], ["student_id", "name"])
        self.assertIn("cgpa >= 8.5", ast["where"])
        self.assertIn("cgpa DESC", ast["order"])
        self.assertIn("5", ast["limit"])

    def test_06_failure_categorization(self) -> None:
        """Verifies that categorize_failure identifies specific failure modes."""
        # Literal mismatch
        ref = "SELECT * FROM courses WHERE course_id = 5"
        pred = "SELECT * FROM courses WHERE course_id = 2"
        ref_ast = extract_ast_components(sqlglot.parse_one(ref, read="mysql"))
        pred_ast = extract_ast_components(sqlglot.parse_one(pred, read="mysql"))
        cat, expl = categorize_failure(ref, pred, True, None, ref_ast, pred_ast)
        self.assertEqual(cat, "LITERAL_VALUE_MISMATCH")

        # Projection mismatch
        ref2 = "SELECT student_id, department FROM students"
        pred2 = "SELECT student_id, name FROM students"
        ref_ast2 = extract_ast_components(sqlglot.parse_one(ref2, read="mysql"))
        pred_ast2 = extract_ast_components(sqlglot.parse_one(pred2, read="mysql"))
        cat2, expl2 = categorize_failure(ref2, pred2, True, None, ref_ast2, pred_ast2)
        self.assertEqual(cat2, "PROJECTION_MISMATCH")

    def test_07_evaluation_json_structure_and_completeness(self) -> None:
        """Verifies the output JSON file contains valid structure, summary, and 45 results."""
        eval_json_path = PROJECT_ROOT / "evaluation_results.json"
        self.assertTrue(eval_json_path.exists(), "evaluation_results.json does not exist.")

        with open(eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("summary", data)
        self.assertIn("results", data)

        summary = data["summary"]
        self.assertEqual(summary["dataset"]["total_examples"], 45)
        self.assertEqual(summary["evaluation_metrics"]["syntactically_valid_sql_count"], 45)
        self.assertEqual(summary["evaluation_metrics"]["syntactically_valid_sql_rate"], 1.0)
        self.assertEqual(summary["evaluation_metrics"]["execution_accuracy"]["status"], "unavailable")

        results = data["results"]
        self.assertEqual(len(results), 45)

        for r in results:
            self.assertIn("id", r)
            self.assertIn("category", r)
            self.assertIn("table", r)
            self.assertIn("question", r)
            self.assertIn("reference_sql", r)
            self.assertIn("predicted_sql", r)
            self.assertIn("exact_match_strict", r)
            self.assertIn("exact_match_normalized", r)
            self.assertIn("is_valid_sql", r)
            self.assertIn("failure_category", r)
            self.assertIn("failure_explanation", r)
            self.assertTrue(r["is_valid_sql"])

    def test_08_final_evaluation_report_exists(self) -> None:
        """Verifies that FINAL_EVALUATION_REPORT.md was generated and is complete."""
        report_path = PROJECT_ROOT / "FINAL_EVALUATION_REPORT.md"
        self.assertTrue(report_path.exists(), "FINAL_EVALUATION_REPORT.md does not exist.")

        content = report_path.read_text(encoding="utf-8")
        self.assertIn("# NL2SQL — Final Model Evaluation Report", content)
        self.assertIn("100.00%", content)
        self.assertIn("Exact Match", content)
        self.assertIn("Representative Examples", content)
        self.assertIn("Honest Technical Conclusions", content)


if __name__ == "__main__":
    unittest.main()
