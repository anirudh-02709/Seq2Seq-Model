"""
Verification Test Suite for Milestone 4, Task 4.1: Standalone Seq2Seq Inference.

Tests:
- Test A: Checkpoint restoration
- Test B: Known examples from held-out test split
- Test C: New (unseen) natural language questions
- Test D: Repeated inference on a single loaded model
- Test E: Comprehensive failure handling and edge cases
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

from src.data.dataset import load_split
from src.data.tokenizer import UNK_IDX, NL2SQLTokenizer
from src.data.vocabulary import Vocabulary
from src.inference.pipeline import (
    DEFAULT_STUDENT_SCHEMA,
    InferenceResult,
    NL2SQLInferencePipeline,
    load_inference_pipeline,
)
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer


class TestStandaloneInference(unittest.TestCase):
    """
    Test suite verifying the standalone inference pipeline for the trained Seq2Seq model.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        cls.assertTrue(
            cls.checkpoint_path.is_file(),
            f"Checkpoint file not found: {cls.checkpoint_path}",
        )
        cls.pipeline = NL2SQLInferencePipeline.from_checkpoint(
            checkpoint_path=str(cls.checkpoint_path),
            device="cpu",
        )

    # =========================================================================
    # Test A: Checkpoint Restoration
    # =========================================================================

    def test_01_checkpoint_restoration(self) -> None:
        """
        Test A: Verify model, parameters, vocabularies, config, and eval mode.
        """
        self.assertIsNotNone(self.pipeline)
        self.assertIsInstance(self.pipeline.model, Seq2Seq)
        self.assertFalse(self.pipeline.model.training, "Model must be in evaluation mode.")

        # Parameter checks
        trainable_params = sum(p.numel() for p in self.pipeline.model.parameters())
        self.assertGreater(trainable_params, 0)

        # Confirm exact shapes from checkpoint
        self.assertEqual(self.pipeline.model.source_vocab_size, 198)
        self.assertEqual(self.pipeline.model.target_vocab_size, 70)
        self.assertEqual(self.pipeline.model.hidden_dim, 128)
        self.assertEqual(self.pipeline.model.encoder.embedding_dim, 64)
        self.assertEqual(self.pipeline.model.decoder.embedding_dim, 64)
        self.assertEqual(self.pipeline.model.decoder.attention_dim, 64)

        # Check vocabulary restoration
        self.assertEqual(len(self.pipeline.src_vocab), 198)
        self.assertEqual(len(self.pipeline.tgt_vocab), 70)

        # Check execution device
        self.assertEqual(self.pipeline.device, torch.device("cpu"))
        for param in self.pipeline.model.parameters():
            self.assertEqual(param.device, torch.device("cpu"))

        # Verify checkpoint info
        self.assertEqual(self.pipeline.checkpoint_info["epoch"], 25)
        self.assertAlmostEqual(self.pipeline.checkpoint_info["val_loss"], 0.6279, places=3)
        self.assertGreater(self.pipeline.checkpoint_info["val_token_acc"], 0.80)

    # =========================================================================
    # Test B: Known Examples from Test Dataset
    # =========================================================================

    def test_02_known_test_examples(self) -> None:
        """
        Test B: Run representative examples from test split and compare against expectations.
        """
        test_records = load_split("test", data_dir=PROJECT_ROOT / "data")

        # Select 6 diverse examples across categories
        selected_ids = [256, 258, 259, 261, 281, 286]
        selected_records = [r for r in test_records if r["id"] in selected_ids]

        results = []
        for rec in selected_records:
            question = rec["question"]
            schema = rec["schema"]
            expected_sql = rec["sql"].strip()

            generated_sql = self.pipeline.generate_sql(question=question, schema=schema)
            self.assertIsInstance(generated_sql, str)
            self.assertTrue(len(generated_sql) > 0, "Generated SQL should not be empty.")

            results.append({
                "id": rec["id"],
                "category": rec["category"],
                "question": question,
                "schema": schema,
                "expected_sql": expected_sql,
                "generated_sql": generated_sql,
                "match": (generated_sql.lower() == expected_sql.lower()),
            })

        # Specifically verify record 259 which was an exact match in the evaluation suite
        rec_259 = next(r for r in results if r["id"] == 259)
        self.assertEqual(
            rec_259["generated_sql"],
            "SELECT enrollment_id, marks FROM enrollments",
        )
        self.assertTrue(rec_259["match"])

        # Ensure all generated queries start with SELECT
        for res in results:
            self.assertTrue(res["generated_sql"].startswith("SELECT"))

    # =========================================================================
    # Test C: New / Unseen Questions
    # =========================================================================

    def test_03_new_questions(self) -> None:
        """
        Test C: Test at least 5 new, realistic natural language questions not in training set.
        """
        new_test_queries = [
            {
                "question": "Show students with CGPA above 8",
                "schema": "students(id, name, department, cgpa)",
            },
            {
                "question": "Find all courses offered with 4 credits",
                "schema": "courses(course_id, course_name, credits)",
            },
            {
                "question": "List students who belong to the CSE department",
                "schema": "students(student_id, name, age, department, cgpa)",
            },
            {
                "question": "Display enrollments where marks are greater than 90",
                "schema": "enrollments(enrollment_id, student_id, course_id, semester, marks)",
            },
            {
                "question": "Retrieve the names of all courses ordered by credits",
                "schema": "courses(course_id, course_name, credits)",
            },
            {
                "question": "Show the top 5 students by CGPA",
                "schema": DEFAULT_STUDENT_SCHEMA,
            },
        ]

        predictions = []
        for item in new_test_queries:
            result = self.pipeline.predict(
                question=item["question"],
                schema=item["schema"],
            )
            self.assertIsInstance(result, InferenceResult)
            self.assertTrue(len(result.sql) > 0)
            self.assertTrue(result.sql.startswith("SELECT"))
            # Confirm syntax validity check
            self.assertTrue(result.is_valid_sql, f"Generated SQL is invalid MySQL: {result.syntax_error}")
            predictions.append({
                "question": item["question"],
                "schema": item["schema"],
                "generated_sql": result.sql,
                "valid": result.is_valid_sql,
            })

        self.assertGreaterEqual(len(predictions), 5)

    # =========================================================================
    # Test D: Repeated Inference
    # =========================================================================

    def test_04_repeated_inference(self) -> None:
        """
        Test D: Process multiple questions in succession using the same loaded instance.
        """
        questions = [
            "List all students",
            "Show all courses",
            "Retrieve all enrollment marks",
            "List all students",  # Determinism check
        ]

        outputs = []
        for q in questions:
            sql = self.pipeline.generate_sql(q)
            self.assertTrue(sql.startswith("SELECT"))
            outputs.append(sql)

        # Check determinism: identical question produces identical SQL
        self.assertEqual(outputs[0], outputs[3])
        self.assertEqual(len(outputs), 4)

    # =========================================================================
    # Test E: Failure Handling and Edge Cases
    # =========================================================================

    def test_05_missing_checkpoint(self) -> None:
        """Missing checkpoint path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            NL2SQLInferencePipeline.from_checkpoint("checkpoints/non_existent_file.pt")

    def test_06_missing_vocabulary(self) -> None:
        """Checkpoint missing vocabulary dictionaries must raise ValueError."""
        # Create a temporary dummy checkpoint without src_vocab / tgt_vocab
        dummy_state = {
            "epoch": 1,
            "model_state_dict": self.pipeline.model.state_dict(),
            "config": self.pipeline.config,
        }
        dummy_path = PROJECT_ROOT / "checkpoints" / "temp_dummy_no_vocab.pt"
        torch.save(dummy_state, dummy_path)

        try:
            with self.assertRaises(ValueError) as ctx:
                NL2SQLInferencePipeline.from_checkpoint(dummy_path)
            self.assertIn("missing required vocabulary artifacts", str(ctx.exception))
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    def test_07_empty_question(self) -> None:
        """Empty or whitespace-only question must raise ValueError."""
        with self.assertRaises(ValueError):
            self.pipeline.generate_sql(question="")

        with self.assertRaises(ValueError):
            self.pipeline.generate_sql(question="   \t\n  ")

    def test_08_invalid_question_type(self) -> None:
        """Non-string question must raise TypeError."""
        with self.assertRaises(TypeError):
            self.pipeline.generate_sql(question=123)  # type: ignore

        with self.assertRaises(TypeError):
            self.pipeline.generate_sql(question=None)  # type: ignore

    def test_09_empty_or_invalid_schema(self) -> None:
        """Explicit empty schema must raise ValueError; non-string must raise TypeError."""
        with self.assertRaises(ValueError):
            self.pipeline.generate_sql(question="List students", schema="")

        with self.assertRaises(ValueError):
            self.pipeline.generate_sql(question="List students", schema="   ")

        with self.assertRaises(TypeError):
            self.pipeline.generate_sql(question="List students", schema=12345)  # type: ignore

    def test_10_unknown_input_tokens(self) -> None:
        """
        Tokens not present in training vocabulary should map to UNK without crashing.
        """
        # "astronauts", "galactic", "quantum" are completely unknown to the vocabulary
        res = self.pipeline.predict(
            question="Find all galactic astronauts studying quantum computing",
            schema="students(student_id, name, department)",
        )
        self.assertGreater(res.source_unk_count, 0)
        self.assertIsInstance(res.sql, str)
        self.assertTrue(len(res.sql) > 0)
        self.assertTrue(res.sql.startswith("SELECT"))

    def test_11_max_generation_length(self) -> None:
        """
        Generation length must be constrained by max_length, and invalid values must raise ValueError.
        """
        # Invalid max_length
        with self.assertRaises(ValueError):
            self.pipeline.generate_sql("List all students", max_length=0)

        with self.assertRaises(ValueError):
            self.pipeline.generate_sql("List all students", max_length=-5)

        # Constrained max_length
        sql_short = self.pipeline.generate_sql("List all students", max_length=3)
        tokens_short = self.pipeline.tokenizer.tokenize_sql(sql_short)
        self.assertLessEqual(len(tokens_short), 3)

    def test_12_model_direct_generate_sql(self) -> None:
        """
        Verifies that model.generate_sql(...) works directly on the restored Seq2Seq model.
        """
        sql = self.pipeline.model.generate_sql(
            question="Show students with CGPA above 8",
            schema="students(id, name, department, cgpa)",
        )
        self.assertIsInstance(sql, str)
        self.assertTrue(sql.startswith("SELECT"))


def run_tests() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStandaloneInference)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
