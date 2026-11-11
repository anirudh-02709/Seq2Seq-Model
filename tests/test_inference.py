"""Consolidated tests for NL2SQL inference pipeline and schema-aware constrained decoding."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.tokenizer import EOS_TOKEN, PAD_TOKEN, SEP_TOKEN, SOS_TOKEN, UNK_TOKEN
from src.data.vocabulary import Vocabulary
from src.inference.constrained_decoder import build_schema_vocab_profile, get_allowed_token_ids
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA, InferenceResult, NL2SQLInferencePipeline
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.validation.validator import SQLValidator


def build_scripted_vocab() -> Vocabulary:
    tokens = [
        PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
        "SELECT", "FROM", "WHERE", "AND", "OR", "ORDER", "BY", "ASC", "DESC", "LIMIT",
        ",", "*", "=", "!=", "<", "<=", ">", ">=", "(", ")",
        "students", "courses", "enrollments",
        "student_id", "name", "age", "department", "cgpa",
        "course_id", "course_name", "credits",
        "enrollment_id", "semester", "marks",
        "students.name", "students.cgpa", "courses.course_name", "enrollments.marks",
        "teachers", "salary",
        "1", "2", "3", "4", "80.0", "8.0", "'CSE'", "'Semester 5'",
    ]
    vocab = Vocabulary()
    vocab.add_tokens(tokens)
    return vocab


def build_scripted_pipeline(policy_fn: Callable[[list[str]], dict[str, float]], **kwargs) -> NL2SQLInferencePipeline:
    """Builds a deterministic Seq2Seq pipeline whose decoder logits are controlled by prefix tokens."""
    vocab = build_scripted_vocab()
    enc = Encoder(vocab=vocab, embedding_dim=8, hidden_dim=32, dropout=0.0)
    orig_enc_forward = enc.forward

    def clean_enc_forward(src, src_lengths):
        outputs, hid = orig_enc_forward(src, src_lengths)
        hid = torch.zeros_like(hid)
        hid._prefix_history = ()
        return outputs, hid

    enc.forward = clean_enc_forward

    class _ScriptedDecoder(Decoder):
        def __init__(self) -> None:
            super().__init__(vocab=vocab, embedding_dim=8, hidden_dim=32, attention_dim=8, dropout=0.0)

        def forward(self, input_token, hidden, encoder_outputs, src_lengths=None, mask=None):
            batch_size = input_token.size(0)
            seq_len = encoder_outputs.size(1)
            prefix_ids = list(getattr(hidden, "_prefix_history", ()))
            in_id = int(input_token[0].item())
            if in_id != vocab.sos_idx:
                prefix_ids.append(in_id)

            logits = torch.full((batch_size, len(vocab)), -50.0, dtype=torch.float32)
            for tok, val in policy_fn([vocab.idx2token[i] for i in prefix_ids]).items():
                if tok in vocab.token2idx:
                    logits[0, vocab.token2idx[tok]] = float(val)

            new_hidden = hidden.clone()
            new_hidden._prefix_history = tuple(prefix_ids)
            return logits, new_hidden, torch.full((batch_size, seq_len), 1.0 / seq_len, dtype=torch.float32)

    model = Seq2Seq(
        encoder=enc,
        decoder=_ScriptedDecoder(),
        pad_idx=vocab.pad_idx,
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
        unk_idx=vocab.unk_idx,
    )
    return NL2SQLInferencePipeline(
        model=model,
        src_vocab=vocab,
        tgt_vocab=vocab,
        device="cpu",
        default_schema=DEFAULT_STUDENT_SCHEMA,
        **kwargs,
    )


class TestInferenceAndConstrainedDecoding(unittest.TestCase):
    """Covers official checkpoint loading, deterministic decoding, and schema constraints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint_path = PROJECT_ROOT / "checkpoints" / "candidates" / "phase2_clean_best.pt"
        cls.pipeline = NL2SQLInferencePipeline.from_checkpoint(device="cpu")

    def test_official_default_checkpoint_loads_and_restores_vocabs(self) -> None:
        self.assertTrue(self.checkpoint_path.is_file())
        self.assertEqual(Path(self.pipeline.checkpoint_info["checkpoint_path"]).resolve(), self.checkpoint_path.resolve())
        self.assertFalse(self.pipeline.model.training)
        self.assertEqual(len(self.pipeline.src_vocab), 290)
        self.assertEqual(len(self.pipeline.tgt_vocab), 96)
        self.assertEqual(self.pipeline.device, torch.device("cpu"))
        self.assertEqual(self.pipeline.checkpoint_info["epoch"], 42)
        for p in self.pipeline.model.parameters():
            self.assertEqual(p.device.type, "cpu")

    def test_valid_sql_generation_determinism_and_unk_handling(self) -> None:
        res1 = self.pipeline.predict(
            "Show student ID, full name, and CGPA for all students.",
            decoding_mode="schema_aware",
            beam_width=5,
        )
        res2 = self.pipeline.predict(
            "Show student ID, full name, and CGPA for all students.",
            decoding_mode="schema_aware",
            beam_width=5,
        )
        self.assertIsInstance(res1, InferenceResult)
        self.assertEqual(res1.sql, "SELECT student_id, name, cgpa FROM students")
        self.assertEqual(res1.sql, res2.sql)
        self.assertTrue(res1.is_valid_sql)
        self.assertTrue(res1.is_schema_valid)

        # Out-of-vocab question words map to UNK without crashing
        unk_res = self.pipeline.predict(
            "Find galactic astronauts studying quantum mechanics",
            schema="students(student_id, name, department)",
        )
        self.assertGreater(unk_res.source_unk_count, 0)
        self.assertTrue(unk_res.sql.startswith("SELECT"))

    def test_checkpoint_and_input_validation_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            NL2SQLInferencePipeline.from_checkpoint("checkpoints/non_existent_checkpoint.pt")

        with tempfile.TemporaryDirectory() as td:
            bad_chk = Path(td) / "missing_vocab.pt"
            torch.save({"epoch": 1, "model_state_dict": self.pipeline.model.state_dict(), "config": self.pipeline.config}, bad_chk)
            with self.assertRaises(ValueError):
                NL2SQLInferencePipeline.from_checkpoint(bad_chk)

        for bad_call, exc in [
            (lambda: self.pipeline.generate_sql(""), ValueError),
            (lambda: self.pipeline.generate_sql("   \t\n "), ValueError),
            (lambda: self.pipeline.generate_sql(123), TypeError),  # type: ignore[arg-type]
            (lambda: self.pipeline.generate_sql("List students", schema="   "), ValueError),
            (lambda: self.pipeline.generate_sql("List students", schema=123), TypeError),  # type: ignore[arg-type]
            (lambda: self.pipeline.generate_sql("List students", max_length=0), ValueError),
            (lambda: self.pipeline.predict("List students", length_penalty_alpha=-0.2), ValueError),
        ]:
            with self.assertRaises(exc):
                bad_call()

        short_sql = self.pipeline.generate_sql("List all students", max_length=3, decoding_mode="greedy")
        self.assertLessEqual(len(self.pipeline.tokenizer.tokenize_sql(short_sql)), 3)

    def test_official_checkpoint_greedy_vs_schema_aware_rescue(self) -> None:
        query = "List the top 2 courses ordered alphabetically by name."
        greedy_res = self.pipeline.predict(query, decoding_mode="greedy")
        self.assertEqual(greedy_res.decoding_mode, "greedy")
        self.assertEqual(greedy_res.sql, "SELECT * FROM courses ORDER BY name ASC")
        self.assertFalse(greedy_res.is_schema_valid)

        aware_res = self.pipeline.predict(query, decoding_mode="schema_aware", beam_width=5)
        self.assertEqual(aware_res.decoding_mode, "schema_aware")
        self.assertEqual(aware_res.sql, "SELECT * FROM courses ORDER BY course_name ASC")
        self.assertTrue(aware_res.is_schema_valid)
        self.assertFalse(aware_res.no_valid_candidate)

    def test_schema_vocab_profile_ownership_rules(self) -> None:
        vocab = build_scripted_vocab()
        profile = build_schema_vocab_profile(vocab, DEFAULT_STUDENT_SCHEMA)
        self.assertIsNotNone(profile)
        assert profile is not None

        allowed_courses_where = {vocab.idx2token[i] for i in get_allowed_token_ids(["SELECT", "*", "FROM", "courses", "WHERE"], profile)}
        self.assertEqual(allowed_courses_where, {"course_id", "course_name", "credits", "courses.course_name", "("})

        allowed_enroll_where = {vocab.idx2token[i] for i in get_allowed_token_ids(["SELECT", "*", "FROM", "enrollments", "WHERE"], profile)}
        self.assertIn("student_id", allowed_enroll_where)
        self.assertNotIn("department", allowed_enroll_where)

        allowed_students_order = {vocab.idx2token[i] for i in get_allowed_token_ids(["SELECT", "*", "FROM", "students", "ORDER", "BY"], profile)}
        self.assertIn("cgpa", allowed_students_order)
        self.assertNotIn("marks", allowed_students_order)
        self.assertNotIn("teachers", allowed_students_order)

        allowed_second_col = {vocab.idx2token[i] for i in get_allowed_token_ids(["SELECT", "name", ","], profile)}
        self.assertIn("cgpa", allowed_second_col)
        self.assertNotIn("name", allowed_second_col)
        self.assertNotIn("credits", allowed_second_col)

        allowed_from = {vocab.idx2token[i] for i in get_allowed_token_ids(["SELECT", "student_id", "FROM"], profile)}
        self.assertEqual(allowed_from, {"students", "enrollments"})

    def test_cross_table_columns_blocked_in_where_and_order_by(self) -> None:
        def order_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"*": 10.0}
            if toks == ["SELECT", "*"]:
                return {"FROM": 10.0}
            if toks[-1] == "FROM":
                return {"courses": 10.0}
            if toks[-1] == "courses":
                return {"ORDER": 10.0}
            if toks[-1] == "ORDER":
                return {"BY": 10.0}
            if toks[-1] == "BY":
                return {"name": 5.0, "course_name": 4.0}
            if toks[-1] in ("name", "course_name"):
                return {"ASC": 10.0}
            if toks[-1] == "ASC":
                return {EOS_TOKEN: 10.0}
            return {EOS_TOKEN: 0.0}

        pipe = build_scripted_pipeline(order_policy, decoding_mode="schema_aware", beam_width=5)
        self.assertEqual(pipe.generate_sql("Order courses by name", decoding_mode="greedy"), "SELECT * FROM courses ORDER BY name ASC")
        res = pipe.predict("Order courses by name", decoding_mode="schema_aware")
        self.assertEqual(res.sql, "SELECT * FROM courses ORDER BY course_name ASC")
        self.assertTrue(res.is_schema_valid)

        def where_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"*": 10.0}
            if toks == ["SELECT", "*"]:
                return {"FROM": 10.0}
            if toks[-1] == "FROM":
                return {"enrollments": 10.0}
            if toks[-1] == "enrollments":
                return {"WHERE": 10.0}
            if toks[-1] == "WHERE":
                return {"marks": 10.0}
            if toks[-1] == "marks":
                return {"=": 10.0}
            if toks[-2:] == ["marks", "="]:
                return {"80.0": 10.0}
            if toks[-1] == "80.0":
                return {"OR": 10.0}
            if toks[-1] == "OR":
                return {"department": 5.0, "semester": 4.0}
            if toks[-1] in ("department", "semester"):
                return {"=": 10.0}
            if toks[-2:] in (["department", "="], ["semester", "="]):
                return {"'Semester 5'": 5.0, "'CSE'": 4.0}
            if toks[-1] in ("'Semester 5'", "'CSE'"):
                return {EOS_TOKEN: 10.0}
            return {EOS_TOKEN: 0.0}

        wpipe = build_scripted_pipeline(where_policy, decoding_mode="schema_aware", beam_width=5)
        self.assertIn("department", wpipe.generate_sql("Enrollments filter", decoding_mode="greedy"))
        wres = wpipe.predict("Enrollments filter", decoding_mode="schema_aware")
        self.assertEqual(wres.sql, "SELECT * FROM enrollments WHERE marks = 80.0 OR semester = 'Semester 5'")
        self.assertTrue(wres.is_schema_valid)

    def test_select_before_from_pruning_and_qualified_columns(self) -> None:
        def select_from_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"course_name": 2.0, "*": 1.8}
            if toks[-1] in ("course_name", "*"):
                return {"FROM": 10.0}
            if toks[-1] == "FROM":
                return {"enrollments": 5.0, "courses": 0.1}
            if toks[-1] in ("enrollments", "courses"):
                return {"WHERE": 10.0}
            if toks[-1] == "WHERE":
                return {"marks": 5.0, "credits": 1.0}
            if toks[-1] in ("marks", "credits"):
                return {">": 10.0}
            if toks[-1] == ">":
                return {"80.0": 5.0, "3": 2.0}
            if toks[-1] in ("80.0", "3"):
                return {EOS_TOKEN: 10.0}
            return {EOS_TOKEN: 0.0}

        pipe = build_scripted_pipeline(select_from_policy, decoding_mode="schema_aware", beam_width=5)
        self.assertFalse(SQLValidator(schema=DEFAULT_STUDENT_SCHEMA).validate(pipe.generate_sql("q", decoding_mode="greedy")).is_valid)
        res = pipe.predict("q", decoding_mode="schema_aware", beam_width=5)
        self.assertEqual(res.sql, "SELECT * FROM enrollments WHERE marks > 80.0")
        self.assertTrue(res.is_schema_valid)

        def qualified_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"students.name": 10.0}
            if toks[-1] == "students.name":
                return {"FROM": 10.0}
            if toks[-1] == "FROM":
                return {"courses": 8.0, "students": 4.0}
            if toks[-1] in ("students", "courses"):
                return {"WHERE": 10.0}
            if toks[-1] == "WHERE":
                return {"enrollments.marks": 8.0, "students.cgpa": 4.0}
            if toks[-1] in ("enrollments.marks", "students.cgpa"):
                return {">": 10.0}
            if toks[-1] == ">":
                return {"8.0": 10.0}
            if toks[-1] == "8.0":
                return {EOS_TOKEN: 10.0}
            return {EOS_TOKEN: 0.0}

        qpipe = build_scripted_pipeline(qualified_policy, decoding_mode="schema_aware", beam_width=5)
        qres = qpipe.predict("Qualified columns", decoding_mode="schema_aware")
        self.assertEqual(qres.sql, "SELECT students.name FROM students WHERE students.cgpa > 8.0")
        self.assertTrue(qres.is_schema_valid)

    def test_beam_ranking_length_penalty_and_no_valid_candidate_rejection(self) -> None:
        def rank_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"*": 5.0, "name": 4.9}
            if toks[-1] in ("*", "name"):
                return {"FROM": 10.0}
            if toks[-1] == "FROM":
                return {"students": 10.0}
            if toks == ["SELECT", "name", "FROM", "students"]:
                return {EOS_TOKEN: 10.0}
            if toks == ["SELECT", "*", "FROM", "students"]:
                return {EOS_TOKEN: 1.0, "WHERE": 0.9}
            if toks[-1] == "WHERE":
                return {"cgpa": 10.0}
            if toks[-1] == "cgpa":
                return {">": 10.0}
            if toks[-1] == ">":
                return {"8.0": 10.0}
            if toks[-1] == "8.0":
                return {EOS_TOKEN: 10.0}
            return {EOS_TOKEN: 0.0}

        pipe = build_scripted_pipeline(rank_policy, decoding_mode="schema_aware", beam_width=5)
        # Higher sequence log-prob candidate wins over greedy initial token '*'
        ranked = pipe.predict("Rank valid candidates", length_penalty_alpha=0.0)
        self.assertTrue(ranked.is_schema_valid)
        self.assertGreaterEqual(ranked.candidates_examined, 2)

        # Length normalization flips preference toward longer WHERE candidate when EOS vs WHERE are close
        def len_policy(toks: list[str]) -> dict[str, float]:
            if toks == []:
                return {"SELECT": 10.0}
            if toks == ["SELECT"]:
                return {"*": 10.0}
            if toks == ["SELECT", "*"]:
                return {"FROM": 10.0}
            if toks == ["SELECT", "*", "FROM"]:
                return {"students": 10.0}
            if toks == ["SELECT", "*", "FROM", "students"]:
                return {EOS_TOKEN: 1.0, "WHERE": 0.9}
            return rank_policy(toks)

        lpipe = build_scripted_pipeline(len_policy, decoding_mode="schema_aware", beam_width=4)
        self.assertEqual(lpipe.predict("cgpa > 8.0", length_penalty_alpha=0.0).sql, "SELECT * FROM students")
        self.assertEqual(lpipe.predict("cgpa > 8.0", length_penalty_alpha=0.6).sql, "SELECT * FROM students WHERE cgpa > 8.0")

        # All candidates invalid under unmatched schema or insufficient max_length -> clean rejection
        res_unmatched = lpipe.predict("List professors", schema="professors(prof_id, building)")
        self.assertFalse(res_unmatched.is_schema_valid)
        self.assertTrue(res_unmatched.no_valid_candidate)
        self.assertIsNotNone(res_unmatched.validation_error)

        res_short = lpipe.predict("List students", max_length=2)
        self.assertFalse(res_short.is_schema_valid)
        self.assertTrue(res_short.no_valid_candidate)


if __name__ == "__main__":
    unittest.main()
