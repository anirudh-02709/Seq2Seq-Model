"""Consolidated tests for dataset splits, preprocessing, v1 validation, and Anti-Leakage Gate v2.0."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts import validate_dataset
from scripts.validate_candidate_dataset import fetch_live_db_catalogs, main as validate_candidate_main
from src.data.anti_leakage import (
    AntiLeakageConfig,
    AntiLeakageGate,
    _extract_projection_nl_span,
    extract_contextual_literals,
    mask_sql_literals,
)
from src.data.dataset import (
    NL2SQLCollator,
    NL2SQLDataset,
    create_all_dataloaders,
    load_all_splits,
    load_jsonl_records,
)
from src.data.tokenizer import EOS_IDX, PAD_IDX, SEP_TOKEN, SOS_IDX, UNK_IDX, NL2SQLTokenizer, detokenize_sql
from src.data.vocabulary import build_vocabularies
from src.database.config import DatabaseConfig
from src.database.executor import ExecutionResult, SQLExecutor
from src.inference.pipeline import DEFAULT_STUDENT_SCHEMA
from src.validation.validator import SQLValidator

CATALOG_STRINGS = {
    "students.department": {"AI&DS", "CSE", "ECE", "ISE", "ME"},
    "courses.course_name": {
        "Database Management Systems",
        "Operating Systems",
        "Computer Networks",
        "Machine Learning",
        "Software Engineering",
        "Design and Analysis of Algorithms",
        "Object Oriented Programming with Java",
        "Cloud Computing",
    },
    "enrollments.semester": {"Semester 5", "Semester 6"},
}

ID_CATALOG = {
    ("students", "student_id"): set(range(1, 21)),
    ("courses", "course_id"): set(range(1, 9)),
    ("enrollments", "enrollment_id"): set(range(1, 50)),
    ("enrollments", "student_id"): set(range(1, 21)),
    ("enrollments", "course_id"): set(range(1, 9)),
}

OBSERVED_NUMERIC_VALUES: Dict[Tuple[str, str], Set[str]] = {
    ("students", "age"): {"20", "21", "22"},
    ("students", "cgpa"): {"6.85", "6.95", "7.8", "8.05", "8.25", "8.3", "8.75", "8.92", "9.05", "9.15", "9.4"},
    ("courses", "credits"): {"3", "4"},
    ("enrollments", "marks"): {"67.5", "70.0", "72.0", "82.5", "87.5", "92.0", "94.0", "96.0"},
}

LIVE_METRIC_BOUNDS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("students", "age"): (20.0, 22.0),
    ("students", "cgpa"): (6.85, 9.40),
    ("courses", "credits"): (3.0, 4.0),
    ("enrollments", "marks"): (67.50, 96.00),
}

TABLE_ROW_COUNTS = {"students": 20, "courses": 8, "enrollments": 49}


class FakeSQLExecutor:
    """Deterministic in-memory mock executor matching SQLExecutor.execute."""

    def __init__(self, empty_sql_substrings: tuple[str, ...] = ()) -> None:
        self.empty_sql_substrings = empty_sql_substrings

    def execute(self, validation_result: Any, approved: bool = False, max_rows: int = 500) -> ExecutionResult:
        sql = getattr(validation_result, "normalized_sql", "") or ""
        if any(sub in sql for sub in self.empty_sql_substrings):
            return ExecutionResult(success=True, executed_sql=sql, columns=["c"], rows=[], row_count=0)
        return ExecutionResult(success=True, executed_sql=sql, columns=["c"], rows=[{"c": 1}], row_count=1)


def _rec(rid: int, q: str, sql: str, tbl: str = "students", cat: str = "WHERE and comparisons") -> Dict[str, Any]:
    return {"id": rid, "schema": DEFAULT_STUDENT_SCHEMA, "question": q, "sql": sql, "category": cat, "table": tbl}


def _lit_entry(
    lid: str,
    val: Any,
    dtype: str,
    tbl: str,
    col: str | None,
    ctx: str,
    rule: str,
    obs: bool,
    dom: Dict[str, Any],
    src: str,
    ex_ids: List[int],
) -> Dict[str, Any]:
    return {
        "literal_id": lid,
        "value": val,
        "data_type": dtype,
        "table": tbl,
        "column": col,
        "sql_context": ctx,
        "generation_rule": rule,
        "is_observed_in_db": obs,
        "approved_domain": dom,
        "domain_source": src,
        "example_ids": ex_ids,
    }


def _build_examples(
    new_records: List[Dict[str, Any]],
    zero_row_ids: Iterable[int] = (),
    reviewed_zero: Optional[Dict[str, Any]] = None,
    reviewed_proj: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    zero_set = set(zero_row_ids)
    rzero = reviewed_zero or {}
    rproj = reviewed_proj or {}
    out: List[Dict[str, Any]] = []
    for r in new_records:
        rid = r["id"]
        tbl = (r.get("table") or "").lower()
        try:
            occs = extract_contextual_literals(r["sql"])
            tmpl = mask_sql_literals(r["sql"])
        except Exception:
            occs, tmpl = [], r["sql"]
        lits_used = []
        for occ in occs:
            ctx, col, val = occ["sql_context"], occ["column"], occ["value"]
            if ctx == "closed_domain_string":
                obs = str(val) in CATALOG_STRINGS.get(f"{tbl}.{col or ''}", set())
            elif ctx == "identifier":
                obs = int(val) in ID_CATALOG.get((tbl, col or ""), set())
            elif ctx == "limit":
                obs = int(val) <= TABLE_ROW_COUNTS.get(tbl, 0)
            else:
                obs = Decimal(str(val)) in {Decimal(x) for x in OBSERVED_NUMERIC_VALUES.get((tbl, col or ""), set())}
            lits_used.append({"value": val, "column": col, "sql_context": ctx, "is_observed_in_db": obs})
        is_zero = rid in zero_set
        z_info = rzero.get(str(rid), {})
        p_info = next((v for k, v in rproj.items() if k == str(rid) or k.startswith(f"{rid}:")), None)
        ex: Dict[str, Any] = {
            "id": rid,
            "question": r["question"],
            "sql": r["sql"],
            "source_method": "unit_test_fixture",
            "table": tbl,
            "columns_referenced": sorted({occ["column"] for occ in occs if occ.get("column")}),
            "literals_used": lits_used,
            "executed": True,
            "execution_classification": "valid_zero_rows" if is_zero else "valid_with_rows",
            "row_count": 0 if is_zero else 1,
            "ast_template": tmpl,
            "overlap_status": "clean",
            "projection_review_status": "approved" if p_info else "clean",
        }
        info = z_info if (is_zero and isinstance(z_info, dict)) else (p_info if isinstance(p_info, dict) else None)
        if info:
            ex["human_disposition"] = info.get("disposition")
            ex["human_rationale"] = info.get("rationale")
        out.append(ex)
    return out


def _manifest(
    literals: List[Dict[str, Any]],
    reviewed: Dict[str, Any] | None = None,
    new_records: Optional[List[Dict[str, Any]]] = None,
    reviewed_proj: Optional[Dict[str, Any]] = None,
    reviewed_zero: Optional[Dict[str, Any]] = None,
    zero_row_ids: Iterable[int] = (),
    schema_version: str = "2.0",
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "schema_version": schema_version,
        "policy_document": "data/candidates/phase2_step7b_projection_policy.md",
        "generation_rules": {
            "comparison_threshold": "deterministic midpoint/metric rules",
            "identifier": "observed catalog PK/FK keys",
            "count": "out of current model scope",
            "limit": "deterministic approved LIMIT set",
            "closed_domain_string": "live MySQL catalog column enumeration",
        },
        "literals": literals,
        "reviewed_soft_flags": reviewed or {},
    }
    if schema_version == "2.0":
        doc["examples"] = _build_examples(new_records or [], zero_row_ids=zero_row_ids, reviewed_zero=reviewed_zero, reviewed_proj=reviewed_proj)
        doc["reviewed_projection_flags"] = reviewed_proj or {}
        doc["reviewed_zero_row_queries"] = reviewed_zero or {}
    return doc


class TestDatasetPreprocessing(unittest.TestCase):
    """Verifies split counts, train-only vocabularies, special tokens, and DataLoader batching."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = PROJECT_ROOT / "data"
        cls.train_records, cls.val_records, cls.test_records = load_all_splits(cls.data_dir)
        cls.tokenizer = NL2SQLTokenizer()
        cls.src_vocab, cls.tgt_vocab = build_vocabularies(cls.train_records, tokenizer=cls.tokenizer)

    def test_splits_counts_fields_and_test_id_isolation(self) -> None:
        self.assertEqual((len(self.train_records), len(self.val_records), len(self.test_records)), (210, 45, 45))
        self.assertEqual([r["id"] for r in self.test_records], list(range(256, 301)))
        required = {"id", "schema", "question", "sql", "category", "table"}
        for rec in self.train_records + self.val_records + self.test_records:
            self.assertTrue(required.issubset(rec.keys()))

    def test_train_only_vocabulary_and_unk_mapping(self) -> None:
        self.assertEqual((PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX), (0, 1, 2, 3))
        for tok in ("72.0", "9.2", "15", "92.0", "8.8"):
            self.assertNotIn(tok, self.tgt_vocab)
        for word in ("carrying", "profiles", "exceeds", "awarded", "studying"):
            self.assertNotIn(word, self.src_vocab)
        self.assertEqual(self.tgt_vocab["nonexistent_xyz"], UNK_IDX)

        val_ds = NL2SQLDataset(self.val_records, src_vocab=self.src_vocab, tgt_vocab=self.tgt_vocab, tokenizer=self.tokenizer)
        self.assertTrue(any(UNK_IDX in item["decoder_target_ids"].tolist() for item in val_ds))

    def test_special_tokens_collation_and_dataloader_roundtrip(self) -> None:
        ds = NL2SQLDataset(self.train_records, src_vocab=self.src_vocab, tgt_vocab=self.tgt_vocab, tokenizer=self.tokenizer)
        sample = ds[5]
        self.assertIsNot(self.src_vocab, self.tgt_vocab)
        self.assertIn(SEP_TOKEN, sample["encoder_tokens"])
        self.assertEqual((sample["decoder_target_ids"][0].item(), sample["decoder_target_ids"][-1].item()), (SOS_IDX, EOS_IDX))
        self.assertNotIn(sample["encoder_input_ids"][0].item(), (SOS_IDX, EOS_IDX))

        batch = NL2SQLCollator(pad_idx=PAD_IDX, batch_first=True)([ds[0], ds[25]])
        max_dec = max(ds[0]["decoder_length"], ds[25]["decoder_length"])
        self.assertEqual(batch["decoder_target_ids"].shape, (2, max_dec))
        shorter = 0 if ds[0]["decoder_length"] < max_dec else 1
        pad_start = min(ds[0]["decoder_length"], ds[25]["decoder_length"])
        self.assertTrue(torch.all(batch["decoder_target_ids"][shorter, pad_start:] == PAD_IDX))
        self.assertTrue(torch.all(batch["decoder_mask"][shorter, pad_start:] == 0))

        tr_loader, val_loader, te_loader, _, tgt_v = create_all_dataloaders(data_dir=self.data_dir, batch_size=16, num_workers=0)
        self.assertEqual((len(tr_loader.dataset), len(val_loader.dataset), len(te_loader.dataset)), (210, 45, 45))
        first_batch = next(iter(tr_loader))
        self.assertTrue(detokenize_sql(tgt_v.to_tokens(first_batch["decoder_target_ids"][0].tolist(), remove_special=True)).startswith("SELECT"))


class TestDatasetV1Validation(unittest.TestCase):
    """Covers v1 AST allowlist, manifest validation, schema parser, and credential hardening."""

    def test_ast_allowlist_and_execution_gating(self) -> None:
        rec = lambda sql, tbl="students", cat="Basic SELECT and projection": {
            "id": 1, "sql": sql, "table": tbl, "category": cat, "_meta": {"filepath": "t.jsonl", "line_idx": 1}
        }
        self.assertTrue(validate_dataset.validate_sql_with_ast_allowlist(rec("SELECT student_id, marks FROM enrollments WHERE course_id = 1 ORDER BY marks DESC LIMIT 5", "enrollments", "Combined SQL patterns"))[0])
        for bad_sql in [
            "SELECT SLEEP(5) FROM students",
            "SELECT * FROM students FOR UPDATE",
            "SELECT * FROM students INTO OUTFILE '/tmp/out.txt'",
            "SELECT @var := 1 FROM students",
            "SELECT DISTINCT department FROM students",
            "SELECT age + 1 FROM students",
            "SELECT * FROM students WHERE name LIKE 'A%'",
            "SELECT * FROM students; DROP TABLE students",
            "SELECT * FROM students -- comment",
        ]:
            with self.subTest(bad_sql=bad_sql):
                self.assertFalse(validate_dataset.validate_sql_with_ast_allowlist(rec(bad_sql))[0])

    def test_manifest_schema_parser_and_credential_injection_guards(self) -> None:
        stats = {
            "total_examples": 300,
            "split_counts": {"train": 210, "validation": 45, "test": 45},
            "category_counts": validate_dataset.PLANNED_CATEGORY_TOTALS,
            "table_counts": validate_dataset.PLANNED_TABLE_TOTALS,
        }
        bad_manifests = [
            ["not_dict"],
            {
                "dataset_name": "T", "description": "D", "schema": validate_dataset.CANONICAL_SCHEMA_STR,
                "total_examples": True, "split_counts": stats["split_counts"],
                "category_distribution": stats["category_counts"], "table_distribution": stats["table_counts"],
                "supported_sql_features": ["SELECT"],
            },
        ]
        for payload in bad_manifests:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump(payload, f)
                tmp = f.name
            try:
                errs: list[str] = []
                validate_dataset.validate_manifest(tmp, stats, errs)
                self.assertGreater(len(errs), 0)
            finally:
                os.remove(tmp)

        self.assertTrue(validate_dataset.parse_and_validate_schema_string(validate_dataset.CANONICAL_SCHEMA_STR)[0])
        for bad_schema in [
            "students(student_id, name, age, department, cgpa) | students(student_id, name, age, department, cgpa)",
            "students(student_id, name, age, department, department, cgpa) | courses(course_id, course_name, credits) | enrollments(enrollment_id, student_id, course_id, semester, marks)",
            "students(student_id, name, age, department, cgpa) | professors(id, name)",
            "students(student_id, name) | courses(course_id, course_name, credits) | enrollments(enrollment_id, student_id, course_id, semester, marks)",
        ]:
            self.assertFalse(validate_dataset.parse_and_validate_schema_string(bad_schema)[0])

        os.environ.update({"MYSQL_USER": "user\n[mysqld]", "MYSQL_PASSWORD": "p", "MYSQL_HOST": "localhost", "MYSQL_PORT": "3306"})
        try:
            res = validate_dataset.validate_mysql([], 0, static_only=False)
            self.assertEqual(res["part_a_status"], "FAILED")
        finally:
            for k in ("MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_HOST", "MYSQL_PORT"):
                os.environ.pop(k, None)


class TestAntiLeakageGate(unittest.TestCase):
    """Covers AntiLeakageGate v2.0 overlap, provenance, projection alignment, and live DB enforcement."""

    def setUp(self) -> None:
        self.gate = AntiLeakageGate(AntiLeakageConfig(schema=DEFAULT_STUDENT_SCHEMA))
        self.orig = [
            _rec(1, "Show all students.", "SELECT * FROM students", "students", "Basic SELECT and projection"),
            _rec(2, "List student names and CGPAs.", "SELECT name, cgpa FROM students", "students", "Basic SELECT and projection"),
        ]
        self.heldout = {
            "validation": [
                _rec(214, "Show credit hours for all courses.", "SELECT credits FROM courses", "courses", "Basic SELECT and projection"),
                _rec(220, "List student IDs for CSE students with CGPA > 8.0.", "SELECT student_id FROM students WHERE department = 'CSE' AND cgpa > 8.0"),
            ],
            "test": [_rec(257, "Display course ID and title for 3-credit courses.", "SELECT course_id, course_name FROM courses WHERE credits = 3", "courses")],
            "benchmark": [
                _rec(1059, "List student ID, course ID, and marks for marks higher than 94.0.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks > 94.0", "enrollments"),
                _rec(1072, "Retrieve the student ID, name, age, and CGPA for all students.", "SELECT student_id, name, age, cgpa FROM students", "students", "Basic SELECT and projection"),
            ],
        }
        self.exec_ok = FakeSQLExecutor()

    def _run_gate(self, candidate, manifest, **kwargs):
        orig = kwargs.pop("orig", self.orig)
        orig_ids = {r["id"] for r in orig}
        if manifest is not None and manifest.get("schema_version") == "2.0" and manifest.get("examples") == []:
            manifest = dict(manifest)
            manifest["examples"] = _build_examples(
                [r for r in candidate if r["id"] not in orig_ids],
                reviewed_zero=manifest.get("reviewed_zero_row_queries"),
                reviewed_proj=manifest.get("reviewed_projection_flags"),
            )
        gate = AntiLeakageGate(kwargs.pop("cfg")) if "cfg" in kwargs else self.gate
        return gate.evaluate(
            candidate_records=candidate,
            original_train_records=orig,
            heldout_splits=kwargs.pop("heldout", self.heldout),
            literal_manifest=manifest,
            catalog_string_values=kwargs.pop("catalog_strings", CATALOG_STRINGS),
            identifier_catalog=kwargs.pop("id_catalog", ID_CATALOG),
            observed_numeric_values=kwargs.pop("observed_nums", OBSERVED_NUMERIC_VALUES),
            live_metric_bounds=kwargs.pop("live_bounds", LIVE_METRIC_BOUNDS),
            table_row_counts=kwargs.pop("row_counts", TABLE_ROW_COUNTS),
            executor=kwargs.pop("executor", self.exec_ok),
            db_unavailable_error=kwargs.pop("db_err", None),
        )

    def test_clean_disjoint_candidate_passes(self) -> None:
        cand = self.orig + [
            _rec(
                301,
                "Show student identifier, branch, and age for AI&DS majors with CGPA strictly above 7.25 ordered by age ascending limit 5.",
                "SELECT student_id, department, age FROM students WHERE department = 'AI&DS' AND cgpa > 7.25 ORDER BY age ASC LIMIT 5",
                "students",
                "Combined SQL patterns",
            )
        ]
        manifest = _manifest([
            _lit_entry("LIT-001", "AI&DS", "string", "students", "department", "closed_domain_string", "live catalog", True, {"values": sorted(CATALOG_STRINGS["students.department"])}, "students.department", [301]),
            _lit_entry("LIT-002", 7.25, "decimal", "students", "cgpa", "comparison_threshold", "cgpa midpoint", False, {"min": 6.85, "max": 9.40}, "students.cgpa", [301]),
            _lit_entry("LIT-003", 5, "integer", "students", None, "limit", "approved deterministic LIMIT set", True, {"values": [5]}, "LIMIT set", [301]),
        ])
        rep = self._run_gate(cand, manifest)
        self.assertTrue(rep.passed, msg=f"Unexpected failures: {rep.hard_failures}")
        self.assertEqual(rep.summary["gate_verdict"], "PASSED")

    def test_synthetic_contaminated_corpus_rejected(self) -> None:
        """Replaces historical skipped archive test: synthetic multi-leakage corpus must fail closed."""
        contaminated = self.orig + [
            _rec(301, "Display credits across catalog subjects.", "SELECT credits FROM courses", "courses"),
            _rec(302, "Output student IDs where GPA exceeds 8.0 and branch is CSE.", "SELECT student_id FROM students WHERE cgpa > 8.0 AND department = 'CSE'"),
            _rec(303, "Return enrollment ID and marks below 94.0.", "SELECT enrollment_id, marks FROM enrollments WHERE marks < 94.0", "enrollments"),
            _rec(304, "Show course ID and credits for Machine Learning.", "SELECT course_id, credits FROM courses WHERE course_name = 'Machine Learning'", "courses"),
            _rec(305, "List student ID, course ID, and marks for marks higher than 82.5.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks > 82.5", "enrollments"),
        ]
        cfg = AntiLeakageConfig(
            schema=DEFAULT_STUDENT_SCHEMA,
            require_literal_manifest=False,
            require_catalog_verification=False,
            require_live_execution=False,
            enforce_projection_policy=False,
        )
        rep = self._run_gate(contaminated, None, executor=None, cfg=cfg)
        self.assertFalse(rep.passed)
        self.assertEqual(rep.summary["gate_verdict"], "FAILED")
        checks = {f["check"] for f in rep.hard_failures}
        self.assertIn("canonical_sql_overlap_validation", checks)
        self.assertEqual(rep.summary["commutative_boolean_overlaps"]["validation"]["population_b_new"]["purely_commutative_count"], 1)
        self.assertIn("heldout_exclusive_numeric_literal_injection", checks)
        self.assertIn("excluded_heldout_literal_violation", checks)
        self.assertIn("hard_paraphrase_similarity", checks)

    def test_commutative_boolean_precedence_and_paraphrase_workflow(self) -> None:
        # Commutative reordering triggers commutative_boolean_overlap_validation hard failure
        cand_comm = self.orig + [
            _rec(301, "Output student IDs where GPA exceeds 8.0 and branch is Computer Science.", "SELECT student_id FROM students WHERE cgpa > 8.0 AND department = 'CSE'", "students", "AND / OR conditions")
        ]
        m_comm = _manifest([
            _lit_entry("L1", 8.0, "decimal", "students", "cgpa", "comparison_threshold", "grid", True, {"min": 6.85, "max": 9.40}, "db", [301]),
            _lit_entry("L2", "CSE", "string", "students", "department", "closed_domain_string", "catalog", True, {"values": sorted(CATALOG_STRINGS["students.department"])}, "db", [301]),
        ])
        rep_comm = self._run_gate(cand_comm, m_comm)
        self.assertFalse(rep_comm.passed)
        self.assertIn("commutative_boolean_overlap_validation", {f["check"] for f in rep_comm.hard_failures})

        # Precedence boundary: (A OR B) AND C != A OR (B AND C)
        ho_prec = {"validation": [_rec(299, "Filter A or B with C.", "SELECT student_id FROM students WHERE (department = 'CSE' OR department = 'ME') AND age = 20")], "test": [], "benchmark": []}
        cand_prec = self.orig + [_rec(301, "Retrieve student IDs with CSE or ME aged 20.", "SELECT student_id FROM students WHERE department = 'CSE' OR (department = 'ME' AND age = 20)")]
        m_prec = _manifest([
            _lit_entry("L1", "CSE", "string", "students", "department", "closed_domain_string", "catalog", True, {"values": ["CSE"]}, "db", [301]),
            _lit_entry("L2", "ME", "string", "students", "department", "closed_domain_string", "catalog", True, {"values": ["ME"]}, "db", [301]),
            _lit_entry("L3", 20, "integer", "students", "age", "comparison_threshold", "catalog", True, {"values": [20]}, "db", [301]),
        ])
        rep_prec = self._run_gate(cand_prec, m_prec, heldout=ho_prec)
        self.assertEqual(rep_prec.summary["commutative_boolean_overlaps"]["validation"]["population_b_new"]["count"], 0)

        # Soft paraphrase requires approved_non_leaking disposition + rationale
        cand_soft = self.orig + [_rec(301, "Print the student ID, name, age, and academic CGPA ordered by CGPA.", "SELECT student_id, name, age, cgpa FROM students ORDER BY cgpa DESC", "students", "ORDER BY")]
        rep_unrev = self._run_gate(cand_soft, _manifest([]))
        self.assertFalse(rep_unrev.passed)
        key = rep_unrev.review_flags[0]["key"]
        self.assertFalse(self._run_gate(cand_soft, _manifest([], {key: {"disposition": "rejected_requires_revision", "rationale": "Redo."}})).passed)
        self.assertTrue(self._run_gate(cand_soft, _manifest([], {key: {"disposition": "approved_non_leaking", "rationale": "Distinct ORDER BY semantics."}})).passed)

    def test_template_coverage_dilution_attack_and_limit_distinction(self) -> None:
        cols = ["student_id", "name", "age", "department", "cgpa"]
        ho_bench, matching_new = [], []
        for idx in range(25):
            c1, c2 = cols[idx % 5], cols[(idx + 1) % 5]
            if c1 == c2:
                c2 = "cgpa"
            op = [">", "<", ">=", "<=", "="][idx % 5]
            lim = idx + 1
            sql_ho = f"SELECT {c1} FROM students WHERE age {op} 20 ORDER BY student_id ASC LIMIT {lim}" if idx < 5 else f"SELECT {c1}, {c2} FROM students WHERE age {op} 20 LIMIT {lim}" if idx < 10 else f"SELECT {c1} FROM students WHERE cgpa {op} 7.4 ORDER BY age DESC LIMIT {lim}" if idx < 15 else f"SELECT {c1}, {c2} FROM students WHERE cgpa {op} 7.4 ORDER BY cgpa ASC LIMIT {lim}" if idx < 20 else f"SELECT {c1} FROM enrollments WHERE marks {op} 70.0 ORDER BY enrollment_id DESC LIMIT {lim}"
            sql_tr = sql_ho.replace("20", "21", 1).replace("7.4", "7.25").replace("70.0", "72.5")
            tbl = "enrollments" if idx >= 20 else "students"
            ho_bench.append(_rec(5000 + idx, f"Heldout {idx}?", sql_ho, tbl))
            matching_new.append(_rec(6000 + idx, f"Train {c1} {idx}.", sql_tr, tbl))
        for idx in range(25, 80):
            ho_bench.append(_rec(5000 + idx, f"Other {idx}?", f"SELECT enrollment_id, student_id, course_id FROM enrollments WHERE enrollment_id = {(idx % 49) + 1} LIMIT {(idx % 12) + 1}", "enrollments"))
        filler = [_rec(7000 + i, "List course identifiers and credits limit 3.", "SELECT course_id, credits FROM courses WHERE credits = 3", "courses") for i in range(75)]

        cfg = AntiLeakageConfig(schema=DEFAULT_STUDENT_SCHEMA, require_literal_manifest=False, require_catalog_verification=False, enforce_projection_policy=False)
        rep = self._run_gate(self.orig + matching_new + filler, None, heldout={"validation": [], "test": [], "benchmark": ho_bench}, cfg=cfg)
        self.assertFalse(rep.passed)
        self.assertIn("excessive_heldout_query_template_coverage_benchmark", {f["check"] for f in rep.hard_failures})

        # LIMIT distinguishes template from Benchmark 1072
        cand_lim = self.orig + [
            _rec(301, "Show student ID, name, age, and CGPA for 9 students.", "SELECT student_id, name, age, cgpa FROM students LIMIT 9", "students", "LIMIT"),
            _rec(302, "Display full name and age for learners aged at most 21.", "SELECT name, age FROM students WHERE age <= 21", "students"),
        ]
        m_lim = _manifest([
            _lit_entry("L1", 9, "integer", "students", None, "limit", "approved LIMIT set", True, {"values": [9]}, "LIMIT", [301]),
            _lit_entry("L2", 21, "integer", "students", "age", "comparison_threshold", "catalog", True, {"values": [20, 21, 22]}, "age", [302]),
        ])
        self.assertEqual(self._run_gate(cand_lim, m_lim).template_overlaps["benchmark"]["population_b_new"]["overlap_ratio"], 0.0)

    def test_literal_provenance_catalogs_and_baseline_protection(self) -> None:
        # Synthetic unobserved threshold in live bounds passes
        cand_synth = self.orig + [_rec(301, "Output student ID and age for CGPAs strictly under 7.25.", "SELECT student_id, age FROM students WHERE cgpa < 7.25")]
        m_synth = _manifest([_lit_entry("L1", 7.25, "decimal", "students", "cgpa", "comparison_threshold", "midpoint", False, {"min": 6.85, "max": 9.40}, "cgpa", [301])])
        self.assertTrue(self._run_gate(cand_synth, m_synth).passed)

        # Wrong observed flag, missing catalogs, wrong string column, count rule on identifier, LIMIT-as-threshold, and out-of-catalog FK all fail
        m_fake_obs = _manifest([_lit_entry("L1", 7.25, "decimal", "students", "cgpa", "comparison_threshold", "midpoint", True, {"min": 6.85, "max": 9.40}, "cgpa", [301])], new_records=[cand_synth[-1]])
        self.assertFalse(self._run_gate(cand_synth, m_fake_obs).passed)
        self.assertIn("missing_live_string_catalog", {f["check"] for f in self._run_gate(cand_synth, m_synth, catalog_strings=None).hard_failures})
        self.assertIn("missing_live_identifier_catalog", {f["check"] for f in self._run_gate(cand_synth, m_synth, id_catalog=None).hard_failures})
        self.assertIn("missing_live_observed_numeric_values", {f["check"] for f in self._run_gate(cand_synth, m_synth, observed_nums=None).hard_failures})

        cand_id_rule = self.orig + [_rec(301, "Return course title for id 3.", "SELECT course_name FROM courses WHERE course_id = 3", "courses")]
        m_id_rule = _manifest([_lit_entry("L1", 3, "integer", "courses", "course_id", "identifier", "count of rows rule", True, {"values": [3]}, "courses.course_id", [301])])
        self.assertIn("literal_context_rule_violations", {f["check"] for f in self._run_gate(cand_id_rule, m_id_rule).hard_failures})

        cand_bad_fk = self.orig + [_rec(301, "Show student name for identifier 21.", "SELECT name FROM students WHERE student_id = 21")]
        m_bad_fk = _manifest([_lit_entry("L1", 21, "integer", "students", "student_id", "identifier", "observed PK key", True, {"values": [21]}, "students.student_id", [301])])
        self.assertIn("literal_context_rule_violations", {f["check"] for f in self._run_gate(cand_bad_fk, m_bad_fk).hard_failures})

        cand_str = self.orig + [_rec(301, "Display scores for CSE semester.", "SELECT enrollment_id, marks FROM enrollments WHERE semester = 'CSE'", "enrollments")]
        m_str = _manifest([_lit_entry("L1", "CSE", "string", "enrollments", "semester", "closed_domain_string", "wrong", True, {"values": ["CSE"]}, "db", [301])])
        self.assertIn("non_catalog_string_literals", {f["check"] for f in self._run_gate(cand_str, m_str).hard_failures})

        cand_lim = self.orig + [_rec(301, "Output name and age limited to 5 rows.", "SELECT name, age FROM students LIMIT 5", "students", "LIMIT")]
        m_bad_lim = _manifest([_lit_entry("L1", 5, "integer", "students", "age", "comparison_threshold", "threshold", True, {"values": [5]}, "db", [301])])
        self.assertFalse(self._run_gate(cand_lim, m_bad_lim).passed)

        # Population A grandfathered overlap does not fail clean B, while missing/mutating/duplicating orig fails
        orig_ho = self.orig + [_rec(10, "Credit hours for all courses.", "SELECT credits FROM courses", "courses", "Basic SELECT and projection")]
        self.assertTrue(self._run_gate(orig_ho + [cand_synth[-1]], m_synth, orig=orig_ho).passed)
        self.assertIn("original_baseline_missing_records", {f["check"] for f in self._run_gate([self.orig[0]], _manifest([])).hard_failures})
        self.assertIn("original_sql_altered", {f["check"] for f in self._run_gate([dict(self.orig[0], sql="SELECT name FROM students"), self.orig[1]], _manifest([])).hard_failures})
        self.assertIn("new_record_duplicates_original_train", {f["check"] for f in self._run_gate(self.orig + [_rec(301, "Learner name and GPA.", "SELECT name, cgpa FROM students")], _manifest([])).hard_failures})

    def test_projection_alignment_decision_a_zero_rows_and_manifest_v2(self) -> None:
        lits = [_lit_entry("L1", 20, "integer", "students", "age", "comparison_threshold", "catalog", True, {"values": [20]}, "age", [301])]
        cand_ord = self.orig + [_rec(301, "Show student name and CGPA with age 20.", "SELECT cgpa, name FROM students WHERE age = 20")]
        rep_ord = self._run_gate(cand_ord, _manifest(lits))
        self.assertFalse(rep_ord.passed)
        self.assertEqual(rep_ord.projection_policy_audit["violations"][0]["violation"], "projection_order_mismatch")
        p_key = rep_ord.projection_policy_audit["review_flags"][0]["key"]
        self.assertTrue(self._run_gate(cand_ord, _manifest(lits, new_records=[cand_ord[-1]], reviewed_proj={p_key: {"disposition": "approved_projection_alignment", "rationale": "Reviewed order."}})).passed)

        cand_star = self.orig + [_rec(301, "Show full details: student name and age for age 20.", "SELECT * FROM students WHERE age = 20")]
        self.assertEqual(self._run_gate(cand_star, _manifest(lits)).projection_policy_audit["violations"][0]["violation"], "wildcard_overrides_explicit_columns")
        cand_guess = self.orig + [_rec(301, "Find students aged 20.", "SELECT name, age FROM students WHERE age = 20")]
        self.assertEqual(self._run_gate(cand_guess, _manifest(lits)).projection_policy_audit["violations"][0]["violation"], "unsupported_explicit_projection")

        # Level-1 duplicate explicit projection never overridable
        cand_dup = self.orig + [_rec(301, "Show student name and age for age 20.", "SELECT name, name FROM students WHERE age = 20")]
        rep_dup = self._run_gate(cand_dup, _manifest(lits, new_records=[cand_dup[-1]], reviewed_proj={"301:duplicate_projection_column": {"disposition": "approved_projection_alignment", "rationale": "Waive."}}))
        self.assertIn("projection_hard_failures", {f["check"] for f in rep_dup.hard_failures})

        # Decision A zero-row policy and DB unavailability fail-closed
        cand_z = self.orig + [_rec(301, "Return student identifier and age for CGPAs under 7.25.", "SELECT student_id, age FROM students WHERE cgpa < 7.25")]
        lit_z = [_lit_entry("L1", 7.25, "decimal", "students", "cgpa", "comparison_threshold", "midpoint", False, {"min": 6.85, "max": 9.40}, "cgpa", [301])]
        exec_zero = FakeSQLExecutor(empty_sql_substrings=("7.25",))
        rep_zunrev = self._run_gate(cand_z, _manifest(lit_z, new_records=[cand_z[-1]], zero_row_ids=[301]), executor=exec_zero)
        self.assertEqual((rep_zunrev.validation_audit["valid_zero_rows_count"], rep_zunrev.validation_audit["invalid_query_count"]), (1, 0))
        self.assertFalse(rep_zunrev.passed)

        m_zapp = _manifest(lit_z, new_records=[cand_z[-1]], zero_row_ids=[301], reviewed_zero={"301": {"disposition": "approved_zero_row", "rationale": "Boundary probe."}})
        self.assertTrue(self._run_gate(cand_z, m_zapp, executor=exec_zero).passed)
        self.assertEqual(self._run_gate(cand_z, m_zapp, db_err="MySQL unreachable").summary["gate_verdict"], "INCOMPLETE / FAILED")

        # Schema 1.0 and tampered row_count rejected
        self.assertFalse(self._run_gate(cand_z, _manifest(lit_z, schema_version="1.0")).passed)
        m_bad_rc = _manifest(lit_z, new_records=[cand_z[-1]])
        m_bad_rc["examples"][0]["row_count"] = 999
        self.assertFalse(self._run_gate(cand_z, m_bad_rc).passed)

        # Projection NL span extraction spot check
        self.assertEqual(_extract_projection_nl_span("Which courses have 4 credits? Show their titles.").rstrip("."), "Show their titles")

    def test_real_mysql_cli_validate_and_executor_integration(self) -> None:
        db_cfg = DatabaseConfig.from_exec_env()
        real_exec = SQLExecutor(config=db_cfg, default_max_rows=500)
        validator = SQLValidator(schema=DEFAULT_STUDENT_SCHEMA, enforce_model_scope=True)
        str_cat, _, obs_nums, bounds, row_cnts = fetch_live_db_catalogs()
        self.assertEqual(row_cnts["students"], 20)
        self.assertIn("8.25", obs_nums[("students", "cgpa")])

        orig_recs = load_jsonl_records("data/train.jsonl")
        new_clean = _rec(
            9991,
            "Report student identifier and department among AI&DS learners whose grade point average exceeds 8.25, capped at 5 entries.",
            "SELECT student_id, department FROM students WHERE department = 'AI&DS' AND cgpa > 8.25 LIMIT 5",
            "students",
            "Combined SQL patterns",
        )
        live_rows = real_exec.execute(validator.validate(new_clean["sql"]), approved=True, max_rows=500).row_count
        m_clean = _manifest(
            [
                _lit_entry("L1", "AI&DS", "string", "students", "department", "closed_domain_string", "live catalog", True, {"values": sorted(str_cat["students.department"])}, "students.department", [9991]),
                _lit_entry("L2", 8.25, "decimal", "students", "cgpa", "comparison_threshold", "observed threshold", True, {"min": bounds[("students", "cgpa")][0], "max": bounds[("students", "cgpa")][1]}, "students.cgpa", [9991]),
                _lit_entry("L3", 5, "integer", "students", None, "limit", "approved LIMIT set", True, {"values": [5]}, "LIMIT", [9991]),
            ],
            new_records=[new_clean],
        )
        m_clean["examples"][0]["row_count"] = live_rows

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cand_p, man_p, rep_p = tdp / "cand.jsonl", tdp / "man.json", tdp / "rep.json"
            cand_p.write_text("\n".join(json.dumps(r) for r in (orig_recs + [new_clean])) + "\n", encoding="utf-8")
            man_p.write_text(json.dumps(m_clean, indent=2), encoding="utf-8")
            rc = validate_candidate_main(["--candidate", str(cand_p), "--provenance-manifest", str(man_p), "--report-out", str(rep_p)])
            self.assertEqual(rc, 0)
            rep_data = json.loads(rep_p.read_text(encoding="utf-8"))
            self.assertEqual(rep_data["summary"]["gate_verdict"], "PASSED")
            self.assertEqual(rep_data["validation_audit"]["executed_count"], len(orig_recs) + 1)


if __name__ == "__main__":
    unittest.main()
