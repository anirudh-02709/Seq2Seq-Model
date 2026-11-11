"""
Comprehensive Dataset Validation Suite for Schema-Aware NL2SQL Mini-Project.

Dataset v1 Strict Standards:
1. Defensive JSON line parsing and record-level validation (never crashes on malformed data).
2. Canonical schema parser rejecting duplicate tables/columns, unknown tables/columns, or malformed schemas.
3. Contiguous ID range (1..300) and uniqueness across all splits.
4. Duplicate (question, SQL) pair and cross-split leakage checks with lexical paraphrase heuristics.
5. Strict Positive AST Allowlist for MySQL dialect (via SQLGlot):
   - Only permitted nodes: Select, From, Table, Column, Identifier, Star, Literal, Where,
     EQ, NEQ, GT, LT, GTE, LTE, And, Or, Paren, Order, Ordered, Limit.
   - Rejects all unknown/unsupported AST nodes, functions (SLEEP, scalar, stored, aggregates),
     DISTINCT, arithmetic expressions, subqueries, JOINs, GROUP BY, HAVING, LIKE, IN, BETWEEN,
     CASE, window functions, user variables, locking clauses (FOR UPDATE), INTO OUTFILE,
     multiple statements, and SQL comments.
6. Execution Gating:
   - Records that fail static AST safety checks are strictly gated and NEVER passed to MySQL EXPLAIN or execution.
7. Exact category distribution adherence (30, 75, 60, 35, 25, 75).
8. Exact primary-table distribution adherence (150, 60, 90).
9. Split coverage validation (all 3 tables and 6 categories in train, val, and test).
10. Defensive Dataset Manifest validation (dataset_manifest.json).
11. Secure and Separated MySQL Server Validation:
    - Layered read-only execution safeguards (AST approval, read-only transactions, rollback).
    - Part A: Syntax and schema validation via EXPLAIN against MySQL Server.
    - Part B: Actual read-only query execution.
    - Secure option file credential isolation with permission enforcement and injection prevention.
12. 3-Tier Overall Status: PASSED, FAILED, or INCOMPLETE.
13. Validation scope and limitations disclosure.
"""

import os
import sys
import json
import re
import tempfile
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import DEFAULT_STUDENT_SCHEMA

# Genuine SQL Parser dependency
try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

# =============================================================================
# 1. SPECIFICATION CONSTANTS
# =============================================================================

CANONICAL_SCHEMA_STR = DEFAULT_STUDENT_SCHEMA

CANONICAL_SCHEMA_TABLES = {
    "students": ["student_id", "name", "age", "department", "cgpa"],
    "courses": ["course_id", "course_name", "credits"],
    "enrollments": ["enrollment_id", "student_id", "course_id", "semester", "marks"]
}

SCHEMA_TABLES = {tbl: set(cols) for tbl, cols in CANONICAL_SCHEMA_TABLES.items()}

PLANNED_CATEGORY_TOTALS = {
    "Basic SELECT and projection": 30,
    "WHERE and comparisons": 75,
    "AND / OR conditions": 60,
    "ORDER BY": 35,
    "LIMIT": 25,
    "Combined SQL patterns": 75
}

PLANNED_TABLE_TOTALS = {
    "students": 150,
    "courses": 60,
    "enrollments": 90
}

PLANNED_SPLIT_COUNTS = {
    "train": 210,
    "validation": 45,
    "test": 45
}

TOTAL_EXPECTED_EXAMPLES = 300

REQUIRED_FIELDS = {"id", "schema", "question", "sql", "category", "table"}

# Supported binary comparison AST types
ALLOWED_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE)

# POSITIVE AST ALLOWLIST: Only these node types may ever appear in a Dataset v1 query AST
ALLOWED_AST_NODE_TYPES = {
    exp.Select,
    exp.From,
    exp.Table,
    exp.Column,
    exp.Identifier,
    exp.Star,
    exp.Literal,
    exp.Where,
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.LT,
    exp.GTE,
    exp.LTE,
    exp.And,
    exp.Or,
    exp.Paren,
    exp.Order,
    exp.Ordered,
    exp.Limit
}

# =============================================================================
# 2. HELPER UTILITIES
# =============================================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def normalize_sql(sql):
    sql = sql.strip().rstrip(";")
    tokens = sql.split()
    return " ".join(tokens).upper()

def tokenize_words(text):
    return set(normalize_text(text).split())

def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def find_mysql_binary():
    import shutil
    binary = shutil.which("mysql")
    if binary:
        return binary
    default_win_path = r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
    if os.path.exists(default_win_path):
        return default_win_path
    return "mysql"

# =============================================================================
# 3. CANONICAL SCHEMA PARSER (REJECTS DUPLICATES & UNKNOWN DEFINITIONS)
# =============================================================================

def parse_and_validate_schema_string(schema_str):
    """
    Parses a schema string like 't1(c1, c2) | t2(c3, c4)'.
    Defensively enforces:
    - Exactly non-empty string.
    - No duplicate table declarations (fails without overwriting).
    - No duplicate column declarations within a table.
    - Exactly canonical tables and columns (no missing, no extra, no unknown).
    Returns (is_valid, error_message).
    """
    if not isinstance(schema_str, str) or not schema_str.strip():
        return False, "Schema is empty or not a string"

    parts = [p.strip() for p in schema_str.split("|")]
    if not parts or any(not p for p in parts):
        return False, "Schema has empty parts separated by '|'"

    seen_tables = set()
    parsed_tables = {}

    for part in parts:
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)$", part)
        if not m:
            return False, f"Malformed table definition in schema: '{part}'"
        tbl = m.group(1).lower()

        if tbl in seen_tables:
            return False, f"Duplicate table definition in schema: '{tbl}'"
        seen_tables.add(tbl)

        if tbl not in CANONICAL_SCHEMA_TABLES:
            return False, f"Unknown table in schema: '{tbl}'. Allowed: {sorted(list(CANONICAL_SCHEMA_TABLES.keys()))}"

        col_parts = [c.strip().lower() for c in m.group(2).split(",")]
        seen_cols = set()
        cols = []
        for c in col_parts:
            if not c:
                return False, f"Empty column identifier in table '{tbl}'"
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", c):
                return False, f"Invalid column name '{c}' in table '{tbl}'"
            if c in seen_cols:
                return False, f"Duplicate column '{c}' in table '{tbl}' schema definition"
            seen_cols.add(c)
            cols.append(c)

        expected_cols = CANONICAL_SCHEMA_TABLES[tbl]
        if cols != expected_cols:
            missing = set(expected_cols) - set(cols)
            extra = set(cols) - set(expected_cols)
            diffs = []
            if missing:
                diffs.append(f"missing {sorted(list(missing))}")
            if extra:
                diffs.append(f"extra {sorted(list(extra))}")
            if not missing and not extra and cols != expected_cols:
                diffs.append(f"order mismatch: got {cols}, expected {expected_cols}")
            return False, f"Table '{tbl}' column definition mismatch: {', '.join(diffs)}"

        parsed_tables[tbl] = cols

    missing_tables = set(CANONICAL_SCHEMA_TABLES.keys()) - seen_tables
    if missing_tables:
        return False, f"Missing table(s) in schema: {sorted(list(missing_tables))}"

    return True, None

def validate_canonical_schema(records, errors):
    for r in records:
        meta = r.get("_meta", {})
        loc = f"{meta.get('filepath', 'unknown')}:{meta.get('line_idx', '?')} (ID: {r.get('id')})"
        schema_val = r.get("schema", "")
        is_valid, schema_err = parse_and_validate_schema_string(schema_val)
        if not is_valid:
            errors.append(f"[{loc}] Schema validation failed: {schema_err}")

# =============================================================================
# 4. ROBUST RECORD INGESTION (NEVER CRASHES ON MALFORMED INPUT)
# =============================================================================

def load_and_validate_records(filepath, split_name, errors):
    accepted_records = []
    all_record_ids = []

    if not os.path.exists(filepath):
        errors.append(f"[{split_name}] Missing dataset file: {filepath}")
        return accepted_records, all_record_ids

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as ex:
        errors.append(f"[{split_name}] Failed to read file {filepath}: {ex}")
        return accepted_records, all_record_ids

    for line_idx, line in enumerate(lines, 1):
        line_str = line.strip()
        if not line_str:
            continue

        try:
            data = json.loads(line_str)
        except json.JSONDecodeError as jde:
            errors.append(f"[{filepath}:{line_idx}] Malformed JSON line: {jde}")
            continue
        except Exception as ex:
            errors.append(f"[{filepath}:{line_idx}] Unexpected JSON parse error: {ex}")
            continue

        if not isinstance(data, dict):
            errors.append(f"[{filepath}:{line_idx}] Line does not parse to a JSON object (got {type(data).__name__})")
            continue

        rec_id = data.get("id")
        loc_desc = f"{filepath}:{line_idx}" if rec_id is None else f"{filepath}:{line_idx} (ID: {rec_id})"

        missing_keys = REQUIRED_FIELDS - set(data.keys())
        if missing_keys:
            errors.append(f"[{loc_desc}] Missing required field(s): {sorted(list(missing_keys))}")
            continue

        if type(rec_id) is not int or isinstance(rec_id, bool):
            errors.append(f"[{filepath}:{line_idx}] Field 'id' must be an integer (rejecting bool), got {type(rec_id).__name__} ({rec_id!r})")
            continue

        all_record_ids.append((rec_id, filepath, line_idx))

        type_error = False
        for str_field in ["schema", "question", "sql", "category", "table"]:
            val = data.get(str_field)
            if not isinstance(val, str):
                errors.append(f"[{loc_desc}] Field '{str_field}' must be a string, got {type(val).__name__}")
                type_error = True
            elif not val.strip():
                errors.append(f"[{loc_desc}] Field '{str_field}' must not be empty or whitespace only")
                type_error = True

        if type_error:
            continue

        cat = data.get("category")
        if cat not in PLANNED_CATEGORY_TOTALS:
            errors.append(f"[{loc_desc}] Invalid category label: '{cat}'. Allowed: {list(PLANNED_CATEGORY_TOTALS.keys())}")
            continue

        tbl = data.get("table")
        if tbl not in SCHEMA_TABLES:
            errors.append(f"[{loc_desc}] Invalid table label: '{tbl}'. Allowed: {list(SCHEMA_TABLES.keys())}")
            continue

        data["_meta"] = {"filepath": filepath, "line_idx": line_idx, "split": split_name}
        accepted_records.append(data)

    return accepted_records, all_record_ids

# =============================================================================
# 5. STRICT POSITIVE AST ALLOWLIST & CATEGORY VALIDATION (SQLGLOT)
# =============================================================================

def validate_sql_with_ast_allowlist(record):
    """
    Validates a single record's SQL query using SQLGlot with a strict positive allowlist.
    Returns:
        (is_approved: bool, record_errors: list)
    """
    record_errors = []
    meta = record.get("_meta", {})
    loc = f"{meta.get('filepath', 'unknown')}:{meta.get('line_idx', '?')} (ID: {record.get('id')})"
    sql = record.get("sql", "").strip()
    table = record.get("table")
    category = record.get("category")

    # 1. Pre-parse checks: Disallow SQL comments and multiple statement delimiters
    if "--" in sql or "/*" in sql or "*/" in sql or "#" in sql:
        record_errors.append(f"[{loc}] SQL comments are forbidden: '{sql}'")
        return False, record_errors

    if ";" in sql.rstrip(";"):
        record_errors.append(f"[{loc}] Multiple statements or unquoted semicolons forbidden: '{sql}'")
        return False, record_errors

    # 2. Parse with SQLGlot in MySQL dialect
    try:
        statements = sqlglot.parse(sql, read="mysql")
    except Exception as ex:
        record_errors.append(f"[{loc}] SQLGlot MySQL parse failure: {ex} (Query: '{sql}')")
        return False, record_errors

    if len(statements) != 1:
        record_errors.append(f"[{loc}] Query must contain exactly 1 statement, found {len(statements)}: '{sql}'")
        return False, record_errors

    ast = statements[0]
    if ast is None:
        record_errors.append(f"[{loc}] SQL parser returned an empty AST: '{sql}'")
        return False, record_errors

    if not isinstance(ast, exp.Select):
        record_errors.append(f"[{loc}] Statement must be a SELECT, found {type(ast).__name__}: '{sql}'")
        return False, record_errors

    # 3. STRICT POSITIVE AST ALLOWLIST: Every node must be in ALLOWED_AST_NODE_TYPES
    for node in ast.walk():
        node_type = type(node)
        if node_type not in ALLOWED_AST_NODE_TYPES:
            record_errors.append(
                f"[{loc}] Unauthorized AST node '{node_type.__name__}' is forbidden under Dataset v1 positive allowlist: '{sql}'"
            )
            return False, record_errors

    # 4. Strict structural checks on Select
    if ast.args.get("distinct"):
        record_errors.append(f"[{loc}] DISTINCT is forbidden in Dataset v1: '{sql}'")
    if ast.args.get("into"):
        record_errors.append(f"[{loc}] SELECT INTO / INTO OUTFILE is forbidden: '{sql}'")
    if ast.args.get("locks"):
        record_errors.append(f"[{loc}] Locking clauses (FOR UPDATE / LOCK IN SHARE MODE) are forbidden: '{sql}'")
    if ast.args.get("group") or ast.args.get("having"):
        record_errors.append(f"[{loc}] GROUP BY and HAVING are forbidden: '{sql}'")
    if ast.args.get("joins"):
        record_errors.append(f"[{loc}] JOIN operations are forbidden: '{sql}'")
    if any(node.comments for node in ast.walk()):
        record_errors.append(f"[{loc}] AST contains comment tokens: '{sql}'")

    all_selects = list(ast.find_all(exp.Select))
    if len(all_selects) > 1:
        record_errors.append(f"[{loc}] Subqueries are forbidden: '{sql}'")

    # 5. Validate single table reference in FROM clause
    tables = list(ast.find_all(exp.Table))
    if len(tables) != 1:
        tbl_names = [t.name for t in tables]
        record_errors.append(f"[{loc}] Query must reference exactly 1 table, found {len(tables)} ({tbl_names}): '{sql}'")
        return False, record_errors

    ast_table = tables[0]
    if ast_table.name.lower() != table.lower():
        record_errors.append(f"[{loc}] FROM table '{ast_table.name}' does not match record table '{table}': '{sql}'")

    if ast_table.db:
        record_errors.append(f"[{loc}] Schema-qualified table references forbidden ('{ast_table.db}.{ast_table.name}'): '{sql}'")

    # 6. Validate column references against schema
    allowed_cols = SCHEMA_TABLES.get(table, set())
    for c in ast.find_all(exp.Column):
        col_name = c.name.lower()
        if col_name not in allowed_cols:
            record_errors.append(f"[{loc}] Column '{c.name}' does not exist in table '{table}' schema: '{sql}'")
        if c.table and c.table.lower() != table.lower():
            record_errors.append(f"[{loc}] Column table qualifier '{c.table}' does not match table '{table}': '{sql}'")
        if c.db:
            record_errors.append(f"[{loc}] Database-qualified column '{c.db}.{c.name}' forbidden: '{sql}'")

    # 7. Validate Category-Specific Structural Rules
    where_node = ast.args.get("where")
    has_where = where_node is not None
    has_order = ast.args.get("order") is not None
    has_limit = ast.args.get("limit") is not None
    has_and_or = bool(list(where_node.find_all(exp.And, exp.Or))) if has_where else False

    if category == "Basic SELECT and projection":
        if has_where:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a WHERE clause: '{sql}'")
        if has_order:
            record_errors.append(f"[{loc}] Category '{category}' must not contain an ORDER BY clause: '{sql}'")
        if has_limit:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a LIMIT clause: '{sql}'")

    elif category == "WHERE and comparisons":
        if not has_where:
            record_errors.append(f"[{loc}] Category '{category}' requires a WHERE clause: '{sql}'")
        else:
            comps = list(where_node.find_all(*ALLOWED_COMPARISONS))
            if not comps:
                record_errors.append(f"[{loc}] Category '{category}' requires at least one valid comparison operator (=, !=, >, <, >=, <=): '{sql}'")
            if has_and_or:
                record_errors.append(f"[{loc}] Category '{category}' must not contain AND or OR operators: '{sql}'")
        if has_order:
            record_errors.append(f"[{loc}] Category '{category}' must not contain an ORDER BY clause: '{sql}'")
        if has_limit:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a LIMIT clause: '{sql}'")

    elif category == "AND / OR conditions":
        if not has_where:
            record_errors.append(f"[{loc}] Category '{category}' requires a WHERE clause: '{sql}'")
        elif not has_and_or:
            record_errors.append(f"[{loc}] Category '{category}' requires at least one AND or OR operator in WHERE: '{sql}'")
        if has_order:
            record_errors.append(f"[{loc}] Category '{category}' must not contain an ORDER BY clause: '{sql}'")
        if has_limit:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a LIMIT clause: '{sql}'")

    elif category == "ORDER BY":
        if not has_order:
            record_errors.append(f"[{loc}] Category '{category}' requires an ORDER BY clause: '{sql}'")
        if has_where:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a WHERE clause: '{sql}'")
        if has_limit:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a LIMIT clause: '{sql}'")

    elif category == "LIMIT":
        if not has_limit:
            record_errors.append(f"[{loc}] Category '{category}' requires a LIMIT clause: '{sql}'")
        if has_where:
            record_errors.append(f"[{loc}] Category '{category}' must not contain a WHERE clause: '{sql}'")
        if has_order:
            record_errors.append(f"[{loc}] Category '{category}' must not contain an ORDER BY clause: '{sql}'")

    elif category == "Combined SQL patterns":
        feature_count = sum([has_where, has_order, has_limit])
        if feature_count < 2:
            record_errors.append(f"[{loc}] Category '{category}' must combine at least two clauses (WHERE, ORDER BY, LIMIT), found {feature_count}: '{sql}'")

    is_approved = len(record_errors) == 0
    return is_approved, record_errors

# =============================================================================
# 6. DUPLICATE AND CROSS-SPLIT LEAKAGE DETECTION
# =============================================================================

def validate_duplicates_and_leakage(records, errors, warnings):
    seen_pairs = defaultdict(list)
    seen_questions = defaultdict(list)
    seen_sqls = defaultdict(list)
    q_tokens_by_split = defaultdict(list)

    for r in records:
        rec_id = r.get("id")
        split = r.get("_meta", {}).get("split", "unknown")
        q_raw = r.get("question", "").strip()
        sql_raw = r.get("sql", "").strip()

        pair_key = (q_raw, normalize_sql(sql_raw))
        seen_pairs[pair_key].append((split, rec_id))

        q_norm = normalize_text(q_raw)
        seen_questions[q_norm].append((split, rec_id, q_raw))

        sql_norm = normalize_sql(sql_raw)
        seen_sqls[sql_norm].append((split, rec_id, sql_raw))

        q_tokens_by_split[split].append((rec_id, tokenize_words(q_raw), q_raw))

    for occurrences in seen_pairs.values():
        if len(occurrences) > 1:
            desc = ", ".join([f"[{sp} ID:{rid}]" for sp, rid in occurrences])
            errors.append(f"Duplicate (question, SQL) pair found across {len(occurrences)} records: {desc}")

    for occurrences in seen_questions.values():
        if len(occurrences) > 1:
            splits_involved = set(sp for sp, _, _ in occurrences)
            desc = ", ".join([f"[{sp} ID:{rid}]" for sp, rid, _ in occurrences])
            if len(splits_involved) > 1:
                errors.append(f"Cross-split question leakage detected across {sorted(list(splits_involved))}: {desc}")
            else:
                errors.append(f"Duplicate question detected within split '{list(splits_involved)[0]}': {desc}")

    for occurrences in seen_sqls.values():
        if len(occurrences) > 1:
            splits_involved = set(sp for sp, _, _ in occurrences)
            if len(splits_involved) > 1:
                desc = ", ".join([f"[{sp} ID:{rid}]" for sp, rid, _ in occurrences])
                errors.append(f"Cross-split identical SQL target leakage detected across {sorted(list(splits_involved))}: {desc}")

    split_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for s1, s2 in split_pairs:
        for id1, tok1, q1 in q_tokens_by_split[s1]:
            for id2, tok2, q2 in q_tokens_by_split[s2]:
                sim = jaccard_similarity(tok1, tok2)
                if sim >= 0.85:
                    warnings.append(
                        f"Potential paraphrase heuristic warning ({sim:.2f} Jaccard): [{s1} ID:{id1}] '{q1}' vs [{s2} ID:{id2}] '{q2}'"
                    )

# =============================================================================
# 7. DEFENSIVE DATASET MANIFEST VALIDATION
# =============================================================================

def validate_manifest(manifest_path, actual_stats, errors):
    if not os.path.exists(manifest_path):
        errors.append(f"Manifest file not found: {manifest_path}")
        return

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as ex:
        errors.append(f"Manifest JSON parse error: {ex}")
        return

    if not isinstance(manifest, dict):
        errors.append(f"Manifest root must be a JSON object, got {type(manifest).__name__}")
        return

    required_manifest_fields = {
        "dataset_name": str,
        "description": str,
        "schema": str,
        "total_examples": int,
        "split_counts": dict,
        "category_distribution": dict,
        "table_distribution": dict,
        "supported_sql_features": list
    }

    type_error_occurred = False
    for field, expected_type in required_manifest_fields.items():
        if field not in manifest:
            errors.append(f"Manifest missing required field: '{field}'")
            type_error_occurred = True
            continue

        val = manifest[field]
        if expected_type is int:
            if type(val) is not int or isinstance(val, bool) or val < 0:
                errors.append(f"Manifest field '{field}' must be a non-negative integer (rejecting bool), got {val!r}")
                type_error_occurred = True
        elif not isinstance(val, expected_type):
            errors.append(f"Manifest field '{field}' must be of type {expected_type.__name__}, got {type(val).__name__}")
            type_error_occurred = True

    if type_error_occurred:
        return

    m_total = manifest.get("total_examples")
    if m_total != actual_stats["total_examples"]:
        errors.append(f"Manifest 'total_examples' ({m_total}) does not match actual total ({actual_stats['total_examples']})")

    m_splits = manifest.get("split_counts", {})
    if isinstance(m_splits, dict):
        for sp, exp_count in actual_stats["split_counts"].items():
            if sp not in m_splits:
                errors.append(f"Manifest 'split_counts' missing split '{sp}'")
                continue
            v = m_splits[sp]
            if type(v) is not int or isinstance(v, bool) or v < 0:
                errors.append(f"Manifest 'split_counts' count for '{sp}' must be a non-negative integer, got {v!r}")
            elif v != exp_count:
                errors.append(f"Manifest split count for '{sp}' ({v}) does not match actual ({exp_count})")
    else:
        errors.append(f"Manifest 'split_counts' must be an object, got {type(m_splits).__name__}")

    m_cats = manifest.get("category_distribution", {})
    if isinstance(m_cats, dict):
        for cat, exp_count in actual_stats["category_counts"].items():
            if cat not in m_cats:
                errors.append(f"Manifest 'category_distribution' missing category '{cat}'")
                continue
            v = m_cats[cat]
            if type(v) is not int or isinstance(v, bool) or v < 0:
                errors.append(f"Manifest 'category_distribution' count for '{cat}' must be a non-negative integer, got {v!r}")
            elif v != exp_count:
                errors.append(f"Manifest category count for '{cat}' ({v}) does not match actual ({exp_count})")
    else:
        errors.append(f"Manifest 'category_distribution' must be an object, got {type(m_cats).__name__}")

    m_tbls = manifest.get("table_distribution", {})
    if isinstance(m_tbls, dict):
        for tbl, exp_count in actual_stats["table_counts"].items():
            if tbl not in m_tbls:
                errors.append(f"Manifest 'table_distribution' missing table '{tbl}'")
                continue
            v = m_tbls[tbl]
            if type(v) is not int or isinstance(v, bool) or v < 0:
                errors.append(f"Manifest 'table_distribution' count for '{tbl}' must be a non-negative integer, got {v!r}")
            elif v != exp_count:
                errors.append(f"Manifest table count for '{tbl}' ({v}) does not match actual ({exp_count})")
    else:
        errors.append(f"Manifest 'table_distribution' must be an object, got {type(m_tbls).__name__}")

    m_features = manifest.get("supported_sql_features", [])
    expected_features = [
        "SELECT", "FROM", "WHERE", "=", "!=", ">", "<", ">=", "<=",
        "AND", "OR", "ORDER BY", "ASC", "DESC", "LIMIT"
    ]
    if isinstance(m_features, list):
        if set(m_features) != set(expected_features):
            errors.append("Manifest 'supported_sql_features' does not match agreed v1 scope")
    else:
        errors.append(f"Manifest 'supported_sql_features' must be a list, got {type(m_features).__name__}")

    m_schema = manifest.get("schema")
    if isinstance(m_schema, str):
        is_valid_schema, schema_err = parse_and_validate_schema_string(m_schema)
        if not is_valid_schema:
            errors.append(f"Manifest 'schema' does not match canonical Dataset V1 schema: {schema_err}")

# =============================================================================
# 8. LAYERED READ-ONLY MYSQL SERVER VALIDATION & SECURE OPTION FILE
# =============================================================================

def validate_mysql(statically_approved_records, total_accepted_count, static_only=False):
    """
    Validates queries against MySQL Server with layered security:
    - Layer 1: Strict AST approval (only statically approved records reach this function).
    - Layer 2: Dedicated MySQL account with SELECT-only privileges (documented).
    - Layer 3: Enforced read-only transaction (SET SESSION TRANSACTION READ ONLY; START TRANSACTION READ ONLY; ...; ROLLBACK;).
    - Layer 4: Secure MySQL option file with restricted permissions and injection prevention.
    - Part A: Syntax and schema resolution via EXPLAIN.
    - Part B: Actual read-only query execution.
    """
    if static_only:
        return {
            "configured": False,
            "part_a_status": "SKIPPED",
            "part_b_status": "SKIPPED",
            "reason": "Skipped via --static-only CLI flag",
            "errors": []
        }

    db_user = os.environ.get("MYSQL_USER") or os.environ.get("DB_USER")
    db_password = os.environ.get("MYSQL_PASSWORD") or os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("MYSQL_HOST") or os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("MYSQL_PORT") or os.environ.get("DB_PORT", "3306")
    db_name = os.environ.get("MYSQL_DATABASE") or os.environ.get("DB_NAME", "nl2sql_student_db")

    if not db_user:
        return {
            "configured": False,
            "part_a_status": "PENDING",
            "part_b_status": "PENDING",
            "reason": "MYSQL_USER environment variable not set",
            "errors": []
        }

    # Security check: Prevent option-file configuration injection
    for val, name in [(db_user, "user"), (db_host, "host"), (str(db_port), "port"), (db_name, "database")]:
        if any(c in val for c in ["\n", "\r", "\0"]):
            return {
                "configured": False,
                "part_a_status": "FAILED",
                "part_b_status": "FAILED",
                "reason": f"Security violation: {name} contains prohibited control characters",
                "errors": [f"Option file configuration injection blocked in {name}"]
            }

    if db_password and any(c in db_password for c in ["\n", "\r", "\0"]):
        return {
            "configured": False,
            "part_a_status": "FAILED",
            "part_b_status": "FAILED",
            "reason": "Security violation: password contains prohibited newline or null characters",
            "errors": ["Option file configuration injection blocked in password"]
        }

    mysql_bin = find_mysql_binary()
    cnf_file = None
    mysql_errors = []

    try:
        # Create temporary MySQL option file with restrictive permissions
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".cnf") as f:
            f.write("[client]\n")
            f.write(f"user = {db_user}\n")
            if db_password:
                # Escape backslashes and double quotes safely
                escaped_pw = db_password.replace("\\", "\\\\").replace('"', '\\"')
                f.write(f'password = "{escaped_pw}"\n')
            f.write(f"host = {db_host}\n")
            f.write(f"port = {db_port}\n")
            cnf_file = f.name

        # Restrict permissions
        if os.name != "nt":
            os.chmod(cnf_file, 0o600)
            st_mode = os.stat(cnf_file).st_mode & 0o777
            if st_mode != 0o600:
                raise PermissionError("Failed to set 0600 permissions on temporary MySQL option file")
        else:
            # On Windows, enforce restrictive ACL to current user
            username = os.environ.get("USERNAME")
            if username:
                cmd = ["icacls", cnf_file, "/inheritance:r", "/grant:r", f"{username}:(R,W)"]
                icacls_res = subprocess.run(cmd, capture_output=True, text=True)
                if icacls_res.returncode != 0:
                    raise PermissionError(f"Failed to restrict file ACLs on Windows: {icacls_res.stderr.strip()}")

        base_cmd = [mysql_bin, f"--defaults-extra-file={cnf_file}", db_name]

        # Test initial connection
        test_res = subprocess.run(base_cmd + ["-e", "SELECT 1;"], capture_output=True, text=True, timeout=10)
        if test_res.returncode != 0:
            clean_err = test_res.stderr.strip().replace(db_password if db_password else "___", "[REDACTED]")
            return {
                "configured": True,
                "part_a_status": "FAILED",
                "part_b_status": "FAILED",
                "reason": f"Connection failed: {clean_err}",
                "errors": [f"MySQL connection error: {clean_err}"]
            }

        # PART A: EXPLAIN validation (Syntax and schema resolution)
        part_a_failures = []
        for r in statically_approved_records:
            sql = r.get("sql", "").strip()
            rec_id = r.get("id")
            cmd = base_cmd + ["-e", f"EXPLAIN {sql};"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                clean_err = res.stderr.strip().replace(db_password if db_password else "___", "[REDACTED]")
                part_a_failures.append(f"Record ID {rec_id} EXPLAIN failed: {clean_err} (Query: '{sql}')")

        part_a_status = "PASSED" if not part_a_failures else "FAILED"
        mysql_errors.extend(part_a_failures)

        # PART B: Actual Read-Only Execution with enforced transaction isolation
        part_b_failures = []
        for r in statically_approved_records:
            sql = r.get("sql", "").strip()
            rec_id = r.get("id")
            # Enforce read-only session & transaction rollback safeguard
            wrapper_sql = f"SET SESSION TRANSACTION READ ONLY; START TRANSACTION READ ONLY; {sql}; ROLLBACK;"
            cmd = base_cmd + ["-e", wrapper_sql]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                clean_err = res.stderr.strip().replace(db_password if db_password else "___", "[REDACTED]")
                part_b_failures.append(f"Record ID {rec_id} execution failed: {clean_err} (Query: '{sql}')")

        part_b_status = "PASSED" if not part_b_failures else "FAILED"
        mysql_errors.extend(part_b_failures)

        # Honest partial validation reporting
        if len(statically_approved_records) < total_accepted_count:
            reason = f"Executed only on {len(statically_approved_records)}/{total_accepted_count} approved records; rejected records were strictly gated"
        else:
            reason = "All approved queries verified against MySQL Server" if not mysql_errors else "Failures encountered during MySQL checks"

        return {
            "configured": True,
            "part_a_status": part_a_status,
            "part_b_status": part_b_status,
            "reason": reason,
            "errors": mysql_errors
        }

    except Exception as ex:
        clean_msg = str(ex).replace(db_password if db_password else "___", "[REDACTED]")
        return {
            "configured": True,
            "part_a_status": "FAILED",
            "part_b_status": "FAILED",
            "reason": f"Exception during MySQL execution: {clean_msg}",
            "errors": [clean_msg]
        }
    finally:
        if cnf_file and os.path.exists(cnf_file):
            try:
                os.remove(cnf_file)
            except Exception:
                pass

# =============================================================================
# 9. MAIN VALIDATION CONTROLLER
# =============================================================================

def run_validation(static_only=False):
    print("=" * 80)
    print("       SCHEMA-AWARE NL2SQL DATASET V1: COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)

    if not HAS_SQLGLOT:
        print("\nCRITICAL ERROR: 'sqlglot' parser library is not installed.")
        print("Please install sqlglot via: pip install sqlglot")
        print("\nOVERALL STATUS: FAILED")
        return False, 1

    errors = []
    warnings = []

    # File paths
    train_path = os.path.join("data", "train.jsonl")
    val_path = os.path.join("data", "validation.jsonl")
    test_path = os.path.join("data", "test.jsonl")
    manifest_path = os.path.join("data", "dataset_manifest.json")

    # 1. Ingest records with JSON and field typing validation
    train_records, train_ids = load_and_validate_records(train_path, "train", errors)
    val_records, val_ids = load_and_validate_records(val_path, "validation", errors)
    test_records, test_ids = load_and_validate_records(test_path, "test", errors)

    all_accepted = train_records + val_records + test_records
    all_raw_ids = train_ids + val_ids + test_ids

    # 1. JSON and Record Integrity Report
    print("\n[1] JSON and Record Integrity:")
    if errors:
        print(f"    - Structural & JSON Errors Found: {len(errors)}")
        for err in errors[:5]:
            print(f"      * {err}")
    else:
        print("    - JSON line syntax & object typing: PASSED")
        print("    - Required field presence & non-empty string types: PASSED")
        print("    - Strict integer ID validation (rejecting bool): PASSED")
        print("    - Valid category & table domain labels: PASSED")

    # 2. Split Counts and ID Coverage
    print("\n[2] Split Counts and ID Coverage:")
    split_counts = {
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records)
    }
    for sp, exp_count in PLANNED_SPLIT_COUNTS.items():
        act_count = split_counts[sp]
        status = "PASSED" if act_count == exp_count else "FAILED"
        print(f"    - {sp.capitalize():<12}: {act_count:>4} (Expected: {exp_count:>4}) -> {status}")
        if act_count != exp_count:
            errors.append(f"Split '{sp}' count {act_count} != expected {exp_count}")

    total_actual = len(all_accepted)
    status_tot = "PASSED" if total_actual == TOTAL_EXPECTED_EXAMPLES else "FAILED"
    print(f"    - {'Total':<12}: {total_actual:>4} (Expected: {TOTAL_EXPECTED_EXAMPLES:>4}) -> {status_tot}")
    if total_actual != TOTAL_EXPECTED_EXAMPLES:
        errors.append(f"Total accepted records {total_actual} != expected {TOTAL_EXPECTED_EXAMPLES}")

    # Check ID uniqueness and continuity
    id_counts = defaultdict(list)
    for rec_id, fp, lno in all_raw_ids:
        id_counts[rec_id].append((fp, lno))

    dup_ids = {rid: locs for rid, locs in id_counts.items() if len(locs) > 1}
    if dup_ids:
        errors.append(f"Duplicate IDs detected: {list(dup_ids.keys())}")
        print(f"    - ID Uniqueness: FAILED ({len(dup_ids)} duplicates)")
    else:
        print("    - ID Uniqueness: PASSED (0 duplicate IDs)")

    expected_ids = set(range(1, TOTAL_EXPECTED_EXAMPLES + 1))
    found_ids = set(id_counts.keys())
    missing_ids = expected_ids - found_ids
    if missing_ids:
        errors.append(f"Missing IDs in expected range 1..{TOTAL_EXPECTED_EXAMPLES}: {sorted(list(missing_ids))}")
        print(f"    - ID Range Coverage (1..{TOTAL_EXPECTED_EXAMPLES}): FAILED (Missing: {len(missing_ids)})")
    else:
        print(f"    - ID Range Coverage (1..{TOTAL_EXPECTED_EXAMPLES}): PASSED (Contiguous and complete)")

    # 3. Canonical Schema Validation
    print("\n[3] Canonical Schema Validation:")
    schema_err_start = len(errors)
    validate_canonical_schema(all_accepted, errors)
    if len(errors) == schema_err_start:
        print("    - Strict Canonical Schema Parser (no duplicate tables/columns, exact matches): PASSED")
    else:
        print(f"    - Schema field mismatches detected: {len(errors) - schema_err_start} error(s)")

    # 4. Duplicate and Cross-Split Leakage Checks
    print("\n[4] Duplicate and Cross-Split Leakage Checks:")
    leakage_err_start = len(errors)
    validate_duplicates_and_leakage(all_accepted, errors, warnings)
    if len(errors) == leakage_err_start:
        print("    - Duplicate (question, SQL) pairs: 0 (PASSED)")
        print("    - Normalized question cross-split leakage: 0 (PASSED)")
        print("    - Identical SQL target cross-split leakage: 0 (PASSED)")
    else:
        print(f"    - Duplication / Leakage errors detected: {len(errors) - leakage_err_start}")

    # 5. Lexical Similarity Paraphrase Warnings
    print("\n[5] Lexical Similarity Paraphrase Warnings:")
    if warnings:
        print(f"    - High Lexical Similarity Warnings (Jaccard >= 0.85): {len(warnings)}")
        for w in warnings[:3]:
            print(f"      * {w}")
        if len(warnings) > 3:
            print(f"      * ... and {len(warnings) - 3} more warnings.")
    else:
        print("    - High Lexical Similarity Warnings (Jaccard >= 0.85): 0 (PASSED)")

    # 6. SQLGlot MySQL Strict AST Positive Allowlist Validation & Execution Gating
    print("\n[6] SQLGlot MySQL Strict Positive AST Allowlist & Execution Gating:")
    statically_approved_records = []
    statically_rejected_records = []

    for r in all_accepted:
        is_approved, ast_errs = validate_sql_with_ast_allowlist(r)
        if is_approved:
            statically_approved_records.append(r)
        else:
            statically_rejected_records.append(r)
            errors.extend(ast_errs)

    if not statically_rejected_records:
        print("    - SQLGlot MySQL Positive AST Allowlist: 300 / 300 queries approved (PASSED)")
        print("    - Disallowed AST nodes (functions, DISTINCT, arithmetic, locks, subqueries): 0 detected (PASSED)")
        print("    - Exactly one SELECT statement per record: PASSED")
        print("    - Exactly one primary table in FROM clause: PASSED")
        print("    - Column membership strictly verified against declared table schema: PASSED")
        print("    - Zero SQL comments or statement delimiters: PASSED")
        print("    - Category clause structure conformance: PASSED")
        print("    - Execution Gating: All 300 records statically approved for MySQL stage.")
    else:
        print(f"    - SQL AST Validation FAILED with {len(statically_rejected_records)} rejected record(s).")
        print(f"    - Execution Gating: {len(statically_rejected_records)} record(s) STRICTLY GATED and blocked from MySQL.")

    # 7. Category Distribution Verification
    print("\n[7] Planned Category Distribution Verification:")
    cat_counts_by_split = defaultdict(lambda: defaultdict(int))
    for r in all_accepted:
        cat_counts_by_split[r["_meta"]["split"]][r["category"]] += 1

    print(f"{'Category':<30} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6} | {'Planned':<7} | {'Status'}")
    print("-" * 80)
    for cat, planned in PLANNED_CATEGORY_TOTALS.items():
        tr = cat_counts_by_split["train"][cat]
        va = cat_counts_by_split["validation"][cat]
        te = cat_counts_by_split["test"][cat]
        tot = tr + va + te
        cat_status = "PASSED" if tot == planned and tr > 0 and va > 0 and te > 0 else "FAILED"
        print(f"{cat:<30} | {tr:<6} | {va:<6} | {te:<6} | {tot:<6} | {planned:<7} | {cat_status}")
        if tot != planned:
            errors.append(f"Category '{cat}' total count {tot} != planned {planned}")
        if tr == 0 or va == 0 or te == 0:
            errors.append(f"Category '{cat}' missing in split coverage (Train:{tr}, Val:{va}, Test:{te})")
    print("-" * 80)

    # 8. Primary-Table Distribution Verification
    print("\n[8] Planned Primary-Table Distribution Verification:")
    tbl_counts_by_split = defaultdict(lambda: defaultdict(int))
    for r in all_accepted:
        tbl_counts_by_split[r["_meta"]["split"]][r["table"]] += 1

    print(f"{'Table':<15} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6} | {'Planned':<7} | {'Status'}")
    print("-" * 65)
    for tbl, planned in PLANNED_TABLE_TOTALS.items():
        tr = tbl_counts_by_split["train"][tbl]
        va = tbl_counts_by_split["validation"][tbl]
        te = tbl_counts_by_split["test"][tbl]
        tot = tr + va + te
        tbl_status = "PASSED" if tot == planned and tr > 0 and va > 0 and te > 0 else "FAILED"
        print(f"{tbl:<15} | {tr:<6} | {va:<6} | {te:<6} | {tot:<6} | {planned:<7} | {tbl_status}")
        if tot != planned:
            errors.append(f"Table '{tbl}' total count {tot} != planned {planned}")
        if tr == 0 or va == 0 or te == 0:
            errors.append(f"Table '{tbl}' missing in split coverage (Train:{tr}, Val:{va}, Test:{te})")
    print("-" * 65)

    # 9. Dataset Manifest Consistency Validation
    print("\n[9] Dataset Manifest Consistency Validation:")
    actual_stats = {
        "total_examples": total_actual,
        "split_counts": split_counts,
        "category_counts": {cat: sum(cat_counts_by_split[sp][cat] for sp in ["train", "validation", "test"]) for cat in PLANNED_CATEGORY_TOTALS},
        "table_counts": {tbl: sum(tbl_counts_by_split[sp][tbl] for sp in ["train", "validation", "test"]) for tbl in PLANNED_TABLE_TOTALS}
    }
    manifest_err_start = len(errors)
    validate_manifest(manifest_path, actual_stats, errors)
    if len(errors) == manifest_err_start:
        print("    - Defensive manifest validation (structure, distributions, types, schema): PASSED")
    else:
        print(f"    - Manifest inconsistency errors: {len(errors) - manifest_err_start}")

    # 10. Layered MySQL Server Validation (Separated: Part A Syntax vs Part B Execution)
    print("\n[10] Layered MySQL Server Validation (Separated):")
    mysql_result = validate_mysql(statically_approved_records, len(all_accepted), static_only=static_only)

    print(f"    [Part A] MySQL Syntax & Schema Validation (EXPLAIN):")
    print(f"             Status: {mysql_result['part_a_status']}")
    print(f"             Detail: {mysql_result['reason']}")

    print(f"    [Part B] MySQL Read-Only Query Execution:")
    print(f"             Status: {mysql_result['part_b_status']}")
    print(f"             Detail: {mysql_result['reason']}")

    if mysql_result["errors"]:
        errors.extend(mysql_result["errors"])

    # 11. Validation Scope and Limitations Notice
    print("\n[11] Validation Scope & Limitations Notice:")
    print("    1. Syntactic AST Allowlist vs. Semantic Equivalence:")
    print("       Passing AST allowlist checks guarantees structural validity and schema adherence,")
    print("       but does not prove that the SQL query semantically answers the natural-language question.")
    print("    2. Lexical Normalization vs. Semantic Paraphrases:")
    print("       Normalized string comparison identifies exact duplicates and token overlaps, but cannot")
    print("       guarantee detection of all semantic paraphrases. Semantic review requires human inspection.")
    print("    3. EXPLAIN Analysis vs. Result Correctness:")
    print("       MySQL EXPLAIN validates query plan resolution and column typing, but does not verify")
    print("       whether returned rows satisfy the question's semantic intent.")
    print("    4. Actual Read-Only Execution vs. Semantic Alignment:")
    print("       Successful read-only query execution guarantees database engine operability without modification,")
    print("       but does not prove that the resulting records match user intent.")
    print("    5. Layered Read-Only Safekeeping:")
    print("       Transaction-level read-only mode is an operational safety layer and not a replacement for")
    print("       proper database user privileges and static SQL validation.")
    print("    6. Manual Verification Requirement:")
    print("       Independent human review of question-query pairs remains required for true ground-truth alignment.")

    # 12. Overall Status Determination
    print("\n" + "=" * 80)
    if errors:
        print(f"OVERALL STATUS: FAILED ({len(errors)} validation error(s) detected)")
        return False, 1
    elif mysql_result["part_a_status"] == "PENDING" or mysql_result["part_b_status"] == "PENDING":
        print("OVERALL STATUS: INCOMPLETE (Static validation passed; MySQL execution validation pending credential configuration)")
        return False, 2
    elif static_only:
        print("OVERALL STATUS: PASSED (Static Dataset & SQL Parser Checks Only; MySQL Validation Skipped via --static-only)")
        return True, 0
    else:
        print("OVERALL STATUS: PASSED (All Static AST, Manifest, EXPLAIN, and Read-Only Execution Checks Completed Successfully)")
        return True, 0

# =============================================================================
# 10. CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Dataset V1 Validator for Schema-Aware NL2SQL.")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run static dataset, manifest, and SQL AST checks only, skipping live MySQL connection."
    )
    args = parser.parse_args()

    _, code = run_validation(static_only=args.static_only)
    sys.exit(code)

if __name__ == "__main__":
    main()
