"""
Automated Anti-Leakage, Provenance, Projection, and Integrity Gate (Phase 2, Step 7B).

Provides the reusable fail-closed acceptance gate validating candidate NL2SQL datasets
against the immutable Population A baseline (`data/train.jsonl`), read-only held-out
evaluation splits (`validation.jsonl`, `test.jsonl`, `benchmark_eval.jsonl`), a formal
machine-verifiable literal provenance manifest, projection-policy rules, and live MySQL
schema/catalog/execution verification via SQLValidator and SQLExecutor.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sqlglot
from sqlglot import exp

try:
    from sqlglot.errors import ParseError, TokenError
except Exception:  # pragma: no cover
    ParseError = Exception  # type: ignore
    TokenError = Exception  # type: ignore

from src.data.dataset import load_jsonl_records
from src.database.executor import SQLExecutor
from src.evaluation.comparator import ASTComparator
from src.validation.validator import SQLValidator, normalize_schema

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "with",
        "who",
        "whose",
        "that",
        "which",
        "where",
        "all",
        "every",
        "each",
        "any",
        "are",
        "is",
        "be",
        "have",
        "having",
        "has",
        "show",
        "list",
        "display",
        "find",
        "get",
        "fetch",
        "retrieve",
        "return",
        "give",
        "provide",
        "print",
        "output",
        "report",
        "pull",
        "me",
        "students",
        "student",
        "courses",
        "course",
        "enrollments",
        "enrollment",
        "learners",
        "learner",
        "subjects",
        "subject",
        "records",
        "record",
        "rows",
        "row",
        "entries",
        "entry",
        "details",
    }
)

# Authoritative schema and literal-context rules verified against nl2sql_student_db.
APPROVED_LIMIT_VALUES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16})

IDENTIFIER_COLUMNS_BY_TABLE: Dict[str, Set[str]] = {
    "students": {"student_id"},
    "courses": {"course_id"},
    "enrollments": {"enrollment_id", "student_id", "course_id"},
}

DEFAULT_IDENTIFIER_CATALOG: Dict[Tuple[str, str], Set[int]] = {
    ("students", "student_id"): set(range(1, 21)),
    ("courses", "course_id"): set(range(1, 9)),
    ("enrollments", "enrollment_id"): set(range(1, 50)),
    ("enrollments", "student_id"): set(range(1, 21)),
    ("enrollments", "course_id"): set(range(1, 9)),
}

COLUMN_SQL_TYPES: Dict[Tuple[str, str], str] = {
    ("students", "student_id"): "integer",
    ("students", "name"): "string",
    ("students", "age"): "integer",
    ("students", "department"): "string",
    ("students", "cgpa"): "decimal",
    ("courses", "course_id"): "integer",
    ("courses", "course_name"): "string",
    ("courses", "credits"): "integer",
    ("enrollments", "enrollment_id"): "integer",
    ("enrollments", "student_id"): "integer",
    ("enrollments", "course_id"): "integer",
    ("enrollments", "semester"): "string",
    ("enrollments", "marks"): "decimal",
}

NUMERIC_METRIC_DOMAINS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("students", "cgpa"): (6.85, 9.40),
    ("students", "age"): (20.0, 22.0),
    ("courses", "credits"): (3.0, 4.0),
    ("enrollments", "marks"): (67.50, 96.00),
}

DISCRETE_METRIC_VALUES: Dict[Tuple[str, str], Set[int]] = {
    ("students", "age"): {20, 21, 22},
    ("courses", "credits"): {3, 4},
}

ALLOWED_SQL_CONTEXTS: frozenset[str] = frozenset(
    {"comparison_threshold", "identifier", "count", "limit", "closed_domain_string"}
)
ALLOWED_DISPOSITIONS: frozenset[str] = frozenset({"approved_non_leaking", "rejected_requires_revision"})
ALLOWED_PROJECTION_DISPOSITIONS: frozenset[str] = frozenset(
    {"approved_projection_alignment", "rejected_requires_revision"}
)
ALLOWED_ZERO_ROW_DISPOSITIONS: frozenset[str] = frozenset(
    {"approved_zero_row", "rejected_requires_revision"}
)
EXCLUDED_NEW_LITERALS_DEFAULT: frozenset[str] = frozenset({"Machine Learning"})

SUPPORTED_AST_TRANSFORMATIONS: List[str] = [
    "SQLGlot MySQL keyword/identifier normalization and canonical formatting",
    "Associative & commutative conjunction flattening: (A AND B) AND C == C AND (B AND A)",
    "Associative & commutative disjunction flattening: (A OR B) OR C == C OR (B OR A)",
    "Precedence-preserving Boolean nesting: (A OR B) AND C != A OR (B AND C)",
    "MySQL inequality operator normalization: != and <> both normalize to AST NEQ",
]

REQUIRED_MANIFEST_FIELDS: Tuple[str, ...] = (
    "literal_id",
    "value",
    "data_type",
    "table",
    "column",
    "sql_context",
    "generation_rule",
    "is_observed_in_db",
    "approved_domain",
    "domain_source",
    "example_ids",
)

REQUIRED_EXAMPLE_FIELDS: Tuple[str, ...] = (
    "id",
    "question",
    "sql",
    "source_method",
    "table",
    "columns_referenced",
    "literals_used",
    "executed",
    "execution_classification",
    "row_count",
    "ast_template",
    "overlap_status",
    "projection_review_status",
)

LEVEL1_PROJECTION_HARD_VIOLATIONS: frozenset[str] = frozenset(
    {"invalid_or_foreign_projection_column", "duplicate_projection_column"}
)

# Ordered projection column phrase patterns (matched greedily on non-overlapping character spans).
PROJECTION_COLUMN_PATTERNS: List[Tuple[str, Optional[str], str]] = [
    # Explicit foreign / composite IDs first
    (r"\b(?:enrollment|registration|entry)\s+(?:ids?|identifiers?|codes?)\b", "enrollments", "enrollment_id"),
    (r"\b(?:enrollment_id)\b", "enrollments", "enrollment_id"),
    (r"\b(?:student|learner)\s+(?:ids?|identifiers?|codes?)\b", None, "student_id"),
    (r"\b(?:student_id)\b", None, "student_id"),
    (r"\b(?:course|subject)\s+(?:ids?|identifiers?|codes?)\b", None, "course_id"),
    (r"\b(?:course_id)\b", None, "course_id"),
    # Multi-word attribute phrases
    (r"\b(?:course|subject)\s+(?:names?|titles?)\b", "courses", "course_name"),
    (r"\b(?:course_name)\b", "courses", "course_name"),
    (r"\b(?:student\s+full\s+names?|learner\s+names?|full\s+names?|student\s+names?|surnames?)\b", "students", "name"),
    (
        r"\b(?:grade\s+point\s+averages?|cumulative\s+grade\s+points?|academic\s+gpas?|cumulative\s+gpas?|grade\s+points?|gpas?|cgpas?)\b",
        "students",
        "cgpa",
    ),
    (r"\b(?:credit\s+hours?|credit\s+units?|credit\s+counts?|credit\s+values?|credit\s+weights?|credits)\b", "courses", "credits"),
    (r"\b(?:academic\s+terms?|semesters?|terms?)\b", "enrollments", "semester"),
    (r"\b(?:exam\s+marks?|exam\s+scores?|marks?|scores?)\b", "enrollments", "marks"),
    (r"\b(?:years\s+of\s+age|ages?)\b", "students", "age"),
    (r"\b(?:department\s+names?|departments?|majors?|branches?|branch|streams?)\b", "students", "department"),
    # Table-scoped bare terms checked strictly on unconsumed spans
    (r"\b(?:titles?)\b", "courses", "course_name"),
    (r"\b(?:names?)\b", "students", "name"),
    (r"\b(?:identifiers?|ids?)\b", "__table_pk__", "__table_pk__"),
]

PROJECTION_VERB_SPLIT = re.compile(
    r"\b(?:show|list|display|return|fetch|retrieve|print|output|report|pull|give|find|get)\b",
    re.IGNORECASE,
)

PROJECTION_TAIL_SPLIT = re.compile(
    r"(?:(?<!along\s)\bwith\b|"
    r"\bafter\s+(?:sorting|ordering|ranking|arranging)\b|"
    r"\b(?:where|who|that|whose|if|scoring|earning|not|without|holding|carrying|having|aged|"
    r"older\s+than|younger\s+than|above|below|over|under|between|greater|less|fewer|"
    r"more\s+than|at\s+(?:least|most)|top\s+\d+|first\s+\d+|"
    r"strictly|outside|excluding|except|other\s+than|differ(?:s)?\s+from|sorted|ordered|ranked|"
    r"arranged|limited\s+to|achieved|recorded|during)\b|"
    r"\bin\s+(?:which|semester\s+\d+|the\s+(?:fifth|sixth|5th|6th)\s+semester|"
    r"cse|ece|ise|me|ai&ds|computer\s+science|mechanical|electronics|information\s+science|"
    r"artificial\s+intelligence)\b|"
    r"\bof\s+(?:the\s+|all\s+|every\s+|each\s+|\d+|non[- ]|(?:students?|courses?|enrollments?|"
    r"learners?|majors?|entries?|records?|rows?|student_id|course_id|enrollment_id)\b)|"
    r"\bfor\s+(?:the\s+|all\s+|every\s+|each\s+|\d+(?:st|nd|rd|th)|fifth|sixth|\d+|non[- ]|"
    r"(?:computer\s+science|mechanical|electronics|information\s+science|artificial\s+intelligence|"
    r"cse|ece|ise|me|ai&ds|semester|student_id|course_id|enrollment_id|"
    r"students?|courses?|enrollments?|learners?|majors?|entries?|records?|rows?|"
    r"cgpas?|gpas?|marks|scores?|credits?|ages?|departments?|semesters?)\b))",
    re.IGNORECASE,
)


def normalize_question(text: str) -> str:
    """Normalizes natural-language text by lowercasing and stripping punctuation."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def normalize_numeric_str(raw: Union[str, int, float]) -> str:
    """
    Canonicalizes numeric token strings while preserving integer vs decimal distinction.

    Examples: '7' -> '7', '7.0' -> '7.0', '8.80' -> '8.8', '70.00' -> '70.0'.
    """
    s = str(raw).strip()
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    if "." in s:
        norm = format(d.normalize(), "f")
        return norm if "." in norm else f"{norm}.0"
    return str(int(d))


def canonicalize_sql(sql: str, dialect: str = "mysql") -> str:
    """Returns canonical single-statement SQL string using SQLGlot MySQL dialect."""
    parsed = sqlglot.parse_one(sql.strip(), read=dialect)
    if not isinstance(parsed, exp.Select):
        raise ValueError(f"Expected SELECT statement, got {type(parsed).__name__}")
    return parsed.sql(dialect=dialect).lower()


def _canonicalize_expression_tuple(
    comparator: ASTComparator,
    sql: str,
    mask_literals: bool = False,
) -> Tuple[Any, ...]:
    """
    Produces a commutative AST signature invariant to AND/OR operand permutation
    while preserving operator precedence, column projection order, and LIMIT presence.
    """
    parsed = sqlglot.parse_one(sql.strip(), read=comparator.dialect)
    if mask_literals:
        def _mask(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Literal):
                return exp.Literal.string("<STR>") if node.is_string else exp.Literal.number("<NUM>")
            return node

        parsed = parsed.transform(_mask)

    clauses = comparator.extract_clauses(parsed)
    limit_sig = "<NUM>" if (mask_literals and parsed.find(exp.Limit) is not None) else clauses["limit"]
    return (
        clauses["table"],
        tuple(clauses["select"]),
        clauses["where_canonical"],
        tuple(clauses["order"]),
        limit_sig,
    )


def mask_sql_literals(sql: str, dialect: str = "mysql") -> str:
    """Replaces numeric literals with <NUM> and string literals with '<STR>' in SQLGlot AST."""
    parsed = sqlglot.parse_one(sql.strip(), read=dialect)

    def _mask(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal):
            return exp.Literal.string("<STR>") if node.is_string else exp.Literal.number("<NUM>")
        return node

    return parsed.transform(_mask).sql(dialect=dialect).lower()


def extract_sql_literals(sql: str, dialect: str = "mysql") -> Tuple[Set[str], Set[str]]:
    """Extracts (normalized_numeric_literals, string_literals) from SQL using SQLGlot AST."""
    parsed = sqlglot.parse_one(sql.strip(), read=dialect)
    nums: set[str] = set()
    strs: set[str] = set()
    for node in parsed.walk():
        if isinstance(node, exp.Literal):
            if node.is_string:
                strs.add(str(node.this))
            else:
                nums.add(normalize_numeric_str(str(node.this)))
    return nums, strs


def extract_contextual_literals(sql: str, dialect: str = "mysql") -> List[Dict[str, Any]]:
    """
    Extracts every SQL literal occurrence annotated with (value, data_type, table, column, sql_context).
    Separates 'limit', 'identifier', 'count', 'comparison_threshold', and 'closed_domain_string'.
    """
    parsed = sqlglot.parse_one(sql.strip(), read=dialect)
    from_expr = parsed.args.get("from_")
    table_name = from_expr.name.lower() if from_expr and hasattr(from_expr, "name") else ""
    id_cols = IDENTIFIER_COLUMNS_BY_TABLE.get(table_name, set())

    occurrences: List[Dict[str, Any]] = []
    for node in parsed.walk():
        if not isinstance(node, exp.Literal):
            continue

        raw_str = str(node.this)
        in_limit = node.find_ancestor(exp.Limit) is not None
        if in_limit:
            try:
                val: Any = int(Decimal(raw_str))
            except (InvalidOperation, ValueError):
                val = raw_str
            occurrences.append(
                {
                    "value": val,
                    "value_norm": normalize_numeric_str(raw_str),
                    "data_type": "integer" if "." not in raw_str else "decimal",
                    "table": table_name,
                    "column": None,
                    "sql_context": "limit",
                }
            )
            continue

        # Check aggregate count comparison (e.g., COUNT(*) > 5)
        comp_anc = node.find_ancestor((exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE))
        has_agg = comp_anc is not None and any(isinstance(a, exp.AggFunc) for a in comp_anc.walk())
        if has_agg:
            try:
                val = int(Decimal(raw_str))
            except (InvalidOperation, ValueError):
                val = raw_str
            occurrences.append(
                {
                    "value": val,
                    "value_norm": normalize_numeric_str(raw_str),
                    "data_type": "integer",
                    "table": table_name,
                    "column": None,
                    "sql_context": "count",
                }
            )
            continue

        col_name = ""
        if comp_anc is not None:
            for c in comp_anc.find_all(exp.Column):
                col_name = c.name.lower()
                break

        if node.is_string:
            occurrences.append(
                {
                    "value": raw_str,
                    "value_norm": raw_str,
                    "data_type": "string",
                    "table": table_name,
                    "column": col_name,
                    "sql_context": "closed_domain_string",
                }
            )
        else:
            is_dec = "." in raw_str
            dtype = "decimal" if is_dec else "integer"
            val = float(raw_str) if is_dec else int(raw_str)
            ctx = "identifier" if col_name in id_cols else "comparison_threshold"
            occurrences.append(
                {
                    "value": val,
                    "value_norm": normalize_numeric_str(raw_str),
                    "data_type": dtype,
                    "table": table_name,
                    "column": col_name,
                    "sql_context": ctx,
                }
            )
    return occurrences


def extract_clause_columns(sql: str, dialect: str = "mysql") -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Extracts (table, sorted_where_columns, order_columns) from SQL."""
    parsed = sqlglot.parse_one(sql.strip(), read=dialect)
    from_expr = parsed.args.get("from_")
    table_name = from_expr.name.lower() if from_expr and hasattr(from_expr, "name") else ""
    where_cols: set[str] = set()
    where_expr = parsed.args.get("where")
    if where_expr is not None:
        for col in where_expr.find_all(exp.Column):
            where_cols.add(col.name.lower())
    order_cols: list[str] = []
    order_expr = parsed.args.get("order")
    if order_expr is not None:
        for ordered in order_expr.expressions:
            for col in ordered.find_all(exp.Column):
                order_cols.append(col.name.lower())
    return table_name, tuple(sorted(where_cols)), tuple(order_cols)


def _extract_projection_nl_span(question: str) -> str:
    """Isolates the projection-bearing clause(s) of a natural-language question."""
    text = question.strip()
    if "?" in text:
        parts = [p.strip() for p in text.split("?") if p.strip()]
        if len(parts) >= 2 and PROJECTION_VERB_SPLIT.search(parts[-1]):
            text = parts[-1]
        else:
            text = parts[0]

    # Advance into "... showing/returning col1, col2" while still stripping any trailing filters/ORDER BY
    m_showing = re.search(r"\b(?:showing|returning)\b", text, re.IGNORECASE)
    if m_showing and m_showing.start() > 15:
        text = text[m_showing.start() :]
    else:
        # Strip leading filter preamble preceding a comma or coordinating "and" + request verb
        m_verb = PROJECTION_VERB_SPLIT.search(text)
        if m_verb:
            prefix = text[: m_verb.start()]
            if "," in prefix or re.search(r"(?:^|\s)and\s*$", prefix, re.IGNORECASE):
                text = text[m_verb.start() :]

    m_tail = PROJECTION_TAIL_SPLIT.search(text)
    if m_tail:
        return text[: m_tail.start()].strip()
    return text.strip()


def _extract_ordered_column_mentions(
    proj_span: str,
    table: str,
    schema_dict: Dict[str, List[str]],
) -> List[Tuple[str, int]]:
    """
    Matches canonical column names & domain synonyms inside proj_span on non-overlapping spans,
    returning [(column_name, start_char_index), ...] ordered by appearance position.
    """
    table_cols = set(schema_dict.get(table, []))
    pk_col = {
        "students": "student_id",
        "courses": "course_id",
        "enrollments": "enrollment_id",
    }.get(table, "")

    occupied = [False] * len(proj_span)
    matches: List[Tuple[str, int]] = []

    for pat, scope_tbl, target_col in PROJECTION_COLUMN_PATTERNS:
        if scope_tbl == "__table_pk__":
            resolved_col = pk_col
        elif scope_tbl is not None and scope_tbl != table:
            continue
        else:
            resolved_col = target_col

        if resolved_col not in table_cols:
            continue

        for m in re.finditer(pat, proj_span, re.IGNORECASE):
            s, e = m.span()
            if any(occupied[s:e]):
                continue
            for idx in range(s, e):
                occupied[idx] = True
            matches.append((resolved_col, s))

    # Fallback: also allow exact schema column names verbatim if not already matched
    for col in schema_dict.get(table, []):
        for m in re.finditer(rf"\b{re.escape(col)}\b", proj_span, re.IGNORECASE):
            s, e = m.span()
            if any(occupied[s:e]):
                continue
            for idx in range(s, e):
                occupied[idx] = True
            matches.append((col, s))

    matches.sort(key=lambda item: item[1])
    return matches


def compute_text_similarities(
    q1: str,
    q2: str,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> Dict[str, float]:
    """Computes complementary token, content-word, bigram, and character sequence similarities."""
    n1 = normalize_question(q1)
    n2 = normalize_question(q2)
    t1 = n1.split()
    t2 = n2.split()
    s1 = set(t1)
    s2 = set(t2)
    c1 = s1 - stopwords
    c2 = s2 - stopwords
    b1 = set(zip(t1, t1[1:])) if len(t1) >= 2 else set(t1)
    b2 = set(zip(t2, t2[1:])) if len(t2) >= 2 else set(t2)

    jaccard = len(s1 & s2) / max(1, len(s1 | s2))
    content_jaccard = len(c1 & c2) / max(1, len(c1 | c2)) if (c1 or c2) else 0.0
    bigram_jaccard = len(b1 & b2) / max(1, len(b1 | b2)) if (b1 or b2) else 0.0
    seq_ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    return {
        "jaccard": round(jaccard, 4),
        "content_jaccard": round(content_jaccard, 4),
        "bigram_jaccard": round(bigram_jaccard, 4),
        "seq_ratio": round(seq_ratio, 4),
    }


@dataclass
class AntiLeakageConfig:
    """Configuration thresholds and policy settings for the automated anti-leakage gate."""

    schema: str
    dialect: str = "mysql"
    soft_jaccard_threshold: float = 0.58
    hard_jaccard_threshold: float = 0.72
    soft_content_jaccard_threshold: float = 0.58
    hard_content_jaccard_threshold: float = 0.75
    soft_bigram_threshold: float = 0.48
    hard_bigram_threshold: float = 0.65
    soft_seq_ratio_threshold: float = 0.72
    hard_seq_ratio_threshold: float = 0.82
    internal_paraphrase_seq_threshold: float = 0.88
    internal_paraphrase_jaccard_threshold: float = 0.82
    max_new_template_overlap_ratio: float = 0.30
    require_literal_manifest: bool = True
    require_manifest_v2: bool = True
    require_catalog_verification: Optional[bool] = None
    fail_on_unreviewed_soft_flags: bool = True
    enforce_projection_policy: bool = True
    use_two_level_projection: bool = True
    require_live_execution: bool = True
    max_zero_row_new_queries: int = 0
    excluded_new_literals: frozenset[str] = EXCLUDED_NEW_LITERALS_DEFAULT
    approved_limit_values: frozenset[int] = APPROVED_LIMIT_VALUES


@dataclass
class AntiLeakageReport:
    """Structured fail-closed evaluation report returned by AntiLeakageGate."""

    passed: bool
    summary: Dict[str, Any]
    hard_failures: List[Dict[str, Any]] = field(default_factory=list)
    review_flags: List[Dict[str, Any]] = field(default_factory=list)
    template_overlaps: Dict[str, Any] = field(default_factory=dict)
    literal_provenance_audit: Dict[str, Any] = field(default_factory=dict)
    projection_policy_audit: Dict[str, Any] = field(default_factory=dict)
    validation_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "hard_failures": self.hard_failures,
            "review_flags": self.review_flags,
            "template_overlaps": self.template_overlaps,
            "literal_provenance_audit": self.literal_provenance_audit,
            "projection_policy_audit": self.projection_policy_audit,
            "validation_audit": self.validation_audit,
        }


class AntiLeakageGate:
    """
    Fail-closed automated dataset integrity and anti-leakage gate.
    Separately evaluates Population A (immutable baseline), Population B (new additions),
    and Combined Corpus totals. Never alters baseline, held-out, or model files.
    """

    def __init__(self, config: AntiLeakageConfig) -> None:
        self.config = config
        self.comparator = ASTComparator(dialect=config.dialect)
        self.validator = SQLValidator(schema=config.schema, enforce_model_scope=True)
        self.schema_dict = normalize_schema(config.schema) or {}

    def evaluate(
        self,
        candidate_records: List[Dict[str, Any]],
        original_train_records: List[Dict[str, Any]],
        heldout_splits: Dict[str, List[Dict[str, Any]]],
        literal_manifest: Optional[Dict[str, Any]] = None,
        reviewed_flag_ids: Optional[Dict[str, Any]] = None,
        reviewed_projection_flags: Optional[Dict[str, Any]] = None,
        reviewed_zero_row_queries: Optional[Dict[str, Any]] = None,
        catalog_string_values: Optional[Dict[str, Set[str]]] = None,
        identifier_catalog: Optional[Dict[Tuple[str, str], Set[int]]] = None,
        observed_numeric_values: Optional[Dict[Tuple[str, str], Set[Any]]] = None,
        live_metric_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
        table_row_counts: Optional[Dict[str, int]] = None,
        executor: Optional[SQLExecutor] = None,
        db_unavailable_error: Optional[str] = None,
    ) -> AntiLeakageReport:
        """
        Runs all fail-closed checks across Population A (baseline), Population B (new additions),
        and Combined Corpus against read-only held-out evaluation splits.
        """
        hard_failures: List[Dict[str, Any]] = []
        review_flags: List[Dict[str, Any]] = []

        raw_reviewed: Dict[str, Any] = {}
        raw_proj_reviewed: Dict[str, Any] = {}
        raw_zero_row_reviewed: Dict[str, Any] = {}
        if literal_manifest:
            if isinstance(literal_manifest.get("reviewed_soft_flags"), dict):
                raw_reviewed.update(literal_manifest["reviewed_soft_flags"])
            if isinstance(literal_manifest.get("reviewed_projection_flags"), dict):
                raw_proj_reviewed.update(literal_manifest["reviewed_projection_flags"])
            if isinstance(literal_manifest.get("reviewed_zero_row_queries"), dict):
                raw_zero_row_reviewed.update(literal_manifest["reviewed_zero_row_queries"])
        if reviewed_flag_ids:
            raw_reviewed.update(reviewed_flag_ids)
        if reviewed_projection_flags:
            raw_proj_reviewed.update(reviewed_projection_flags)
        if reviewed_zero_row_queries:
            raw_zero_row_reviewed.update(reviewed_zero_row_queries)

        orig_by_id = {r["id"]: r for r in original_train_records}
        orig_records = [r for r in candidate_records if r["id"] in orig_by_id]
        new_records = [r for r in candidate_records if r["id"] not in orig_by_id]

        # 1. SQLGlot MySQL parse, SQLValidator & Live MySQL execution (4-way classification + Decision A)
        validation_audit, exec_status_by_id = self._check_sql_and_execution(
            candidate_records=candidate_records,
            orig_by_id=orig_by_id,
            executor=executor,
            db_unavailable_error=db_unavailable_error,
            reviewed_zero_row_map=raw_zero_row_reviewed,
            hard_failures=hard_failures,
        )

        # Precompute canonical representations
        cand_meta = self._precompute_records(candidate_records)
        orig_ref_meta = self._precompute_records(original_train_records)
        ho_meta = {
            split_name: self._precompute_records(recs)
            for split_name, recs in heldout_splits.items()
        }

        # 2. Population A Baseline & Population B Internal Integrity
        baseline_integrity, new_integrity = self._check_population_integrity(
            candidate_records, orig_records, new_records, orig_by_id, orig_ref_meta, cand_meta, hard_failures
        )

        # 3. Question, Canonical SQL & Commutative Boolean Overlaps (Pop A vs Pop B vs Combined)
        q_overlaps, canon_sql_overlaps, comm_ast_overlaps = self._check_question_sql_and_commutative_overlaps(
            orig_records, new_records, cand_meta, heldout_splits, ho_meta, hard_failures
        )

        # 4. Literal-Masked AST Template Overlaps (dual non-dilutable heldout ratios + new-record ratio)
        template_overlaps = self._check_template_overlaps(
            orig_records, new_records, cand_meta, heldout_splits, ho_meta, hard_failures
        )

        # 5. NL Paraphrase (hard/soft w/ disposition+rationale) & Structure+Literal Leakage
        self._check_paraphrase_and_structure_literal(
            new_records=new_records,
            orig_records=orig_records,
            cand_meta=cand_meta,
            heldout_splits=heldout_splits,
            ho_meta=ho_meta,
            literal_manifest=literal_manifest,
            reviewed_map=raw_reviewed,
            hard_failures=hard_failures,
            review_flags=review_flags,
        )

        # 6. Five-Category Machine-Verifiable Literal & Example Traceability Audit (Live DB verified)
        literal_audit = self._check_literal_provenance(
            new_records=new_records,
            original_train_records=original_train_records,
            heldout_splits=heldout_splits,
            literal_manifest=literal_manifest,
            cand_meta=cand_meta,
            exec_status_by_id=exec_status_by_id,
            reviewed_zero_row_map=raw_zero_row_reviewed,
            catalog_string_values=catalog_string_values,
            identifier_catalog=identifier_catalog,
            observed_numeric_values=observed_numeric_values,
            live_metric_bounds=live_metric_bounds,
            table_row_counts=table_row_counts,
            hard_failures=hard_failures,
        )

        # 7. Two-Level Projection-Policy Audit (Population B only; Population A untouched)
        projection_audit = self._check_projection_policy(
            new_records=new_records,
            cand_meta=cand_meta,
            reviewed_projection_map=raw_proj_reviewed,
            hard_failures=hard_failures,
        )

        # Fail closed on any soft flag lacking valid approved disposition + rationale
        unreviewed_soft = [rf for rf in review_flags if not rf.get("approved")]
        if self.config.fail_on_unreviewed_soft_flags and unreviewed_soft:
            hard_failures.append(
                {
                    "check": "unreviewed_soft_flags",
                    "count": len(unreviewed_soft),
                    "message": (
                        f"{len(unreviewed_soft)} soft similarity flag(s) lack an approved disposition "
                        "and rationale."
                    ),
                }
            )

        pop_b_q_fail = sum(
            q_overlaps[s]["population_b_new"]["exact_q"] + q_overlaps[s]["population_b_new"]["norm_q"]
            for s in heldout_splits
        )
        pop_b_csql_fail = sum(
            canon_sql_overlaps[s]["population_b_new"]["count"] for s in heldout_splits
        )
        pop_b_comm_fail = sum(
            comm_ast_overlaps[s]["population_b_new"]["count"] for s in heldout_splits
        )

        infra_incomplete = validation_audit["live_execution_status"] != "complete"
        passed = (
            len(hard_failures) == 0
            and validation_audit["invalid_query_count"] == 0
            and validation_audit["execution_infrastructure_failure_count"] == 0
            and not infra_incomplete
            and pop_b_q_fail == 0
            and pop_b_csql_fail == 0
            and pop_b_comm_fail == 0
        )

        if passed:
            gate_verdict = "PASSED"
        elif infra_incomplete or validation_audit["execution_infrastructure_failure_count"] > 0:
            gate_verdict = "INCOMPLETE / FAILED"
        else:
            gate_verdict = "FAILED"

        failure_reasons = [hf.get("check", "unknown_failure") for hf in hard_failures]
        summary = {
            "total_records": len(candidate_records),
            "population_a_baseline_count": len(orig_records),
            "population_b_new_count": len(new_records),
            "original_baseline_integrity": baseline_integrity,
            "new_addition_integrity": new_integrity,
            "supported_ast_transformations": list(SUPPORTED_AST_TRANSFORMATIONS),
            "exact_question_overlaps": q_overlaps,
            "canonical_sql_overlaps": canon_sql_overlaps,
            "commutative_boolean_overlaps": comm_ast_overlaps,
            "hard_failure_count": len(hard_failures),
            "soft_review_flag_count": len(review_flags),
            "unreviewed_soft_flag_count": len(unreviewed_soft),
            "projection_hard_failure_count": projection_audit.get("hard_failure_count", 0),
            "projection_review_flag_count": projection_audit.get("review_flag_count", 0),
            "unreviewed_projection_flag_count": projection_audit.get("unreviewed_flag_count", 0),
            "zero_row_new_query_count": validation_audit.get("zero_row_new_query_count", 0),
            "unjustified_zero_row_new_count": validation_audit.get("unjustified_zero_row_new_count", 0),
            "failure_reasons": sorted(set(failure_reasons)),
            "gate_verdict": gate_verdict,
        }
        return AntiLeakageReport(
            passed=passed,
            summary=summary,
            hard_failures=hard_failures,
            review_flags=review_flags,
            template_overlaps=template_overlaps,
            literal_provenance_audit=literal_audit,
            projection_policy_audit=projection_audit,
            validation_audit=validation_audit,
        )

    def _precompute_records(self, records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        meta: Dict[int, Dict[str, Any]] = {}
        for r in records:
            rid = r["id"]
            sql = r.get("sql") or r.get("reference_sql", "")
            q = r.get("question", "")
            try:
                csql = canonicalize_sql(sql, dialect=self.config.dialect)
                cast_sig = _canonicalize_expression_tuple(self.comparator, sql, mask_literals=False)
                tsql = mask_sql_literals(sql, dialect=self.config.dialect)
                tast_sig = _canonicalize_expression_tuple(self.comparator, sql, mask_literals=True)
                nums, strs = extract_sql_literals(sql, dialect=self.config.dialect)
                tbl, wcols, ocols = extract_clause_columns(sql, dialect=self.config.dialect)
                ctx_lits = extract_contextual_literals(sql, dialect=self.config.dialect)
            except (ParseError, TokenError, ValueError):
                csql = f"INVALID::{sql}"
                cast_sig = ("INVALID", sql)
                tsql = f"INVALID::{sql}"
                tast_sig = ("INVALID", sql)
                nums, strs = set(), set()
                tbl, wcols, ocols = "", (), ()
                ctx_lits = []
            meta[rid] = {
                "raw_q": q.strip(),
                "norm_q": normalize_question(q),
                "sql": sql,
                "canon_sql": csql,
                "canon_ast_sig": cast_sig,
                "template_sql": tsql,
                "template_ast_sig": tast_sig,
                "nums": nums,
                "strs": strs,
                "ctx_lits": ctx_lits,
                "table": tbl,
                "where_cols": wcols,
                "order_cols": ocols,
            }
        return meta

    def _check_sql_and_execution(
        self,
        candidate_records: List[Dict[str, Any]],
        orig_by_id: Dict[int, Dict[str, Any]],
        executor: Optional[SQLExecutor],
        db_unavailable_error: Optional[str],
        reviewed_zero_row_map: Dict[str, Any],
        hard_failures: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        parse_failures: List[Dict[str, Any]] = []
        validator_failures: List[Dict[str, Any]] = []
        valid_with_rows_count = 0
        valid_zero_rows: List[Dict[str, Any]] = []
        zero_row_new_ids: List[int] = []
        approved_zero_row_new_ids: List[int] = []
        unjustified_zero_row_new: List[Dict[str, Any]] = []
        infra_failures: List[Dict[str, Any]] = []
        exec_status_by_id: Dict[int, Dict[str, Any]] = {}
        executed_count = 0

        if self.config.require_live_execution and (executor is None or db_unavailable_error):
            msg = db_unavailable_error or "Live SQLExecutor is required; database execution cannot be skipped."
            infra_failures.append({"id": None, "error": msg, "sql": None})
            hard_failures.append(
                {"check": "execution_infrastructure_failure", "count": 1, "details": infra_failures}
            )

        for r in candidate_records:
            rid = r["id"]
            sql = r.get("sql", "")
            rec_errs: List[str] = []

            if not r.get("question", "").strip():
                rec_errs.append("Empty question")
            if not r.get("category", "").strip():
                rec_errs.append("Empty category")
            if not r.get("table", "").strip():
                rec_errs.append("Empty table")
            if r.get("schema") != self.config.schema:
                rec_errs.append("Schema string mismatch vs canonical schema")

            ok_parse, parse_err, parsed = self.comparator.parse_sql(sql)
            if not ok_parse or parsed is None:
                parse_failures.append({"id": rid, "error": str(parse_err), "sql": sql})
                rec_errs.append(f"SQLGlot parse error: {parse_err}")

            vres = self.validator.validate(sql)
            if not vres.is_valid:
                rec_errs.append(f"SQLValidator error: {vres.error_message}")
            elif r.get("table") and [t.lower() for t in vres.ast_details.get("tables", [])] != [
                r["table"].lower()
            ]:
                rec_errs.append(
                    f"Table field mismatch: record='{r.get('table')}' vs SQL={vres.ast_details.get('tables', [])}"
                )

            if rec_errs:
                validator_failures.append({"id": rid, "errors": rec_errs, "sql": sql})
                exec_status_by_id[rid] = {
                    "executed": False,
                    "classification": "invalid_query",
                    "row_count": 0,
                }
                continue

            if executor is not None:
                eres = executor.execute(vres, approved=True, max_rows=500)
                executed_count += 1
                if not eres.success:
                    infra_failures.append({"id": rid, "error": eres.error, "sql": sql})
                    exec_status_by_id[rid] = {
                        "executed": True,
                        "classification": "execution_infrastructure_failure",
                        "row_count": 0,
                    }
                elif len(eres.rows) == 0:
                    pop = "baseline" if rid in orig_by_id else "new"
                    is_zero_approved = False
                    zero_disp = None
                    zero_rat = None
                    if pop == "new":
                        zero_row_new_ids.append(rid)
                        rev = reviewed_zero_row_map.get(str(rid))
                        if isinstance(rev, dict):
                            zero_disp = rev.get("disposition")
                            zero_rat = (rev.get("rationale") or "").strip()
                            is_zero_approved = (
                                zero_disp == "approved_zero_row"
                                and bool(zero_rat)
                                and len(zero_rat) >= 10
                            )
                        if is_zero_approved:
                            approved_zero_row_new_ids.append(rid)
                        else:
                            unjustified_zero_row_new.append(
                                {"id": rid, "question": r["question"], "sql": sql, "disposition": zero_disp}
                            )

                    valid_zero_rows.append(
                        {
                            "id": rid,
                            "population": pop,
                            "question": r["question"],
                            "sql": sql,
                            "approved_zero_row": is_zero_approved if pop == "new" else True,
                            "disposition": zero_disp,
                            "rationale": zero_rat,
                        }
                    )
                    exec_status_by_id[rid] = {
                        "executed": True,
                        "classification": "valid_zero_rows",
                        "row_count": 0,
                    }
                else:
                    valid_with_rows_count += 1
                    exec_status_by_id[rid] = {
                        "executed": True,
                        "classification": "valid_with_rows",
                        "row_count": int(eres.row_count if hasattr(eres, "row_count") and eres.row_count is not None else len(eres.rows)),
                    }
            else:
                exec_status_by_id[rid] = {
                    "executed": False,
                    "classification": "execution_infrastructure_failure",
                    "row_count": 0,
                }

        if validator_failures:
            hard_failures.append(
                {
                    "check": "sql_validity_and_schema",
                    "count": len(validator_failures),
                    "details": validator_failures[:20],
                }
            )
        if infra_failures and not any(hf["check"] == "execution_infrastructure_failure" for hf in hard_failures):
            hard_failures.append(
                {
                    "check": "execution_infrastructure_failure",
                    "count": len(infra_failures),
                    "details": infra_failures[:20],
                }
            )
        if unjustified_zero_row_new:
            hard_failures.append(
                {
                    "check": "unjustified_zero_row_new_queries",
                    "count": len(unjustified_zero_row_new),
                    "details": unjustified_zero_row_new[:20],
                    "message": (
                        f"{len(unjustified_zero_row_new)} new Population B query/queries returned 0 live DB rows "
                        "without an approved_zero_row disposition and rationale."
                    ),
                }
            )
        elif (
            self.config.max_zero_row_new_queries >= 0
            and len(zero_row_new_ids) > self.config.max_zero_row_new_queries
            and len(approved_zero_row_new_ids) < len(zero_row_new_ids)
        ):
            hard_failures.append(
                {
                    "check": "excessive_zero_row_new_queries",
                    "count": len(zero_row_new_ids),
                    "limit": self.config.max_zero_row_new_queries,
                }
            )

        total = len(candidate_records)
        valid_count = total - len(validator_failures)
        if self.config.require_live_execution:
            live_status = "complete" if (executor is not None and not db_unavailable_error and not infra_failures) else "incomplete_or_unavailable"
        else:
            live_status = "complete" if not infra_failures else "incomplete_or_unavailable"

        audit = {
            "total_checked": total,
            "sqlglot_parse_pass_count": total - len(parse_failures),
            "sqlglot_parse_rate": round((total - len(parse_failures)) / max(1, total), 4),
            "sqlvalidator_pass_count": valid_count,
            "sqlvalidator_compliance_rate": round(valid_count / max(1, total), 4),
            "valid_with_rows_count": valid_with_rows_count,
            "valid_zero_rows_count": len(valid_zero_rows),
            "valid_zero_rows_queries": valid_zero_rows,
            "zero_row_new_query_count": len(zero_row_new_ids),
            "approved_zero_row_new_count": len(approved_zero_row_new_ids),
            "unjustified_zero_row_new_count": len(unjustified_zero_row_new),
            "invalid_query_count": len(validator_failures),
            "invalid_records": validator_failures,
            "executed_count": executed_count,
            "execution_infrastructure_failure_count": len(infra_failures),
            "execution_infrastructure_failures": infra_failures,
            "live_execution_status": live_status,
        }
        return audit, exec_status_by_id

    def _check_population_integrity(
        self,
        records: List[Dict[str, Any]],
        orig_records: List[Dict[str, Any]],
        new_records: List[Dict[str, Any]],
        orig_by_id: Dict[int, Dict[str, Any]],
        orig_ref_meta: Dict[int, Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        hard_failures: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        id_counts = Counter(r["id"] for r in records)
        dup_ids = [i for i, c in id_counts.items() if c > 1]
        if dup_ids:
            hard_failures.append({"check": "internal_duplicate_ids", "ids": dup_ids})

        # Population A check: all baseline IDs must remain present and SQL canonically unchanged
        present_orig_ids = {r["id"] for r in orig_records}
        missing_orig_ids = sorted(set(orig_by_id.keys()) - present_orig_ids)
        if missing_orig_ids:
            hard_failures.append(
                {
                    "check": "original_baseline_missing_records",
                    "count": len(missing_orig_ids),
                    "ids": missing_orig_ids[:20],
                }
            )

        sql_mutated_orig = []
        clarified_orig_qs = []
        for r in orig_records:
            rid = r["id"]
            expected_sql = orig_ref_meta[rid]["canon_sql"]
            if cand_meta[rid]["canon_sql"] != expected_sql:
                sql_mutated_orig.append(
                    {"id": rid, "orig_sql": expected_sql, "new_sql": cand_meta[rid]["canon_sql"]}
                )
            elif cand_meta[rid]["raw_q"] != orig_ref_meta[rid]["raw_q"]:
                clarified_orig_qs.append(rid)

        if sql_mutated_orig:
            hard_failures.append({"check": "original_sql_altered", "details": sql_mutated_orig})

        # Population B vs Combined duplicate questions & internal near-paraphrases
        orig_norm_q = {orig_ref_meta[oid]["norm_q"]: oid for oid in orig_by_id}
        norm_q_map: Dict[str, List[int]] = defaultdict(list)
        canon_sql_groups: Dict[Any, List[int]] = defaultdict(list)
        for r in records:
            m = cand_meta[r["id"]]
            norm_q_map[m["norm_q"]].append(r["id"])
            canon_sql_groups[m["canon_ast_sig"]].append(r["id"])

        dup_norm_qs = {k: v for k, v in norm_q_map.items() if len(v) > 1}
        if dup_norm_qs:
            hard_failures.append({"check": "internal_duplicate_questions", "details": dup_norm_qs})

        new_dup_orig_sql = []
        orig_ast_map = {orig_ref_meta[oid]["canon_ast_sig"]: oid for oid in orig_by_id}
        for nr in new_records:
            nid = nr["id"]
            if cand_meta[nid]["canon_ast_sig"] in orig_ast_map:
                new_dup_orig_sql.append(
                    {"new_id": nid, "orig_id": orig_ast_map[cand_meta[nid]["canon_ast_sig"]], "sql": nr["sql"]}
                )
            elif cand_meta[nid]["norm_q"] in orig_norm_q:
                new_dup_orig_sql.append(
                    {"new_id": nid, "orig_id": orig_norm_q[cand_meta[nid]["norm_q"]], "sql": nr["sql"]}
                )

        if new_dup_orig_sql:
            hard_failures.append({"check": "new_record_duplicates_original_train", "details": new_dup_orig_sql})

        near_dup_internal: List[Dict[str, Any]] = []
        for id_list in canon_sql_groups.values():
            if len(id_list) > 1:
                for i in range(len(id_list)):
                    for j in range(i + 1, len(id_list)):
                        id1, id2 = id_list[i], id_list[j]
                        sims = compute_text_similarities(cand_meta[id1]["raw_q"], cand_meta[id2]["raw_q"])
                        if (
                            sims["seq_ratio"] >= self.config.internal_paraphrase_seq_threshold
                            or sims["jaccard"] >= self.config.internal_paraphrase_jaccard_threshold
                        ):
                            near_dup_internal.append(
                                {"id_a": id1, "id_b": id2, "sims": sims, "sql": cand_meta[id1]["sql"]}
                            )
        if near_dup_internal:
            hard_failures.append({"check": "internal_near_duplicates", "details": near_dup_internal})

        baseline_integrity = {
            "expected_baseline_count": len(orig_by_id),
            "present_baseline_count": len(orig_records),
            "missing_baseline_count": len(missing_orig_ids),
            "original_sql_altered_count": len(sql_mutated_orig),
            "question_clarified_ids": sorted(clarified_orig_qs),
        }
        new_integrity = {
            "new_record_count": len(new_records),
            "duplicate_id_count": len(dup_ids),
            "duplicate_norm_question_count": len(dup_norm_qs),
            "new_duplicates_original_count": len(new_dup_orig_sql),
            "internal_near_duplicate_count": len(near_dup_internal),
            "shared_canon_sql_groups_count": sum(1 for v in canon_sql_groups.values() if len(v) > 1),
        }
        return baseline_integrity, new_integrity

    def _check_question_sql_and_commutative_overlaps(
        self,
        orig_records: List[Dict[str, Any]],
        new_records: List[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        heldout_splits: Dict[str, List[Dict[str, Any]]],
        ho_meta: Dict[str, Dict[int, Dict[str, Any]]],
        hard_failures: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        def _index_pop(recs: List[Dict[str, Any]]) -> Dict[str, Dict[Any, List[int]]]:
            rq: Dict[Any, List[int]] = defaultdict(list)
            nq: Dict[Any, List[int]] = defaultdict(list)
            cs: Dict[Any, List[int]] = defaultdict(list)
            ca: Dict[Any, List[int]] = defaultdict(list)
            for r in recs:
                m = cand_meta[r["id"]]
                rq[m["raw_q"]].append(r["id"])
                nq[m["norm_q"]].append(r["id"])
                cs[m["canon_sql"]].append(r["id"])
                ca[m["canon_ast_sig"]].append(r["id"])
            return {"raw_q": rq, "norm_q": nq, "canon_sql": cs, "canon_ast_sig": ca}

        pop_a = _index_pop(orig_records)
        pop_b = _index_pop(new_records)

        q_report: Dict[str, Any] = {}
        csql_report: Dict[str, Any] = {}
        comm_report: Dict[str, Any] = {}

        for sname, hrecs in heldout_splits.items():
            hm_map = ho_meta[sname]

            def _scan_split(pop_idx: Dict[str, Dict[Any, List[int]]]) -> Dict[str, Any]:
                ex_q, nm_q, cs_hits, ca_hits = [], [], [], []
                for h in hrecs:
                    hm = hm_map[h["id"]]
                    if hm["raw_q"] in pop_idx["raw_q"]:
                        ex_q.append({"ho_id": h["id"], "train_ids": pop_idx["raw_q"][hm["raw_q"]]})
                    if hm["norm_q"] in pop_idx["norm_q"]:
                        nm_q.append({"ho_id": h["id"], "train_ids": pop_idx["norm_q"][hm["norm_q"]]})
                    if hm["canon_sql"] in pop_idx["canon_sql"]:
                        cs_hits.append(
                            {"ho_id": h["id"], "train_ids": pop_idx["canon_sql"][hm["canon_sql"]], "sql": hm["sql"]}
                        )
                    if hm["canon_ast_sig"] in pop_idx["canon_ast_sig"]:
                        pure_comm = hm["canon_sql"] not in pop_idx["canon_sql"]
                        ca_hits.append(
                            {
                                "ho_id": h["id"],
                                "train_ids": pop_idx["canon_ast_sig"][hm["canon_ast_sig"]],
                                "sql": hm["sql"],
                                "purely_commutative_reordering": pure_comm,
                            }
                        )
                return {"exact_q": ex_q, "norm_q": nm_q, "canon_sql": cs_hits, "comm_ast": ca_hits}

            res_a = _scan_split(pop_a)
            res_b = _scan_split(pop_b)

            q_report[sname] = {
                "population_a_baseline": {
                    "classification": "grandfathered_baseline",
                    "exact_q": len(res_a["exact_q"]),
                    "norm_q": len(res_a["norm_q"]),
                    "hits": res_a["norm_q"],
                },
                "population_b_new": {
                    "exact_q": len(res_b["exact_q"]),
                    "norm_q": len(res_b["norm_q"]),
                    "hits": res_b["norm_q"],
                },
                "combined_corpus": {
                    "exact_q": len(res_a["exact_q"]) + len(res_b["exact_q"]),
                    "norm_q": len(res_a["norm_q"]) + len(res_b["norm_q"]),
                },
            }
            csql_report[sname] = {
                "population_a_baseline": {
                    "classification": "grandfathered_baseline",
                    "count": len(res_a["canon_sql"]),
                    "hits": res_a["canon_sql"],
                },
                "population_b_new": {
                    "count": len(res_b["canon_sql"]),
                    "hits": res_b["canon_sql"],
                },
                "combined_corpus": {
                    "count": len(res_a["canon_sql"]) + len(res_b["canon_sql"]),
                },
            }
            comm_report[sname] = {
                "population_a_baseline": {
                    "classification": "grandfathered_baseline",
                    "count": len(res_a["comm_ast"]),
                    "purely_commutative_count": sum(1 for h in res_a["comm_ast"] if h["purely_commutative_reordering"]),
                    "hits": res_a["comm_ast"],
                },
                "population_b_new": {
                    "count": len(res_b["comm_ast"]),
                    "purely_commutative_count": sum(1 for h in res_b["comm_ast"] if h["purely_commutative_reordering"]),
                    "hits": res_b["comm_ast"],
                },
                "combined_corpus": {
                    "count": len(res_a["comm_ast"]) + len(res_b["comm_ast"]),
                },
            }

            if res_b["exact_q"] or res_b["norm_q"]:
                hard_failures.append({"check": f"question_overlap_{sname}", "hits": res_b["norm_q"]})
            if res_b["canon_sql"]:
                hard_failures.append({"check": f"canonical_sql_overlap_{sname}", "hits": res_b["canon_sql"]})
            elif res_b["comm_ast"]:
                hard_failures.append({"check": f"commutative_boolean_overlap_{sname}", "hits": res_b["comm_ast"]})

        return q_report, csql_report, comm_report

    def _check_template_overlaps(
        self,
        orig_records: List[Dict[str, Any]],
        new_records: List[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        heldout_splits: Dict[str, List[Dict[str, Any]]],
        ho_meta: Dict[str, Dict[int, Dict[str, Any]]],
        hard_failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        orig_tasts = {cand_meta[r["id"]]["template_ast_sig"] for r in orig_records}
        new_unique_tasts = {cand_meta[r["id"]]["template_ast_sig"] for r in new_records}
        new_only_tasts = new_unique_tasts - orig_tasts
        total_new = len(new_records)
        total_orig = len(orig_records)

        report: Dict[str, Any] = {}
        for sname, hrecs in heldout_splits.items():
            ho_tasts = {ho_meta[sname][h["id"]]["template_ast_sig"] for h in hrecs}
            total_ho = len(hrecs)
            total_unique_ho = len(ho_tasts)

            orig_matching_ids = [r["id"] for r in orig_records if cand_meta[r["id"]]["template_ast_sig"] in ho_tasts]
            new_matching_ids = [r["id"] for r in new_records if cand_meta[r["id"]]["template_ast_sig"] in ho_tasts]

            unique_matching_new_templates = {cand_meta[rid]["template_ast_sig"] for rid in new_matching_ids}
            matched_ho_by_orig = [h["id"] for h in hrecs if ho_meta[sname][h["id"]]["template_ast_sig"] in orig_tasts]
            matched_unique_ho_by_orig = ho_tasts & orig_tasts

            matched_ho_by_new_only = [
                {
                    "ho_id": h["id"],
                    "template": ho_meta[sname][h["id"]]["template_sql"],
                }
                for h in hrecs
                if ho_meta[sname][h["id"]]["template_ast_sig"] in new_only_tasts
            ]
            matched_unique_ho_by_new_only = ho_tasts & new_only_tasts
            matched_ho_by_new_all = [
                h["id"] for h in hrecs if ho_meta[sname][h["id"]]["template_ast_sig"] in unique_matching_new_templates
            ]

            overlap_ratio = len(new_matching_ids) / total_new if total_new > 0 else 0.0
            query_coverage_ratio = len(matched_ho_by_new_only) / total_ho if total_ho > 0 else 0.0
            unique_template_coverage_ratio = (
                len(matched_unique_ho_by_new_only) / total_unique_ho if total_unique_ho > 0 else 0.0
            )
            all_new_query_coverage_ratio = len(matched_ho_by_new_all) / total_ho if total_ho > 0 else 0.0

            report[sname] = {
                "ratio_definitions": {
                    "overlap_ratio": "matching_new_records / total_new_records",
                    "query_coverage_ratio": "new_only_matched_heldout_queries / total_heldout_queries",
                    "unique_template_coverage_ratio": "new_only_matched_unique_heldout_templates / total_unique_heldout_templates",
                },
                "total_heldout": total_ho,
                "total_heldout_queries": total_ho,
                "total_unique_heldout_templates": total_unique_ho,
                "total_new_records": total_new,
                "unique_templates_in_new": len(new_unique_tasts),
                "population_a_baseline": {
                    "matching_orig_records": len(orig_matching_ids),
                    "total_orig_records": total_orig,
                    "matched_heldout_count": len(matched_ho_by_orig),
                    "matched_unique_templates_count": len(matched_unique_ho_by_orig),
                },
                "population_b_new": {
                    "matching_new_records": len(new_matching_ids),
                    "matching_new_record_ids": new_matching_ids,
                    "unique_matching_templates_count": len(unique_matching_new_templates),
                    "new_only_matched_heldout_count": len(matched_ho_by_new_only),
                    "new_only_matched_unique_templates_count": len(matched_unique_ho_by_new_only),
                    "overlap_ratio": round(overlap_ratio, 4),
                    "new_record_template_ratio": round(overlap_ratio, 4),
                    "query_coverage_ratio": round(query_coverage_ratio, 4),
                    "unique_template_coverage_ratio": round(unique_template_coverage_ratio, 4),
                    "all_new_matched_heldout_count": len(matched_ho_by_new_all),
                    "all_new_query_coverage_ratio": round(all_new_query_coverage_ratio, 4),
                    "new_only_details": matched_ho_by_new_only,
                },
                "combined_corpus": {
                    "matching_total_records": len(orig_matching_ids) + len(new_matching_ids),
                    "total_records": total_orig + total_new,
                },
            }
            if overlap_ratio > self.config.max_new_template_overlap_ratio:
                hard_failures.append(
                    {
                        "check": f"excessive_new_template_overlap_{sname}",
                        "overlap_ratio": round(overlap_ratio, 4),
                        "limit": self.config.max_new_template_overlap_ratio,
                        "matching_new_records": len(new_matching_ids),
                        "total_new_records": total_new,
                    }
                )
            if total_ho > 0 and query_coverage_ratio > self.config.max_new_template_overlap_ratio:
                hard_failures.append(
                    {
                        "check": f"excessive_heldout_query_template_coverage_{sname}",
                        "query_coverage_ratio": round(query_coverage_ratio, 4),
                        "limit": self.config.max_new_template_overlap_ratio,
                        "new_only_matched_heldout_queries": len(matched_ho_by_new_only),
                        "total_heldout_queries": total_ho,
                    }
                )
            if total_unique_ho > 0 and unique_template_coverage_ratio > self.config.max_new_template_overlap_ratio:
                hard_failures.append(
                    {
                        "check": f"excessive_heldout_unique_template_coverage_{sname}",
                        "unique_template_coverage_ratio": round(unique_template_coverage_ratio, 4),
                        "limit": self.config.max_new_template_overlap_ratio,
                        "new_only_matched_unique_templates": len(matched_unique_ho_by_new_only),
                        "total_unique_heldout_templates": total_unique_ho,
                    }
                )
        return report

    def _check_paraphrase_and_structure_literal(
        self,
        new_records: List[Dict[str, Any]],
        orig_records: List[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        heldout_splits: Dict[str, List[Dict[str, Any]]],
        ho_meta: Dict[str, Dict[int, Dict[str, Any]]],
        literal_manifest: Optional[Dict[str, Any]],
        reviewed_map: Dict[str, Any],
        hard_failures: List[Dict[str, Any]],
        review_flags: List[Dict[str, Any]],
    ) -> None:
        manifest_lits = (literal_manifest or {}).get("literals", [])
        str_manifest_by_val = {
            str(item.get("value")): item
            for item in manifest_lits
            if isinstance(item, dict) and item.get("sql_context") == "closed_domain_string"
        }

        orig_nums: set[str] = set()
        orig_strs: set[str] = set()
        for r in orig_records:
            orig_nums |= cand_meta[r["id"]]["nums"]
            orig_strs |= cand_meta[r["id"]]["strs"]

        for tr in new_records:
            tm = cand_meta[tr["id"]]
            tr_lits = tm["nums"] | tm["strs"]
            for sname, hrecs in heldout_splits.items():
                hmeta = ho_meta[sname]
                for hr in hrecs:
                    hm = hmeta[hr["id"]]
                    sims = compute_text_similarities(tm["raw_q"], hm["raw_q"])

                    # Structure + literal coincidence
                    ho_lits = hm["nums"] | hm["strs"]
                    shared_lits = sorted(tr_lits & ho_lits)
                    same_struct = (
                        tm["template_ast_sig"] == hm["template_ast_sig"]
                        or (
                            tm["table"] == hm["table"]
                            and tm["where_cols"] == hm["where_cols"]
                            and len(tm["where_cols"]) > 0
                            and tm["order_cols"] == hm["order_cols"]
                        )
                    )
                    nontrivial_shared_nums = (tm["nums"] & hm["nums"]) - orig_nums
                    non_catalog_shared_strs = {
                        s
                        for s in ((tm["strs"] & hm["strs"]) - orig_strs)
                        if str_manifest_by_val.get(s, {}).get("sql_context") != "closed_domain_string"
                    }
                    same_tmpl_all_lits = (
                        tm["template_ast_sig"] == hm["template_ast_sig"]
                        and len(ho_lits) > 0
                        and ho_lits == tr_lits
                    )
                    struct_plus_lit = same_struct and (
                        bool(nontrivial_shared_nums)
                        or bool(non_catalog_shared_strs)
                        or same_tmpl_all_lits
                        or (len(shared_lits) >= 2 and sims["seq_ratio"] >= 0.65)
                        or (len(shared_lits) >= 1 and sims["content_jaccard"] >= 0.65)
                    )
                    if struct_plus_lit:
                        hard_failures.append(
                            {
                                "check": "structure_plus_literal_overlap",
                                "train_id": tr["id"],
                                "split": sname,
                                "heldout_id": hr["id"],
                                "shared_lits": shared_lits,
                                "sims": sims,
                                "tr_q": tm["raw_q"],
                                "ho_q": hm["raw_q"],
                                "tr_sql": tm["sql"],
                                "ho_sql": hm["sql"],
                            }
                        )

                    # Hard paraphrase check
                    is_hard_para = (
                        sims["seq_ratio"] >= self.config.hard_seq_ratio_threshold
                        or sims["jaccard"] >= self.config.hard_jaccard_threshold
                        or (
                            sims["content_jaccard"] >= self.config.hard_content_jaccard_threshold
                            and sims["bigram_jaccard"] >= self.config.hard_bigram_threshold
                        )
                    )
                    if is_hard_para:
                        hard_failures.append(
                            {
                                "check": "hard_paraphrase_similarity",
                                "train_id": tr["id"],
                                "split": sname,
                                "heldout_id": hr["id"],
                                "sims": sims,
                                "tr_q": tm["raw_q"],
                                "ho_q": hm["raw_q"],
                                "tr_sql": tm["sql"],
                                "ho_sql": hm["sql"],
                            }
                        )
                        continue

                    # Soft review flag check (requires recorded disposition + rationale)
                    is_soft_para = (
                        sims["seq_ratio"] >= self.config.soft_seq_ratio_threshold
                        or sims["jaccard"] >= self.config.soft_jaccard_threshold
                        or sims["bigram_jaccard"] >= self.config.soft_bigram_threshold
                        or (
                            tm["table"] == hm["table"]
                            and sims["content_jaccard"] >= self.config.soft_content_jaccard_threshold
                            and (tm["template_ast_sig"] == hm["template_ast_sig"] or sims["seq_ratio"] >= 0.65)
                        )
                    )
                    if is_soft_para:
                        key = f"{tr['id']}->{sname}:{hr['id']}"
                        rev = reviewed_map.get(key)
                        disposition = None
                        rationale = None
                        if isinstance(rev, dict):
                            disposition = rev.get("disposition")
                            rationale = (rev.get("rationale") or "").strip()
                        elif isinstance(rev, str) and rev.strip():
                            disposition = "approved_non_leaking"
                            rationale = rev.strip()
                        is_approved = (
                            disposition == "approved_non_leaking"
                            and bool(rationale)
                            and len(rationale) >= 10
                        )
                        review_flags.append(
                            {
                                "key": key,
                                "train_id": tr["id"],
                                "split": sname,
                                "heldout_id": hr["id"],
                                "sims": sims,
                                "same_template": tm["template_ast_sig"] == hm["template_ast_sig"],
                                "shared_lits": shared_lits,
                                "tr_q": tm["raw_q"],
                                "ho_q": hm["raw_q"],
                                "tr_sql": tm["sql"],
                                "ho_sql": hm["sql"],
                                "disposition": disposition,
                                "rationale": rationale,
                                "approved": is_approved,
                            }
                        )

    def _check_literal_provenance(
        self,
        new_records: List[Dict[str, Any]],
        original_train_records: List[Dict[str, Any]],
        heldout_splits: Dict[str, List[Dict[str, Any]]],
        literal_manifest: Optional[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        exec_status_by_id: Dict[int, Dict[str, Any]],
        reviewed_zero_row_map: Dict[str, Any],
        catalog_string_values: Optional[Dict[str, Set[str]]],
        identifier_catalog: Optional[Dict[Tuple[str, str], Set[int]]],
        observed_numeric_values: Optional[Dict[Tuple[str, str], Set[Any]]],
        live_metric_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]],
        table_row_counts: Optional[Dict[str, int]],
        hard_failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        orig_nums: set[str] = set()
        orig_strs: set[str] = set()
        for r in original_train_records:
            n, s = extract_sql_literals(r["sql"], dialect=self.config.dialect)
            orig_nums |= n
            orig_strs |= s

        enforce_catalogs = (
            self.config.require_catalog_verification
            if self.config.require_catalog_verification is not None
            else (self.config.require_literal_manifest or self.config.require_live_execution)
        )
        if enforce_catalogs:
            if catalog_string_values is None:
                hard_failures.append(
                    {
                        "check": "missing_live_string_catalog",
                        "message": "catalog_string_values is required for closed-domain string verification.",
                    }
                )
            if identifier_catalog is None:
                hard_failures.append(
                    {
                        "check": "missing_live_identifier_catalog",
                        "message": "identifier_catalog is required for PK/FK identifier verification.",
                    }
                )
            if observed_numeric_values is None:
                hard_failures.append(
                    {
                        "check": "missing_live_observed_numeric_values",
                        "message": "observed_numeric_values is required to verify is_observed_in_db flags.",
                    }
                )

        excluded_lower = {s.strip().lower() for s in self.config.excluded_new_literals}
        excluded_literal_violations: List[Dict[str, Any]] = []

        # Extract contextual literals used in Population B (new additions)
        new_nums: Counter[str] = Counter()
        new_strs: Counter[str] = Counter()
        used_ctx_map: Dict[Tuple[str, str, str, str], Set[int]] = defaultdict(set)
        used_occs_by_id: Dict[int, List[Dict[str, Any]]] = {}
        all_new_ids = {r["id"] for r in new_records}

        for r in new_records:
            rid = r["id"]
            try:
                occs = extract_contextual_literals(r["sql"], dialect=self.config.dialect)
            except (ParseError, TokenError, ValueError):
                occs = []
            used_occs_by_id[rid] = occs
            for occ in occs:
                if occ["data_type"] == "string":
                    new_strs[occ["value_norm"]] += 1
                    if occ["value_norm"].strip().lower() in excluded_lower:
                        excluded_literal_violations.append(
                            {
                                "id": rid,
                                "value": occ["value_norm"],
                                "table": occ["table"],
                                "column": occ["column"],
                                "source": "sql",
                            }
                        )
                else:
                    new_nums[occ["value_norm"]] += 1
                key = (
                    occ["value_norm"],
                    occ["table"],
                    occ["column"] or "",
                    occ["sql_context"],
                )
                used_ctx_map[key].add(rid)

        ho_nums: set[str] = set()
        ho_strs: set[str] = set()
        for hrecs in heldout_splits.values():
            for hr in hrecs:
                try:
                    n, s = extract_sql_literals(
                        hr.get("sql") or hr.get("reference_sql", ""),
                        dialect=self.config.dialect,
                    )
                    ho_nums |= n
                    ho_strs |= s
                except (ParseError, TokenError, ValueError):
                    pass

        ho_exclusive_nums = ho_nums - orig_nums
        ho_exclusive_strs = ho_strs - orig_strs
        added_new_nums = set(new_nums.keys()) - orig_nums
        added_new_strs = set(new_strs.keys()) - orig_strs

        # 1. Held-out-exclusive numeric literal injection check (0 tolerance)
        leaked_nums = sorted(added_new_nums & ho_exclusive_nums)
        if leaked_nums:
            hard_failures.append(
                {
                    "check": "heldout_exclusive_numeric_literal_injection",
                    "tokens": leaked_nums,
                    "message": "Numeric literal(s) absent from baseline train match held-out OOV targets.",
                }
            )

        schema_violations: List[Dict[str, Any]] = []
        context_violations: List[Dict[str, Any]] = []
        undeclared_usages: List[Dict[str, Any]] = []
        unused_or_mismatched_declarations: List[Dict[str, Any]] = []
        example_traceability_violations: List[Dict[str, Any]] = []
        non_catalog_strings: List[Dict[str, Any]] = []
        unjustified_ho_strings: List[str] = []
        observed_count = 0
        synthetic_threshold_count = 0

        declared_by_ctx: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

        if self.config.require_literal_manifest:
            if not literal_manifest:
                hard_failures.append(
                    {
                        "check": "missing_literal_provenance_manifest",
                        "message": "Machine-verifiable literal provenance manifest is required.",
                    }
                )
            else:
                schema_violations.extend(self._validate_manifest_schema(literal_manifest))
                seen_lit_ids: Set[str] = set()
                for entry in literal_manifest.get("literals", []):
                    if not isinstance(entry, dict):
                        continue
                    lid = entry.get("literal_id", "")
                    if lid in seen_lit_ids:
                        schema_violations.append({"literal_id": lid, "error": f"Duplicate literal_id '{lid}'"})
                    seen_lit_ids.add(lid)

                    val = entry.get("value")
                    dtype = entry.get("data_type", "")
                    tbl = (entry.get("table") or "").lower()
                    col = entry.get("column")
                    col_str = col.lower() if isinstance(col, str) else ""
                    ctx = entry.get("sql_context", "")
                    rule = str(entry.get("generation_rule") or "")
                    dom_src = str(entry.get("domain_source") or "")
                    app_dom = entry.get("approved_domain")
                    obs_db = entry.get("is_observed_in_db")
                    ex_ids = entry.get("example_ids", [])

                    if obs_db is True:
                        observed_count += 1
                    elif obs_db is False:
                        synthetic_threshold_count += 1

                    if isinstance(val, str) and val.strip().lower() in excluded_lower:
                        excluded_literal_violations.append(
                            {"literal_id": lid, "value": val, "table": tbl, "column": col_str, "source": "manifest"}
                        )

                    # Guard against held-out leakage via provenance notes
                    combined_notes = f"{rule} {dom_src}".lower()
                    if any(w in combined_notes for w in ("validation.jsonl", "test.jsonl", "benchmark_eval", "heldout", "held-out", "copied from benchmark")):
                        context_violations.append(
                            {"literal_id": lid, "error": f"Prohibited held-out reference in provenance rule/source: '{dom_src}'"}
                        )

                    val_norm = str(val) if dtype == "string" else normalize_numeric_str(val)
                    ctx_key = (val_norm, tbl, col_str, ctx)
                    if ctx_key in declared_by_ctx:
                        schema_violations.append(
                            {"literal_id": lid, "error": f"Duplicate manifest declaration for key {ctx_key}"}
                        )
                    declared_by_ctx[ctx_key] = entry

                    # Check bidirectional usage with Population B example_ids
                    declared_ids = set(ex_ids) if isinstance(ex_ids, list) else set()
                    invalid_ex_ids = sorted(declared_ids - all_new_ids)
                    actual_ids = used_ctx_map.get(ctx_key, set())
                    if invalid_ex_ids or not declared_ids or declared_ids != actual_ids:
                        unused_or_mismatched_declarations.append(
                            {
                                "literal_id": lid,
                                "key": list(ctx_key),
                                "declared_example_ids": sorted(declared_ids),
                                "actual_used_example_ids": sorted(actual_ids),
                                "invalid_non_population_b_ids": invalid_ex_ids,
                            }
                        )

                    # Validate context-specific rules against live DB catalogs/observations
                    err = self._validate_manifest_entry_context(
                        lid=lid,
                        val=val,
                        dtype=dtype,
                        tbl=tbl,
                        col=col,
                        col_str=col_str,
                        ctx=ctx,
                        rule=rule,
                        obs_db=obs_db,
                        app_dom=app_dom,
                        catalog_string_values=catalog_string_values,
                        id_cat=identifier_catalog,
                        observed_numeric_values=observed_numeric_values,
                        live_metric_bounds=live_metric_bounds,
                        table_row_counts=table_row_counts,
                    )
                    if err:
                        context_violations.append({"literal_id": lid, "value": val, "sql_context": ctx, "error": err})

                # Every literal occurrence in Population B must be declared under its exact context
                for ctx_key, uids in sorted(used_ctx_map.items()):
                    if ctx_key not in declared_by_ctx:
                        undeclared_usages.append(
                            {
                                "value": ctx_key[0],
                                "table": ctx_key[1],
                                "column": ctx_key[2] or None,
                                "sql_context": ctx_key[3],
                                "example_ids": sorted(uids),
                            }
                        )

                # Schema v2.0 per-example traceability cross-check
                if self.config.require_manifest_v2:
                    example_traceability_violations.extend(
                        self._validate_examples_manifest(
                            manifest=literal_manifest,
                            new_records=new_records,
                            cand_meta=cand_meta,
                            used_occs_by_id=used_occs_by_id,
                            exec_status_by_id=exec_status_by_id,
                            reviewed_zero_row_map=reviewed_zero_row_map,
                            observed_numeric_values=observed_numeric_values,
                            catalog_string_values=catalog_string_values,
                            identifier_catalog=identifier_catalog,
                            table_row_counts=table_row_counts,
                        )
                    )

        # Live DB column-level catalog string enforcement (even if manifest disabled)
        for occ_key, uids in sorted(used_ctx_map.items()):
            val_norm, tbl, col_str, ctx = occ_key
            if ctx == "closed_domain_string" and catalog_string_values is not None:
                full_col = f"{tbl}.{col_str}"
                allowed = catalog_string_values.get(full_col, set())
                if val_norm not in allowed:
                    non_catalog_strings.append(
                        {"value": val_norm, "column": full_col, "example_ids": sorted(uids)}
                    )

        # Held-out-exclusive strings allowed ONLY when verified in live column catalog (and not excluded)
        for ctx_key in sorted(used_ctx_map.keys()):
            val_norm, tbl, col_str, ctx = ctx_key
            if ctx == "closed_domain_string" and val_norm in ho_exclusive_strs:
                entry = declared_by_ctx.get(ctx_key, {})
                in_col_cat = (
                    catalog_string_values is not None
                    and val_norm in catalog_string_values.get(f"{tbl}.{col_str}", set())
                )
                if entry.get("sql_context") != "closed_domain_string" or entry.get("is_observed_in_db") is not True or (
                    catalog_string_values is not None and not in_col_cat
                ):
                    unjustified_ho_strings.append(val_norm)

        if excluded_literal_violations:
            hard_failures.append(
                {
                    "check": "excluded_heldout_literal_violation",
                    "count": len(excluded_literal_violations),
                    "details": excluded_literal_violations[:20],
                    "message": (
                        "Decision B excludes benchmark literal(s) such as 'Machine Learning' from all "
                        "Population B additions."
                    ),
                }
            )
        if schema_violations:
            hard_failures.append(
                {"check": "manifest_schema_violations", "count": len(schema_violations), "details": schema_violations[:20]}
            )
        if example_traceability_violations:
            hard_failures.append(
                {
                    "check": "example_traceability_violations",
                    "count": len(example_traceability_violations),
                    "details": example_traceability_violations[:20],
                }
            )
        if context_violations:
            hard_failures.append(
                {"check": "literal_context_rule_violations", "count": len(context_violations), "details": context_violations[:20]}
            )
        if undeclared_usages:
            hard_failures.append(
                {"check": "undeclared_new_literals", "count": len(undeclared_usages), "details": undeclared_usages[:20]}
            )
        if unused_or_mismatched_declarations:
            hard_failures.append(
                {
                    "check": "unused_or_mismatched_manifest_declarations",
                    "count": len(unused_or_mismatched_declarations),
                    "details": unused_or_mismatched_declarations[:20],
                }
            )
        if non_catalog_strings:
            hard_failures.append(
                {"check": "non_catalog_string_literals", "count": len(non_catalog_strings), "details": non_catalog_strings[:20]}
            )
        if unjustified_ho_strings:
            hard_failures.append(
                {"check": "heldout_exclusive_string_literal_injection", "tokens": sorted(set(unjustified_ho_strings))}
            )

        return {
            "orig_numeric_count": len(orig_nums),
            "orig_string_count": len(orig_strs),
            "added_numeric_tokens": sorted(added_new_nums),
            "added_string_tokens": sorted(added_new_strs),
            "heldout_exclusive_numeric_tokens": sorted(ho_exclusive_nums),
            "heldout_exclusive_string_tokens": sorted(ho_exclusive_strs),
            "leaked_heldout_numeric_tokens": leaked_nums,
            "excluded_literal_violations": excluded_literal_violations,
            "manifest_declared_count": len(declared_by_ctx),
            "observed_literal_declarations": observed_count,
            "synthetic_threshold_declarations": synthetic_threshold_count,
            "undeclared_usage_count": len(undeclared_usages),
            "unused_or_mismatched_declaration_count": len(unused_or_mismatched_declarations),
            "example_traceability_violation_count": len(example_traceability_violations),
            "context_violation_count": len(context_violations),
            "non_catalog_string_count": len(non_catalog_strings),
            "unjustified_heldout_string_tokens": sorted(set(unjustified_ho_strings)),
        }

    def _validate_manifest_schema(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        errs: List[Dict[str, Any]] = []
        if self.config.require_manifest_v2:
            required_top = (
                "schema_version",
                "policy_document",
                "generation_rules",
                "literals",
                "examples",
                "reviewed_soft_flags",
                "reviewed_projection_flags",
                "reviewed_zero_row_queries",
            )
            expected_ver = "2.0"
        else:
            required_top = ("schema_version", "policy_document", "generation_rules", "literals", "reviewed_soft_flags")
            expected_ver = "1.0"

        for req in required_top:
            if req not in manifest:
                errs.append({"field": req, "error": f"Missing top-level manifest field '{req}'"})
        if manifest.get("schema_version") != expected_ver:
            errs.append({"field": "schema_version", "error": f"Expected schema_version == '{expected_ver}'"})

        gen_rules = manifest.get("generation_rules")
        if not isinstance(gen_rules, dict):
            errs.append({"field": "generation_rules", "error": "'generation_rules' must be an object"})
        else:
            for ctx in ALLOWED_SQL_CONTEXTS:
                if not str(gen_rules.get(ctx) or "").strip():
                    errs.append({"field": f"generation_rules.{ctx}", "error": f"Missing generation rule for '{ctx}'"})

        lits = manifest.get("literals")
        if not isinstance(lits, list):
            errs.append({"field": "literals", "error": "'literals' must be a list"})
            return errs
        for idx, item in enumerate(lits):
            if not isinstance(item, dict):
                errs.append({"index": idx, "error": "Literal entry must be an object"})
                continue
            missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in item]
            if missing:
                errs.append({"index": idx, "literal_id": item.get("literal_id"), "error": f"Missing fields: {missing}"})

        if self.config.require_manifest_v2:
            examples = manifest.get("examples")
            if not isinstance(examples, list):
                errs.append({"field": "examples", "error": "'examples' must be a list"})
            else:
                for idx, ex in enumerate(examples):
                    if not isinstance(ex, dict):
                        errs.append({"examples_index": idx, "error": "Example entry must be an object"})
                        continue
                    missing_ex = [f for f in REQUIRED_EXAMPLE_FIELDS if f not in ex]
                    if missing_ex:
                        errs.append(
                            {"examples_index": idx, "id": ex.get("id"), "error": f"Missing example fields: {missing_ex}"}
                        )
        return errs

    def _is_actually_observed_in_db(
        self,
        *,
        val: Any,
        tbl: str,
        col_str: str,
        ctx: str,
        observed_numeric_values: Optional[Dict[Tuple[str, str], Set[Any]]],
        catalog_string_values: Optional[Dict[str, Set[str]]],
        id_cat: Optional[Dict[Tuple[str, str], Set[int]]],
        table_row_counts: Optional[Dict[str, int]],
    ) -> Optional[bool]:
        if ctx == "closed_domain_string":
            if catalog_string_values is None:
                return None
            return str(val) in catalog_string_values.get(f"{tbl}.{col_str}", set())
        if ctx == "identifier":
            if id_cat is None:
                return None
            return isinstance(val, int) and val in id_cat.get((tbl, col_str), set())
        if ctx == "limit":
            if table_row_counts is None or tbl not in table_row_counts:
                return None
            try:
                return 1 <= int(val) <= int(table_row_counts[tbl])
            except (TypeError, ValueError):
                return False
        if ctx == "comparison_threshold":
            if observed_numeric_values is None:
                return None
            obs_set = observed_numeric_values.get((tbl, col_str))
            if obs_set is None:
                return None
            try:
                val_dec = Decimal(str(val))
                obs_decs = {Decimal(str(x)) for x in obs_set}
                return val_dec in obs_decs
            except (InvalidOperation, ValueError):
                return False
        return None

    def _validate_examples_manifest(
        self,
        *,
        manifest: Dict[str, Any],
        new_records: List[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        used_occs_by_id: Dict[int, List[Dict[str, Any]]],
        exec_status_by_id: Dict[int, Dict[str, Any]],
        reviewed_zero_row_map: Dict[str, Any],
        observed_numeric_values: Optional[Dict[Tuple[str, str], Set[Any]]],
        catalog_string_values: Optional[Dict[str, Set[str]]],
        identifier_catalog: Optional[Dict[Tuple[str, str], Set[int]]],
        table_row_counts: Optional[Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        errs: List[Dict[str, Any]] = []
        examples = manifest.get("examples")
        if not isinstance(examples, list):
            return [{"error": "Manifest 'examples' must be a list in schema_version 2.0"}]

        new_by_id = {r["id"]: r for r in new_records}
        ex_by_id: Dict[int, Dict[str, Any]] = {}
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            rid = ex.get("id")
            if not isinstance(rid, int):
                errs.append({"id": rid, "error": "Example manifest entry missing integer id"})
                continue
            if rid in ex_by_id:
                errs.append({"id": rid, "error": f"Duplicate id {rid} in manifest 'examples'"})
                continue
            ex_by_id[rid] = ex

        missing_ex_ids = sorted(set(new_by_id.keys()) - set(ex_by_id.keys()))
        extra_ex_ids = sorted(set(ex_by_id.keys()) - set(new_by_id.keys()))
        if missing_ex_ids:
            errs.append({"error": f"Population B ID(s) missing from manifest 'examples': {missing_ex_ids[:20]}"})
        if extra_ex_ids:
            errs.append({"error": f"Manifest 'examples' contains non-Population-B ID(s): {extra_ex_ids[:20]}"})

        for rid, rec in sorted(new_by_id.items()):
            ex = ex_by_id.get(rid)
            if not ex:
                continue
            meta = cand_meta[rid]
            if ex.get("question") != rec.get("question"):
                errs.append({"id": rid, "error": "Manifest example question does not match dataset record"})
            if ex.get("sql") != rec.get("sql"):
                errs.append({"id": rid, "error": "Manifest example SQL does not match dataset record"})
            if str(ex.get("table") or "").lower() != meta["table"]:
                errs.append(
                    {"id": rid, "error": f"Manifest example table '{ex.get('table')}' != AST table '{meta['table']}'"}
                )
            if ex.get("ast_template") != meta["template_sql"]:
                errs.append({"id": rid, "error": "Manifest example ast_template does not match canonical SQL template"})

            actual_lits = used_occs_by_id.get(rid, [])
            decl_lits = ex.get("literals_used")
            if not isinstance(decl_lits, list) or len(decl_lits) != len(actual_lits):
                errs.append(
                    {
                        "id": rid,
                        "error": (
                            f"Manifest example literals_used count ({len(decl_lits) if isinstance(decl_lits, list) else 'invalid'}) "
                            f"!= SQL AST literal count ({len(actual_lits)})"
                        ),
                    }
                )
            else:
                def _occ_sig(d: Dict[str, Any]) -> Tuple[str, str, str]:
                    v = d.get("value")
                    ctx_d = str(d.get("sql_context") or "")
                    vn = str(v) if ctx_d == "closed_domain_string" else normalize_numeric_str(v)
                    return (vn, str(d.get("column") or "").lower(), ctx_d)

                actual_sigs = Counter(_occ_sig(o) for o in actual_lits)
                decl_sigs = Counter(_occ_sig(o) for o in decl_lits if isinstance(o, dict))
                if actual_sigs != decl_sigs:
                    errs.append(
                        {"id": rid, "error": "Manifest example literals_used mismatch vs SQL AST literal occurrences"}
                    )
                for item in decl_lits:
                    if not isinstance(item, dict):
                        continue
                    act_obs = self._is_actually_observed_in_db(
                        val=item.get("value"),
                        tbl=meta["table"],
                        col_str=str(item.get("column") or "").lower(),
                        ctx=str(item.get("sql_context") or ""),
                        observed_numeric_values=observed_numeric_values,
                        catalog_string_values=catalog_string_values,
                        id_cat=identifier_catalog,
                        table_row_counts=table_row_counts,
                    )
                    if act_obs is not None and item.get("is_observed_in_db") != act_obs:
                        errs.append(
                            {
                                "id": rid,
                                "error": (
                                    f"Manifest example literal {item.get('value')} has is_observed_in_db="
                                    f"{item.get('is_observed_in_db')} != live DB ({act_obs})"
                                ),
                            }
                        )

            # Live execution classification & row_count cross-verification
            if rid in exec_status_by_id:
                act_ex = exec_status_by_id[rid]
                if ex.get("executed") != act_ex["executed"]:
                    errs.append(
                        {"id": rid, "error": f"Manifest example executed={ex.get('executed')} != live {act_ex['executed']}"}
                    )
                if ex.get("execution_classification") != act_ex["classification"]:
                    errs.append(
                        {
                            "id": rid,
                            "error": (
                                f"Manifest example execution_classification='{ex.get('execution_classification')}' "
                                f"!= live '{act_ex['classification']}'"
                            ),
                        }
                    )
                if ex.get("row_count") != act_ex["row_count"]:
                    errs.append(
                        {
                            "id": rid,
                            "error": f"Manifest example row_count={ex.get('row_count')} != live {act_ex['row_count']}",
                        }
                    )
                if act_ex["classification"] == "valid_zero_rows":
                    zrev = reviewed_zero_row_map.get(str(rid))
                    if not isinstance(zrev, dict) or zrev.get("disposition") != "approved_zero_row":
                        errs.append({"id": rid, "error": "Zero-row example missing reviewed_zero_row_queries approval"})
                    elif ex.get("human_disposition") != "approved_zero_row" or not str(ex.get("human_rationale") or "").strip():
                        errs.append({"id": rid, "error": "Zero-row example missing human_disposition/human_rationale in 'examples'"})
        return errs

    def _validate_manifest_entry_context(
        self,
        *,
        lid: str,
        val: Any,
        dtype: str,
        tbl: str,
        col: Any,
        col_str: str,
        ctx: str,
        rule: str,
        obs_db: Any,
        app_dom: Any,
        catalog_string_values: Optional[Dict[str, Set[str]]],
        id_cat: Optional[Dict[Tuple[str, str], Set[int]]],
        observed_numeric_values: Optional[Dict[Tuple[str, str], Set[Any]]],
        live_metric_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]],
        table_row_counts: Optional[Dict[str, int]],
    ) -> Optional[str]:
        if ctx not in ALLOWED_SQL_CONTEXTS:
            return f"Invalid sql_context '{ctx}'."
        if dtype not in {"integer", "decimal", "string"}:
            return f"Invalid data_type '{dtype}'."
        if tbl not in self.schema_dict:
            return f"Unknown table '{tbl}'."
        if not isinstance(obs_db, bool):
            return "'is_observed_in_db' must be a boolean."
        if not rule.strip():
            return "'generation_rule' must be non-empty."

        rule_l = rule.lower()

        if ctx == "limit":
            if col is not None:
                return "LIMIT literal must have column=null."
            if dtype != "integer" or not isinstance(val, int) or isinstance(val, bool):
                return "LIMIT value must have data_type='integer' and an integer value."
            if val not in self.config.approved_limit_values:
                return f"LIMIT value {val} not in approved deterministic LIMIT set {sorted(self.config.approved_limit_values)}."
            if any(k in rule_l for k in ("threshold", "cgpa", "marks", "identifier", "count")):
                return f"LIMIT rule '{rule}' must not reuse threshold, identifier, or count generation rules."
            if table_row_counts is not None and tbl in table_row_counts:
                expected_within = int(val) <= int(table_row_counts[tbl])
                if obs_db != expected_within:
                    return (
                        f"LIMIT value {val} on {tbl} (live row_count={table_row_counts[tbl]}) has "
                        f"is_observed_in_db={obs_db}, expected {expected_within}."
                    )
            return None

        if ctx == "count":
            if dtype != "integer" or not isinstance(val, int) or isinstance(val, bool) or val < 0:
                return "Count literal must be a non-negative integer."
            return None

        # Non-limit / non-agg predicate contexts require a valid column belonging to tbl
        if not col_str or col_str not in self.schema_dict.get(tbl, []):
            return f"Column '{col}' does not exist in table '{tbl}'."

        expected_col_type = COLUMN_SQL_TYPES.get((tbl, col_str))
        if expected_col_type and expected_col_type != dtype:
            return f"Declared data_type '{dtype}' incompatible with column {tbl}.{col_str} ({expected_col_type})."

        if ctx == "identifier":
            if col_str not in IDENTIFIER_COLUMNS_BY_TABLE.get(tbl, set()):
                return f"Column {tbl}.{col_str} is not a schema identifier (PK/FK) column."
            if dtype != "integer" or not isinstance(val, int) or isinstance(val, bool):
                return "Identifier literal must be an integer."
            if obs_db is not True:
                return "Identifier literals must have is_observed_in_db=true (synthetic IDs forbidden)."
            if any(k in rule_l for k in ("count", "num_rows", "row_count", "threshold", "midpoint", "limit")):
                return f"Identifier generation_rule '{rule}' must not derive IDs from count/threshold/limit rules."
            if id_cat is not None:
                valid_ids = id_cat.get((tbl, col_str))
                if valid_ids is None or val not in valid_ids:
                    return f"Identifier {val} does not exist in verified live catalog PK/FK keys for {tbl}.{col_str}."
            return None

        if ctx == "comparison_threshold":
            if col_str in IDENTIFIER_COLUMNS_BY_TABLE.get(tbl, set()):
                return f"Identifier column {tbl}.{col_str} cannot be declared as comparison_threshold; use 'identifier'."
            if dtype == "string" or isinstance(val, bool) or not isinstance(val, (int, float)):
                return "comparison_threshold value must be numeric."
            if any(k in rule_l for k in ("limit", "identifier", "pk", "count")):
                return f"comparison_threshold rule '{rule}' conflates threshold with limit/identifier/count."
            if (tbl, col_str) in DISCRETE_METRIC_VALUES:
                if int(val) != val or int(val) not in DISCRETE_METRIC_VALUES[(tbl, col_str)]:
                    return f"Value {val} outside discrete integer domain {sorted(DISCRETE_METRIC_VALUES[(tbl, col_str)])} for {tbl}.{col_str}."
            bounds = (live_metric_bounds or {}).get((tbl, col_str)) or NUMERIC_METRIC_DOMAINS.get((tbl, col_str))
            if bounds and not (bounds[0] <= float(val) <= bounds[1]):
                return f"Value {val} outside verified {tbl}.{col_str} domain [{bounds[0]}, {bounds[1]}]."
            if isinstance(app_dom, dict) and "min" in app_dom and "max" in app_dom:
                if not (app_dom["min"] <= float(val) <= app_dom["max"]):
                    return f"Value {val} outside manifest approved_domain [{app_dom['min']}, {app_dom['max']}]."
            if observed_numeric_values is not None:
                obs_set = observed_numeric_values.get((tbl, col_str))
                if obs_set is None:
                    return f"Missing live observed numeric catalog for {tbl}.{col_str}."
                actual_obs = Decimal(str(val)) in {Decimal(str(x)) for x in obs_set}
                if obs_db != actual_obs:
                    return (
                        f"Declared is_observed_in_db={obs_db} contradicts live DB observed values for "
                        f"{tbl}.{col_str} literal {val} (actual_observed={actual_obs})."
                    )
            return None

        if ctx == "closed_domain_string":
            if dtype != "string" or not isinstance(val, str):
                return "closed_domain_string must be a string value."
            if obs_db is not True:
                return "closed_domain_string must have is_observed_in_db=true."
            if catalog_string_values is not None:
                allowed = catalog_string_values.get(f"{tbl}.{col_str}", set())
                if val not in allowed:
                    return f"String '{val}' is not in verified DB catalog for {tbl}.{col_str}."
            return None

        return None

    def _check_projection_policy(
        self,
        new_records: List[Dict[str, Any]],
        cand_meta: Dict[int, Dict[str, Any]],
        reviewed_projection_map: Dict[str, Any],
        hard_failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.config.enforce_projection_policy:
            return {
                "enabled": False,
                "checked_count": 0,
                "hard_failure_count": 0,
                "review_flag_count": 0,
                "unreviewed_flag_count": 0,
                "violation_count": 0,
                "hard_failures": [],
                "review_flags": [],
                "violations": [],
            }

        level1_hard: List[Dict[str, Any]] = []
        level2_flags: List[Dict[str, Any]] = []
        unresolved_violations: List[Dict[str, Any]] = []

        def _emit_finding(rid: int, code: str, details: Dict[str, Any]) -> None:
            payload = {"id": rid, "violation": code, **details}
            is_l1 = code in LEVEL1_PROJECTION_HARD_VIOLATIONS or not self.config.use_two_level_projection
            if is_l1:
                payload["level"] = "hard_failure"
                level1_hard.append(payload)
                unresolved_violations.append(payload)
                return

            key = f"{rid}:{code}"
            rev = reviewed_projection_map.get(key)
            if rev is None:
                rev = reviewed_projection_map.get(str(rid))
            disposition = None
            rationale = None
            if isinstance(rev, dict):
                disposition = rev.get("disposition")
                rationale = (rev.get("rationale") or "").strip()
            is_approved = (
                disposition == "approved_projection_alignment"
                and bool(rationale)
                and len(rationale) >= 10
            )
            flag_entry = {
                **payload,
                "level": "review_flag",
                "key": key,
                "disposition": disposition,
                "rationale": rationale,
                "approved": is_approved,
            }
            level2_flags.append(flag_entry)
            if not is_approved:
                unresolved_violations.append(flag_entry)

        for r in new_records:
            rid = r["id"]
            tbl = (r.get("table") or "").lower()
            table_cols = set(self.schema_dict.get(tbl, []))
            try:
                parsed = sqlglot.parse_one(r["sql"].strip(), read=self.config.dialect)
            except (ParseError, TokenError, ValueError):
                # Already recorded under sql_validity_and_schema
                continue
            if not isinstance(parsed, exp.Select):
                continue
            proj_cols = []
            has_star = False
            for expr in parsed.expressions:
                if isinstance(expr, exp.Star):
                    has_star = True
                elif isinstance(expr, exp.Column):
                    proj_cols.append(expr.name.lower())

            proj_span = _extract_projection_nl_span(r["question"])
            ordered_mentions = _extract_ordered_column_mentions(proj_span, tbl, self.schema_dict)

            if has_star:
                # SELECT * cannot silently override an explicit strict-subset column request
                mentioned_cols = {c for c, _ in ordered_mentions}
                if mentioned_cols and len(mentioned_cols) < len(table_cols):
                    _emit_finding(
                        rid,
                        "wildcard_overrides_explicit_columns",
                        {
                            "question": r["question"],
                            "sql": r["sql"],
                            "mentioned_columns": [c for c, _ in ordered_mentions],
                        },
                    )
                continue

            # Level 1 deterministic structural checks (never overridable)
            if not proj_cols or any(c not in table_cols for c in proj_cols):
                _emit_finding(
                    rid,
                    "invalid_or_foreign_projection_column",
                    {"question": r["question"], "sql": r["sql"]},
                )
                continue

            if len(proj_cols) != len(set(proj_cols)):
                _emit_finding(
                    rid,
                    "duplicate_projection_column",
                    {"question": r["question"], "sql": r["sql"]},
                )
                continue

            # Level 2 heuristic NL-alignment checks
            mentioned_cols_set = {c for c, _ in ordered_mentions}
            if not mentioned_cols_set:
                _emit_finding(
                    rid,
                    "unsupported_explicit_projection",
                    {"question": r["question"], "sql": r["sql"], "proj_span": proj_span},
                )
                continue

            if set(proj_cols) != mentioned_cols_set:
                _emit_finding(
                    rid,
                    "projection_arity_or_column_mismatch",
                    {
                        "question": r["question"],
                        "sql": r["sql"],
                        "sql_cols": proj_cols,
                        "nl_cols": [c for c, _ in ordered_mentions],
                    },
                )
                continue

            # Sequence check: SQL column order must match first mention offsets in question
            first_pos_by_col: Dict[str, int] = {}
            for col_name, pos in ordered_mentions:
                if col_name not in first_pos_by_col:
                    first_pos_by_col[col_name] = pos
            positions = [first_pos_by_col[c] for c in proj_cols]
            if positions != sorted(positions):
                _emit_finding(
                    rid,
                    "projection_order_mismatch",
                    {
                        "question": r["question"],
                        "sql": r["sql"],
                        "sql_cols": proj_cols,
                        "nl_ordered_cols": [c for c, _ in ordered_mentions],
                    },
                )

        if level1_hard:
            hard_failures.append(
                {
                    "check": "projection_hard_failures",
                    "count": len(level1_hard),
                    "details": level1_hard[:20],
                }
            )
        unreviewed_l2 = [fl for fl in level2_flags if not fl.get("approved")]
        if unreviewed_l2:
            hard_failures.append(
                {
                    "check": "unreviewed_projection_flags",
                    "count": len(unreviewed_l2),
                    "details": unreviewed_l2[:20],
                }
            )

        return {
            "enabled": True,
            "two_level_mode": self.config.use_two_level_projection,
            "checked_count": len(new_records),
            "hard_failure_count": len(level1_hard),
            "review_flag_count": len(level2_flags),
            "unreviewed_flag_count": len(unreviewed_l2),
            "violation_count": len(unresolved_violations),
            "hard_failures": level1_hard,
            "review_flags": level2_flags,
            "violations": unresolved_violations,
        }


def load_heldout_splits(data_dir: Union[str, Path] = "data") -> Dict[str, List[Dict[str, Any]]]:
    """Loads held-out validation, test, and benchmark splits strictly read-only for gate auditing."""
    d = Path(data_dir)
    return {
        "validation": load_jsonl_records(d / "validation.jsonl"),
        "test": load_jsonl_records(d / "test.jsonl"),
        "benchmark": load_jsonl_records(d / "benchmark_eval.jsonl"),
    }
