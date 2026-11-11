"""
Comparison infrastructure for independent NL2SQL evaluation benchmark (Phase 1, Step 2).

Provides:
1. ASTComparator: AST-based query parser, clause extractor, and canonicalized comparator
   using SQLGlot with MySQL dialect.
2. ExecutionComparator: Database execution result set comparator supporting unordered multiset
   comparison, ordered sequence comparison, NULL handling, and Decimal numeric precision.
"""

from __future__ import annotations

import decimal
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import sqlglot
from sqlglot import exp

from src.data.tokenizer import DEFAULT_STUDENT_SCHEMA
from src.database.executor import ExecutionResult, SQLExecutor
from src.validation.validator import SQLValidator

logger = logging.getLogger("src.evaluation.comparator")


@dataclass
class ASTComparisonResult:
    """
    Structured outcome of AST-based query comparison.
    """
    is_valid_sql: bool
    syntax_error: Optional[str] = None
    strict_exact_match: bool = False
    normalized_exact_match: bool = False
    select_match: bool = False
    where_match: bool = False
    order_match: bool = False
    limit_match: bool = False
    table_match: bool = False
    clause_diffs: dict[str, Any] = field(default_factory=dict)
    failure_mode: str = "UNKNOWN"
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid_sql": self.is_valid_sql,
            "syntax_error": self.syntax_error,
            "strict_exact_match": self.strict_exact_match,
            "normalized_exact_match": self.normalized_exact_match,
            "select_match": self.select_match,
            "where_match": self.where_match,
            "order_match": self.order_match,
            "limit_match": self.limit_match,
            "table_match": self.table_match,
            "clause_diffs": dict(self.clause_diffs),
            "failure_mode": self.failure_mode,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ExecutionComparisonResult:
    """
    Structured outcome of database execution comparison.
    """
    executed: bool
    ref_success: bool
    pred_success: bool
    results_match: bool
    ref_row_count: int = 0
    pred_row_count: int = 0
    has_order_by: bool = False
    ref_error: Optional[str] = None
    pred_error: Optional[str] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "ref_success": self.ref_success,
            "pred_success": self.pred_success,
            "results_match": self.results_match,
            "ref_row_count": self.ref_row_count,
            "pred_row_count": self.pred_row_count,
            "has_order_by": self.has_order_by,
            "ref_error": self.ref_error,
            "pred_error": self.pred_error,
            "failure_reason": self.failure_reason,
        }


class ASTComparator:
    """
    AST-based SQL comparator specialized for MySQL dialect via SQLGlot.

    Supports:
    - Formal AST parsing.
    - Table, Column, and Clause extraction.
    - Conjunction-order invariant comparison (A AND B == B AND A).
    - Disjunction-order invariant comparison (A OR B == B OR A).
    - Preservation of operator precedence and parenthetical grouping.
    - Independent clause-level evaluation (SELECT, WHERE, ORDER BY, LIMIT).
    """

    def __init__(self, dialect: str = "mysql") -> None:
        self.dialect = dialect

    def parse_sql(self, sql_str: str) -> Tuple[bool, Optional[str], Optional[exp.Select]]:
        """
        Parses SQL string into a SQLGlot Select expression.
        """
        if not sql_str or not sql_str.strip():
            return False, "SQL string is empty", None
        try:
            parsed = sqlglot.parse_one(sql_str.strip(), read=self.dialect)
            if not isinstance(parsed, exp.Select):
                return False, f"Expected Select statement, got {type(parsed).__name__}", None
            return True, None, parsed
        except Exception as exc:
            return False, str(exc), None

    @classmethod
    def canonicalize_ast_node(cls, node: Optional[exp.Expression]) -> Any:
        """
        Recursively converts AST expressions into a canonical Python data structure
        that is invariant to conjunction/disjunction operand ordering while strictly
        preserving boolean operators, literal values, column names, and parenthetical grouping.
        """
        if node is None:
            return None

        # Unwrap unnecessary outer parentheses
        while isinstance(node, exp.Paren):
            node = node.this

        if isinstance(node, exp.And):
            # Flatten associative and commutative conjunctions
            conjuncts: list[Any] = []
            def collect_ands(curr: exp.Expression) -> None:
                while isinstance(curr, exp.Paren):
                    curr = curr.this
                if isinstance(curr, exp.And):
                    collect_ands(curr.left)
                    collect_ands(curr.right)
                else:
                    conjuncts.append(cls.canonicalize_ast_node(curr))

            collect_ands(node)
            # Sort conjuncts deterministically by string representation
            conjuncts.sort(key=lambda item: str(item))
            return ("AND", tuple(conjuncts))

        elif isinstance(node, exp.Or):
            # Flatten associative and commutative disjunctions
            disjuncts: list[Any] = []
            def collect_ors(curr: exp.Expression) -> None:
                while isinstance(curr, exp.Paren):
                    curr = curr.this
                if isinstance(curr, exp.Or):
                    collect_ors(curr.left)
                    collect_ors(curr.right)
                else:
                    disjuncts.append(cls.canonicalize_ast_node(curr))

            collect_ors(node)
            # Sort disjuncts deterministically by string representation
            disjuncts.sort(key=lambda item: str(item))
            return ("OR", tuple(disjuncts))

        elif isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            # Binary comparison operator
            op_name = type(node).__name__
            left_sql = cls._normalize_leaf(node.left)
            right_sql = cls._normalize_leaf(node.right)
            return ("COMP", op_name, left_sql, right_sql)

        elif isinstance(node, exp.Not):
            return ("NOT", cls.canonicalize_ast_node(node.this))

        else:
            # Atomic leaf expression
            return ("LEAF", cls._normalize_leaf(node))

    @staticmethod
    def _normalize_leaf(node: Optional[exp.Expression]) -> str:
        if node is None:
            return ""
        # Format via SQLGlot in MySQL dialect and lowercase
        sql_val = node.sql(dialect="mysql").strip()
        # Canonicalize spaces
        return " ".join(sql_val.lower().split())

    def extract_clauses(self, parsed: Optional[exp.Select]) -> dict[str, Any]:
        """
        Extracts structured representations of query clauses from a Select AST.
        """
        if parsed is None or not isinstance(parsed, exp.Select):
            return {
                "table": None,
                "select": [],
                "where_raw": None,
                "where_canonical": None,
                "order": [],
                "limit": None,
            }

        # 1. Table
        tables = [t.name.lower() for t in parsed.find_all(exp.Table)]
        table_name = tables[0] if tables else None

        # 2. SELECT / Projections
        projections: list[str] = []
        for expr in parsed.expressions:
            if isinstance(expr, exp.Star):
                projections.append("*")
            else:
                projections.append(" ".join(expr.sql(dialect=self.dialect).lower().split()))

        # 3. WHERE Clause
        where_node = parsed.find(exp.Where)
        if where_node and where_node.this:
            where_raw = " ".join(where_node.this.sql(dialect=self.dialect).split())
            where_canonical = self.canonicalize_ast_node(where_node.this)
        else:
            where_raw = None
            where_canonical = None

        # 4. ORDER BY
        order_list: list[tuple[str, str]] = []
        order_node = parsed.find(exp.Order)
        if order_node:
            for ordered in order_node.expressions:
                col_name = " ".join(ordered.this.sql(dialect=self.dialect).lower().split())
                # Default direction in MySQL is ASC
                direction = "DESC" if ordered.args.get("desc") else "ASC"
                order_list.append((col_name, direction))

        # 5. LIMIT
        limit_val: Optional[int] = None
        limit_node = parsed.find(exp.Limit)
        if limit_node and limit_node.expression:
            try:
                limit_val = int(limit_node.expression.name)
            except (ValueError, TypeError, AttributeError):
                limit_val = None

        return {
            "table": table_name,
            "select": projections,
            "where_raw": where_raw,
            "where_canonical": where_canonical,
            "order": order_list,
            "limit": limit_val,
        }

    def compare(self, ref_sql: str, pred_sql: str) -> ASTComparisonResult:
        """
        Performs comprehensive AST-level clause comparison between reference and predicted SQL.
        """
        strict_em = (ref_sql.strip() == pred_sql.strip())

        is_valid, syntax_err, pred_ast = self.parse_sql(pred_sql)
        _, _, ref_ast = self.parse_sql(ref_sql)

        if not is_valid:
            return ASTComparisonResult(
                is_valid_sql=False,
                syntax_error=syntax_err,
                strict_exact_match=False,
                normalized_exact_match=False,
                failure_mode="SYNTAX_ERROR",
                failure_reason=f"Predicted SQL syntax error: {syntax_err}",
            )

        ref_clauses = self.extract_clauses(ref_ast)
        pred_clauses = self.extract_clauses(pred_ast)

        # Clause 1: Table match
        table_match = (ref_clauses["table"] == pred_clauses["table"])

        # Clause 2: Projection / SELECT match
        select_match = (ref_clauses["select"] == pred_clauses["select"])

        # Clause 3: WHERE clause match (canonicalized, conjunction/disjunction order-invariant)
        where_match = (ref_clauses["where_canonical"] == pred_clauses["where_canonical"])

        # Clause 4: ORDER BY match
        order_match = (ref_clauses["order"] == pred_clauses["order"])

        # Clause 5: LIMIT match
        limit_match = (ref_clauses["limit"] == pred_clauses["limit"])

        normalized_em = (
            table_match and select_match and where_match and order_match and limit_match
        )

        clause_diffs: dict[str, Any] = {
            "table": {"ref": ref_clauses["table"], "pred": pred_clauses["table"], "match": table_match},
            "select": {"ref": ref_clauses["select"], "pred": pred_clauses["select"], "match": select_match},
            "where": {"ref": ref_clauses["where_raw"], "pred": pred_clauses["where_raw"], "match": where_match},
            "order": {"ref": ref_clauses["order"], "pred": pred_clauses["order"], "match": order_match},
            "limit": {"ref": ref_clauses["limit"], "pred": pred_clauses["limit"], "match": limit_match},
        }

        # Classify Failure Mode if not exact match
        failure_mode, failure_reason = self._diagnose_failure(
            ref_clauses=ref_clauses,
            pred_clauses=pred_clauses,
            table_match=table_match,
            select_match=select_match,
            where_match=where_match,
            order_match=order_match,
            limit_match=limit_match,
            normalized_em=normalized_em,
        )

        return ASTComparisonResult(
            is_valid_sql=True,
            syntax_error=None,
            strict_exact_match=strict_em,
            normalized_exact_match=normalized_em,
            select_match=select_match,
            where_match=where_match,
            order_match=order_match,
            limit_match=limit_match,
            table_match=table_match,
            clause_diffs=clause_diffs,
            failure_mode=failure_mode,
            failure_reason=failure_reason,
        )

    def _diagnose_failure(
        self,
        ref_clauses: dict[str, Any],
        pred_clauses: dict[str, Any],
        table_match: bool,
        select_match: bool,
        where_match: bool,
        order_match: bool,
        limit_match: bool,
        normalized_em: bool,
    ) -> Tuple[str, str]:
        if normalized_em:
            return "EXACT_MATCH", "Predicted SQL matches reference SQL in all clauses."

        if not table_match:
            return "TABLE_MISMATCH", f"Target table expected {ref_clauses['table']}, got {pred_clauses['table']}."

        # Count how many clauses differed
        diffs = [not select_match, not where_match, not order_match, not limit_match]
        if sum(diffs) > 1:
            return "COMPOUND_MISMATCH", "Multiple clauses differed across projection, filtering, ordering, or limit."

        if not select_match:
            return "PROJECTION_MISMATCH", f"Projection mismatch: expected {ref_clauses['select']}, got {pred_clauses['select']}."

        if not order_match:
            return "ORDER_BY_MISMATCH", f"ORDER BY mismatch: expected {ref_clauses['order']}, got {pred_clauses['order']}."

        if not limit_match:
            return "LIMIT_MISMATCH", f"LIMIT mismatch: expected {ref_clauses['limit']}, got {pred_clauses['limit']}."

        if not where_match:
            r_where = ref_clauses["where_raw"]
            p_where = pred_clauses["where_raw"]
            if r_where is None and p_where is not None:
                return "EXTRA_WHERE_CLAUSE", f"Generated unrequested WHERE clause: '{p_where}'."
            if r_where is not None and p_where is None:
                return "MISSING_WHERE_CLAUSE", f"Omitted required WHERE clause: '{r_where}'."

            # Check for duplicate / redundant conditions
            p_canon = pred_clauses["where_canonical"]
            if isinstance(p_canon, tuple) and p_canon[0] == "AND":
                # Check for repeated elements in predicted conjuncts
                conjunct_counts = Counter(p_canon[1])
                if any(cnt > 1 for cnt in conjunct_counts.values()):
                    return "REDUNDANT_WHERE_CONDITION", f"Generated duplicate redundant condition in WHERE clause: '{p_where}'."

            return "WHERE_CONDITION_MISMATCH", f"WHERE mismatch: expected '{r_where}', got '{p_where}'."

        return "UNKNOWN_MISMATCH", "Discrepancy detected between queries."


class ExecutionComparator:
    """
    Result set comparator for SQL query execution on MySQL.

    Enforces:
    - Dedicated read-only credentials via SQLExecutor.
    - Zero write operations permitted.
    - Result comparison:
      - With ORDER BY: strictly ordered sequence comparison.
      - Without ORDER BY: unordered multiset comparison (preserves duplicates).
    - Type-safe numeric normalization (Decimal, int, float).
    - Preserves NULL semantics (None matches None; None != 0).
    """

    def __init__(
        self,
        executor: Optional[SQLExecutor] = None,
        validator: Optional[SQLValidator] = None,
        schema: Optional[str] = None,
        max_rows: int = 100,
    ) -> None:
        self.executor = executor
        target_schema = schema or DEFAULT_STUDENT_SCHEMA
        self.validator = (
            validator
            if validator is not None
            else SQLValidator(schema=target_schema, enforce_model_scope=True)
        )
        self.max_rows = max_rows

    @staticmethod
    def normalize_cell(val: Any) -> Any:
        """
        Normalizes single cell values for safe cross-driver comparison.
        """
        if val is None:
            return None
        if isinstance(val, decimal.Decimal):
            # Convert decimal to float
            return float(val)
        if isinstance(val, (int, float)):
            return float(val) if isinstance(val, float) else val
        return str(val)

    @classmethod
    def row_to_canonical_tuple(cls, row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        """
        Converts a row dict into a canonical sorted tuple of (lowercased_column_name, normalized_value).
        """
        return tuple(sorted((k.lower(), cls.normalize_cell(v)) for k, v in row.items()))

    def compare_result_rows(
        self,
        ref_rows: list[dict[str, Any]],
        pred_rows: list[dict[str, Any]],
        has_order_by: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Compares two lists of row dictionaries according to benchmark rules.
        """
        if len(ref_rows) != len(pred_rows):
            return (
                False,
                f"Row count mismatch: expected {len(ref_rows)} rows, got {len(pred_rows)} rows",
            )

        if has_order_by:
            # Ordered sequence comparison
            for idx, (r_row, p_row) in enumerate(zip(ref_rows, pred_rows)):
                r_tuple = self.row_to_canonical_tuple(r_row)
                p_tuple = self.row_to_canonical_tuple(p_row)
                if r_tuple != p_tuple:
                    return (
                        False,
                        f"Ordered row mismatch at row index {idx}: expected {r_row}, got {p_row}",
                    )
            return True, None
        else:
            # Unordered multiset comparison
            ref_multiset = Counter(self.row_to_canonical_tuple(r) for r in ref_rows)
            pred_multiset = Counter(self.row_to_canonical_tuple(p) for p in pred_rows)
            if ref_multiset != pred_multiset:
                return False, "Result rows do not match as an unordered multiset."
            return True, None

    def execute_and_compare(
        self,
        ref_sql: str,
        pred_sql: str,
        is_pred_valid: bool = True,
        has_order_by: Optional[bool] = None,
    ) -> ExecutionComparisonResult:
        """
        Executes reference and predicted SQL queries and compares their result sets.
        """
        if self.executor is None:
            return ExecutionComparisonResult(
                executed=False,
                ref_success=False,
                pred_success=False,
                results_match=False,
                failure_reason="SQLExecutor is unconfigured or database access is unavailable.",
            )

        # Detect ORDER BY if not explicitly provided
        if has_order_by is None:
            has_order_by = "order by" in ref_sql.lower()

        # 1. Execute Reference Query
        ref_val = self.validator.validate(ref_sql)
        if not ref_val.is_valid:
            return ExecutionComparisonResult(
                executed=False,
                ref_success=False,
                pred_success=False,
                results_match=False,
                ref_error=f"Reference query failed validation: {ref_val.error_message}",
                failure_reason="Invalid reference query",
            )

        ref_exec: ExecutionResult = self.executor.execute(
            validation_result=ref_val,
            approved=True,
            max_rows=self.max_rows,
        )

        if not ref_exec.success:
            return ExecutionComparisonResult(
                executed=True,
                ref_success=False,
                pred_success=False,
                results_match=False,
                ref_error=ref_exec.error,
                failure_reason=f"Reference query execution failed: {ref_exec.error}",
            )

        # 2. Check and Execute Predicted Query
        if not is_pred_valid:
            return ExecutionComparisonResult(
                executed=True,
                ref_success=True,
                pred_success=False,
                results_match=False,
                ref_row_count=ref_exec.row_count,
                pred_row_count=0,
                has_order_by=has_order_by,
                failure_reason="Predicted query failed syntactic or safety validation; execution skipped.",
            )

        pred_val = self.validator.validate(pred_sql)
        if not pred_val.is_valid:
            return ExecutionComparisonResult(
                executed=True,
                ref_success=True,
                pred_success=False,
                results_match=False,
                ref_row_count=ref_exec.row_count,
                pred_row_count=0,
                has_order_by=has_order_by,
                pred_error=pred_val.error_message,
                failure_reason=f"Predicted query failed validation: {pred_val.error_message}",
            )

        pred_exec: ExecutionResult = self.executor.execute(
            validation_result=pred_val,
            approved=True,
            max_rows=self.max_rows,
        )

        if not pred_exec.success:
            return ExecutionComparisonResult(
                executed=True,
                ref_success=True,
                pred_success=False,
                results_match=False,
                ref_row_count=ref_exec.row_count,
                pred_row_count=0,
                has_order_by=has_order_by,
                pred_error=pred_exec.error,
                failure_reason=f"Predicted query execution failed in MySQL: {pred_exec.error}",
            )

        # 3. Compare Result Sets
        match, diff_reason = self.compare_result_rows(
            ref_rows=ref_exec.rows,
            pred_rows=pred_exec.rows,
            has_order_by=has_order_by,
        )

        return ExecutionComparisonResult(
            executed=True,
            ref_success=True,
            pred_success=True,
            results_match=match,
            ref_row_count=ref_exec.row_count,
            pred_row_count=pred_exec.row_count,
            has_order_by=has_order_by,
            failure_reason=diff_reason if not match else None,
        )
