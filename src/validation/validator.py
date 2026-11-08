"""
SQL Validation and Read-Only Safety Layer for NL2SQL (Milestone 4, Task 4.3).

This module provides dedicated, AST-based SQL inspection using SQLGlot with
the MySQL dialect (read="mysql"). It enforces:
1. Syntactic validity as MySQL SQL.
2. Strict read-only single-statement execution (permits only a single SELECT;
   forbids INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, REPLACE,
   GRANT, REVOKE, CALL, EXECUTE, locking clauses, and SELECT INTO).
3. Model capability boundaries (current model supports SELECT, WHERE,
   comparison operators, AND/OR, ORDER BY, LIMIT; rejects/flags aggregations,
   GROUP BY, HAVING, JOIN, multiple FROM tables, and subqueries).
4. Database schema compatibility against live or canonical schema metadata
   (validates tables, columns, aliases, qualifiers, wildcards, and detects
   ambiguous column references).

Parser Rationale:
-----------------
SQLGlot was selected over regex or naive tokenization because:
- It produces a formal Abstract Syntax Tree (AST) specialized for MySQL dialect.
- It parses structural statements without executing subprocesses or commands.
- It is resilient to obfuscation, inline comments (/* */, --), casing, and whitespace.
- It explicitly differentiates statement types and provides specialized AST nodes
  for locking clauses (exp.Lock), table references (exp.Table), and mutations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Silence non-critical sqlglot parser fallbacks
logging.getLogger("sqlglot").setLevel(logging.ERROR)

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError, TokenError
    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover
    HAS_SQLGLOT = False
    sqlglot = None  # type: ignore
    exp = None  # type: ignore
    ParseError = Exception  # type: ignore
    TokenError = Exception  # type: ignore

from src.database.schema import DatabaseSchemaInspector, DatabaseSchemaMetadata, TableMetadata
from src.validation.result import ValidationResult

# Forbidden AST statement or operation types that mutate state or grant access
_FORBIDDEN_TYPE_NAMES: list[str] = [
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Alter",
    "Create",
    "TruncateTable",
    "Command",
    "Transaction",
    "Commit",
    "Rollback",
    "Set",
    "Grant",
    "Revoke",
    "Kill",
    "LoadData",
]
FORBIDDEN_AST_NODES: tuple[type, ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_TYPE_NAMES if hasattr(exp, name)
)

# Functions that present security risks (execution delays, filesystem access, privilege probes)
FORBIDDEN_FUNCTIONS: set[str] = {
    "sleep",
    "benchmark",
    "load_file",
    "sys_exec",
    "sys_eval",
    "get_lock",
    "release_lock",
    "version",
    "currentversion",
    "current_version",
    "user",
    "currentuser",
    "current_user",
    "systemuser",
    "system_user",
    "sessionuser",
    "session_user",
    "database",
    "currentschema",
    "current_schema",
    "schema",
}

# Aggregate functions outside current model training scope
AGGREGATE_FUNCTIONS: set[str] = {
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "std",
    "variance",
    "group_concat",
}

# Vetted, pure deterministic read-only functions permitted in general SQL mode.
# Any function not in this set (and any unvetted, user-defined, or stored function) is strictly rejected.
ALLOWED_READONLY_FUNCTIONS: set[str] = {
    "abs",
    "round",
    "ceil",
    "floor",
    "lower",
    "upper",
    "trim",
    "length",
    "concat",
    "coalesce",
    # Read-only aggregates (permitted as read-only, gated under scope checking)
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "std",
    "variance",
    "group_concat",
}


def normalize_schema(
    schema: Optional[Union[DatabaseSchemaMetadata, DatabaseSchemaInspector, dict[str, Any], str]]
) -> Optional[dict[str, list[str]]]:
    """
    Normalizes diverse schema representations into a canonical dictionary
    mapping lowercase table names to lowercase column name lists.

    Supported input formats:
    - DatabaseSchemaMetadata instance (from src.database.schema)
    - DatabaseSchemaInspector instance (calls inspect_all())
    - Canonical schema string: 'table1(col1, col2) | table2(...)'
    - Dictionary: {'students': ['student_id', 'name'], ...}
    - None or empty structures -> returns None

    Args:
        schema: Raw schema metadata or string representation.

    Returns:
        Dictionary {table_name: [column_names]} in lowercase, or None if invalid/empty.
    """
    if schema is None:
        return None

    # Case 1: DatabaseSchemaInspector
    if isinstance(schema, DatabaseSchemaInspector):
        try:
            if hasattr(schema, "inspect_database"):
                metadata = schema.inspect_database()
            elif hasattr(schema, "inspect_all"):
                metadata = schema.inspect_all()
            else:
                return None
            return normalize_schema(metadata)
        except Exception:
            return None

    # Case 2: DatabaseSchemaMetadata
    if isinstance(schema, DatabaseSchemaMetadata):
        if not schema.tables:
            return None
        normalized: dict[str, list[str]] = {}
        for t_name, t_meta in schema.tables.items():
            normalized[t_name.lower().strip()] = [
                c.lower().strip() for c in t_meta.column_names
            ]
        return normalized

    # Case 3: Canonical schema string, e.g.:
    # "students(student_id, name, age) | courses(course_id, course_name)"
    if isinstance(schema, str):
        cleaned_str = schema.strip()
        if not cleaned_str:
            return None
        normalized_str_schema: dict[str, list[str]] = {}
        table_chunks = [t.strip() for t in re.split(r"[|\n]", cleaned_str) if t.strip()]
        for chunk in table_chunks:
            match = re.match(r"^([a-zA-Z0-9_]+)\s*\((.*?)\)$", chunk)
            if match:
                table_name = match.group(1).lower().strip()
                cols_str = match.group(2)
                cols = [c.strip().lower() for c in cols_str.split(",") if c.strip()]
                normalized_str_schema[table_name] = cols
        return normalized_str_schema if normalized_str_schema else None

    # Case 4: Dictionary mapping
    if isinstance(schema, dict):
        if not schema:
            return None
        normalized_dict_schema: dict[str, list[str]] = {}
        for k, v in schema.items():
            t_name = str(k).lower().strip()
            if isinstance(v, TableMetadata):
                normalized_dict_schema[t_name] = [c.lower().strip() for c in v.column_names]
            elif isinstance(v, (list, tuple, set)):
                normalized_dict_schema[t_name] = [str(c).lower().strip() for c in v]
            elif isinstance(v, dict):
                normalized_dict_schema[t_name] = [str(c).lower().strip() for c in v.keys()]
            else:
                normalized_dict_schema[t_name] = []
        return normalized_dict_schema if normalized_dict_schema else None

    return None


class SQLValidator:
    """
    Dedicated SQL Validator and Read-Only Safety Layer for NL2SQL.

    Inspects SQL queries using SQLGlot AST analysis to verify:
    - Syntactic validity under MySQL dialect.
    - Single-statement read-only execution.
    - Conformance to model scope (SELECT, WHERE, comparisons, AND/OR, ORDER BY, LIMIT).
    - Database schema compatibility against target schema metadata.
    """

    def __init__(
        self,
        schema: Optional[Union[DatabaseSchemaMetadata, DatabaseSchemaInspector, dict[str, Any], str]] = None,
        enforce_scope: bool = True,
        enforce_model_scope: bool = True,
    ) -> None:
        """
        Initializes the validator with an optional default schema and scope enforcement setting.

        Args:
            schema: Optional schema metadata (metadata object, dict, or canonical string).
            enforce_scope: Whether to treat queries outside general supported read-only scope as failures.
            enforce_model_scope: Whether to strictly enforce the trained Seq2Seq model vocabulary boundary.
                Defaults to True because our immediate workflow validates model-generated SQL.
                May be explicitly set to False only for manual/general SQL inspection; must NOT
                be disabled for model-generated SQL.
        """
        self.default_schema = schema
        self.enforce_scope = enforce_scope
        self.enforce_model_scope = enforce_model_scope

    def validate(
        self,
        sql: Any,
        schema: Optional[Union[DatabaseSchemaMetadata, DatabaseSchemaInspector, dict[str, Any], str]] = None,
        enforce_scope: Optional[bool] = None,
        enforce_model_scope: Optional[bool] = None,
    ) -> ValidationResult:
        """
        Validates an SQL query against syntax, read-only safety, scope, and schema rules.

        Args:
            sql: SQL query string to inspect.
            schema: Optional schema metadata. Overrides default_schema if supplied.
            enforce_scope: Optional boolean overriding the instance general scope enforcement setting.
            enforce_model_scope: Optional boolean overriding the instance model scope enforcement setting.

        Returns:
            ValidationResult with detailed breakdown of validation checks.
        """
        should_enforce_scope = self.enforce_scope if enforce_scope is None else enforce_scope
        should_enforce_model_scope = self.enforce_model_scope if enforce_model_scope is None else enforce_model_scope
        active_schema = schema if schema is not None else self.default_schema

        errors: list[str] = []
        warnings: list[str] = []
        ast_details: dict[str, Any] = {
            "statement_type": None,
            "tables": [],
            "columns": [],
            "aliases": {},
            "has_where": False,
            "has_order_by": False,
            "has_limit": False,
            "has_joins": False,
            "has_aggregations": False,
            "has_group_by": False,
            "has_having": False,
            "has_subqueries": False,
            "has_locks": False,
            "has_into": False,
        }

        # -------------------------------------------------------------
        # Stage 1: Input Sanity & Type Check
        # -------------------------------------------------------------
        if not isinstance(sql, str):
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=str(sql) if sql is not None else None,
                errors=["SQL input must be a non-null string."],
                warnings=warnings,
                ast_details=ast_details,
            )

        raw_sql = sql
        stripped_sql = sql.strip()

        if not stripped_sql:
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=["SQL query string is empty or contains only whitespace."],
                warnings=warnings,
                ast_details=ast_details,
            )

        # Pre-check for raw patterns that represent file dumps or variable writes
        if re.search(r"\bINTO\s+(OUTFILE|DUMPFILE)\b", stripped_sql, re.IGNORECASE):
            errors.append("SELECT INTO OUTFILE/DUMPFILE file-dumping operations are strictly prohibited.")

        # -------------------------------------------------------------
        # Stage 2: MySQL Dialect Parsing & Statement Count
        # -------------------------------------------------------------
        if not HAS_SQLGLOT:  # pragma: no cover
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=["SQLGlot parser dependency is not available."],
                warnings=warnings,
                ast_details=ast_details,
            )

        try:
            parsed_list = sqlglot.parse(stripped_sql, read="mysql")
        except (ParseError, TokenError) as err:
            clean_err = self._sanitize_parser_error(str(err))
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=[f"SQL syntax error: {clean_err}"],
                warnings=warnings,
                ast_details=ast_details,
            )
        except Exception as err:
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=[f"Failed to parse SQL: {self._sanitize_parser_error(str(err))}"],
                warnings=warnings,
                ast_details=ast_details,
            )

        # Filter out empty statements or isolated comments/semicolons
        substantive = [
            s for s in parsed_list
            if s is not None and not isinstance(s, exp.Semicolon)
        ]

        if not substantive:
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=["Query contains no executable SQL statements."],
                warnings=warnings,
                ast_details=ast_details,
            )

        if len(substantive) > 1:
            errors.append(
                f"Multiple SQL statements detected (found {len(substantive)} statements). "
                "Only a single statement is permitted."
            )
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=True,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=errors,
                warnings=warnings,
                ast_details=ast_details,
            )

        statement: exp.Expression = substantive[0]
        ast_details["statement_type"] = type(statement).__name__.upper()

        # Check top-level statement type
        if not isinstance(statement, exp.Select):
            errors.append(
                f"Forbidden statement type: expected SELECT, but got {ast_details['statement_type']}. "
                "Only single read-only SELECT queries are permitted."
            )
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=True,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=errors,
                warnings=warnings,
                ast_details=ast_details,
            )

        # Check projection expressions exist
        if not statement.expressions:
            errors.append("SELECT statement has no projected expressions or columns.")
            return ValidationResult(
                is_valid=False,
                is_syntax_valid=False,
                is_permitted_read_only=False,
                is_schema_compatible=False,
                is_in_supported_scope=False,
                is_in_model_scope=False,
                normalized_sql=None,
                raw_sql=raw_sql,
                errors=errors,
                warnings=warnings,
                ast_details=ast_details,
            )

        is_syntax_valid = True

        # -------------------------------------------------------------
        # Stage 3: Read-Only Safety & Disallowed Clauses
        # -------------------------------------------------------------
        is_permitted_read_only = True

        # Check for mutation/administrative nodes anywhere in AST
        for node in statement.walk():
            if isinstance(node, FORBIDDEN_AST_NODES):
                is_permitted_read_only = False
                op_name = type(node).__name__.upper()
                errors.append(f"Forbidden operation '{op_name}' detected in query.")
                break

        # Check for locking clauses (FOR UPDATE, LOCK IN SHARE MODE, FOR SHARE)
        locks = list(statement.find_all(exp.Lock))
        if locks or re.search(r"\bFOR\s+UPDATE\b|\bLOCK\s+IN\s+SHARE\s+MODE\b|\bFOR\s+SHARE\b", stripped_sql, re.IGNORECASE):
            is_permitted_read_only = False
            ast_details["has_locks"] = True
            errors.append("Locking clauses (FOR UPDATE, LOCK IN SHARE MODE, FOR SHARE) are strictly prohibited.")

        # Check for INTO clauses (INTO OUTFILE, INTO DUMPFILE, INTO @var)
        intos = list(statement.find_all(exp.Into))
        if intos:
            is_permitted_read_only = False
            ast_details["has_into"] = True
            errors.append("SELECT INTO clauses (such as variables or file output) are strictly prohibited.")

        # Check for system variables and session parameters (@@global, @my_var)
        variables = list(statement.find_all((exp.SessionParameter, exp.Parameter)))
        if variables:
            is_permitted_read_only = False
            errors.append("System variables (@@) and session parameters (@) are strictly prohibited.")

        # Check functions against conservative allowlist and denylist
        for func in statement.find_all((exp.Func, exp.Anonymous)):
            if isinstance(func, (exp.And, exp.Or)):
                continue

            func_names = self._get_function_names(func)

            # 1. Reject explicitly forbidden functions first
            matched_forbidden = func_names.intersection(FORBIDDEN_FUNCTIONS)
            if matched_forbidden:
                is_permitted_read_only = False
                forbidden_name = sorted(list(matched_forbidden))[0]
                errors.append(f"Call to prohibited or security-sensitive function '{forbidden_name}' is not allowed.")
                continue

            # 2. Reject any anonymous/unrecognized/user-defined function
            if isinstance(func, exp.Anonymous):
                is_permitted_read_only = False
                errors.append(
                    f"Call to unapproved, stored, or user-defined function '{func.name}' is strictly prohibited."
                )
                continue

            # 3. Reject any function not present in the vetted read-only allowlist
            if not func_names.intersection(ALLOWED_READONLY_FUNCTIONS):
                is_permitted_read_only = False
                fname = getattr(func, "name", None) or type(func).__name__.lower()
                errors.append(
                    f"Call to unapproved function '{fname}' is not permitted. "
                    f"Allowed read-only functions: {', '.join(sorted(ALLOWED_READONLY_FUNCTIONS))}."
                )

        # -------------------------------------------------------------
        # Stage 4: Scope Boundaries Check
        # Current parser-supported scope: SELECT, WHERE, comparisons, AND/OR, ORDER BY, LIMIT
        # Out-of-scope: Aggregations, GROUP BY, HAVING, JOIN, multiple FROM tables, UNION, subqueries
        # Trained Model scope: strictly 70-token target vocabulary
        # -------------------------------------------------------------
        is_in_supported_scope = True
        out_of_scope_features: list[str] = []

        # 1. Aggregations
        aggs = list(statement.find_all(exp.AggFunc))
        for func in statement.find_all((exp.Func, exp.Anonymous)):
            if self._get_function_names(func).intersection(AGGREGATE_FUNCTIONS):
                aggs.append(func)
        if aggs:
            ast_details["has_aggregations"] = True
            out_of_scope_features.append("Aggregations (COUNT, SUM, AVG, MIN, MAX)")

        # 2. GROUP BY
        groups = list(statement.find_all(exp.Group))
        if groups or statement.args.get("group") is not None:
            ast_details["has_group_by"] = True
            out_of_scope_features.append("GROUP BY")

        # 3. HAVING
        havings = list(statement.find_all(exp.Having))
        if havings or statement.args.get("having") is not None:
            ast_details["has_having"] = True
            out_of_scope_features.append("HAVING")

        # 4. Joins and Multi-Table FROM
        joins = list(statement.find_all(exp.Join))
        table_nodes = list(statement.find_all(exp.Table))
        if joins or len(table_nodes) > 1:
            ast_details["has_joins"] = True
            out_of_scope_features.append("JOIN / Multi-table query")

        # 5. Subqueries
        subqueries = list(statement.find_all(exp.Subquery))
        if subqueries:
            ast_details["has_subqueries"] = True
            out_of_scope_features.append("Subqueries")

        # 6. Set operations (UNION, INTERSECT, EXCEPT)
        set_ops = list(statement.find_all((exp.Union, exp.Intersect, exp.Except)))
        if set_ops:
            out_of_scope_features.append("Set operations (UNION, INTERSECT, EXCEPT)")

        # 7. Window functions
        windows = list(statement.find_all(exp.Window))
        if windows:
            out_of_scope_features.append("Window functions")

        # 8. DISTINCT
        if statement.args.get("distinct") is not None or list(statement.find_all(exp.Distinct)):
            out_of_scope_features.append("DISTINCT")

        if out_of_scope_features:
            is_in_supported_scope = False
            msg = (
                f"Query contains SQL features outside the general supported read-only scope: "
                f"{', '.join(out_of_scope_features)}. Supported scope is limited to: "
                "SELECT, WHERE, comparison operators, AND/OR, ORDER BY, and LIMIT."
            )
            if should_enforce_scope:
                errors.append(msg)
            else:
                warnings.append(f"{msg} (Allowed because general scope enforcement is disabled).")

        # Model Scope Inspection (strictly evaluating against 70-token target vocabulary)
        # Features absent from model target vocabulary:
        # LIKE, BETWEEN, IN, IS NULL/IS NOT NULL, NOT, Table Aliases, AS aliases, Functions
        is_in_model_scope = is_in_supported_scope
        out_of_model_scope_features: list[str] = list(out_of_scope_features)

        if list(statement.find_all(exp.Like)):
            out_of_model_scope_features.append("LIKE operator")
        if list(statement.find_all(exp.Between)):
            out_of_model_scope_features.append("BETWEEN operator")
        if list(statement.find_all(exp.In)):
            out_of_model_scope_features.append("IN operator")
        if list(statement.find_all(exp.Is)):
            out_of_model_scope_features.append("IS NULL / IS NOT NULL")
        if list(statement.find_all(exp.Not)):
            out_of_model_scope_features.append("NOT operator")
        if list(statement.find_all(exp.TableAlias)):
            out_of_model_scope_features.append("Table Aliases")
        if any(isinstance(s, exp.Alias) for s in statement.selects):
            out_of_model_scope_features.append("Column AS Aliases")
        actual_funcs = [
            f for f in statement.find_all((exp.Func, exp.Anonymous))
            if not isinstance(f, (exp.And, exp.Or))
        ]
        if actual_funcs:
            out_of_model_scope_features.append("Function Calls")

        if out_of_model_scope_features:
            is_in_model_scope = False
            unique_model_features = sorted(list(set(out_of_model_scope_features)))
            ast_details["out_of_model_scope_features"] = unique_model_features
            if should_enforce_model_scope:
                msg_model = (
                    f"Query contains features outside the trained Seq2Seq model vocabulary: "
                    f"{', '.join(unique_model_features)}. The trained model supports only single-table "
                    "SELECT queries with =, !=, <, <=, >, >=, AND, OR, ORDER BY, and LIMIT."
                )
                errors.append(msg_model)
            else:
                warnings.append(
                    f"Query contains features outside the trained model vocabulary: "
                    f"{', '.join(unique_model_features)} (permitted because enforce_model_scope=False opt-out was specified)."
                )

        # Update AST details
        ast_details["has_where"] = statement.args.get("where") is not None
        ast_details["has_order_by"] = statement.args.get("order") is not None
        ast_details["has_limit"] = statement.args.get("limit") is not None

        # -------------------------------------------------------------
        # Stage 5: Schema Compatibility Validation
        # -------------------------------------------------------------
        norm_schema = normalize_schema(active_schema)
        is_schema_compatible = True

        if norm_schema is None or len(norm_schema) == 0:
            is_schema_compatible = False
            errors.append(
                "Schema metadata was not provided or is empty. "
                "Schema compatibility cannot be verified."
            )
        else:
            # Validate tables and resolve aliases
            tbl_nodes = list(statement.find_all(exp.Table))
            if not tbl_nodes:
                is_schema_compatible = False
                errors.append("Query does not reference any database tables.")
            else:
                referenced_tables: list[str] = []
                alias_map: dict[str, str] = {}

                for tbl in tbl_nodes:
                    tbl_name = tbl.name.lower().strip()
                    referenced_tables.append(tbl_name)
                    if tbl.alias:
                        alias_map[tbl.alias.lower().strip()] = tbl_name
                    # Table name itself acts as a qualifier
                    alias_map[tbl_name] = tbl_name

                    if tbl_name not in norm_schema:
                        is_schema_compatible = False
                        errors.append(f"Table '{tbl.name}' does not exist in the database schema.")

                ast_details["tables"] = sorted(list(set(referenced_tables)))
                ast_details["aliases"] = alias_map

                # Collect projection aliases (e.g., SELECT col AS col_alias)
                projection_aliases: set[str] = set()
                for sel in statement.selects:
                    if isinstance(sel, exp.Alias):
                        projection_aliases.add(sel.alias.lower().strip())

                # Validate columns
                col_nodes = list(statement.find_all(exp.Column))
                extracted_col_names: list[str] = []

                for col in col_nodes:
                    c_table = col.table.lower().strip() if col.table and col.table.strip() else None
                    c_name = col.name.lower().strip()
                    if c_name != "*":
                        extracted_col_names.append(c_name)

                    # Wildcard with qualifier (e.g. s.* or students.*)
                    if c_name == "*":
                        if c_table:
                            if c_table not in alias_map:
                                is_schema_compatible = False
                                errors.append(
                                    f"Table or alias '{col.table}' in '{col.sql()}' "
                                    "is not defined in the query FROM clause."
                                )
                        continue

                    # Case A: Column is qualified with a table or alias (e.g. students.name or s.name)
                    if c_table is not None:
                        if c_table not in alias_map:
                            is_schema_compatible = False
                            errors.append(
                                f"Table or alias '{col.table}' in '{col.sql()}' "
                                "is not defined in the query FROM clause."
                            )
                        else:
                            target_tbl = alias_map[c_table]
                            if target_tbl in norm_schema:
                                if c_name not in norm_schema[target_tbl]:
                                    is_schema_compatible = False
                                    errors.append(
                                        f"Column '{col.name}' does not exist in table '{target_tbl}'."
                                    )
                    # Case B: Column is unqualified (e.g. name, age)
                    else:
                        # Check if column is a recognized projection alias in ORDER BY / expressions
                        if c_name in projection_aliases:
                            continue

                        # Find matching tables among referenced tables
                        matching_tables = [
                            t for t in set(referenced_tables)
                            if t in norm_schema and c_name in norm_schema[t]
                        ]

                        if len(matching_tables) == 0:
                            is_schema_compatible = False
                            ref_str = ", ".join(f"'{t}'" for t in sorted(list(set(referenced_tables))))
                            errors.append(
                                f"Column '{col.name}' does not exist in referenced table(s): {ref_str}."
                            )
                        elif len(matching_tables) > 1:
                            is_schema_compatible = False
                            match_str = ", ".join(f"'{t}'" for t in sorted(matching_tables))
                            errors.append(
                                f"Ambiguous column reference: '{col.name}' exists in multiple "
                                f"referenced tables ({match_str})."
                            )

                ast_details["columns"] = sorted(list(set(extracted_col_names)))

        # -------------------------------------------------------------
        # Stage 6: Normalization and Final Assembly
        # -------------------------------------------------------------
        normalized_sql: Optional[str] = None
        try:
            normalized_sql = statement.sql(dialect="mysql", comments=False)
        except Exception:
            normalized_sql = stripped_sql

        is_valid = (
            is_syntax_valid
            and is_permitted_read_only
            and is_schema_compatible
            and (is_in_supported_scope if should_enforce_scope else True)
            and (is_in_model_scope if should_enforce_model_scope else True)
            and len(errors) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            is_syntax_valid=is_syntax_valid,
            is_permitted_read_only=is_permitted_read_only,
            is_schema_compatible=is_schema_compatible,
            is_in_supported_scope=is_in_supported_scope,
            is_in_model_scope=is_in_model_scope,
            normalized_sql=normalized_sql if is_valid or is_syntax_valid else None,
            raw_sql=raw_sql,
            errors=errors,
            warnings=warnings,
            ast_details=ast_details,
        )

    @staticmethod
    def _get_function_names(func_node: exp.Expression) -> set[str]:
        """Extracts normalized names, keys, and aliases for a function node."""
        names: set[str] = set()
        if isinstance(func_node, exp.Anonymous):
            names.add(func_node.name.lower().strip())
        if hasattr(func_node, "key") and func_node.key:
            names.add(str(func_node.key).lower().strip())
        sql_name = getattr(func_node, "sql_name", None)
        if callable(sql_name):
            try:
                names.add(str(sql_name()).lower().strip())
            except Exception:
                pass
        name = getattr(func_node, "name", None)
        if name and isinstance(name, str):
            names.add(name.lower().strip())
        names.add(type(func_node).__name__.lower().strip())
        return names

    @staticmethod
    def _sanitize_parser_error(error_msg: str) -> str:
        """Strips stack traces, internal paths, and sensitive tokens from parser errors."""
        # Remove absolute file paths if any appear in error strings
        cleaned = re.sub(r"[A-Za-z]:\\[\w\s\\.-]+", "[path]", error_msg)
        cleaned = re.sub(r"/[\w\s/.-]+", "[path]", cleaned)
        # Keep only the essential message (first 1-2 lines)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return " ".join(lines[:2]) if lines else "Invalid SQL syntax"


def validate_sql(
    sql: Any,
    schema: Optional[Union[DatabaseSchemaMetadata, DatabaseSchemaInspector, dict[str, Any], str]] = None,
    enforce_scope: bool = True,
    enforce_model_scope: bool = True,
) -> ValidationResult:
    """
    Functional convenience entry point for SQL validation.

    Args:
        sql: SQL query string to validate.
        schema: Optional schema metadata (metadata object, dict, or canonical string).
        enforce_scope: Whether to require strict adherence to general supported read-only scope.
        enforce_model_scope: Whether to strictly enforce the trained Seq2Seq model vocabulary boundary.
            Defaults to True to guarantee that queries stay within the model's 70-token generative capability.
            Explicit opt-out (False) is reserved for future manual SQL testing and must NOT be used
            for model-generated SQL.

    Returns:
        ValidationResult object.
    """
    validator = SQLValidator(schema=schema, enforce_scope=enforce_scope, enforce_model_scope=enforce_model_scope)
    return validator.validate(sql)
