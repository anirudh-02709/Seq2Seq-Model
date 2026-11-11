"""
SQL Validation Result Data Structures (Milestone 4, Task 4.3).

Encapsulates the structured results of SQL validation, including syntax
correctness, read-only safety permissions, schema compatibility, and model scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ValidationResult:
    """
    Structured outcome of SQL validation against syntax, safety, scope, and schema rules.

    Attributes:
        is_valid: True if all required validation criteria passed.
        is_syntax_valid: True if the input is parseable as standard MySQL SQL.
        is_permitted_read_only: True if the statement is strictly a read-only single SELECT.
        is_schema_compatible: True if all referenced tables/columns exist and match schema metadata.
        is_in_supported_scope: True if all clauses and operations fall within general supported read-only scope.
        is_in_model_scope: True if all tokens/clauses strictly fall within the trained Seq2Seq model's vocabulary and capability.
        normalized_sql: Canonical MySQL-formatted SQL string, or None if validation/parsing failed.
        raw_sql: Original query string before normalization.
        errors: List of error descriptions detailing why validation failed.
        warnings: List of non-fatal warnings or informational notices.
        ast_details: Detailed structural information extracted from the parsed SQL AST.
    """

    is_valid: bool
    is_syntax_valid: bool
    is_permitted_read_only: bool
    is_schema_compatible: bool
    is_in_supported_scope: bool
    is_in_model_scope: bool = True
    normalized_sql: Optional[str] = None
    raw_sql: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ast_details: dict[str, Any] = field(default_factory=dict)

    @property
    def error_message(self) -> Optional[str]:
        """Returns the primary (first) validation error, or None if validation passed."""
        return self.errors[0] if self.errors else None

    def to_dict(self) -> dict[str, Any]:
        """Converts the validation result to a dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "is_syntax_valid": self.is_syntax_valid,
            "is_permitted_read_only": self.is_permitted_read_only,
            "is_schema_compatible": self.is_schema_compatible,
            "is_in_supported_scope": self.is_in_supported_scope,
            "is_in_model_scope": self.is_in_model_scope,
            "normalized_sql": self.normalized_sql,
            "raw_sql": self.raw_sql,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "ast_details": dict(self.ast_details),
            "error_message": self.error_message,
        }
