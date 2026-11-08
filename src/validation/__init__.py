"""
SQL Validation and Read-Only Safety Layer (Milestone 4, Task 4.3).

Provides AST-based SQL validation, schema compatibility checks,
read-only permission verification, and model capability boundary enforcement.
"""

from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator, normalize_schema, validate_sql

__all__ = [
    "ValidationResult",
    "SQLValidator",
    "validate_sql",
    "normalize_schema",
]
