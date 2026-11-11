"""
Pydantic Request and Response Schemas for NL2SQL FastAPI Backend (Milestone 5.2).

Defines strongly-typed data contracts for all API endpoints:
- GET  /api/health
- GET  /api/schema
- POST /api/generate
- POST /api/execute
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# Health Check Models
# ==============================================================================

class ModelHealthInfo(BaseModel):
    """Diagnostic health information for the Seq2Seq model."""
    model_config = ConfigDict(protected_namespaces=())

    loaded: bool = Field(..., description="Whether the Seq2Seq model is loaded into memory.")
    checkpoint: Optional[str] = Field(None, description="Sanitized checkpoint reference.")
    source_vocab_size: Optional[int] = Field(None, description="Number of tokens in source vocabulary.")
    target_vocab_size: Optional[int] = Field(None, description="Number of tokens in target vocabulary.")
    device: Optional[str] = Field(None, description="Execution device (e.g. 'cpu', 'cuda').")


class DatabaseHealthInfo(BaseModel):
    """Diagnostic health information for MySQL database connectivity."""
    model_config = ConfigDict(protected_namespaces=())

    connected: bool = Field(..., description="Whether the database connection is currently alive.")
    database: Optional[str] = Field(None, description="Name of the target database.")
    host: Optional[str] = Field(None, description="Database server hostname or IP.")
    port: Optional[int] = Field(None, description="Database server port.")
    exec_user: Optional[str] = Field(None, description="Dedicated read-only execution account.")
    mysql_version: Optional[str] = Field(None, description="MySQL server engine version.")
    error: Optional[str] = Field(None, description="Sanitized error description if connection failed.")


class HealthResponse(BaseModel):
    """Response payload for GET /api/health."""
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Overall service status ('healthy', 'degraded', 'unhealthy').")
    model: ModelHealthInfo = Field(..., description="Model initialization status.")
    database: DatabaseHealthInfo = Field(..., description="Database connectivity status.")


# ==============================================================================
# Schema Inspection Models
# ==============================================================================

class ColumnInfo(BaseModel):
    """Column metadata for a database table."""
    name: str = Field(..., description="Column name.")
    data_type: str = Field(..., description="Base SQL data type (e.g. int, varchar).")
    column_type: str = Field(..., description="Full column specification (e.g. varchar(100)).")
    is_nullable: bool = Field(..., description="Whether NULL values are permitted.")
    is_primary_key: bool = Field(False, description="Whether the column is part of the primary key.")
    column_default: Optional[str] = Field(None, description="Default value expression, if any.")
    extra: Optional[str] = Field(None, description="Additional column metadata (e.g. auto_increment).")


class ForeignKeyInfo(BaseModel):
    """Foreign key relationship metadata."""
    constraint_name: str = Field(..., description="Foreign key constraint name.")
    column_name: str = Field(..., description="Local column name.")
    referenced_table: str = Field(..., description="Target referenced table.")
    referenced_column: str = Field(..., description="Target referenced column.")
    update_rule: Optional[str] = Field(None, description="ON UPDATE action rule.")
    delete_rule: Optional[str] = Field(None, description="ON DELETE action rule.")


class TableInfo(BaseModel):
    """Metadata describing a single database table."""
    table_name: str = Field(..., description="Table name.")
    columns: list[ColumnInfo] = Field(default_factory=list, description="Ordered list of column definitions.")
    primary_keys: list[str] = Field(default_factory=list, description="Primary key column names.")
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list, description="Foreign key constraints.")


class SchemaResponse(BaseModel):
    """Response payload for GET /api/schema."""
    database_name: str = Field(..., description="Name of the inspected database.")
    canonical_schema: str = Field(..., description="Canonical schema string for model inference.")
    table_count: int = Field(..., description="Total number of base tables.")
    foreign_key_count: int = Field(0, description="Total number of foreign key constraints.")
    tables: dict[str, TableInfo] = Field(default_factory=dict, description="Dictionary mapping table name to metadata.")


# ==============================================================================
# SQL Generation Models
# ==============================================================================

class GenerateRequest(BaseModel):
    """Request payload for POST /api/generate."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question to convert into SQL.",
        examples=["Find all students with CGPA above 8.5"],
    )
    schema_override: Optional[str] = Field(
        None,
        alias="schema",
        description="Optional schema signature override. Defaults to configured student DB schema.",
    )
    max_length: int = Field(
        60,
        ge=1,
        le=200,
        description="Maximum number of tokens to generate.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return cleaned


class ValidationSummary(BaseModel):
    """Structured breakdown of SQL validation checks."""
    is_valid: bool = Field(..., description="True if query passes all syntax, safety, scope, and schema checks.")
    is_syntax_valid: bool = Field(..., description="True if query parses as valid MySQL SQL.")
    is_permitted_read_only: bool = Field(..., description="True if query is strictly a single read-only SELECT.")
    is_schema_compatible: bool = Field(..., description="True if referenced tables and columns exist.")
    is_in_supported_scope: bool = Field(..., description="True if all clauses are within general read-only scope.")
    is_in_model_scope: bool = Field(True, description="True if query adheres strictly to the model's closed target vocabulary.")
    normalized_sql: Optional[str] = Field(None, description="MySQL-canonicalized SQL statement.")
    raw_sql: Optional[str] = Field(None, description="Unnormalized generated SQL statement.")
    errors: list[str] = Field(default_factory=list, description="Validation failure reasons.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warning notices.")
    ast_details: dict[str, Any] = Field(default_factory=dict, description="Structural AST inspection metadata.")


class InferenceDiagnostics(BaseModel):
    """Diagnostic metadata from Seq2Seq model inference."""
    tokens: list[str] = Field(default_factory=list, description="Generated target tokens.")
    source_tokens: list[str] = Field(default_factory=list, description="Input encoder tokens.")
    source_unk_count: int = Field(0, description="Number of unknown tokens (<UNK>) in input.")
    target_unk_count: int = Field(0, description="Number of unknown tokens (<UNK>) in output.")


class GenerateResponse(BaseModel):
    """Response payload for POST /api/generate."""
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    question: str = Field(..., description="Input natural language question.")
    schema_signature: str = Field(..., alias="schema", description="Database schema signature used for inference and validation.")
    generated_sql: Optional[str] = Field(None, description="Model-generated SQL query.")
    validation: ValidationSummary = Field(..., description="SQL validation and safety audit outcome.")
    inference_diagnostics: Optional[InferenceDiagnostics] = Field(None, description="Token-level model diagnostics.")
    stage_reached: str = Field(
        "execution_pending",
        description="Current pipeline stage ('execution_pending', 'validation', 'error').",
    )
    can_execute: bool = Field(
        ...,
        description="True if validation passed and query is eligible for execution approval.",
    )
    error: Optional[str] = Field(None, description="Error message if generation or validation failed.")


# ==============================================================================
# SQL Execution Models
# ==============================================================================

class ExecuteRequest(BaseModel):
    """Request payload for POST /api/execute."""
    sql: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="SQL query submitted for approved execution against MySQL.",
        examples=["SELECT * FROM students WHERE cgpa > 8.5"],
    )
    schema_override: Optional[str] = Field(
        None,
        alias="schema",
        description="Optional schema signature override for server-side revalidation.",
    )
    approved: bool = Field(
        True,
        description="Explicit user approval gate. Must be True to execute against MySQL.",
    )
    max_rows: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of rows to return (cursor-level truncation).",
    )

    @field_validator("sql")
    @classmethod
    def validate_sql_string(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("SQL statement cannot be empty or contain only whitespace.")
        return cleaned


class ExecuteResponse(BaseModel):
    """Response payload for POST /api/execute."""
    success: bool = Field(..., description="True if query was validated, approved, and executed successfully.")
    executed_sql: Optional[str] = Field(None, description="Exact canonical SQL executed on MySQL.")
    columns: list[str] = Field(default_factory=list, description="Result set column names.")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Result set records as JSON objects.")
    row_count: int = Field(0, description="Number of rows returned in this response.")
    execution_time_ms: float = Field(0.0, description="Server-side execution time in milliseconds.")
    is_truncated: bool = Field(False, description="True if matching rows exceeded max_rows limit.")
    warning: Optional[str] = Field(None, description="Warning notice (e.g. row truncation alert).")
    error: Optional[str] = Field(None, description="Error description if validation or execution failed.")
    validation: Optional[ValidationSummary] = Field(None, description="Server-side revalidation details.")


# ==============================================================================
# Generic Error Response Model
# ==============================================================================

class ErrorDetail(BaseModel):
    """Structured error detail payload."""
    detail: str = Field(..., description="High-level description of the error.")
    errors: list[str] = Field(default_factory=list, description="Specific error messages or diagnostics.")
