"""
FastAPI Route Handlers for NL2SQL Backend (Milestone 5.2).

Exposes:
- GET  /api/health   -> Check model status and MySQL connectivity
- GET  /api/schema   -> Retrieve inspected database schema metadata
- POST /api/generate -> Generate SQL and perform AST validation (does not execute)
- POST /api/execute  -> Re-validate submitted SQL and execute on MySQL with approval
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import (
    ColumnInfo,
    DatabaseHealthInfo,
    ExecuteRequest,
    ExecuteResponse,
    ForeignKeyInfo,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    InferenceDiagnostics,
    ModelHealthInfo,
    SchemaResponse,
    TableInfo,
    ValidationSummary,
)
from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.connection import DatabaseConnection, sanitize_db_error
from src.database.schema import DatabaseSchemaInspector
from src.pipeline.orchestrator import NL2SQLPipeline
from src.validation.result import ValidationResult

logger = logging.getLogger("src.api.routes")

router = APIRouter(prefix="/api", tags=["NL2SQL"])


def _to_validation_summary(val_res: ValidationResult) -> ValidationSummary:
    """Helper to convert a ValidationResult dataclass into a Pydantic ValidationSummary."""
    return ValidationSummary(
        is_valid=val_res.is_valid,
        is_syntax_valid=val_res.is_syntax_valid,
        is_permitted_read_only=val_res.is_permitted_read_only,
        is_schema_compatible=val_res.is_schema_compatible,
        is_in_supported_scope=val_res.is_in_supported_scope,
        is_in_model_scope=val_res.is_in_model_scope,
        normalized_sql=val_res.normalized_sql,
        raw_sql=val_res.raw_sql,
        errors=list(val_res.errors),
        warnings=list(val_res.warnings),
        ast_details=dict(val_res.ast_details),
    )


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHECKPOINT_LABEL = "checkpoints/candidates/phase2_clean_best.pt"


def _resolve_checkpoint_label(pipeline: NL2SQLPipeline) -> str:
    """Returns a sanitized repository-relative checkpoint path when available."""
    inf_pipe = getattr(pipeline, "inference_pipeline", None)
    info = getattr(inf_pipe, "checkpoint_info", None)
    raw_path = info.get("checkpoint_path") if isinstance(info, dict) else None
    if not isinstance(raw_path, str) or not raw_path:
        return _DEFAULT_CHECKPOINT_LABEL
    try:
        return Path(raw_path).resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(raw_path).name


def _get_pipeline(request: Request) -> NL2SQLPipeline:
    """Retrieves the pre-initialized NL2SQLPipeline from application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NL2SQL pipeline is not initialized or failed to load on startup.",
        )
    return pipeline


# ==============================================================================
# 1. Health & Diagnostics Endpoint
# ==============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns operational status of the Seq2Seq model and MySQL database connectivity.",
)
def get_health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    model_loaded = pipeline is not None

    model_info = ModelHealthInfo(
        loaded=model_loaded,
        checkpoint=_resolve_checkpoint_label(pipeline) if model_loaded else None,
        source_vocab_size=len(pipeline.inference_pipeline.src_vocab) if model_loaded else None,
        target_vocab_size=len(pipeline.inference_pipeline.tgt_vocab) if model_loaded else None,
        device=str(pipeline.inference_pipeline.device) if model_loaded else None,
    )

    # Safely verify database connectivity without leaking credentials
    db_connected = False
    db_name = None
    db_host = None
    db_port = None
    exec_user = None
    mysql_ver = None
    db_err = None

    try:
        # Check execution account configuration
        db_cfg = DatabaseConfig.from_exec_env()
        db_name = db_cfg.database
        db_host = db_cfg.host
        db_port = db_cfg.port
        exec_user = db_cfg.user

        conn = DatabaseConnection(config=db_cfg)
        status_info = conn.verify_connection()
        if status_info.get("connected"):
            db_connected = True
            mysql_ver = status_info.get("mysql_version")
        else:
            db_err = status_info.get("error")
    except DatabaseConfigurationError as err:
        db_err = f"Execution configuration error: {err}"
    except Exception as err:
        db_err = sanitize_db_error(err)

    db_info = DatabaseHealthInfo(
        connected=db_connected,
        database=db_name,
        host=db_host,
        port=db_port,
        exec_user=exec_user,
        mysql_version=mysql_ver,
        error=db_err,
    )

    if model_loaded and db_connected:
        overall_status = "healthy"
    elif model_loaded and not db_connected:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        model=model_info,
        database=db_info,
    )


# ==============================================================================
# 2. Schema Inspection Endpoint
# ==============================================================================

@router.get(
    "/schema",
    response_model=SchemaResponse,
    summary="Get Database Schema Metadata",
    description="Inspects and returns the live database schema via INFORMATION_SCHEMA.",
)
def get_schema(request: Request) -> SchemaResponse:
    try:
        inspector = DatabaseSchemaInspector()
        schema_meta = inspector.inspect_database()
    except Exception as err:
        clean_err = sanitize_db_error(err)
        logger.error(f"Failed to inspect database schema: {clean_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database schema inspection failed: {clean_err}",
        )

    tables_dict: dict[str, TableInfo] = {}
    for t_name, t_meta in schema_meta.tables.items():
        cols = [
            ColumnInfo(
                name=col.name,
                data_type=col.data_type,
                column_type=col.column_type,
                is_nullable=col.is_nullable,
                is_primary_key=col.is_primary_key,
                column_default=col.column_default,
                extra=col.extra,
            )
            for col in t_meta.columns
        ]
        fks = [
            ForeignKeyInfo(
                constraint_name=fk.constraint_name,
                column_name=fk.column_name,
                referenced_table=fk.referenced_table,
                referenced_column=fk.referenced_column,
                update_rule=fk.update_rule,
                delete_rule=fk.delete_rule,
            )
            for fk in t_meta.foreign_keys
        ]
        tables_dict[t_name] = TableInfo(
            table_name=t_name,
            columns=cols,
            primary_keys=list(t_meta.primary_keys),
            foreign_keys=fks,
        )

    return SchemaResponse(
        database_name=schema_meta.database_name,
        canonical_schema=schema_meta.format_schema_summary(),
        table_count=schema_meta.total_tables,
        foreign_key_count=schema_meta.total_foreign_keys,
        tables=tables_dict,
    )


# ==============================================================================
# 3. SQL Generation & AST Validation Endpoint
# ==============================================================================

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate and Validate SQL",
    description="Converts natural language question into SQL and performs AST validation. Does NOT execute.",
)
def generate_sql(request: Request, body: GenerateRequest) -> GenerateResponse:
    pipeline = _get_pipeline(request)
    target_schema = body.schema_override or pipeline.default_schema

    try:
        inf_res = pipeline.generate_sql(
            question=body.question,
            schema=target_schema,
            max_length=body.max_length,
        )
    except Exception as err:
        logger.error(f"Inference error for question '{body.question}': {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {err}",
        )

    val_res = pipeline.validate_sql(
        sql=inf_res.sql,
        schema=target_schema,
        enforce_model_scope=True,
    )

    validation_summary = _to_validation_summary(val_res)
    inf_diag = InferenceDiagnostics(
        tokens=list(inf_res.tokens),
        source_tokens=list(inf_res.source_tokens),
        source_unk_count=inf_res.source_unk_count,
        target_unk_count=inf_res.target_unk_count,
    )

    stage_reached = "execution_pending" if val_res.is_valid else "validation"
    error_msg = None if val_res.is_valid else val_res.error_message

    return GenerateResponse(
        question=inf_res.question,
        schema_signature=inf_res.schema,
        generated_sql=inf_res.sql,
        validation=validation_summary,
        inference_diagnostics=inf_diag,
        stage_reached=stage_reached,
        can_execute=val_res.is_valid,
        error=error_msg,
    )


# ==============================================================================
# 4. Approved SQL Execution Endpoint
# ==============================================================================

@router.post(
    "/execute",
    response_model=ExecuteResponse,
    summary="Execute Approved SQL Query",
    description="Re-validates submitted SQL on the server and executes it against MySQL with explicit approval.",
)
def execute_sql(request: Request, body: ExecuteRequest) -> ExecuteResponse:
    # 1. Enforce explicit caller approval gate
    if not body.approved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Execution approval required. Set 'approved: true' to execute query against database.",
        )

    pipeline = _get_pipeline(request)
    target_schema = body.schema_override or pipeline.default_schema

    # 2. Mandatory Server-Side Re-Validation (Never trust client-side validation)
    val_res = pipeline.validate_sql(
        sql=body.sql,
        schema=target_schema,
        enforce_model_scope=True,
    )

    validation_summary = _to_validation_summary(val_res)

    if not val_res.is_valid:
        error_detail = val_res.error_message or "Submitted SQL failed server-side validation."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Execution rejected: {error_detail}",
        )

    # 3. Safe database execution using pre-validated AST
    exec_res = pipeline.execute_sql(
        validation_result=val_res,
        approved=True,
        max_rows=body.max_rows,
    )

    return ExecuteResponse(
        success=exec_res.success,
        executed_sql=exec_res.executed_sql,
        columns=list(exec_res.columns),
        rows=list(exec_res.rows),
        row_count=exec_res.row_count,
        execution_time_ms=exec_res.execution_time_ms,
        is_truncated=exec_res.is_truncated,
        warning=exec_res.warning,
        error=exec_res.error,
        validation=validation_summary,
    )
