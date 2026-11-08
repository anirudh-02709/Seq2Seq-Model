"""
End-to-End NL2SQL Pipeline Orchestrator (Milestone 4, Task 4.4).

Connects:
1. Model inference (natural language question -> model-generated SQL).
2. SQL validation & safety inspection (AST validation, model-scope enforcement, schema compatibility).
3. Explicit execution approval gate (refuses execution without caller confirmation).
4. Isolated database execution & result retrieval (read-only transaction, timeout, max_rows limit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.database.config import DatabaseConfig, DatabaseConfigurationError
from src.database.executor import ExecutionResult, SQLExecutor
from src.inference.pipeline import InferenceResult, NL2SQLInferencePipeline
from src.validation.result import ValidationResult
from src.validation.validator import SQLValidator


def _inference_result_to_dict(res: Optional[InferenceResult]) -> Optional[dict[str, Any]]:
    """Converts an InferenceResult dataclass instance to a dictionary."""
    if res is None:
        return None
    return {
        "sql": res.sql,
        "tokens": list(res.tokens),
        "source_tokens": list(res.source_tokens),
        "source_unk_count": res.source_unk_count,
        "target_unk_count": res.target_unk_count,
        "is_valid_sql": res.is_valid_sql,
        "syntax_error": res.syntax_error,
        "generated_token_ids": list(res.generated_token_ids),
        "question": res.question,
        "schema": res.schema,
    }


@dataclass
class PipelineResult:
    """
    Structured outcome of the end-to-end NL2SQL pipeline.

    Attributes:
        question: Input natural language question.
        schema: Database schema representation used for inference and validation.
        generated_sql: Model-generated SQL query string.
        inference_result: Diagnostic details from the Seq2Seq inference stage.
        validation_result: Outcome of SQL syntax, read-only safety, and schema validation.
        execution_result: Outcome of MySQL database execution (None if unapproved or blocked).
        approved: True if execution approval was granted by caller.
        stage_reached: Pipeline stage reached ('inference', 'validation', 'execution_pending', 'executed', 'error').
        error: High-level error summary if any stage failed.
    """

    question: str
    schema: Optional[str] = None
    generated_sql: Optional[str] = None
    inference_result: Optional[InferenceResult] = None
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[ExecutionResult] = None
    approved: bool = False
    stage_reached: str = "init"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Converts the pipeline result to a JSON-serializable dictionary."""
        return {
            "question": self.question,
            "schema": self.schema,
            "generated_sql": self.generated_sql,
            "inference_result": _inference_result_to_dict(self.inference_result),
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "approved": self.approved,
            "stage_reached": self.stage_reached,
            "error": self.error,
        }


class NL2SQLPipeline:
    """
    End-to-End Orchestrator for the NL2SQL System.

    Coordinates model inference, SQL validation, caller approval, and database execution.
    Maintains clean architectural boundaries between layers and never bypasses validation.
    """

    def __init__(
        self,
        inference_pipeline: NL2SQLInferencePipeline,
        validator: Optional[SQLValidator] = None,
        executor: Optional[SQLExecutor] = None,
        default_schema: Optional[str] = None,
    ) -> None:
        """
        Initializes the pipeline with inference, validation, and execution components.

        Args:
            inference_pipeline: Restored NL2SQLInferencePipeline instance.
            validator: SQLValidator instance (created with default_schema if None).
            executor: SQLExecutor instance (optional; can be provided or lazy loaded).
            default_schema: Optional default schema string for inference and validation.
        """
        if not isinstance(inference_pipeline, NL2SQLInferencePipeline):
            raise TypeError(
                f"Expected inference_pipeline to be NL2SQLInferencePipeline, got {type(inference_pipeline)}."
            )

        self.inference_pipeline = inference_pipeline
        self.default_schema = (
            default_schema if default_schema is not None else inference_pipeline.default_schema
        )
        self.validator = (
            validator
            if validator is not None
            else SQLValidator(schema=self.default_schema, enforce_model_scope=True)
        )
        self.executor = executor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
        db_config: Optional[DatabaseConfig] = None,
        executor: Optional[SQLExecutor] = None,
        default_schema: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> NL2SQLPipeline:
        """
        Factory method to instantiate a complete NL2SQLPipeline from a trained checkpoint.

        Args:
            checkpoint_path: Path to PyTorch model checkpoint.
            db_config: Optional database configuration for execution layer.
            executor: Optional preconfigured SQLExecutor instance.
            default_schema: Optional default database schema string.
            device: Optional torch.device or device string.

        Returns:
            Configured NL2SQLPipeline instance.
        """
        inf_pipe = NL2SQLInferencePipeline.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            default_schema=default_schema,
        )

        resolved_schema = default_schema or inf_pipe.default_schema
        validator = SQLValidator(schema=resolved_schema, enforce_model_scope=True)

        exec_instance = executor
        if exec_instance is None and db_config is not None:
            exec_instance = SQLExecutor(config=db_config)

        return cls(
            inference_pipeline=inf_pipe,
            validator=validator,
            executor=exec_instance,
            default_schema=resolved_schema,
        )

    def generate_sql(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
    ) -> InferenceResult:
        """
        Runs model inference to generate an SQL query from a question.
        This stage NEVER performs validation or database execution.

        Args:
            question: Natural language query.
            schema: Database schema definition (or None for default).
            max_length: Maximum number of tokens to generate.

        Returns:
            InferenceResult dataclass.
        """
        target_schema = schema if schema is not None else self.default_schema
        return self.inference_pipeline.predict(
            question=question,
            schema=target_schema,
            max_length=max_length,
        )

    def validate_sql(
        self,
        sql: str,
        schema: Optional[Any] = None,
        enforce_model_scope: bool = True,
    ) -> ValidationResult:
        """
        Inspects SQL query using AST validation and schema compatibility checking.
        Strictly enforces model-scope boundaries by default.

        Args:
            sql: SQL query string to validate.
            schema: Database schema definition (or None for default).
            enforce_model_scope: Whether to enforce model vocabulary bounds (defaults to True).

        Returns:
            ValidationResult dataclass.
        """
        target_schema = schema if schema is not None else self.default_schema
        return self.validator.validate(
            sql=sql,
            schema=target_schema,
            enforce_model_scope=enforce_model_scope,
        )

    def execute_sql(
        self,
        validation_result: ValidationResult,
        approved: bool = False,
        max_rows: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Executes a validated query against MySQL with explicit approval.

        Args:
            validation_result: Outcome of prior validate_sql() call.
            approved: Explicit caller approval required to execute against database.
            max_rows: Optional maximum rows to return.

        Returns:
            ExecutionResult dataclass.
        """
        if self.executor is None:
            # Attempt to instantiate executor with dedicated execution config
            try:
                self.executor = SQLExecutor()
            except DatabaseConfigurationError as err:
                return ExecutionResult(
                    success=False,
                    executed_sql=validation_result.normalized_sql if validation_result else None,
                    error=f"Executor configuration error: {err}",
                )

        return self.executor.execute(
            validation_result=validation_result,
            approved=approved,
            max_rows=max_rows,
        )

    def process(
        self,
        question: str,
        schema: Optional[str] = None,
        approved: bool = False,
        max_rows: Optional[int] = None,
        max_length: int = 60,
    ) -> PipelineResult:
        """
        Executes the full NL2SQL pipeline from natural language to retrieved records.

        Steps:
        1. Model inference generates candidate SQL.
        2. SQLValidator inspects syntax, safety, scope, and schema.
        3. If validation fails, pipeline halts immediately.
        4. If approved=False, pipeline halts at execution_pending stage and returns the validated SQL.
        5. If approved=True, executes query against MySQL and returns records.

        Args:
            question: Natural language question.
            schema: Optional schema definition override.
            approved: Whether caller explicitly approves execution if validation passes.
            max_rows: Maximum records to retrieve.
            max_length: Maximum inference generation length.

        Returns:
            PipelineResult containing outcomes of all attempted stages.
        """
        target_schema = schema if schema is not None else self.default_schema

        # Stage 1: Model Inference
        try:
            inf_res = self.generate_sql(
                question=question,
                schema=target_schema,
                max_length=max_length,
            )
        except Exception as err:
            return PipelineResult(
                question=question,
                schema=target_schema,
                approved=approved,
                stage_reached="inference",
                error=f"Inference stage failed: {err}",
            )

        gen_sql = inf_res.sql

        # Stage 2: SQL Validation & Safety Audit
        val_res = self.validate_sql(
            sql=gen_sql,
            schema=target_schema,
            enforce_model_scope=True,
        )

        if not val_res.is_valid:
            return PipelineResult(
                question=question,
                schema=target_schema,
                generated_sql=gen_sql,
                inference_result=inf_res,
                validation_result=val_res,
                approved=approved,
                stage_reached="validation",
                error=f"Validation stage failed: {val_res.error_message}",
            )

        # Stage 3: Approval Check
        if not approved:
            return PipelineResult(
                question=question,
                schema=target_schema,
                generated_sql=gen_sql,
                inference_result=inf_res,
                validation_result=val_res,
                execution_result=None,
                approved=False,
                stage_reached="execution_pending",
                error=None,
            )

        # Stage 4: Database Execution
        exec_res = self.execute_sql(
            validation_result=val_res,
            approved=True,
            max_rows=max_rows,
        )

        stage = "executed"
        error_summary = None if exec_res.success else f"Execution stage failed: {exec_res.error}"

        return PipelineResult(
            question=question,
            schema=target_schema,
            generated_sql=gen_sql,
            inference_result=inf_res,
            validation_result=val_res,
            execution_result=exec_res,
            approved=True,
            stage_reached=stage,
            error=error_summary,
        )
