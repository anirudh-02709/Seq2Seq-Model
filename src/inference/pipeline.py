"""
Standalone Inference Pipeline for NL2SQL Attention-Based Seq2Seq Model (Milestone 4, Task 4.1).

This module provides a clean, production-grade standalone inference interface
for the trained Seq2Seq model. It loads the model, configuration, and vocabulary
artifacts once from the saved checkpoint, accepts new natural-language questions
and database schema representations, and generates SQL queries without gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover
    HAS_SQLGLOT = False

from src.data.tokenizer import (
    EOS_IDX,
    PAD_IDX,
    UNK_IDX,
    NL2SQLTokenizer,
    combine_schema_and_question,
    detokenize_sql,
)
from src.data.vocabulary import Vocabulary
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer

# Standard student database schema string matching the training distribution
DEFAULT_STUDENT_SCHEMA = (
    "students(student_id, name, age, department, cgpa) | "
    "courses(course_id, course_name, credits) | "
    "enrollments(enrollment_id, student_id, course_id, semester, marks)"
)


@dataclass
class InferenceResult:
    """
    Structured result returned by detailed model inference.
    """
    sql: str
    tokens: list[str]
    source_tokens: list[str]
    source_unk_count: int
    target_unk_count: int
    is_valid_sql: Optional[bool]
    syntax_error: Optional[str]
    generated_token_ids: list[int]
    question: str
    schema: str


class NL2SQLInferencePipeline:
    """
    Standalone inference pipeline for the trained Seq2Seq NL2SQL model.

    Loads the trained model checkpoint once, preprocesses natural language
    questions and schema representations, and generates SQL queries.

    Example usage:
        >>> pipeline = NL2SQLInferencePipeline.from_checkpoint("checkpoints/best_model.pt")
        >>> sql = pipeline.generate_sql(
        ...     question="Show students with CGPA above 8",
        ...     schema="students(id, name, department, cgpa)"
        ... )
        >>> print(sql)
    """

    DEFAULT_SCHEMA = DEFAULT_STUDENT_SCHEMA

    def __init__(
        self,
        model: Seq2Seq,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        tokenizer: Optional[NL2SQLTokenizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        default_schema: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        checkpoint_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the inference pipeline with restored components.

        Args:
            model: Restored Seq2Seq model instance.
            src_vocab: Restored source vocabulary (198 tokens).
            tgt_vocab: Restored target vocabulary (70 tokens).
            tokenizer: Optional NL2SQLTokenizer instance.
            device: Optional torch.device or device string ('cpu', 'cuda').
            default_schema: Default database schema to use when not specified in query.
            config: Optional model configuration dictionary.
            checkpoint_info: Optional metadata from the checkpoint.
        """
        if not isinstance(model, Seq2Seq):
            raise TypeError(f"Expected model to be an instance of Seq2Seq, got {type(model)}.")
        if not isinstance(src_vocab, Vocabulary):
            raise TypeError(f"Expected src_vocab to be an instance of Vocabulary, got {type(src_vocab)}.")
        if not isinstance(tgt_vocab, Vocabulary):
            raise TypeError(f"Expected tgt_vocab to be an instance of Vocabulary, got {type(tgt_vocab)}.")

        # Resolve execution device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()  # Strictly enforce evaluation mode

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer if tokenizer is not None else NL2SQLTokenizer()

        self.default_schema = (
            default_schema if default_schema is not None else self.DEFAULT_SCHEMA
        )
        self.config = config or {}
        self.checkpoint_info = checkpoint_info or {}

        # Validate that model dimensions match vocabulary sizes
        if model.source_vocab_size != len(src_vocab):
            raise ValueError(
                f"Model source_vocab_size ({model.source_vocab_size}) does not match "
                f"src_vocab size ({len(src_vocab)})."
            )
        if model.target_vocab_size != len(tgt_vocab):
            raise ValueError(
                f"Model target_vocab_size ({model.target_vocab_size}) does not match "
                f"tgt_vocab size ({len(tgt_vocab)})."
            )

        # Attach vocabulary references to model for direct access
        self.model.src_vocab = self.src_vocab
        self.model.tgt_vocab = self.tgt_vocab
        self.model.tokenizer = self.tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
        device: Optional[Union[torch.device, str]] = None,
        default_schema: Optional[str] = None,
    ) -> NL2SQLInferencePipeline:
        """
        Loads the trained model and all required artifacts from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Optional target device ('cpu', 'cuda', etc.).
            default_schema: Optional default database schema string.

        Returns:
            Configured and ready-to-use NL2SQLInferencePipeline instance.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

        # Delegate checkpoint loading to Trainer's existing restoration method
        model, chk_data, src_vocab, tgt_vocab = Trainer.load_model_from_checkpoint(
            checkpoint_path=str(path),
            device=device,
        )

        # Verify that vocabulary artifacts were stored in checkpoint
        missing_artifacts: list[str] = []
        if src_vocab is None:
            missing_artifacts.append("src_vocab")
        if tgt_vocab is None:
            missing_artifacts.append("tgt_vocab")

        if missing_artifacts:
            raise ValueError(
                f"Checkpoint at '{path.resolve()}' is missing required vocabulary artifacts: "
                f"{', '.join(missing_artifacts)}. Standalone inference cannot proceed without them."
            )

        checkpoint_info = {
            "epoch": chk_data.get("epoch"),
            "val_loss": chk_data.get("val_loss"),
            "val_token_acc": chk_data.get("val_token_acc"),
            "train_loss": chk_data.get("train_loss"),
            "is_best": chk_data.get("is_best"),
            "checkpoint_path": str(path.resolve()),
        }

        return cls(
            model=model,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            default_schema=default_schema,
            config=chk_data.get("config", {}),
            checkpoint_info=checkpoint_info,
        )

    def _prepare_inputs(
        self,
        question: str,
        schema: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, list[str], str, str]:
        """
        Validates, combines, tokenizes, and formats question and schema inputs.

        Args:
            question: Natural language question string.
            schema: Database schema string (or None to use default_schema).

        Returns:
            Tuple of (src_tensor, src_lengths, src_tokens, resolved_schema, cleaned_question).
        """
        # Validate question
        if not isinstance(question, str):
            raise TypeError(f"Question must be a string, got {type(question).__name__}.")
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty or contain only whitespace.")

        # Resolve and validate schema
        if schema is None:
            resolved_schema = self.default_schema
        else:
            if not isinstance(schema, str):
                raise TypeError(f"Schema must be a string, got {type(schema).__name__}.")
            resolved_schema = schema.strip()
            if not resolved_schema:
                raise ValueError("Database schema cannot be empty or contain only whitespace.")

        # Tokenize encoder record: schema + <SEP> + question
        src_tokens = self.tokenizer.tokenize_encoder_record(resolved_schema, cleaned_question)
        if not src_tokens:
            raise ValueError("Tokenization produced empty sequence for input question and schema.")

        # Convert tokens to indices using the source vocabulary
        src_ids = self.src_vocab.to_indices(src_tokens, add_sos=False, add_eos=False)

        # Build PyTorch tensors of shape (1, seq_len) and (1,)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=self.device)

        return src_tensor, src_lengths, src_tokens, resolved_schema, cleaned_question

    def generate_sql(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
    ) -> str:
        """
        Generates an SQL query string from a natural language question and database schema.

        This is the primary standalone inference method.

        Args:
            question: Natural language query (e.g., 'Show students with CGPA above 8').
            schema: Database schema definition (e.g., 'students(id, name, department, cgpa)').
                    If None, uses the default student database schema.
            max_length: Maximum number of tokens to generate (default: 60).

        Returns:
            Generated SQL query string (e.g., 'SELECT name, cgpa FROM students WHERE cgpa > 8.0').
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}.")

        src_tensor, src_lengths, _, _, _ = self._prepare_inputs(question, schema)

        # Autoregressive greedy generation with gradients disabled
        self.model.eval()
        with torch.no_grad():
            gen_ids = self.model.generate(
                src=src_tensor,
                src_lengths=src_lengths,
                max_length=max_length,
            )[0].tolist()

        # Filter out EOS and any trailing PAD tokens
        filtered_ids: list[int] = []
        for tid in gen_ids:
            if tid == EOS_IDX:
                break
            if tid != PAD_IDX:
                filtered_ids.append(tid)

        # Convert token IDs to tokens and detokenize to SQL string
        pred_tokens = self.tgt_vocab.to_tokens(filtered_ids)
        pred_sql = self.tokenizer.detokenize_sql(pred_tokens).strip()

        return pred_sql

    def predict(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
    ) -> InferenceResult:
        """
        Generates SQL and returns detailed diagnostic metadata.

        Args:
            question: Natural language query.
            schema: Database schema definition (or None for default).
            max_length: Maximum generation length.

        Returns:
            InferenceResult dataclass containing SQL, tokens, UNK counts, and syntax validity.
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}.")

        src_tensor, src_lengths, src_tokens, resolved_schema, cleaned_question = (
            self._prepare_inputs(question, schema)
        )

        src_unk_count = sum(1 for tid in src_tensor[0].tolist() if tid == UNK_IDX)

        self.model.eval()
        with torch.no_grad():
            gen_ids = self.model.generate(
                src=src_tensor,
                src_lengths=src_lengths,
                max_length=max_length,
            )[0].tolist()

        filtered_ids: list[int] = []
        for tid in gen_ids:
            if tid == EOS_IDX:
                break
            if tid != PAD_IDX:
                filtered_ids.append(tid)

        tgt_unk_count = sum(1 for tid in filtered_ids if tid == UNK_IDX)
        pred_tokens = self.tgt_vocab.to_tokens(filtered_ids)
        pred_sql = self.tokenizer.detokenize_sql(pred_tokens).strip()

        # Syntax validity checking via sqlglot if installed
        is_valid_sql: Optional[bool] = None
        syntax_error: Optional[str] = None
        if HAS_SQLGLOT:
            if not pred_sql:
                is_valid_sql = False
                syntax_error = "Generated SQL string is empty."
            else:
                try:
                    parsed = sqlglot.parse_one(pred_sql, read="mysql")
                    if not isinstance(parsed, exp.Select):
                        is_valid_sql = False
                        syntax_error = f"Expression parsed as {type(parsed).__name__}, not Select."
                    else:
                        is_valid_sql = True
                except Exception as err:
                    is_valid_sql = False
                    syntax_error = str(err)

        return InferenceResult(
            sql=pred_sql,
            tokens=pred_tokens,
            source_tokens=src_tokens,
            source_unk_count=src_unk_count,
            target_unk_count=tgt_unk_count,
            is_valid_sql=is_valid_sql,
            syntax_error=syntax_error,
            generated_token_ids=filtered_ids,
            question=cleaned_question,
            schema=resolved_schema,
        )

    def batch_generate_sql(
        self,
        items: list[dict[str, str]],
        max_length: int = 60,
    ) -> list[str]:
        """
        Processes a list of items sequentially using the single loaded model.

        Args:
            items: List of dicts, each with 'question' and optional 'schema'.
            max_length: Maximum tokens per generation.

        Returns:
            List of generated SQL query strings.
        """
        results: list[str] = []
        for item in items:
            q = item["question"]
            s = item.get("schema")
            sql = self.generate_sql(question=q, schema=s, max_length=max_length)
            results.append(sql)
        return results

    def __call__(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
    ) -> str:
        """Convenience callable: pipeline(question, schema) -> sql."""
        return self.generate_sql(question=question, schema=schema, max_length=max_length)


# Convenient Aliases
NL2SQLPredictor = NL2SQLInferencePipeline
Seq2SeqInference = NL2SQLInferencePipeline


def load_inference_pipeline(
    checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
    device: Optional[Union[torch.device, str]] = None,
    default_schema: Optional[str] = None,
) -> NL2SQLInferencePipeline:
    """
    Convenience function to restore and return an NL2SQLInferencePipeline.
    """
    return NL2SQLInferencePipeline.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        default_schema=default_schema,
    )
