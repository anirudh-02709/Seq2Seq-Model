"""
Standalone Inference Pipeline for NL2SQL Attention-Based Seq2Seq Model (Milestone 4, Task 4.1).

This module provides a clean, production-grade standalone inference interface
for the trained Seq2Seq model. It loads the model, configuration, and vocabulary
artifacts once from the saved checkpoint, accepts new natural-language questions
and database schema representations, and generates SQL queries without gradients.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover
    HAS_SQLGLOT = False

from src.data.tokenizer import (
    DEFAULT_STUDENT_SCHEMA,
    EOS_IDX,
    PAD_IDX,
    UNK_IDX,
    NL2SQLTokenizer,
)
from src.data.vocabulary import Vocabulary
from src.inference.constrained_decoder import (
    ConstrainedDecodingResult,
    ConstrainedSchemaDecoder,
)
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer
from src.validation.validator import SQLValidator

VALID_DECODING_MODES: tuple[str, str] = ("schema_aware", "greedy")


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
    decoding_mode: str = "greedy"
    beam_width: int = 1
    is_schema_valid: Optional[bool] = None
    no_valid_candidate: bool = False
    fallback_used: bool = False
    validation_error: Optional[str] = None
    candidates_examined: int = 1
    candidates_rejected: int = 0
    decoding_latency_ms: Optional[float] = None


class NL2SQLInferencePipeline:
    """
    Standalone inference pipeline for the trained Seq2Seq NL2SQL model.

    Loads the trained model checkpoint once, preprocesses natural language
    questions and schema representations, and generates SQL queries.

    Example usage:
        >>> pipeline = NL2SQLInferencePipeline.from_checkpoint("checkpoints/candidates/phase2_clean_best.pt")
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
        decoding_mode: str = "schema_aware",
        beam_width: int = 5,
        length_penalty_alpha: float = 0.0,
    ) -> None:
        """
        Initializes the inference pipeline with restored components.

        Args:
            model: Restored Seq2Seq model instance.
            src_vocab: Restored source vocabulary.
            tgt_vocab: Restored target vocabulary.
            tokenizer: Optional NL2SQLTokenizer instance.
            device: Optional torch.device or device string ('cpu', 'cuda').
            default_schema: Default database schema to use when not specified in query.
            config: Optional model configuration dictionary.
            checkpoint_info: Optional metadata from the checkpoint.
            decoding_mode: Decoding strategy ('schema_aware' or 'greedy'). Defaults to 'schema_aware'.
            beam_width: Candidate beam width for 'schema_aware' decoding (default: 5).
            length_penalty_alpha: Optional beam length-normalization exponent alpha (default: 0.0).
        """
        if not isinstance(model, Seq2Seq):
            raise TypeError(f"Expected model to be an instance of Seq2Seq, got {type(model)}.")
        if not isinstance(src_vocab, Vocabulary):
            raise TypeError(f"Expected src_vocab to be an instance of Vocabulary, got {type(src_vocab)}.")
        if not isinstance(tgt_vocab, Vocabulary):
            raise TypeError(f"Expected tgt_vocab to be an instance of Vocabulary, got {type(tgt_vocab)}.")
        if decoding_mode not in VALID_DECODING_MODES:
            raise ValueError(
                f"decoding_mode must be one of {VALID_DECODING_MODES}, got {decoding_mode!r}."
            )
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
            raise ValueError(f"beam_width must be a positive integer, got {beam_width}.")
        if (
            isinstance(length_penalty_alpha, bool)
            or not isinstance(length_penalty_alpha, (int, float))
            or length_penalty_alpha < 0.0
        ):
            raise ValueError(f"length_penalty_alpha must be non-negative, got {length_penalty_alpha}.")

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
        self.decoding_mode = decoding_mode
        self.beam_width = beam_width
        self.length_penalty_alpha = float(length_penalty_alpha)

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

        self.constrained_decoder = ConstrainedSchemaDecoder(
            model=self.model,
            tgt_vocab=self.tgt_vocab,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.last_decoding_result: Optional[ConstrainedDecodingResult] = None
        self._validator_cache: dict[str, SQLValidator] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = "checkpoints/candidates/phase2_clean_best.pt",
        device: Optional[Union[torch.device, str]] = None,
        default_schema: Optional[str] = None,
        decoding_mode: str = "schema_aware",
        beam_width: int = 5,
        length_penalty_alpha: float = 0.0,
    ) -> NL2SQLInferencePipeline:
        """
        Loads the trained model and all required artifacts from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Optional target device ('cpu', 'cuda', etc.).
            default_schema: Optional default database schema string.
            decoding_mode: Decoding strategy ('schema_aware' or 'greedy').
            beam_width: Candidate beam width for 'schema_aware' decoding.

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
            decoding_mode=decoding_mode,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
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

    def _resolve_decoding_params(
        self,
        decoding_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
    ) -> Tuple[str, int]:
        mode = self.decoding_mode if decoding_mode is None else decoding_mode
        if mode not in VALID_DECODING_MODES:
            raise ValueError(f"decoding_mode must be one of {VALID_DECODING_MODES}, got {mode!r}.")
        width = self.beam_width if beam_width is None else beam_width
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError(f"beam_width must be a positive integer, got {width}.")
        return mode, width

    @staticmethod
    def _check_mysql_syntax(sql: str) -> Tuple[Optional[bool], Optional[str]]:
        if not HAS_SQLGLOT:
            return None, None
        if not sql:
            return False, "Generated SQL string is empty."
        try:
            parsed = sqlglot.parse_one(sql, read="mysql")
            if not isinstance(parsed, exp.Select):
                return False, f"Expression parsed as {type(parsed).__name__}, not Select."
            return True, None
        except Exception as err:
            return False, str(err)

    def generate_sql(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
        decoding_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
        length_penalty_alpha: Optional[float] = None,
    ) -> str:
        """
        Generates an SQL query string from a natural language question and database schema.

        Args:
            question: Natural language query (e.g., 'Show students with CGPA above 8').
            schema: Database schema definition (or None for default schema).
            max_length: Maximum number of tokens to generate (default: 60).
            decoding_mode: Optional override ('schema_aware' or 'greedy').
            beam_width: Optional override for schema-aware beam width.
            length_penalty_alpha: Optional override for beam length-normalization alpha.

        Returns:
            Generated SQL query string.
        """
        return self.predict(
            question=question,
            schema=schema,
            max_length=max_length,
            decoding_mode=decoding_mode,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
        ).sql

    def generate_greedy(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
    ) -> str:
        """Explicitly runs original greedy autoregressive decoding baseline."""
        return self.generate_sql(
            question=question,
            schema=schema,
            max_length=max_length,
            decoding_mode="greedy",
        )

    def predict(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
        decoding_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
        length_penalty_alpha: Optional[float] = None,
    ) -> InferenceResult:
        """
        Generates SQL and returns detailed diagnostic and schema-validation metadata.

        Args:
            question: Natural language query.
            schema: Database schema definition (or None for default).
            max_length: Maximum generation length.
            decoding_mode: Optional override ('schema_aware' or 'greedy').
            beam_width: Optional override for schema-aware beam width.
            length_penalty_alpha: Optional override for beam length-normalization alpha.

        Returns:
            InferenceResult dataclass containing SQL, tokens, UNK counts, syntax validity,
            and schema-aware candidate diagnostics.
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}.")

        active_alpha = self.length_penalty_alpha if length_penalty_alpha is None else float(length_penalty_alpha)
        if active_alpha < 0.0:
            raise ValueError(f"length_penalty_alpha must be non-negative, got {active_alpha}.")

        active_mode, active_beam = self._resolve_decoding_params(decoding_mode, beam_width)
        src_tensor, src_lengths, src_tokens, resolved_schema, cleaned_question = (
            self._prepare_inputs(question, schema)
        )
        src_unk_count = sum(1 for tid in src_tensor[0].tolist() if tid == UNK_IDX)

        if active_mode == "schema_aware":
            dec_res = self.constrained_decoder.decode(
                src_tensor=src_tensor,
                src_lengths=src_lengths,
                schema=resolved_schema,
                beam_width=active_beam,
                max_length=max_length,
                length_penalty_alpha=active_alpha,
            )
            self.last_decoding_result = dec_res
            pred_sql = dec_res.selected_sql
            pred_tokens = dec_res.selected_tokens
            filtered_ids = dec_res.selected_token_ids
            tgt_unk_count = sum(1 for tid in filtered_ids if tid == UNK_IDX)
            is_valid_sql, syntax_error = self._check_mysql_syntax(pred_sql)
            return InferenceResult(
                sql=pred_sql,
                tokens=pred_tokens,
                source_tokens=src_tokens,
                source_unk_count=src_unk_count,
                target_unk_count=tgt_unk_count,
                is_valid_sql=is_valid_sql,
                syntax_error=syntax_error if not is_valid_sql else None,
                generated_token_ids=filtered_ids,
                question=cleaned_question,
                schema=resolved_schema,
                decoding_mode="schema_aware",
                beam_width=active_beam,
                is_schema_valid=dec_res.is_schema_valid,
                no_valid_candidate=dec_res.no_valid_candidate,
                fallback_used=dec_res.fallback_used,
                validation_error=dec_res.validation_error,
                candidates_examined=dec_res.candidates_examined,
                candidates_rejected=dec_res.candidates_rejected,
                decoding_latency_ms=dec_res.latency_ms,
            )

        # Original greedy autoregressive baseline path
        self.last_decoding_result = None
        t0 = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            gen_ids = self.model.generate(
                src=src_tensor,
                src_lengths=src_lengths,
                max_length=max_length,
            )[0].tolist()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        filtered_ids: list[int] = []
        for tid in gen_ids:
            if tid == EOS_IDX:
                break
            if tid != PAD_IDX:
                filtered_ids.append(tid)

        tgt_unk_count = sum(1 for tid in filtered_ids if tid == UNK_IDX)
        pred_tokens = self.tgt_vocab.to_tokens(filtered_ids)
        pred_sql = self.tokenizer.detokenize_sql(pred_tokens).strip()
        is_valid_sql, syntax_error = self._check_mysql_syntax(pred_sql)

        if resolved_schema not in self._validator_cache:
            self._validator_cache[resolved_schema] = SQLValidator(
                schema=resolved_schema, enforce_model_scope=True
            )
        val_res = self._validator_cache[resolved_schema].validate(pred_sql)

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
            decoding_mode="greedy",
            beam_width=1,
            is_schema_valid=val_res.is_valid,
            no_valid_candidate=not val_res.is_valid,
            fallback_used=False,
            validation_error=val_res.error_message,
            candidates_examined=1,
            candidates_rejected=0 if val_res.is_valid else 1,
            decoding_latency_ms=latency_ms,
        )

    def batch_generate_sql(
        self,
        items: list[dict[str, str]],
        max_length: int = 60,
        decoding_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
    ) -> list[str]:
        """
        Processes a list of items sequentially using the single loaded model.

        Args:
            items: List of dicts, each with 'question' and optional 'schema'.
            max_length: Maximum tokens per generation.
            decoding_mode: Optional override ('schema_aware' or 'greedy').
            beam_width: Optional override for schema-aware beam width.

        Returns:
            List of generated SQL query strings.
        """
        results: list[str] = []
        for item in items:
            q = item["question"]
            s = item.get("schema")
            sql = self.generate_sql(
                question=q,
                schema=s,
                max_length=max_length,
                decoding_mode=decoding_mode,
                beam_width=beam_width,
            )
            results.append(sql)
        return results

    def __call__(
        self,
        question: str,
        schema: Optional[str] = None,
        max_length: int = 60,
        decoding_mode: Optional[str] = None,
        beam_width: Optional[int] = None,
    ) -> str:
        """Convenience callable: pipeline(question, schema) -> sql."""
        return self.generate_sql(
            question=question,
            schema=schema,
            max_length=max_length,
            decoding_mode=decoding_mode,
            beam_width=beam_width,
        )


# Convenient Aliases
NL2SQLPredictor = NL2SQLInferencePipeline
Seq2SeqInference = NL2SQLInferencePipeline


def load_inference_pipeline(
    checkpoint_path: Union[str, Path] = "checkpoints/candidates/phase2_clean_best.pt",
    device: Optional[Union[torch.device, str]] = None,
    default_schema: Optional[str] = None,
    decoding_mode: str = "schema_aware",
    beam_width: int = 5,
) -> NL2SQLInferencePipeline:
    """
    Convenience function to restore and return an NL2SQLInferencePipeline.
    """
    return NL2SQLInferencePipeline.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        default_schema=default_schema,
        decoding_mode=decoding_mode,
        beam_width=beam_width,
    )
