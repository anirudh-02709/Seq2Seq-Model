"""
Trainer module for NL2SQL Seq2Seq model (Phase 3, Step 3.6).

Features:
- Adam optimizer and CrossEntropyLoss with ignore_index for PAD tokens.
- Training and evaluation loops with device management (CUDA if available, else CPU).
- Gradient clipping to prevent gradient explosion.
- Strict evaluation with teacher-forced next-token prediction and PAD-excluded token accuracy.
- Best model checkpointing (only updates when validation loss improves).
- Epoch-wise history recording to training_history.json.
- Checkpoint saving and full model reconstruction from checkpoints.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tokenizer import DEFAULT_STUDENT_SCHEMA, EOS_IDX, PAD_IDX, NL2SQLTokenizer
from src.data.vocabulary import Vocabulary
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq


class Trainer:
    """
    Manages the training, validation, checkpointing, and metric logging
    for the attention-based Seq2Seq NL2SQL model.
    """

    def __init__(
        self,
        model: Seq2Seq,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[Union[torch.device, str]] = None,
        learning_rate: float = 1e-3,
        clip_norm: float = 1.0,
        teacher_forcing_ratio: float = 0.50,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_name: str = "best_model.pt",
        history_path: Union[str, Path] = "training_history.json",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[NL2SQLTokenizer] = None,
        schema: Optional[str] = None,
        train_eval_loader: Optional[DataLoader] = None,
        selection_metric: str = "val_exact_match",
        early_stopping_patience: Optional[int] = None,
        min_em_delta: float = 1e-6,
        min_loss_delta: float = 1e-4,
        eval_max_length: int = 60,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model: Seq2Seq model instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optional PyTorch optimizer (defaults to Adam).
            criterion: Optional loss function (defaults to CrossEntropyLoss with ignore_index).
            device: Target device (defaults to CUDA if available, else CPU).
            learning_rate: Learning rate for Adam optimizer.
            clip_norm: Maximum norm for gradient clipping.
            teacher_forcing_ratio: Teacher forcing probability for training (default: 0.50).
            checkpoint_dir: Directory where checkpoints are stored.
            checkpoint_name: Filename for the saved best checkpoint within checkpoint_dir.
            history_path: Filepath for saving JSON training history.
            src_vocab: Source Vocabulary object for serialization into checkpoints.
            tgt_vocab: Target Vocabulary object for serialization into checkpoints.
            tokenizer: Optional NL2SQLTokenizer for SQL detokenization during greedy evaluation.
            schema: Optional database schema definition for SQLValidator checks.
            train_eval_loader: Optional unshuffled training DataLoader for non-intrusive train EM/acc metrics.
            selection_metric: Primary checkpoint selection metric ('val_exact_match' or 'val_loss').
            early_stopping_patience: Epochs without improvement before stopping (None or 0 disables).
            min_em_delta: Minimum exact match increase required to count as primary improvement.
            min_loss_delta: Minimum validation loss decrease required for tie-breaking / val_loss selection.
            eval_max_length: Maximum decoding sequence length for greedy validation evaluation.
            seed: Optional random seed recorded in checkpoint metadata.
            batch_size: Optional batch size recorded in checkpoint metadata.
            config: Hyperparameter dictionary stored in checkpoints.
        """
        if not isinstance(model, Seq2Seq):
            raise TypeError(f"Expected model to be an instance of Seq2Seq, got {type(model)}.")

        if isinstance(teacher_forcing_ratio, bool) or not isinstance(teacher_forcing_ratio, (int, float)):
            raise TypeError(
                f"teacher_forcing_ratio must be a float or int, got {type(teacher_forcing_ratio).__name__}."
            )
        teacher_forcing_ratio = float(teacher_forcing_ratio)
        if not (0.0 <= teacher_forcing_ratio <= 1.0):
            raise ValueError(
                f"teacher_forcing_ratio must be between 0.0 and 1.0, got {teacher_forcing_ratio}."
            )

        if selection_metric not in {"val_exact_match", "val_loss"}:
            raise ValueError(
                f"selection_metric must be 'val_exact_match' or 'val_loss', got '{selection_metric}'."
            )

        if early_stopping_patience is not None and early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative or None, got {early_stopping_patience}."
            )

        # Device assignment
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = train_eval_loader

        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.pad_idx = model.pad_idx

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function with ignore_index for target PAD tokens
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name
        self.best_checkpoint_path = self.checkpoint_dir / self.checkpoint_name
        self.history_path = Path(history_path)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer if tokenizer is not None else (NL2SQLTokenizer() if tgt_vocab is not None else None)
        self.schema = schema
        self.selection_metric = selection_metric
        self.early_stopping_patience = early_stopping_patience
        self.min_em_delta = float(min_em_delta)
        self.min_loss_delta = float(min_loss_delta)
        self.eval_max_length = eval_max_length
        self.seed = seed
        self.batch_size = batch_size

        self._ast_comparator = None
        self._sql_validator = None

        # Populate config with model dimensions if not explicitly provided
        if config is None:
            self.config = {
                "source_vocab_size": model.source_vocab_size,
                "target_vocab_size": model.target_vocab_size,
                "embedding_dim": model.encoder.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "attention_dim": model.decoder.attention_dim,
                "dropout": getattr(model.encoder, "dropout_p", 0.0),
                "optimizer": type(self.optimizer).__name__,
                "learning_rate": self.learning_rate,
                "clip_norm": self.clip_norm,
                "teacher_forcing_ratio": self.teacher_forcing_ratio,
                "selection_metric": self.selection_metric,
                "early_stopping_patience": self.early_stopping_patience,
                "min_em_delta": self.min_em_delta,
                "min_loss_delta": self.min_loss_delta,
            }
            if self.seed is not None:
                self.config["seed"] = self.seed
            if self.batch_size is not None:
                self.config["batch_size"] = self.batch_size
        else:
            self.config = dict(config)
            # Ensure essential dimensions and training controls are always present
            self.config.setdefault("source_vocab_size", model.source_vocab_size)
            self.config.setdefault("target_vocab_size", model.target_vocab_size)
            self.config.setdefault("embedding_dim", model.encoder.embedding_dim)
            self.config.setdefault("hidden_dim", model.hidden_dim)
            self.config.setdefault("attention_dim", model.decoder.attention_dim)
            self.config.setdefault("optimizer", type(self.optimizer).__name__)
            self.config.setdefault("learning_rate", self.learning_rate)
            self.config.setdefault("clip_norm", self.clip_norm)
            self.config["teacher_forcing_ratio"] = self.teacher_forcing_ratio
            self.config.setdefault("selection_metric", self.selection_metric)
            self.config.setdefault("early_stopping_patience", self.early_stopping_patience)
            self.config.setdefault("min_em_delta", self.min_em_delta)
            self.config.setdefault("min_loss_delta", self.min_loss_delta)
            if self.seed is not None:
                self.config.setdefault("seed", self.seed)
            if self.batch_size is not None:
                self.config.setdefault("batch_size", self.batch_size)

        # Tracking state
        self.best_val_loss = float("inf")
        self.best_val_em = -1.0
        self.best_val_token_acc = 0.0
        self.best_epoch = 0
        self.best_metrics: dict[str, Any] = {}
        self.early_stopped = False
        self.stopped_epoch = 0
        self.completed_epochs = 0
        self.max_epochs = 0
        self.history: list[dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> float:
        """
        Runs one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            avg_train_loss: Average training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            src = batch["encoder_input_ids"].to(self.device)
            src_lengths = batch["encoder_lengths"].to(self.device)
            tgt = batch["decoder_target_ids"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through Seq2Seq
            # outputs: (batch_size, tgt_len - 1, target_vocab_size)
            outputs = self.model(
                src=src,
                src_lengths=src_lengths,
                tgt=tgt,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
            )

            # Targets are shifted by 1 (ignoring initial <SOS>)
            # targets: (batch_size, tgt_len - 1)
            targets = tgt[:, 1:]

            output_dim = outputs.size(-1)
            loss = self.criterion(
                outputs.reshape(-1, output_dim),
                targets.reshape(-1),
            )

            loss.backward()

            # Gradient clipping
            if self.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _get_ast_comparator(self):
        if self._ast_comparator is None:
            from src.evaluation.comparator import ASTComparator

            self._ast_comparator = ASTComparator(dialect="mysql")
        return self._ast_comparator

    def _get_sql_validator(self):
        if self._sql_validator is None:
            from src.validation.validator import SQLValidator

            resolved_schema = self.schema if self.schema is not None else DEFAULT_STUDENT_SCHEMA
            self._sql_validator = SQLValidator(schema=resolved_schema, enforce_model_scope=True)
        return self._sql_validator

    @staticmethod
    def _trim_sequence(token_ids: list[int], eos_idx: int, pad_idx: int) -> list[int]:
        trimmed: list[int] = []
        for tid in token_ids:
            if tid == eos_idx:
                break
            if tid != pad_idx:
                trimmed.append(int(tid))
        return trimmed

    def evaluate_tf(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates model on a DataLoader using teacher-forced next-token prediction.

        Returns:
            avg_loss: Average cross-entropy loss across batches.
            token_acc: Token-level prediction accuracy ignoring PAD tokens.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = len(loader)

        with torch.no_grad():
            for batch in loader:
                src = batch["encoder_input_ids"].to(self.device)
                src_lengths = batch["encoder_lengths"].to(self.device)
                tgt = batch["decoder_target_ids"].to(self.device)

                # Standard validation uses teacher forcing (ratio = 1.0) for cross-entropy comparability
                outputs = self.model(
                    src=src,
                    src_lengths=src_lengths,
                    tgt=tgt,
                    teacher_forcing_ratio=1.0,
                )

                targets = tgt[:, 1:]
                output_dim = outputs.size(-1)

                loss = self.criterion(
                    outputs.reshape(-1, output_dim),
                    targets.reshape(-1),
                )
                total_loss += loss.item()

                preds = outputs.argmax(dim=-1)
                valid_mask = targets != self.pad_idx
                correct = (preds == targets) & valid_mask

                total_correct += correct.sum().item()
                total_tokens += valid_mask.sum().item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        token_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return avg_loss, token_acc

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluates model on validation DataLoader.

        Returns:
            avg_val_loss: Average validation loss.
            val_token_acc: Token-level prediction accuracy ignoring PAD tokens.
        """
        return self.evaluate_tf(self.val_loader)

    def evaluate_greedy(self, loader: DataLoader) -> dict[str, Optional[float]]:
        """
        Evaluates autoregressive greedy decoding on a DataLoader.

        Computes normalized AST exact match (or trimmed token-sequence exact match when
        raw SQL strings are unavailable, such as in synthetic test DataLoaders), strict exact
        match, SQLGlot syntactic validity, and schema compatibility via SQLValidator.
        """
        self.model.eval()
        total_examples = 0
        exact_match_count = 0
        strict_match_count = 0
        valid_syntax_count = 0
        schema_compat_count = 0
        evaluated_sql = False

        eos_idx = getattr(self.model, "eos_idx", EOS_IDX)

        with torch.no_grad():
            for batch in loader:
                src = batch["encoder_input_ids"].to(self.device)
                src_lengths = batch["encoder_lengths"].to(self.device)
                tgt = batch["decoder_target_ids"].to(self.device)

                gen_batch = self.model.generate(
                    src=src,
                    src_lengths=src_lengths,
                    max_length=self.eval_max_length,
                ).tolist()

                has_sql_refs = "sqls" in batch and self.tgt_vocab is not None and self.tokenizer is not None
                if has_sql_refs:
                    evaluated_sql = True
                    ast_comp = self._get_ast_comparator()
                    sql_validator = self._get_sql_validator()
                    for gen_ids, ref_sql in zip(gen_batch, batch["sqls"]):
                        trimmed = self._trim_sequence(gen_ids, eos_idx, self.pad_idx)
                        pred_tokens = self.tgt_vocab.to_tokens(trimmed)
                        pred_sql = self.tokenizer.detokenize_sql(pred_tokens).strip()
                        ref_clean = ref_sql.strip()

                        ast_res = ast_comp.compare(ref_sql=ref_clean, pred_sql=pred_sql)
                        val_res = sql_validator.validate(pred_sql)

                        if ast_res.strict_exact_match:
                            strict_match_count += 1
                        if ast_res.normalized_exact_match:
                            exact_match_count += 1
                        if ast_res.is_valid_sql:
                            valid_syntax_count += 1
                        if val_res.is_valid:
                            schema_compat_count += 1
                        total_examples += 1
                else:
                    targets = tgt[:, 1:].tolist()
                    for gen_ids, ref_ids in zip(gen_batch, targets):
                        pred_trim = self._trim_sequence(gen_ids, eos_idx, self.pad_idx)
                        ref_trim = self._trim_sequence(ref_ids, eos_idx, self.pad_idx)
                        is_match = pred_trim == ref_trim
                        if is_match:
                            strict_match_count += 1
                            exact_match_count += 1
                        total_examples += 1

        if total_examples == 0:
            return {
                "exact_match": 0.0,
                "strict_exact_match": 0.0,
                "sql_validity_rate": 0.0 if evaluated_sql else None,
                "schema_compatibility_rate": 0.0 if evaluated_sql else None,
            }

        return {
            "exact_match": exact_match_count / total_examples,
            "strict_exact_match": strict_match_count / total_examples,
            "sql_validity_rate": (valid_syntax_count / total_examples) if evaluated_sql else None,
            "schema_compatibility_rate": (schema_compat_count / total_examples) if evaluated_sql else None,
        }

    def _is_checkpoint_improvement(self, val_em: float, val_loss: float, epoch: int) -> bool:
        """
        Determines whether current validation metrics improve over the best saved checkpoint.
        """
        if epoch == 1 or self.best_epoch == 0:
            return True
        if self.selection_metric == "val_loss":
            return val_loss < (self.best_val_loss - self.min_loss_delta)

        # Primary: validation exact match; Tie-breaker: lower validation loss
        if val_em > (self.best_val_em + self.min_em_delta):
            return True
        if abs(val_em - self.best_val_em) <= self.min_em_delta:
            return val_loss < (self.best_val_loss - self.min_loss_delta)
        return False

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_token_acc: float,
        is_best: bool = False,
        val_exact_match: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Saves model and training state to a checkpoint file.

        Args:
            filepath: Destination path for checkpoint.
            epoch: Completed epoch number.
            train_loss: Epoch training loss.
            val_loss: Validation loss.
            val_token_acc: Validation token accuracy.
            is_best: Whether this checkpoint is the current best.
            val_exact_match: Optional validation greedy exact match rate.
            extra_metadata: Optional additional metadata stored at checkpoint root.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_em": self.best_val_em,
            "val_loss": val_loss,
            "val_token_acc": val_token_acc,
            "val_exact_match": val_exact_match,
            "train_loss": train_loss,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
            "config": self.config,
            "is_best": is_best,
            "early_stopping": {
                "patience": self.early_stopping_patience,
                "min_em_delta": self.min_em_delta,
                "min_loss_delta": self.min_loss_delta,
                "selection_metric": self.selection_metric,
                "early_stopped": self.early_stopped,
                "stopped_epoch": self.stopped_epoch,
            },
            "max_epochs": self.max_epochs,
            "completed_epochs": self.completed_epochs,
            "training_history": list(self.history),
        }

        if self.seed is not None:
            checkpoint_data["seed"] = self.seed
        if self.batch_size is not None:
            checkpoint_data["batch_size"] = self.batch_size
        if extra_metadata:
            checkpoint_data.update(extra_metadata)

        # Include vocabulary dictionaries and sizes if available
        if self.src_vocab is not None:
            checkpoint_data["src_vocab"] = self.src_vocab.to_dict()
            checkpoint_data["src_vocab_size"] = len(self.src_vocab)
        if self.tgt_vocab is not None:
            checkpoint_data["tgt_vocab"] = self.tgt_vocab.to_dict()
            checkpoint_data["tgt_vocab_size"] = len(self.tgt_vocab)

        torch.save(checkpoint_data, path)

    def restore_best_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Restores model parameters from the best saved validation checkpoint.
        """
        target_path = Path(checkpoint_path) if checkpoint_path is not None else self.best_checkpoint_path
        if not target_path.is_file():
            return False
        chk_data = torch.load(target_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(chk_data["model_state_dict"])
        self.model.to(self.device)
        return True

    def fit(self, epochs: int, print_progress: bool = True) -> list[dict[str, Any]]:
        """
        Executes the training and validation schedule with checkpoint selection and optional early stopping.

        Args:
            epochs: Maximum number of training epochs to execute.
            print_progress: Whether to print progress each epoch.

        Returns:
            history: List of epoch metric dictionaries.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}.")

        self.best_val_loss = float("inf")
        self.best_val_em = -1.0
        self.best_val_token_acc = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        self.early_stopped = False
        self.stopped_epoch = 0
        self.completed_epochs = 0
        self.max_epochs = epochs
        self.history = []

        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_token_acc = self.evaluate()

            # Preserve RNG state across extra evaluation DataLoader passes so logging greedy
            # and training diagnostics never alters subsequent epoch sampling or TF coin flips.
            torch_rng_state = torch.get_rng_state()
            py_rng_state = random.getstate()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                val_greedy = self.evaluate_greedy(self.val_loader)
                val_em = float(val_greedy["exact_match"] or 0.0)

                train_token_acc: Optional[float] = None
                train_exact_match: Optional[float] = None
                if self.train_eval_loader is not None:
                    _, tr_acc = self.evaluate_tf(self.train_eval_loader)
                    tr_greedy = self.evaluate_greedy(self.train_eval_loader)
                    train_token_acc = round(tr_acc, 4)
                    train_exact_match = round(float(tr_greedy["exact_match"] or 0.0), 4)
            finally:
                torch.set_rng_state(torch_rng_state)
                random.setstate(py_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)

            current_lr = self.optimizer.param_groups[0]["lr"]

            is_best = self._is_checkpoint_improvement(val_em=val_em, val_loss=val_loss, epoch=epoch)
            epoch_metrics: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_token_acc": round(val_token_acc, 4),
                "val_exact_match": round(val_em, 4),
                "val_strict_exact_match": round(float(val_greedy["strict_exact_match"] or 0.0), 4),
                "val_sql_validity": (
                    round(float(val_greedy["sql_validity_rate"]), 4)
                    if val_greedy["sql_validity_rate"] is not None
                    else None
                ),
                "val_schema_compatibility": (
                    round(float(val_greedy["schema_compatibility_rate"]), 4)
                    if val_greedy["schema_compatibility_rate"] is not None
                    else None
                ),
                "train_token_acc": train_token_acc,
                "train_exact_match": train_exact_match,
                "teacher_forcing_ratio": self.teacher_forcing_ratio,
                "learning_rate": current_lr,
                "is_best": is_best,
            }
            self.history.append(epoch_metrics)

            marker = ""
            if is_best:
                self.best_val_em = val_em
                self.best_val_loss = val_loss
                self.best_val_token_acc = val_token_acc
                self.best_epoch = epoch
                self.best_metrics = dict(epoch_metrics)
                self.completed_epochs = epoch
                self.save_checkpoint(
                    self.best_checkpoint_path,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_token_acc=val_token_acc,
                    is_best=True,
                    val_exact_match=val_em,
                )
                no_improve_epochs = 0
                marker = " [BEST]"
            else:
                no_improve_epochs += 1

            # Save epoch-wise history JSON
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            if print_progress:
                schema_str = (
                    f" | Val Schema: {val_greedy['schema_compatibility_rate']:.4f}"
                    if val_greedy["schema_compatibility_rate"] is not None
                    else ""
                )
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_token_acc:.4f} | "
                    f"Val EM: {val_em:.4f}{schema_str} | "
                    f"LR: {current_lr:.6f}{marker}"
                )

            if (
                self.early_stopping_patience is not None
                and self.early_stopping_patience > 0
                and no_improve_epochs >= self.early_stopping_patience
            ):
                self.early_stopped = True
                self.stopped_epoch = epoch
                if print_progress:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={self.early_stopping_patience}, best epoch={self.best_epoch})."
                    )
                break

        self.completed_epochs = len(self.history)

        # Restore best checkpoint weights and finalize lifecycle metadata in best checkpoint
        if self.best_checkpoint_path.is_file():
            self.restore_best_checkpoint(self.best_checkpoint_path)
            chk_data = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
            chk_data["completed_epochs"] = self.completed_epochs
            chk_data["max_epochs"] = self.max_epochs
            chk_data["training_history"] = list(self.history)
            chk_data["early_stopping"] = {
                "patience": self.early_stopping_patience,
                "min_em_delta": self.min_em_delta,
                "min_loss_delta": self.min_loss_delta,
                "selection_metric": self.selection_metric,
                "early_stopped": self.early_stopped,
                "stopped_epoch": self.stopped_epoch,
            }
            torch.save(chk_data, self.best_checkpoint_path)

        return self.history

    @classmethod
    def load_model_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[Seq2Seq, dict[str, Any], Optional[Vocabulary], Optional[Vocabulary]]:
        """
        Reconstructs and restores a Seq2Seq model from a saved checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Optional target device.

        Returns:
            Tuple of (model, checkpoint_data, src_vocab, tgt_vocab).
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

        target_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint_data = torch.load(path, map_location=target_device, weights_only=False)
        config = checkpoint_data.get("config", {})
        state_dict = checkpoint_data["model_state_dict"]

        # Reconstruct vocabularies if present
        src_vocab = (
            Vocabulary.from_dict(checkpoint_data["src_vocab"])
            if "src_vocab" in checkpoint_data
            else None
        )
        tgt_vocab = (
            Vocabulary.from_dict(checkpoint_data["tgt_vocab"])
            if "tgt_vocab" in checkpoint_data
            else None
        )

        # Build architecture, inferring from state_dict if missing from config
        src_dim = config.get(
            "source_vocab_size",
            len(src_vocab) if src_vocab else state_dict["encoder.embedding.weight"].shape[0],
        )
        tgt_dim = config.get(
            "target_vocab_size",
            len(tgt_vocab) if tgt_vocab else state_dict["decoder.embedding.weight"].shape[0],
        )
        emb_dim = config.get(
            "embedding_dim",
            state_dict["encoder.embedding.weight"].shape[1],
        )
        hid_dim = config.get(
            "hidden_dim",
            state_dict["encoder.gru.weight_hh_l0"].shape[1],
        )
        attn_dim = config.get(
            "attention_dim",
            state_dict["decoder.attention.v.weight"].shape[1],
        )
        dropout = config.get("dropout", 0.0)

        encoder = Encoder(
            input_dim=src_dim,
            embedding_dim=emb_dim,
            hidden_dim=hid_dim,
            pad_idx=src_vocab.pad_idx if src_vocab else PAD_IDX,
            dropout=dropout,
        )
        decoder = Decoder(
            output_dim=tgt_dim,
            embedding_dim=emb_dim,
            hidden_dim=hid_dim,
            attention_dim=attn_dim,
            pad_idx=tgt_vocab.pad_idx if tgt_vocab else PAD_IDX,
            dropout=dropout,
        )
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            pad_idx=tgt_vocab.pad_idx if tgt_vocab else PAD_IDX,
            sos_idx=tgt_vocab.sos_idx if tgt_vocab else 2,
            eos_idx=tgt_vocab.eos_idx if tgt_vocab else 3,
        )

        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.to(target_device)
        model.src_vocab = src_vocab
        model.tgt_vocab = tgt_vocab
        return model, checkpoint_data, src_vocab, tgt_vocab
