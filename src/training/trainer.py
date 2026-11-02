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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tokenizer import PAD_IDX
from src.data.vocabulary import Vocabulary
from src.models.attention import BahdanauAttention
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
        teacher_forcing_ratio: float = 1.0,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        history_path: Union[str, Path] = "training_history.json",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
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
            teacher_forcing_ratio: Teacher forcing probability for training.
            checkpoint_dir: Directory where checkpoints are stored.
            history_path: Filepath for saving JSON training history.
            src_vocab: Source Vocabulary object for serialization into checkpoints.
            tgt_vocab: Target Vocabulary object for serialization into checkpoints.
            config: Hyperparameter dictionary stored in checkpoints.
        """
        if not isinstance(model, Seq2Seq):
            raise TypeError(f"Expected model to be an instance of Seq2Seq, got {type(model)}.")

        # Device assignment
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

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
        self.history_path = Path(history_path)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Populate config with model dimensions if not explicitly provided
        if config is None:
            self.config = {
                "source_vocab_size": model.source_vocab_size,
                "target_vocab_size": model.target_vocab_size,
                "embedding_dim": model.encoder.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "attention_dim": model.decoder.attention_dim,
                "dropout": getattr(model.encoder, "dropout_p", 0.0),
                "learning_rate": self.learning_rate,
                "clip_norm": self.clip_norm,
                "teacher_forcing_ratio": self.teacher_forcing_ratio,
            }
        else:
            self.config = dict(config)
            # Ensure essential dimensions are always present
            self.config.setdefault("source_vocab_size", model.source_vocab_size)
            self.config.setdefault("target_vocab_size", model.target_vocab_size)
            self.config.setdefault("embedding_dim", model.encoder.embedding_dim)
            self.config.setdefault("hidden_dim", model.hidden_dim)
            self.config.setdefault("attention_dim", model.decoder.attention_dim)

        # Tracking state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
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

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluates model on validation DataLoader.

        Returns:
            avg_val_loss: Average validation loss.
            val_token_acc: Token-level prediction accuracy ignoring PAD tokens.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["encoder_input_ids"].to(self.device)
                src_lengths = batch["encoder_lengths"].to(self.device)
                tgt = batch["decoder_target_ids"].to(self.device)

                # Validation uses teacher forcing (ratio = 1.0) for standard next-token loss
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

                # Calculate token accuracy ignoring PAD tokens
                preds = outputs.argmax(dim=-1)
                valid_mask = targets != self.pad_idx
                correct = (preds == targets) & valid_mask

                total_correct += correct.sum().item()
                total_tokens += valid_mask.sum().item()

        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_token_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

        return avg_val_loss, val_token_acc

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_token_acc: float,
        is_best: bool = False,
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
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "val_token_acc": val_token_acc,
            "train_loss": train_loss,
            "config": self.config,
            "is_best": is_best,
        }

        # Include vocabulary dictionaries if available
        if self.src_vocab is not None:
            checkpoint_data["src_vocab"] = self.src_vocab.to_dict()
        if self.tgt_vocab is not None:
            checkpoint_data["tgt_vocab"] = self.tgt_vocab.to_dict()

        torch.save(checkpoint_data, path)

    def fit(self, epochs: int, print_progress: bool = True) -> list[dict[str, Any]]:
        """
        Executes the full training and validation schedule.

        Args:
            epochs: Total number of training epochs to execute.
            print_progress: Whether to print progress each epoch.

        Returns:
            history: List of epoch metric dictionaries.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_token_acc = self.evaluate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record epoch metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_token_acc": round(val_token_acc, 4),
                "learning_rate": current_lr,
            }
            self.history.append(epoch_metrics)

            is_best = val_loss < self.best_val_loss
            marker = ""
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_token_acc=val_token_acc,
                    is_best=True,
                )
                marker = " [BEST]"

            # Save epoch-wise history JSON
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            if print_progress:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_token_acc:.4f} | "
                    f"LR: {current_lr:.6f}{marker}"
                )

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
        return model, checkpoint_data, src_vocab, tgt_vocab
