"""
Encoder component for custom attention-based Seq2Seq NL2SQL model (Phase 3, Step 3.2).

Architecture:
- Unidirectional GRU with batch_first=True.
- Embedding layer with padding_idx for source tokens.
- Dynamic sequence packing (pack_padded_sequence and pad_packed_sequence) to ensure
  padded positions do not contaminate the final hidden state and are zeroed out in outputs.
- Compatible with downstream Attention mechanisms.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.tokenizer import PAD_IDX
from src.data.vocabulary import Vocabulary


class Encoder(nn.Module):
    """
    Unidirectional GRU-based Encoder for NL2SQL Seq2Seq model.

    Converts variable-length source sequences (schema + question token IDs) into:
    - encoder_outputs: dense representations for each time-step (for attention).
    - hidden: the final hidden state of the sequence (for decoder initialization).
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        pad_idx: Optional[int] = None,
        dropout: float = 0.0,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        """
        Initializes the Encoder.

        Args:
            input_dim: Vocabulary size for source tokens. If None, derived from `vocab`.
            embedding_dim: Dimensionality of token embedding vectors.
            hidden_dim: Dimensionality of GRU hidden state.
            pad_idx: Index of PAD token in source vocabulary. If None, derived from `vocab` or PAD_IDX.
            dropout: Dropout probability applied to embeddings. Default: 0.0.
            vocab: Optional Vocabulary object to obtain input_dim and pad_idx automatically.
        """
        super().__init__()

        # Derive input_dim and pad_idx from Vocabulary object if provided
        if vocab is not None:
            if input_dim is None:
                input_dim = len(vocab)
            if pad_idx is None:
                pad_idx = vocab.pad_idx

        if input_dim is None:
            raise ValueError("Must provide either 'input_dim' or 'vocab' to Encoder.")

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}.")

        if pad_idx is None:
            pad_idx = PAD_IDX

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.dropout_p = dropout

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Union[torch.Tensor, list[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Encoder.

        Args:
            src: Source token IDs tensor of shape `(batch_size, source_sequence_length)`.
            src_lengths: 1D tensor or list of integers of shape `(batch_size,)` containing
                actual lengths of each sequence in the batch.

        Returns:
            outputs: Tensor of shape `(batch_size, source_sequence_length, hidden_dim)`
                containing GRU outputs for each step, with padded positions zeroed out.
            hidden: Final hidden state tensor of shape `(1, batch_size, hidden_dim)`.
        """
        # Validate src tensor
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Expected 'src' to be a torch.Tensor, got {type(src)}.")

        if src.dim() != 2:
            raise ValueError(
                f"Expected 'src' to be a 2D tensor of shape (batch_size, seq_len), "
                f"got shape {tuple(src.shape)}."
            )

        if src.size(0) == 0 or src.size(1) == 0:
            raise ValueError(f"Expected non-empty 'src' tensor, got shape {tuple(src.shape)}.")

        # Ensure src is integer / long
        if src.dtype != torch.long:
            src = src.long()

        # Validate and format src_lengths
        if not isinstance(src_lengths, torch.Tensor):
            src_lengths = torch.as_tensor(src_lengths, dtype=torch.long, device=src.device)

        if src_lengths.dim() != 1:
            raise ValueError(
                f"Expected 'src_lengths' to be a 1D tensor of shape (batch_size,), "
                f"got shape {tuple(src_lengths.shape)}."
            )

        if src.size(0) != src_lengths.size(0):
            raise ValueError(
                f"Batch size mismatch: 'src' has batch size {src.size(0)}, "
                f"but 'src_lengths' has length {src_lengths.size(0)}."
            )

        if torch.any(src_lengths <= 0):
            raise ValueError("All sequence lengths in 'src_lengths' must be strictly positive (> 0).")

        if torch.any(src_lengths > src.size(1)):
            raise ValueError(
                f"A sequence length in 'src_lengths' exceeds source sequence dimension ({src.size(1)})."
            )

        batch_size, seq_len = src.shape

        # 1. Embedding lookup: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(src))

        # 2. Pack padded sequence to ignore padding positions in GRU
        # PyTorch requires lengths to be CPU int64 tensor
        lengths_cpu = src_lengths.to("cpu", dtype=torch.int64)
        packed_embedded = pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        # 3. GRU pass:
        # packed_outputs contains concatenated step outputs for unpadded tokens
        # hidden has shape (1, batch_size, hidden_dim)
        packed_outputs, hidden = self.gru(packed_embedded)

        # 4. Unpack and restore original batch sequence length
        # outputs shape: (batch_size, seq_len, hidden_dim)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=seq_len,
        )

        return outputs, hidden
