"""
GRU-based Decoder with Bahdanau Attention for NL2SQL Seq2Seq model (Phase 3, Step 3.4).

Architecture:
- Target embedding layer with padding_idx for target tokens.
- Single-step GRU layer.
- Existing BahdanauAttention component to align decoder hidden state with encoder outputs.
- Output projection layer producing unnormalized logits over target vocabulary.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.data.tokenizer import PAD_IDX
from src.data.vocabulary import Vocabulary
from src.models.attention import BahdanauAttention


class Decoder(nn.Module):
    """
    GRU-based Decoder with Bahdanau additive attention for one decoding step.

    Operates autoregressively step-by-step:
    1. Embeds the previous target token.
    2. Updates decoder hidden state with GRU.
    3. Calculates Bahdanau attention over encoder outputs using the updated hidden state.
    4. Combines GRU step output and context vector.
    5. Projects to unnormalized logits over target vocabulary.
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        attention_dim: Optional[int] = None,
        enc_hidden_dim: Optional[int] = None,
        pad_idx: Optional[int] = None,
        dropout: float = 0.0,
        attention: Optional[BahdanauAttention] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        """
        Initializes the Decoder.

        Args:
            output_dim: Target vocabulary size. If None, derived from `vocab`.
            embedding_dim: Dimensionality of target token embeddings.
            hidden_dim: Dimensionality of decoder GRU hidden state.
            attention_dim: Dimensionality for internal attention projection (default: hidden_dim).
            enc_hidden_dim: Dimensionality of encoder outputs (default: hidden_dim).
            pad_idx: Index of PAD token in target vocabulary. If None, derived from `vocab` or PAD_IDX.
            dropout: Dropout probability applied to embeddings and combined context. Default: 0.0.
            attention: Optional pre-instantiated BahdanauAttention instance.
            vocab: Optional target Vocabulary object to obtain output_dim and pad_idx automatically.
        """
        super().__init__()

        # Derive output_dim and pad_idx from target Vocabulary if provided
        if vocab is not None:
            if output_dim is None:
                output_dim = len(vocab)
            if pad_idx is None:
                pad_idx = vocab.pad_idx

        if output_dim is None:
            raise ValueError("Must provide either 'output_dim' or 'vocab' to Decoder.")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout}.")

        if pad_idx is None:
            pad_idx = PAD_IDX

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.enc_hidden_dim = enc_hidden_dim if enc_hidden_dim is not None else hidden_dim
        self.attention_dim = attention_dim if attention_dim is not None else hidden_dim
        self.pad_idx = pad_idx
        self.dropout_p = dropout

        # 1. Target embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.output_dim,
            embedding_dim=self.embedding_dim,
            padding_idx=self.pad_idx,
        )

        self.dropout = nn.Dropout(p=dropout)

        # 2. GRU layer (unidirectional, batch_first)
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 3. Existing BahdanauAttention component
        if attention is not None:
            self.attention = attention
        else:
            self.attention = BahdanauAttention(
                hidden_dim=self.hidden_dim,
                attention_dim=self.attention_dim,
                enc_hidden_dim=self.enc_hidden_dim,
                dec_hidden_dim=self.hidden_dim,
            )

        # 4. Output projection layer: projects [gru_output, context] to target vocabulary logits
        self.fc_out = nn.Linear(
            self.hidden_dim + self.enc_hidden_dim,
            self.output_dim,
        )

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: Optional[Union[torch.Tensor, list[int]]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single decoding step.

        Args:
            input_token: Previous target token IDs tensor of shape `(batch_size,)` or `(batch_size, 1)`.
            hidden: Previous decoder hidden state tensor of shape `(1, batch_size, hidden_dim)`
                or `(batch_size, hidden_dim)`.
            encoder_outputs: Encoder step representations tensor of shape
                `(batch_size, source_sequence_length, enc_hidden_dim)`.
            src_lengths: Optional source sequence lengths `(batch_size,)`.
            mask: Optional source padding mask `(batch_size, source_sequence_length)`.

        Returns:
            logits: Unnormalized logits over target vocabulary of shape `(batch_size, output_dim)`.
            hidden: Updated decoder hidden state of shape `(1, batch_size, hidden_dim)`.
            attention_weights: Normalized attention weights over source sequence positions
                of shape `(batch_size, source_sequence_length)`.
        """
        # --- 1. Validate input_token ---
        if not isinstance(input_token, torch.Tensor):
            raise TypeError(f"Expected 'input_token' to be a torch.Tensor, got {type(input_token)}.")

        if input_token.dim() == 1:
            input_token_2d = input_token.unsqueeze(1)
        elif input_token.dim() == 2 and input_token.size(1) == 1:
            input_token_2d = input_token
        else:
            raise ValueError(
                f"Expected 'input_token' to have shape (batch_size,) or (batch_size, 1), "
                f"got {tuple(input_token.shape)}."
            )

        batch_size = input_token_2d.size(0)
        if batch_size == 0:
            raise ValueError("Expected non-empty 'input_token' tensor.")

        if input_token_2d.dtype != torch.long:
            input_token_2d = input_token_2d.long()

        # --- 2. Validate hidden ---
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Expected 'hidden' to be a torch.Tensor, got {type(hidden)}.")

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        elif hidden.dim() == 3:
            if hidden.size(0) != 1:
                raise ValueError(
                    f"Expected 'hidden' to have shape (1, batch_size, hidden_dim), "
                    f"got {tuple(hidden.shape)}."
                )
        else:
            raise ValueError(
                f"Expected 'hidden' to have shape (1, batch_size, hidden_dim) or (batch_size, hidden_dim), "
                f"got {tuple(hidden.shape)}."
            )

        if hidden.size(1) != batch_size:
            raise ValueError(
                f"Batch size mismatch: 'input_token' has batch size {batch_size}, "
                f"but 'hidden' has {hidden.size(1)}."
            )

        if hidden.size(2) != self.hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self.hidden_dim}, got {hidden.size(2)}."
            )

        # --- 3. Validate encoder_outputs ---
        if not isinstance(encoder_outputs, torch.Tensor):
            raise TypeError(f"Expected 'encoder_outputs' to be a torch.Tensor, got {type(encoder_outputs)}.")

        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"Expected 'encoder_outputs' to be a 3D tensor (batch_size, seq_len, enc_hidden_dim), "
                f"got shape {tuple(encoder_outputs.shape)}."
            )

        if encoder_outputs.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: 'input_token' has batch size {batch_size}, "
                f"but 'encoder_outputs' has {encoder_outputs.size(0)}."
            )

        if encoder_outputs.size(1) == 0:
            raise ValueError("Source sequence length in 'encoder_outputs' must be greater than 0.")

        if encoder_outputs.size(2) != self.enc_hidden_dim:
            raise ValueError(
                f"encoder_outputs hidden dimension mismatch: expected {self.enc_hidden_dim}, "
                f"got {encoder_outputs.size(2)}."
            )

        # --- 4. Embed target token ---
        # input_token_2d: (batch_size, 1) -> embedded: (batch_size, 1, embedding_dim)
        embedded = self.dropout(self.embedding(input_token_2d))

        # --- 5. Step GRU ---
        # embedded: (batch_size, 1, embedding_dim), hidden: (1, batch_size, hidden_dim)
        # gru_output: (batch_size, 1, hidden_dim), new_hidden: (1, batch_size, hidden_dim)
        gru_output, new_hidden = self.gru(embedded, hidden)

        # --- 6. Bahdanau Attention over encoder outputs ---
        # Uses updated decoder hidden state new_hidden: (1, batch_size, hidden_dim)
        # context: (batch_size, enc_hidden_dim), attention_weights: (batch_size, seq_len)
        context, attention_weights = self.attention(
            new_hidden,
            encoder_outputs,
            src_lengths=src_lengths,
            mask=mask,
        )

        # --- 7. Combine GRU output and context vector ---
        # gru_output.squeeze(1): (batch_size, hidden_dim)
        # context: (batch_size, enc_hidden_dim)
        # combined: (batch_size, hidden_dim + enc_hidden_dim)
        combined = torch.cat([gru_output.squeeze(1), context], dim=1)
        combined = self.dropout(combined)

        # --- 8. Output projection layer ---
        # logits: (batch_size, output_dim) - unnormalized raw logits
        logits = self.fc_out(combined)

        return logits, new_hidden, attention_weights
