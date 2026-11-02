"""
Bahdanau (Additive) Attention module for NL2SQL Seq2Seq model (Phase 3, Step 3.3).

Formulation:
    score(s_t, h_i) = v^T tanh(W_d s_t + W_e h_i)
    attention_weights = softmax(scores)
    context = sum(attention_weights_i * encoder_output_i)

Features:
- Configurable hidden_dim and attention_dim.
- Flexible decoder hidden state input: (1, batch_size, hidden_dim) or (batch_size, hidden_dim).
- Strict padding masking: padded positions receive exactly zero attention weight.
- Fully compatible with encoder outputs and lengths from Step 3.2.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Aligns the decoder's current hidden state with encoder outputs across
    all source sequence time steps to compute a weighted context vector.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: Optional[int] = None,
        enc_hidden_dim: Optional[int] = None,
        dec_hidden_dim: Optional[int] = None,
    ) -> None:
        """
        Initializes Bahdanau Attention.

        Args:
            hidden_dim: Default hidden dimensionality for both encoder and decoder.
            attention_dim: Internal attention projection dimensionality (default: hidden_dim).
            enc_hidden_dim: Optional encoder hidden state dimension (default: hidden_dim).
            dec_hidden_dim: Optional decoder hidden state dimension (default: hidden_dim).
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.enc_hidden_dim = enc_hidden_dim if enc_hidden_dim is not None else hidden_dim
        self.dec_hidden_dim = dec_hidden_dim if dec_hidden_dim is not None else hidden_dim
        self.attention_dim = attention_dim if attention_dim is not None else hidden_dim

        if self.enc_hidden_dim <= 0:
            raise ValueError(f"enc_hidden_dim must be positive, got {self.enc_hidden_dim}.")
        if self.dec_hidden_dim <= 0:
            raise ValueError(f"dec_hidden_dim must be positive, got {self.dec_hidden_dim}.")
        if self.attention_dim <= 0:
            raise ValueError(f"attention_dim must be positive, got {self.attention_dim}.")

        # Learned linear projections: W_d for decoder hidden state, W_e for encoder outputs
        self.W_d = nn.Linear(self.dec_hidden_dim, self.attention_dim, bias=False)
        self.W_e = nn.Linear(self.enc_hidden_dim, self.attention_dim, bias=False)

        # Projection vector v: projects tanh result to scalar energy score
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lengths: Optional[Union[torch.Tensor, list[int]]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes context vector and attention weights.

        Args:
            decoder_hidden: Current decoder hidden state of shape `(1, batch_size, hidden_dim)`
                or `(batch_size, hidden_dim)`.
            encoder_outputs: Full encoder sequence outputs of shape
                `(batch_size, source_sequence_length, enc_hidden_dim)`.
            src_lengths: Optional 1D tensor or list of integers `(batch_size,)` containing
                valid unpadded sequence lengths for each sample.
            mask: Optional boolean or numeric mask tensor of shape
                `(batch_size, source_sequence_length)`, where True/1 indicates valid tokens
                and False/0 indicates padded positions.

        Returns:
            context: Context vector tensor of shape `(batch_size, enc_hidden_dim)`.
            attention_weights: Normalized attention probability distribution over source
                positions of shape `(batch_size, source_sequence_length)`.
        """
        # --- 1. Validate and normalize decoder_hidden ---
        if not isinstance(decoder_hidden, torch.Tensor):
            raise TypeError(f"Expected decoder_hidden to be torch.Tensor, got {type(decoder_hidden)}.")

        if decoder_hidden.dim() == 3:
            if decoder_hidden.size(0) != 1:
                # If multi-layer hidden state passed, select top layer
                s_t = decoder_hidden[-1]
            else:
                s_t = decoder_hidden.squeeze(0)
        elif decoder_hidden.dim() == 2:
            s_t = decoder_hidden
        else:
            raise ValueError(
                f"Expected decoder_hidden to have 2 or 3 dimensions, "
                f"got shape {tuple(decoder_hidden.shape)}."
            )

        # --- 2. Validate encoder_outputs ---
        if not isinstance(encoder_outputs, torch.Tensor):
            raise TypeError(f"Expected encoder_outputs to be torch.Tensor, got {type(encoder_outputs)}.")

        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"Expected encoder_outputs to be a 3D tensor (batch_size, seq_len, enc_hidden_dim), "
                f"got shape {tuple(encoder_outputs.shape)}."
            )

        batch_size, seq_len, enc_h_dim = encoder_outputs.shape

        if batch_size == 0 or seq_len == 0:
            raise ValueError(
                f"Expected non-empty encoder_outputs tensor, got shape {tuple(encoder_outputs.shape)}."
            )

        if s_t.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: decoder_hidden has batch size {s_t.size(0)}, "
                f"but encoder_outputs has batch size {batch_size}."
            )

        if s_t.size(1) != self.dec_hidden_dim:
            raise ValueError(
                f"decoder_hidden dimension mismatch: expected {self.dec_hidden_dim}, "
                f"got {s_t.size(1)}."
            )

        if enc_h_dim != self.enc_hidden_dim:
            raise ValueError(
                f"encoder_outputs hidden dimension mismatch: expected {self.enc_hidden_dim}, "
                f"got {enc_h_dim}."
            )

        # --- 3. Validate and construct boolean mask ---
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Expected mask to be torch.Tensor, got {type(mask)}.")
            if mask.dim() != 2:
                raise ValueError(
                    f"Expected mask to be a 2D tensor of shape (batch_size, seq_len), "
                    f"got shape {tuple(mask.shape)}."
                )
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} does not match (batch_size={batch_size}, seq_len={seq_len})."
                )
            mask_bool = mask.bool()
        elif src_lengths is not None:
            if not isinstance(src_lengths, torch.Tensor):
                src_lengths = torch.as_tensor(src_lengths, dtype=torch.long, device=encoder_outputs.device)
            if src_lengths.dim() != 1:
                raise ValueError(
                    f"Expected src_lengths to be 1D tensor of shape (batch_size,), "
                    f"got shape {tuple(src_lengths.shape)}."
                )
            if src_lengths.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: src_lengths has size {src_lengths.size(0)}, "
                    f"but batch size is {batch_size}."
                )
            if torch.any(src_lengths <= 0):
                raise ValueError("All sequence lengths in src_lengths must be strictly positive (> 0).")
            if torch.any(src_lengths > seq_len):
                raise ValueError(
                    f"A sequence length in src_lengths exceeds encoder sequence length ({seq_len})."
                )

            # Generate boolean mask: True for valid tokens, False for padding
            cols = torch.arange(seq_len, device=src_lengths.device).unsqueeze(0)
            mask_bool = cols < src_lengths.unsqueeze(1)
        else:
            # No padding mask specified; all positions treated as valid
            mask_bool = torch.ones((batch_size, seq_len), dtype=torch.bool, device=encoder_outputs.device)

        # Ensure no sample has 0 valid tokens
        valid_counts = mask_bool.sum(dim=-1)
        if torch.any(valid_counts == 0):
            raise ValueError("Every sequence in the batch must have at least one valid (unpadded) position.")

        # --- 4. Bahdanau Energy Scores ---
        # W_d(s_t): (batch_size, attention_dim) -> unsqueeze(1) -> (batch_size, 1, attention_dim)
        proj_dec = self.W_d(s_t).unsqueeze(1)
        # W_e(encoder_outputs): (batch_size, seq_len, attention_dim)
        proj_enc = self.W_e(encoder_outputs)

        # Broadcast and add: (batch_size, seq_len, attention_dim)
        energy = torch.tanh(proj_dec + proj_enc)

        # Project to scalar score: (batch_size, seq_len, 1) -> squeeze(-1) -> (batch_size, seq_len)
        scores = self.v(energy).squeeze(-1)

        # --- 5. Masked Softmax ---
        # Set scores of padded positions to large negative value before softmax
        masked_scores = scores.masked_fill(~mask_bool, -1e9)
        attention_weights = F.softmax(masked_scores, dim=-1)

        # Guard against any numerical precision residual at padded positions
        attention_weights = attention_weights * mask_bool.float()
        # Re-normalize to ensure exact sum to 1 over valid positions
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        # --- 6. Context Vector ---
        # Weighted sum: (batch_size, 1, seq_len) x (batch_size, seq_len, enc_hidden_dim)
        # -> (batch_size, 1, enc_hidden_dim) -> squeeze(1) -> (batch_size, enc_hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
