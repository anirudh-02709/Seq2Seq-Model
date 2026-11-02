"""
Complete Sequence-to-Sequence (Seq2Seq) architecture for NL2SQL (Phase 3, Step 3.5).

Integrates:
- Unidirectional GRU Encoder (src/models/encoder.py)
- Bahdanau Attention (src/models/attention.py)
- GRU Decoder (src/models/decoder.py)

Supports:
- Forward pass with configurable teacher forcing ratio.
- Autoregressive greedy generation for inference with EOS early stopping.
- Comprehensive input validation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX
from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Seq2Seq model with Bahdanau attention for NL2SQL translation.

    Encodes database schema + natural language question, and decodes
    SQL query token sequences with attention and teacher forcing.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_idx: int = PAD_IDX,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
        unk_idx: int = UNK_IDX,
    ) -> None:
        """
        Initializes the Seq2Seq model.

        Args:
            encoder: Configured Encoder instance.
            decoder: Configured Decoder instance.
            pad_idx: Special PAD token ID.
            sos_idx: Special Start-of-Sequence token ID.
            eos_idx: Special End-of-Sequence token ID.
            unk_idx: Special Unknown token ID.
        """
        super().__init__()

        if not isinstance(encoder, Encoder):
            raise TypeError(f"Expected encoder to be an instance of Encoder, got {type(encoder)}.")
        if not isinstance(decoder, Decoder):
            raise TypeError(f"Expected decoder to be an instance of Decoder, got {type(decoder)}.")

        # Ensure compatibility of hidden dimensions between encoder and decoder
        if encoder.hidden_dim != decoder.hidden_dim:
            raise ValueError(
                f"Dimension mismatch: encoder hidden_dim ({encoder.hidden_dim}) "
                f"does not match decoder hidden_dim ({decoder.hidden_dim})."
            )
        if encoder.hidden_dim != decoder.enc_hidden_dim:
            raise ValueError(
                f"Dimension mismatch: encoder hidden_dim ({encoder.hidden_dim}) "
                f"does not match decoder enc_hidden_dim ({decoder.enc_hidden_dim})."
            )

        self.encoder = encoder
        self.decoder = decoder

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.hidden_dim = encoder.hidden_dim
        self.source_vocab_size = encoder.input_dim
        self.target_vocab_size = decoder.output_dim

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Union[torch.Tensor, list[int]],
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Executes a complete forward pass through encoder and decoder.

        Args:
            src: Padded source token IDs of shape `(batch_size, src_len)`.
            src_lengths: 1D tensor or list of integers `(batch_size,)` containing
                valid unpadded sequence lengths.
            tgt: Padded target token IDs beginning with `<SOS>` of shape `(batch_size, tgt_len)`.
            teacher_forcing_ratio: Probability of using ground-truth target token as next input
                during decoding (between 0.0 and 1.0).

        Returns:
            outputs: Unnormalized logits of shape `(batch_size, tgt_len - 1, target_vocab_size)`.
        """
        # --- 1. Input Validation ---
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Expected 'src' to be a torch.Tensor, got {type(src)}.")
        if not isinstance(tgt, torch.Tensor):
            raise TypeError(f"Expected 'tgt' to be a torch.Tensor, got {type(tgt)}.")

        if src.dim() != 2:
            raise ValueError(
                f"Expected 'src' to be a 2D tensor (batch_size, src_len), got shape {tuple(src.shape)}."
            )
        if tgt.dim() != 2:
            raise ValueError(
                f"Expected 'tgt' to be a 2D tensor (batch_size, tgt_len), got shape {tuple(tgt.shape)}."
            )

        batch_size, src_len = src.shape
        tgt_batch_size, tgt_len = tgt.shape

        if batch_size == 0 or src_len == 0:
            raise ValueError(f"Expected non-empty 'src' tensor, got shape {tuple(src.shape)}.")
        if tgt_len < 2:
            raise ValueError(
                f"Expected 'tgt' to have sequence length >= 2 (starts with <SOS> and at least 1 prediction step), "
                f"got tgt_len={tgt_len}."
            )

        if batch_size != tgt_batch_size:
            raise ValueError(
                f"Batch size mismatch: 'src' has batch size {batch_size}, "
                f"but 'tgt' has batch size {tgt_batch_size}."
            )

        if not isinstance(teacher_forcing_ratio, (int, float)):
            raise TypeError(
                f"teacher_forcing_ratio must be a float or int, got {type(teacher_forcing_ratio)}."
            )
        if not (0.0 <= float(teacher_forcing_ratio) <= 1.0):
            raise ValueError(
                f"teacher_forcing_ratio must be between 0.0 and 1.0, got {teacher_forcing_ratio}."
            )
        teacher_forcing_ratio = float(teacher_forcing_ratio)

        # Validate src_lengths
        if not isinstance(src_lengths, torch.Tensor):
            src_lengths_tensor = torch.as_tensor(src_lengths, dtype=torch.long, device=src.device)
        else:
            src_lengths_tensor = src_lengths

        if src_lengths_tensor.dim() != 1:
            raise ValueError(
                f"Expected 'src_lengths' to be a 1D tensor (batch_size,), got shape {tuple(src_lengths_tensor.shape)}."
            )
        if src_lengths_tensor.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: 'src' has batch size {batch_size}, "
                f"but 'src_lengths' has length {src_lengths_tensor.size(0)}."
            )
        if torch.any(src_lengths_tensor <= 0):
            raise ValueError("All sequence lengths in 'src_lengths' must be strictly positive (> 0).")
        if torch.any(src_lengths_tensor > src_len):
            raise ValueError(
                f"A sequence length in 'src_lengths' exceeds source sequence length ({src_len})."
            )

        # Ensure long dtypes for token indices
        if src.dtype != torch.long:
            src = src.long()
        if tgt.dtype != torch.long:
            tgt = tgt.long()

        # --- 2. Encoder Step ---
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)
        encoder_outputs, hidden = self.encoder(src, src_lengths_tensor)

        # --- 3. Allocate Logits Tensor ---
        # Shape: (batch_size, tgt_len - 1, target_vocab_size)
        outputs = torch.zeros(
            batch_size,
            tgt_len - 1,
            self.target_vocab_size,
            device=src.device,
            dtype=encoder_outputs.dtype,
        )

        # --- 4. Autoregressive Decoding Loop ---
        # Initial input token: <SOS> at column index 0 of tgt
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            # Step decoder: receives input_token (batch_size,), hidden, encoder_outputs, src_lengths
            step_logits, hidden, _ = self.decoder(
                input_token=input_token,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                src_lengths=src_lengths_tensor,
            )

            # Store logits at index t - 1
            outputs[:, t - 1] = step_logits

            # Determine whether to use teacher forcing for next input
            if teacher_forcing_ratio == 1.0:
                teacher_force = True
            elif teacher_forcing_ratio == 0.0:
                teacher_force = False
            else:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # Predict highest-scoring token
            top1 = step_logits.argmax(dim=-1)

            # Select next input token: ground truth if teacher forcing, else model prediction
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def generate(
        self,
        src: torch.Tensor,
        src_lengths: Union[torch.Tensor, list[int]],
        max_length: int = 50,
        sos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Executes greedy generation for inference.

        Args:
            src: Source token IDs tensor of shape `(batch_size, src_len)`.
            src_lengths: 1D tensor or list of integers `(batch_size,)`.
            max_length: Maximum number of tokens to generate.
            sos_idx: Optional override for <SOS> token ID (defaults to self.sos_idx).
            eos_idx: Optional override for <EOS> token ID (defaults to self.eos_idx).

        Returns:
            generated_ids: Tensor of shape `(batch_size, gen_length)` containing generated
                token IDs (excluding the initial <SOS> token).
        """
        # Validate inputs
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Expected 'src' to be a torch.Tensor, got {type(src)}.")
        if src.dim() != 2:
            raise ValueError(
                f"Expected 'src' to be a 2D tensor (batch_size, src_len), got shape {tuple(src.shape)}."
            )
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}.")

        sos_token_id = sos_idx if sos_idx is not None else self.sos_idx
        eos_token_id = eos_idx if eos_idx is not None else self.eos_idx

        batch_size, src_len = src.shape
        if batch_size == 0 or src_len == 0:
            raise ValueError(f"Expected non-empty 'src' tensor, got shape {tuple(src.shape)}.")

        if not isinstance(src_lengths, torch.Tensor):
            src_lengths_tensor = torch.as_tensor(src_lengths, dtype=torch.long, device=src.device)
        else:
            src_lengths_tensor = src_lengths

        if src_lengths_tensor.dim() != 1 or src_lengths_tensor.size(0) != batch_size:
            raise ValueError(
                f"Expected 'src_lengths' of shape ({batch_size},), got {tuple(src_lengths_tensor.shape)}."
            )
        if torch.any(src_lengths_tensor <= 0) or torch.any(src_lengths_tensor > src_len):
            raise ValueError(
                f"Invalid sequence length in 'src_lengths': must be in range [1, {src_len}]."
            )

        if src.dtype != torch.long:
            src = src.long()

        # 1. Encode source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths_tensor)

        # 2. Initialize generation
        curr_token = torch.full((batch_size,), sos_token_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        generated_steps: list[torch.Tensor] = []

        # 3. Autoregressive Greedy Decoding
        for _ in range(max_length):
            step_logits, hidden, _ = self.decoder(
                input_token=curr_token,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                src_lengths=src_lengths_tensor,
            )

            # Greedily choose highest scoring token
            pred = step_logits.argmax(dim=-1)

            # Mask finished sequences to PAD
            step_tokens = pred.clone()
            step_tokens[finished] = self.pad_idx
            generated_steps.append(step_tokens)

            # Mark sequences that produced EOS
            has_eos = pred == eos_token_id
            finished = finished | has_eos

            # Early stopping if all sequences in batch reached EOS
            if finished.all():
                break

            curr_token = pred

        if not generated_steps:
            return torch.empty((batch_size, 0), dtype=torch.long, device=src.device)

        return torch.stack(generated_steps, dim=1)
