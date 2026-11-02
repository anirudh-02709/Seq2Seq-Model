"""
Focused Test Suite for GRU Decoder with Bahdanau Attention (Phase 3, Step 3.4).

Verifies:
1. Correct logits dimensions (batch_size, target_vocab_size).
2. Correct hidden-state dimensions (1, batch_size, hidden_dim).
3. Correct attention-weight dimensions (batch_size, source_sequence_length).
4. Attention weights sum to one for every sample in the batch.
5. Padded source positions receive exactly zero attention.
6. Forward pass works seamlessly with actual Encoder outputs from Step 3.2.
7. Backward pass produces non-zero gradients for all trainable Decoder parameters.
8. Decoder works with different batch sizes (including batch_size=1).
9. Invalid input dimensions, batch mismatches, and invalid lengths produce clear errors.
10. Target PAD embedding uses the correct padding index and exhibits zero gradient behavior.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import create_all_dataloaders
from src.data.tokenizer import PAD_IDX, SOS_IDX
from src.models.attention import BahdanauAttention
from src.models.decoder import Decoder
from src.models.encoder import Encoder


class TestDecoder(unittest.TestCase):
    """Test suite for Decoder component."""

    def setUp(self):
        torch.manual_seed(42)
        self.output_dim = 50
        self.embedding_dim = 16
        self.hidden_dim = 32
        self.attention_dim = 16
        self.pad_idx = 0

        self.decoder = Decoder(
            output_dim=self.output_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
            pad_idx=self.pad_idx,
            dropout=0.0,
        )

    def test_01_logits_dimensions(self):
        """Test 1: Correct logits dimensions (batch_size, target_vocab_size)."""
        batch_size = 4
        seq_len = 7
        input_token_1d = torch.tensor([5, 12, 1, 9], dtype=torch.long)
        hidden = torch.randn(1, batch_size, self.hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)

        # 1. 1D input_token: (batch_size,)
        logits_1d, _, _ = self.decoder(input_token_1d, hidden, encoder_outputs)
        self.assertEqual(
            logits_1d.shape,
            (batch_size, self.output_dim),
            f"Expected logits shape {(batch_size, self.output_dim)}, got {tuple(logits_1d.shape)}",
        )

        # 2. 2D input_token: (batch_size, 1)
        input_token_2d = input_token_1d.unsqueeze(1)
        logits_2d, _, _ = self.decoder(input_token_2d, hidden, encoder_outputs)
        self.assertEqual(logits_2d.shape, (batch_size, self.output_dim))
        self.assertTrue(torch.allclose(logits_1d, logits_2d, atol=1e-6))

    def test_02_hidden_state_dimensions(self):
        """Test 2: Correct hidden-state dimensions (1, batch_size, hidden_dim)."""
        test_cases = [
            (1, 16),
            (3, 32),
            (5, 64),
        ]
        for b_size, h_dim in test_cases:
            dec = Decoder(
                output_dim=self.output_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=h_dim,
                pad_idx=self.pad_idx,
            )
            input_token = torch.randint(1, self.output_dim, (b_size,), dtype=torch.long)
            hidden = torch.randn(1, b_size, h_dim)
            encoder_outputs = torch.randn(b_size, 6, h_dim)

            _, new_hidden, _ = dec(input_token, hidden, encoder_outputs)
            self.assertEqual(
                new_hidden.shape,
                (1, b_size, h_dim),
                f"Expected hidden shape (1, {b_size}, {h_dim}), got {tuple(new_hidden.shape)}",
            )

    def test_03_attention_weight_dimensions(self):
        """Test 3: Correct attention-weight dimensions (batch_size, source_sequence_length)."""
        batch_size = 4
        seq_len = 11
        input_token = torch.tensor([2, 5, 8, 14], dtype=torch.long)
        hidden = torch.randn(1, batch_size, self.hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([11, 8, 6, 2], dtype=torch.long)

        _, _, weights = self.decoder(input_token, hidden, encoder_outputs, src_lengths=lengths)
        self.assertEqual(
            weights.shape,
            (batch_size, seq_len),
            f"Expected weights shape {(batch_size, seq_len)}, got {tuple(weights.shape)}",
        )

    def test_04_attention_weights_sum_to_one(self):
        """Test 4: Attention weights sum to one for every sample in the batch."""
        batch_size = 4
        seq_len = 9
        input_token = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        hidden = torch.randn(1, batch_size, self.hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([9, 6, 4, 1], dtype=torch.long)

        _, _, weights = self.decoder(input_token, hidden, encoder_outputs, src_lengths=lengths)

        for i in range(batch_size):
            weight_sum = weights[i].sum().item()
            self.assertAlmostEqual(
                weight_sum,
                1.0,
                places=5,
                msg=f"Sample {i} attention weights do not sum to 1.0 (got {weight_sum})",
            )

    def test_05_padded_source_positions_receive_zero_attention(self):
        """Test 5: Padded source positions receive exactly zero attention."""
        batch_size = 3
        seq_len = 8
        input_token = torch.tensor([7, 10, 15], dtype=torch.long)
        hidden = torch.randn(1, batch_size, self.hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([8, 5, 2], dtype=torch.long)

        _, _, weights = self.decoder(input_token, hidden, encoder_outputs, src_lengths=lengths)

        # Check padded positions
        self.assertTrue(torch.all(weights[0, 8:] == 0.0))  # no padding
        self.assertTrue(torch.all(weights[1, 5:] == 0.0))
        self.assertTrue(torch.all(weights[2, 2:] == 0.0))

        # Check unpadded positions are strictly positive
        self.assertTrue(torch.all(weights[0, :8] > 0.0))
        self.assertTrue(torch.all(weights[1, :5] > 0.0))
        self.assertTrue(torch.all(weights[2, :2] > 0.0))

    def test_06_forward_pass_with_actual_encoder_outputs(self):
        """Test 6: Forward pass works with actual Encoder outputs from existing Encoder."""
        data_dir = PROJECT_ROOT / "data"
        train_loader, _, _, src_vocab, tgt_vocab = create_all_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=0,
        )

        hidden_dim = 128
        embedding_dim = 64
        attention_dim = 64

        encoder = Encoder(
            vocab=src_vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        decoder = Decoder(
            vocab=tgt_vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        batch = next(iter(train_loader))
        src = batch["encoder_input_ids"]
        src_lengths = batch["encoder_lengths"]
        batch_size, max_src_len = src.shape

        # Run Encoder
        enc_outputs, enc_hidden = encoder(src, src_lengths)

        # Initial decoder input: SOS token for each item in the batch
        sos_token = torch.tensor([tgt_vocab.sos_idx] * batch_size, dtype=torch.long)

        # Run Decoder one step
        logits, dec_hidden, attn_weights = decoder(
            input_token=sos_token,
            hidden=enc_hidden,
            encoder_outputs=enc_outputs,
            src_lengths=src_lengths,
        )

        self.assertEqual(logits.shape, (batch_size, len(tgt_vocab)))
        self.assertEqual(dec_hidden.shape, (1, batch_size, hidden_dim))
        self.assertEqual(attn_weights.shape, (batch_size, max_src_len))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(dec_hidden).any())
        self.assertFalse(torch.isnan(attn_weights).any())

    def test_07_backward_pass_produces_gradients(self):
        """Test 7: Backward pass produces gradients for trainable Decoder parameters."""
        batch_size = 2
        seq_len = 5
        input_token = torch.tensor([4, 9], dtype=torch.long)
        hidden = torch.randn(1, batch_size, self.hidden_dim, requires_grad=True)
        encoder_outputs = torch.randn(batch_size, seq_len, self.hidden_dim, requires_grad=True)
        lengths = torch.tensor([5, 3], dtype=torch.long)

        logits, new_hidden, _ = self.decoder(
            input_token,
            hidden,
            encoder_outputs,
            src_lengths=lengths,
        )

        # Smoke loss
        loss = logits.sum() + new_hidden.sum()
        loss.backward()

        # Check gradients for all decoder parameter groups
        for name, param in self.decoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Decoder parameter '{name}' did not receive a gradient.")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient of '{name}'.")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf in gradient of '{name}'.")

        # Check gradients backpropagated to inputs
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(encoder_outputs.grad)

    def test_08_works_with_different_batch_sizes(self):
        """Test 8: Decoder works with different batch sizes, including batch_size=1."""
        test_batch_sizes = [1, 2, 7, 16]
        seq_len = 6

        for b_size in test_batch_sizes:
            input_token = torch.randint(1, self.output_dim, (b_size,), dtype=torch.long)
            hidden = torch.randn(1, b_size, self.hidden_dim)
            encoder_outputs = torch.randn(b_size, seq_len, self.hidden_dim)

            logits, new_hidden, weights = self.decoder(input_token, hidden, encoder_outputs)

            self.assertEqual(logits.shape, (b_size, self.output_dim))
            self.assertEqual(new_hidden.shape, (1, b_size, self.hidden_dim))
            self.assertEqual(weights.shape, (b_size, seq_len))

    def test_09_invalid_input_dimensions_produce_clear_errors(self):
        """Test 9: Invalid input dimensions produce clear errors."""
        hidden = torch.randn(1, 2, self.hidden_dim)
        enc_outputs = torch.randn(2, 5, self.hidden_dim)

        # 1. input_token 3D
        with self.assertRaises(ValueError) as ctx:
            self.decoder(torch.ones(2, 1, 1, dtype=torch.long), hidden, enc_outputs)
        self.assertIn("shape", str(ctx.exception))

        # 2. input_token 2D but size(1) != 1
        with self.assertRaises(ValueError) as ctx:
            self.decoder(torch.ones(2, 3, dtype=torch.long), hidden, enc_outputs)
        self.assertIn("shape", str(ctx.exception))

        # 3. hidden wrong dimension (1D or 4D)
        with self.assertRaises(ValueError) as ctx:
            self.decoder(torch.tensor([1, 2]), torch.randn(2), enc_outputs)
        self.assertIn("shape", str(ctx.exception))

        # 4. hidden with size(0) != 1
        with self.assertRaises(ValueError) as ctx:
            self.decoder(torch.tensor([1, 2]), torch.randn(2, 2, self.hidden_dim), enc_outputs)
        self.assertIn("shape", str(ctx.exception))

        # 5. Batch size mismatch between input_token and hidden
        with self.assertRaises(ValueError) as ctx:
            self.decoder(torch.tensor([1, 2, 3]), hidden, enc_outputs)
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 6. Batch size mismatch between input_token and encoder_outputs
        with self.assertRaises(ValueError) as ctx:
            self.decoder(
                torch.tensor([1, 2]),
                hidden,
                torch.randn(3, 5, self.hidden_dim),
            )
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 7. Hidden dimension mismatch
        with self.assertRaises(ValueError) as ctx:
            self.decoder(
                torch.tensor([1, 2]),
                torch.randn(1, 2, 64),
                enc_outputs,
            )
        self.assertIn("dimension mismatch", str(ctx.exception))

        # 8. Non-tensor input
        with self.assertRaises(TypeError):
            self.decoder("invalid_token", hidden, enc_outputs)

        # 9. Constructor argument validation
        with self.assertRaises(ValueError):
            Decoder(output_dim=None, vocab=None)
        with self.assertRaises(ValueError):
            Decoder(output_dim=-5)
        with self.assertRaises(ValueError):
            Decoder(output_dim=20, embedding_dim=0)
        with self.assertRaises(ValueError):
            Decoder(output_dim=20, hidden_dim=-1)
        with self.assertRaises(ValueError):
            Decoder(output_dim=20, dropout=2.0)

    def test_10_target_pad_embedding_uses_correct_padding_index(self):
        """Test 10: Target PAD embedding uses the correct padding index."""
        custom_pad_idx = 3
        dec = Decoder(
            output_dim=25,
            embedding_dim=16,
            hidden_dim=32,
            pad_idx=custom_pad_idx,
        )

        # Verify pad_idx stored and configured in embedding layer
        self.assertEqual(dec.pad_idx, custom_pad_idx)
        self.assertEqual(dec.embedding.padding_idx, custom_pad_idx)

        # Initial embedding vector at padding_idx is all zeros
        pad_vector = dec.embedding.weight[custom_pad_idx]
        self.assertTrue(torch.all(pad_vector == 0.0))

        # Forward pass with target PAD token
        input_token = torch.tensor([custom_pad_idx, 1], dtype=torch.long)
        hidden = torch.randn(1, 2, 32)
        encoder_outputs = torch.randn(2, 4, 32)

        logits, new_hidden, _ = dec(input_token, hidden, encoder_outputs)
        loss = logits.sum() + new_hidden.sum()
        loss.backward()

        self.assertIsNotNone(dec.embedding.weight.grad)
        pad_grad = dec.embedding.weight.grad[custom_pad_idx]
        self.assertTrue(
            torch.all(pad_grad == 0.0),
            f"Expected zero gradient at padding_idx, got {pad_grad}",
        )

        # Non-pad token receives non-zero gradient
        token_grad = dec.embedding.weight.grad[1]
        self.assertFalse(torch.all(token_grad == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
