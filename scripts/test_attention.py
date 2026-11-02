"""
Focused Test Suite for Bahdanau Attention (Phase 3, Step 3.3).

Verifies:
1. Correct context-vector dimensions.
2. Correct attention-weight dimensions.
3. Attention weights sum to 1 for every sample in the batch.
4. Padded positions receive exactly zero attention weight.
5. Different source lengths in the same batch work correctly (unsorted, variable padding).
6. Context vector equals the explicit weighted sum of encoder outputs.
7. Backward pass produces gradients for all trainable attention parameters.
8. Invalid dimensions, batch mismatches, and invalid lengths produce clear errors.
9. Attention works seamlessly with actual outputs from the existing Encoder.
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
from src.models.attention import BahdanauAttention
from src.models.encoder import Encoder


class TestBahdanauAttention(unittest.TestCase):
    """Test suite for BahdanauAttention module."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_dim = 32
        self.attention_dim = 16
        self.attention = BahdanauAttention(
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
        )

    def test_01_context_vector_dimensions(self):
        """Test 1: Correct context-vector dimensions (batch_size, hidden_dim)."""
        test_cases = [
            (2, 6, 32),
            (4, 10, 32),
            (1, 5, 32),
        ]
        for b_size, seq_len, h_dim in test_cases:
            dec_hidden = torch.randn(b_size, h_dim)
            enc_outputs = torch.randn(b_size, seq_len, h_dim)
            lengths = torch.tensor([seq_len] * b_size, dtype=torch.long)

            context, _ = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)
            self.assertEqual(
                context.shape,
                (b_size, h_dim),
                f"Expected context shape {(b_size, h_dim)}, got {tuple(context.shape)}",
            )

    def test_02_attention_weight_dimensions(self):
        """Test 2: Correct attention-weight dimensions (batch_size, source_sequence_length)."""
        batch_size = 4
        seq_len = 9
        # Test with 2D decoder hidden: (batch_size, hidden_dim)
        dec_hidden_2d = torch.randn(batch_size, self.hidden_dim)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([9, 7, 5, 3], dtype=torch.long)

        _, weights_2d = self.attention(dec_hidden_2d, enc_outputs, src_lengths=lengths)
        self.assertEqual(
            weights_2d.shape,
            (batch_size, seq_len),
            f"Expected weights shape {(batch_size, seq_len)}, got {tuple(weights_2d.shape)}",
        )

        # Test with 3D decoder hidden: (1, batch_size, hidden_dim)
        dec_hidden_3d = dec_hidden_2d.unsqueeze(0)
        _, weights_3d = self.attention(dec_hidden_3d, enc_outputs, src_lengths=lengths)
        self.assertEqual(weights_3d.shape, (batch_size, seq_len))
        self.assertTrue(torch.allclose(weights_2d, weights_3d, atol=1e-6))

    def test_03_attention_weights_sum_to_one(self):
        """Test 3: Attention weights sum to 1 for every sample in the batch."""
        batch_size = 5
        seq_len = 10
        dec_hidden = torch.randn(batch_size, self.hidden_dim)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([10, 8, 5, 3, 1], dtype=torch.long)

        _, weights = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)

        for i in range(batch_size):
            weight_sum = weights[i].sum().item()
            self.assertAlmostEqual(
                weight_sum,
                1.0,
                places=5,
                msg=f"Sample {i} attention weights do not sum to 1.0 (got {weight_sum})",
            )

    def test_04_padded_positions_receive_exactly_zero_weight(self):
        """Test 4: Padded positions receive exactly zero attention weight."""
        batch_size = 3
        seq_len = 8
        dec_hidden = torch.randn(batch_size, self.hidden_dim)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([5, 3, 2], dtype=torch.long)

        _, weights = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)

        # Verify padded positions are strictly zero
        self.assertTrue(
            torch.all(weights[0, 5:] == 0.0),
            f"Expected padded positions for sample 0 to be zero, got {weights[0, 5:]}",
        )
        self.assertTrue(
            torch.all(weights[1, 3:] == 0.0),
            f"Expected padded positions for sample 1 to be zero, got {weights[1, 3:]}",
        )
        self.assertTrue(
            torch.all(weights[2, 2:] == 0.0),
            f"Expected padded positions for sample 2 to be zero, got {weights[2, 2:]}",
        )

        # Verify unpadded positions are strictly positive
        self.assertTrue(torch.all(weights[0, :5] > 0.0))
        self.assertTrue(torch.all(weights[1, :3] > 0.0))
        self.assertTrue(torch.all(weights[2, :2] > 0.0))

        # Test identical behavior when passing explicit boolean mask
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        _, weights_mask = self.attention(dec_hidden, enc_outputs, mask=mask)
        self.assertTrue(torch.allclose(weights, weights_mask, atol=1e-6))

    def test_05_different_source_lengths_in_same_batch(self):
        """Test 5: Different source lengths in the same batch work correctly (unsorted order)."""
        batch_size = 4
        seq_len = 12
        dec_hidden = torch.randn(batch_size, self.hidden_dim)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        # Unsorted sequence lengths
        lengths = torch.tensor([4, 12, 1, 7], dtype=torch.long)

        context, weights = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)

        self.assertEqual(context.shape, (batch_size, self.hidden_dim))
        self.assertEqual(weights.shape, (batch_size, seq_len))

        # Check exact zero on padded positions for unsorted batch
        self.assertTrue(torch.all(weights[0, 4:] == 0.0))
        self.assertTrue(torch.all(weights[1, 12:] == 0.0))  # no padding
        self.assertTrue(torch.all(weights[2, 1:] == 0.0))   # only 1 valid token
        self.assertTrue(torch.all(weights[3, 7:] == 0.0))

        # Single-token item must assign weight 1.0 to its only token
        self.assertAlmostEqual(weights[2, 0].item(), 1.0, places=5)
        # And its context vector must equal that position's encoder output
        self.assertTrue(torch.allclose(context[2], enc_outputs[2, 0], atol=1e-5))

    def test_06_context_vector_equals_weighted_sum_of_encoder_outputs(self):
        """Test 6: Context vector equals the explicit weighted sum of encoder outputs."""
        batch_size = 3
        seq_len = 6
        dec_hidden = torch.randn(batch_size, self.hidden_dim)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim)
        lengths = torch.tensor([6, 4, 2], dtype=torch.long)

        context, weights = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)

        # Manually compute weighted sum for each batch element
        manual_context = torch.zeros(batch_size, self.hidden_dim)
        for b in range(batch_size):
            for i in range(seq_len):
                manual_context[b] += weights[b, i] * enc_outputs[b, i]

        self.assertTrue(
            torch.allclose(context, manual_context, atol=1e-5),
            f"Max difference between context and manual weighted sum: {torch.max(torch.abs(context - manual_context)).item()}",
        )

    def test_07_backward_pass_produces_gradients(self):
        """Test 7: Backward pass produces gradients for trainable attention parameters."""
        batch_size = 2
        seq_len = 5
        dec_hidden = torch.randn(batch_size, self.hidden_dim, requires_grad=True)
        enc_outputs = torch.randn(batch_size, seq_len, self.hidden_dim, requires_grad=True)
        lengths = torch.tensor([5, 3], dtype=torch.long)

        context, weights = self.attention(dec_hidden, enc_outputs, src_lengths=lengths)
        loss = context.sum()
        loss.backward()

        # Check gradients for attention parameters
        for name, param in self.attention.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter '{name}' did not receive a gradient.")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient of '{name}'.")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf in gradient of '{name}'.")
            self.assertFalse(torch.all(param.grad == 0.0), f"Zero gradient for '{name}'.")

        # Check gradients backpropagated to inputs
        self.assertIsNotNone(dec_hidden.grad)
        self.assertIsNotNone(enc_outputs.grad)

    def test_08_invalid_inputs_produce_clear_errors(self):
        """Test 8: Invalid dimensions, batch mismatches, and invalid lengths produce clear errors."""
        # 1. decoder_hidden wrong dimensions (1D or 4D)
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(10), torch.randn(2, 5, self.hidden_dim))
        self.assertIn("dimensions", str(ctx.exception))

        # 2. encoder_outputs wrong dimensions (2D or 4D)
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(2, self.hidden_dim), torch.randn(2, self.hidden_dim))
        self.assertIn("3D", str(ctx.exception))

        # 3. Batch size mismatch
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(3, self.hidden_dim), torch.randn(2, 5, self.hidden_dim))
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 4. Hidden dimension mismatch for decoder_hidden
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(2, 64), torch.randn(2, 5, self.hidden_dim))
        self.assertIn("dimension mismatch", str(ctx.exception))

        # 5. Hidden dimension mismatch for encoder_outputs
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(2, self.hidden_dim), torch.randn(2, 5, 64))
        self.assertIn("dimension mismatch", str(ctx.exception))

        # 6. Zero sequence length in encoder_outputs
        with self.assertRaises(ValueError) as ctx:
            self.attention(torch.randn(2, self.hidden_dim), torch.empty(2, 0, self.hidden_dim))
        self.assertIn("non-empty", str(ctx.exception))

        # 7. src_lengths mismatch
        with self.assertRaises(ValueError) as ctx:
            self.attention(
                torch.randn(2, self.hidden_dim),
                torch.randn(2, 5, self.hidden_dim),
                src_lengths=torch.tensor([5, 5, 5]),
            )
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 8. src_lengths exceeding sequence length
        with self.assertRaises(ValueError) as ctx:
            self.attention(
                torch.randn(2, self.hidden_dim),
                torch.randn(2, 5, self.hidden_dim),
                src_lengths=torch.tensor([5, 8]),
            )
        self.assertIn("exceeds", str(ctx.exception))

        # 9. Non-positive sequence length in src_lengths
        with self.assertRaises(ValueError) as ctx:
            self.attention(
                torch.randn(2, self.hidden_dim),
                torch.randn(2, 5, self.hidden_dim),
                src_lengths=torch.tensor([4, 0]),
            )
        self.assertIn("positive", str(ctx.exception))

        # 10. Non-tensor input
        with self.assertRaises(TypeError):
            self.attention("not_a_tensor", torch.randn(2, 5, self.hidden_dim))

        # 11. Constructor argument validation
        with self.assertRaises(ValueError):
            BahdanauAttention(hidden_dim=0)
        with self.assertRaises(ValueError):
            BahdanauAttention(hidden_dim=32, attention_dim=-5)

    def test_09_works_with_actual_encoder_outputs(self):
        """Test 9: Attention works seamlessly with actual Encoder outputs from the existing Encoder."""
        data_dir = PROJECT_ROOT / "data"
        train_loader, _, _, src_vocab, _ = create_all_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=0,
        )

        hidden_dim = 128
        attention_dim = 64

        encoder = Encoder(
            vocab=src_vocab,
            embedding_dim=64,
            hidden_dim=hidden_dim,
        )
        attention = BahdanauAttention(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        batch = next(iter(train_loader))
        src = batch["encoder_input_ids"]
        src_lengths = batch["encoder_lengths"]
        src_mask = batch["encoder_mask"]

        # Run Encoder forward
        enc_outputs, enc_hidden = encoder(src, src_lengths)
        batch_size, seq_len, _ = enc_outputs.shape

        self.assertEqual(enc_outputs.shape, (batch_size, seq_len, hidden_dim))
        self.assertEqual(enc_hidden.shape, (1, batch_size, hidden_dim))

        # Run Attention with src_lengths
        context_lens, weights_lens = attention(
            decoder_hidden=enc_hidden,
            encoder_outputs=enc_outputs,
            src_lengths=src_lengths,
        )
        self.assertEqual(context_lens.shape, (batch_size, hidden_dim))
        self.assertEqual(weights_lens.shape, (batch_size, seq_len))
        self.assertFalse(torch.isnan(context_lens).any())
        self.assertFalse(torch.isnan(weights_lens).any())

        # Run Attention with explicit mask
        context_mask, weights_mask = attention(
            decoder_hidden=enc_hidden,
            encoder_outputs=enc_outputs,
            mask=src_mask,
        )
        self.assertTrue(torch.allclose(weights_lens, weights_mask, atol=1e-5))
        self.assertTrue(torch.allclose(context_lens, context_mask, atol=1e-5))

        # Test end-to-end backward pass through both Attention and Encoder
        loss = context_lens.sum()
        loss.backward()

        self.assertIsNotNone(encoder.embedding.weight.grad)
        self.assertIsNotNone(attention.W_d.weight.grad)
        self.assertIsNotNone(attention.W_e.weight.grad)
        self.assertIsNotNone(attention.v.weight.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
