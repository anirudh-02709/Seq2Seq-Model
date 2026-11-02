"""
Focused Test Suite for NL2SQL Seq2Seq Encoder (Phase 3, Step 3.2).

Verifies:
1. Correct output tensor dimensions (batch_size, source_sequence_length, hidden_dim).
2. Correct hidden-state dimensions (1, batch_size, hidden_dim).
3. Different sequence lengths within the same batch (unsorted, variable padding).
4. Padding positions do not affect the final hidden state.
5. PAD embedding uses the correct padding index and zero gradient behavior.
6. Forward pass works with a real batch produced by the existing DataLoader.
7. Model parameters receive gradients during a backward-pass smoke test.
8. Invalid input dimensions or invalid sequence lengths produce clear errors.
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
import torch.nn as nn

from src.data.tokenizer import PAD_IDX
from src.data.vocabulary import Vocabulary
from src.data.dataset import create_all_dataloaders
from src.models.encoder import Encoder


class TestEncoder(unittest.TestCase):
    """Unit and integration test cases for Encoder."""

    def setUp(self):
        # Set manual seed for reproducibility
        torch.manual_seed(42)
        self.input_dim = 100
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.pad_idx = 0
        self.encoder = Encoder(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=self.pad_idx,
            dropout=0.0,
        )

    def test_01_output_tensor_dimensions(self):
        """Test 1: Correct output tensor dimensions (batch_size, seq_len, hidden_dim)."""
        batch_size = 4
        seq_len = 10
        src = torch.randint(1, self.input_dim, (batch_size, seq_len), dtype=torch.long)
        lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long)

        outputs, _ = self.encoder(src, lengths)

        self.assertEqual(
            outputs.shape,
            (batch_size, seq_len, self.hidden_dim),
            f"Expected outputs shape {(batch_size, seq_len, self.hidden_dim)}, got {tuple(outputs.shape)}",
        )

    def test_02_hidden_state_dimensions(self):
        """Test 2: Correct hidden-state dimensions (1, batch_size, hidden_dim)."""
        test_cases = [(1, 8), (3, 16), (5, 64)]
        for b_size, h_dim in test_cases:
            enc = Encoder(
                input_dim=self.input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=h_dim,
                pad_idx=self.pad_idx,
            )
            src = torch.randint(1, self.input_dim, (b_size, 7), dtype=torch.long)
            lengths = torch.tensor([7] * b_size, dtype=torch.long)

            _, hidden = enc(src, lengths)

            self.assertEqual(
                hidden.shape,
                (1, b_size, h_dim),
                f"Expected hidden shape (1, {b_size}, {h_dim}), got {tuple(hidden.shape)}",
            )

    def test_03_different_sequence_lengths_within_batch(self):
        """Test 3: Different sequence lengths within the same batch (variable-length, unsorted)."""
        # Batch of 4 with lengths: 8, 4, 2, 6 (unsorted order)
        seq_len = 8
        src = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],  # length 8
                [20, 21, 22, 23, 0, 0, 0, 0],       # length 4
                [30, 31, 0, 0, 0, 0, 0, 0],          # length 2
                [40, 41, 42, 43, 44, 45, 0, 0],     # length 6
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([8, 4, 2, 6], dtype=torch.long)

        outputs, hidden = self.encoder(src, lengths)

        self.assertEqual(outputs.shape, (4, seq_len, self.hidden_dim))
        self.assertEqual(hidden.shape, (1, 4, self.hidden_dim))

        # Padded positions in outputs must be zeroed out
        self.assertTrue(torch.all(outputs[1, 4:] == 0.0), "Padded steps for item 1 not zeroed out")
        self.assertTrue(torch.all(outputs[2, 2:] == 0.0), "Padded steps for item 2 not zeroed out")
        self.assertTrue(torch.all(outputs[3, 6:] == 0.0), "Padded steps for item 3 not zeroed out")

        # Valid positions must be non-zero
        self.assertFalse(torch.all(outputs[1, :4] == 0.0), "Valid steps for item 1 should be non-zero")
        self.assertFalse(torch.all(outputs[2, :2] == 0.0), "Valid steps for item 2 should be non-zero")
        self.assertFalse(torch.all(outputs[3, :6] == 0.0), "Valid steps for item 3 should be non-zero")

    def test_04_padding_does_not_affect_final_hidden_state(self):
        """Test 4: Padding positions do not affect the final hidden state."""
        # Define a base sequence of tokens: length 3
        base_tokens = [15, 25, 35]

        # Case A: Sequence run alone with length 3
        src_unpadded = torch.tensor([base_tokens], dtype=torch.long)
        lengths_unpadded = torch.tensor([3], dtype=torch.long)
        _, hidden_unpadded = self.encoder(src_unpadded, lengths_unpadded)

        # Case B: The exact same sequence padded with PAD tokens up to length 9
        src_padded = torch.tensor([base_tokens + [0, 0, 0, 0, 0, 0]], dtype=torch.long)
        lengths_padded = torch.tensor([3], dtype=torch.long)
        _, hidden_padded = self.encoder(src_padded, lengths_padded)

        # Final hidden states must be identical
        self.assertTrue(
            torch.allclose(hidden_unpadded, hidden_padded, atol=1e-6),
            f"Hidden state differed when padded: max diff = {torch.max(torch.abs(hidden_unpadded - hidden_padded)).item()}",
        )

        # Case C: Same sequence embedded in a multi-item batch alongside a longer item
        longer_seq = [5, 6, 7, 8, 9, 10, 11, 12, 13]
        multi_batch = torch.tensor(
            [
                longer_seq,
                base_tokens + [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        multi_lengths = torch.tensor([9, 3], dtype=torch.long)
        _, hidden_multi = self.encoder(multi_batch, multi_lengths)

        self.assertTrue(
            torch.allclose(hidden_unpadded[0, 0], hidden_multi[0, 1], atol=1e-6),
            "Hidden state of sequence differed when batched with longer sequence",
        )

    def test_05_pad_embedding_uses_correct_padding_index(self):
        """Test 5: PAD embedding uses correct padding index and zero gradient behavior."""
        custom_pad_idx = 3
        enc = Encoder(
            input_dim=20,
            embedding_dim=16,
            hidden_dim=32,
            pad_idx=custom_pad_idx,
        )

        # 1. Padding index is correctly stored and configured in embedding layer
        self.assertEqual(enc.pad_idx, custom_pad_idx)
        self.assertEqual(enc.embedding.padding_idx, custom_pad_idx)

        # 2. Embedding vector at padding_idx is initialized to zeros
        pad_vector = enc.embedding.weight[custom_pad_idx]
        self.assertTrue(
            torch.all(pad_vector == 0.0),
            f"Embedding at padding_idx should be zeros, got {pad_vector}",
        )

        # 3. PAD token embedding does not accumulate gradients during backward pass
        src = torch.tensor([[1, 2, custom_pad_idx, custom_pad_idx]], dtype=torch.long)
        lengths = torch.tensor([2], dtype=torch.long)
        outputs, hidden = enc(src, lengths)
        loss = outputs.sum() + hidden.sum()
        loss.backward()

        self.assertIsNotNone(enc.embedding.weight.grad)
        pad_grad = enc.embedding.weight.grad[custom_pad_idx]
        self.assertTrue(
            torch.all(pad_grad == 0.0),
            f"Expected zero gradient at padding_idx, got {pad_grad}",
        )

        # Non-padding indices present in the sequence DO receive non-zero gradients
        token_grad = enc.embedding.weight.grad[1]
        self.assertFalse(torch.all(token_grad == 0.0), "Expected non-zero gradient for token ID 1")

    def test_06_forward_pass_with_existing_dataloader_batch(self):
        """Test 6: Forward pass works with a batch produced by the existing DataLoader."""
        data_dir = PROJECT_ROOT / "data"
        train_loader, val_loader, _, src_vocab, _ = create_all_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=0,
        )

        # Initialize Encoder using the source Vocabulary object
        enc = Encoder(
            vocab=src_vocab,
            embedding_dim=64,
            hidden_dim=128,
            dropout=0.1,
        )
        enc.eval()

        # Check vocab-derived attributes
        self.assertEqual(enc.input_dim, len(src_vocab))
        self.assertEqual(enc.pad_idx, src_vocab.pad_idx)

        # 1. Test on train_loader batch
        train_batch = next(iter(train_loader))
        enc_input_ids = train_batch["encoder_input_ids"]
        enc_lengths = train_batch["encoder_lengths"]

        train_outputs, train_hidden = enc(enc_input_ids, enc_lengths)

        batch_size, max_seq_len = enc_input_ids.shape
        self.assertEqual(train_outputs.shape, (batch_size, max_seq_len, 128))
        self.assertEqual(train_hidden.shape, (1, batch_size, 128))
        self.assertFalse(torch.isnan(train_outputs).any())
        self.assertFalse(torch.isnan(train_hidden).any())

        # 2. Test on val_loader batch
        val_batch = next(iter(val_loader))
        val_outputs, val_hidden = enc(val_batch["encoder_input_ids"], val_batch["encoder_lengths"])
        val_batch_size, val_max_seq_len = val_batch["encoder_input_ids"].shape
        self.assertEqual(val_outputs.shape, (val_batch_size, val_max_seq_len, 128))
        self.assertEqual(val_hidden.shape, (1, val_batch_size, 128))

    def test_07_backward_pass_smoke_test_gradients(self):
        """Test 7: Model parameters receive gradients during a backward-pass smoke test."""
        enc = Encoder(
            input_dim=50,
            embedding_dim=16,
            hidden_dim=32,
            pad_idx=self.pad_idx,
            dropout=0.0,
        )
        src = torch.tensor([[5, 12, 8, 0], [4, 9, 2, 7]], dtype=torch.long)
        lengths = torch.tensor([3, 4], dtype=torch.long)

        outputs, hidden = enc(src, lengths)
        loss = outputs.sum() + hidden.sum()
        loss.backward()

        # Verify all trainable parameters have gradients populated
        param_count = 0
        for name, param in enc.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter '{name}' did not receive a gradient")
            self.assertFalse(
                torch.isnan(param.grad).any(),
                f"Parameter '{name}' gradient contains NaN",
            )
            self.assertFalse(
                torch.isinf(param.grad).any(),
                f"Parameter '{name}' gradient contains Inf",
            )
            param_count += 1

        self.assertGreaterEqual(param_count, 5, "Expected at least 5 parameter tensors (emb + GRU weights/biases)")

    def test_08_invalid_input_dimensions_and_lengths(self):
        """Test 8: Invalid input dimensions or invalid sequence lengths produce clear errors."""
        # 1. 1D src tensor
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([3]))
        self.assertIn("2D", str(ctx.exception))

        # 2. 3D src tensor
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(2, 3, 4, dtype=torch.long), torch.tensor([3, 3]))
        self.assertIn("2D", str(ctx.exception))

        # 3. Empty src tensor
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.empty(0, 5, dtype=torch.long), torch.empty(0, dtype=torch.long))
        self.assertIn("non-empty", str(ctx.exception))

        # 4. Non-tensor src
        with self.assertRaises(TypeError) as ctx:
            self.encoder([[1, 2, 3]], torch.tensor([3]))
        self.assertIn("torch.Tensor", str(ctx.exception))

        # 5. Batch size mismatch
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(3, 5, dtype=torch.long), torch.tensor([5, 5]))
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 6. 2D src_lengths
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(2, 5, dtype=torch.long), torch.ones(2, 1, dtype=torch.long))
        self.assertIn("1D", str(ctx.exception))

        # 7. Sequence length exceeds source sequence dimension
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(2, 4, dtype=torch.long), torch.tensor([4, 6], dtype=torch.long))
        self.assertIn("exceeds", str(ctx.exception))

        # 8. Non-positive sequence length (<= 0)
        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(2, 4, dtype=torch.long), torch.tensor([3, 0], dtype=torch.long))
        self.assertIn("positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.encoder(torch.ones(2, 4, dtype=torch.long), torch.tensor([3, -1], dtype=torch.long))
        self.assertIn("positive", str(ctx.exception))

        # 9. Constructor argument validation
        with self.assertRaises(ValueError):
            Encoder(input_dim=None, vocab=None)  # neither input_dim nor vocab

        with self.assertRaises(ValueError):
            Encoder(input_dim=-5, embedding_dim=16, hidden_dim=32)

        with self.assertRaises(ValueError):
            Encoder(input_dim=50, embedding_dim=0, hidden_dim=32)

        with self.assertRaises(ValueError):
            Encoder(input_dim=50, embedding_dim=16, hidden_dim=-1)

        with self.assertRaises(ValueError):
            Encoder(input_dim=50, embedding_dim=16, hidden_dim=32, dropout=1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
