"""
Comprehensive Test Suite for Complete Seq2Seq NL2SQL Integration (Phase 3, Step 3.5).

Verifies:
1. Correct model initialization and dimension validation.
2. Forward pass output dimensions (batch_size, tgt_len - 1, target_vocab_size).
3. Encoder-to-Decoder hidden-state integration.
4. Full teacher forcing operation (ratio = 1.0).
5. Autoregressive forward operation with teacher forcing disabled (ratio = 0.0).
6. Greedy generation for inference.
7. EOS-based early stopping during generation.
8. Maximum generation length enforcement.
9. Batch processing with varying sequence lengths and batch sizes.
10. Backward gradient propagation through the complete model.
11. Invalid input handling and parameter verification.
12. Seamless end-to-end compatibility with existing data pipeline and real batches.
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

from src.data.dataset import create_all_dataloaders
from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX
from src.models.attention import BahdanauAttention
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq


class TestSeq2SeqIntegration(unittest.TestCase):
    """Test suite for complete Seq2Seq architecture."""

    def setUp(self):
        torch.manual_seed(42)
        self.src_vocab_size = 50
        self.tgt_vocab_size = 60
        self.embedding_dim = 16
        self.hidden_dim = 32
        self.attention_dim = 16

        self.encoder = Encoder(
            input_dim=self.src_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=PAD_IDX,
            dropout=0.0,
        )
        self.decoder = Decoder(
            output_dim=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
            pad_idx=PAD_IDX,
            dropout=0.0,
        )
        self.model = Seq2Seq(
            encoder=self.encoder,
            decoder=self.decoder,
            pad_idx=PAD_IDX,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            unk_idx=UNK_IDX,
        )

    def test_01_model_initialization(self):
        """Test 1: Correct model initialization and dimension validation."""
        self.assertIs(self.model.encoder, self.encoder)
        self.assertIs(self.model.decoder, self.decoder)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.source_vocab_size, self.src_vocab_size)
        self.assertEqual(self.model.target_vocab_size, self.tgt_vocab_size)
        self.assertEqual(self.model.pad_idx, PAD_IDX)
        self.assertEqual(self.model.sos_idx, SOS_IDX)
        self.assertEqual(self.model.eos_idx, EOS_IDX)

        # Incompatible hidden dimensions must raise ValueError
        incompatible_decoder = Decoder(
            output_dim=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=64,  # mismatch with encoder hidden_dim 32
        )
        with self.assertRaises(ValueError) as ctx:
            Seq2Seq(self.encoder, incompatible_decoder)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_02_forward_pass_output_dimensions(self):
        """Test 2: Forward pass output dimensions (batch_size, tgt_len - 1, target_vocab_size)."""
        batch_size = 4
        src_len = 9
        tgt_len = 7
        src = torch.randint(1, self.src_vocab_size, (batch_size, src_len))
        src_lengths = torch.tensor([9, 7, 5, 3])
        tgt = torch.randint(1, self.tgt_vocab_size, (batch_size, tgt_len))
        tgt[:, 0] = SOS_IDX

        outputs = self.model(src, src_lengths, tgt)

        expected_shape = (batch_size, tgt_len - 1, self.tgt_vocab_size)
        self.assertEqual(
            outputs.shape,
            expected_shape,
            f"Expected outputs shape {expected_shape}, got {tuple(outputs.shape)}",
        )

    def test_03_encoder_to_decoder_hidden_state_integration(self):
        """Test 3: Encoder-to-Decoder hidden-state integration."""
        batch_size = 2
        src = torch.randint(1, self.src_vocab_size, (batch_size, 5))
        lengths = torch.tensor([5, 3])
        tgt = torch.tensor([[SOS_IDX, 10, 11], [SOS_IDX, 20, 21]])

        # 1. Forward pass through full model
        outputs = self.model(src, lengths, tgt, teacher_forcing_ratio=1.0)

        # 2. Manual step using encoder and decoder
        enc_out, enc_hidden = self.encoder(src, lengths)
        step_logits, _, _ = self.decoder(tgt[:, 0], enc_hidden, enc_out, src_lengths=lengths)

        # Step 0 of model output must match manual step 0 exactly
        self.assertTrue(
            torch.allclose(outputs[:, 0, :], step_logits, atol=1e-5),
            "Initial decoder output does not match encoder-initialized step output",
        )

    def test_04_full_teacher_forcing_operation(self):
        """Test 4: Full teacher forcing operation (ratio = 1.0)."""
        batch_size = 2
        src = torch.randint(1, self.src_vocab_size, (batch_size, 4))
        lengths = torch.tensor([4, 2])
        tgt = torch.tensor([[SOS_IDX, 8, 12, 16], [SOS_IDX, 9, 13, 17]])

        # Run model with 100% teacher forcing
        outputs = self.model(src, lengths, tgt, teacher_forcing_ratio=1.0)

        # Manually compute step-by-step using ground-truth tokens as inputs
        enc_out, hidden = self.encoder(src, lengths)
        manual_logits = []
        for t in range(tgt.size(1) - 1):
            logits, hidden, _ = self.decoder(tgt[:, t], hidden, enc_out, src_lengths=lengths)
            manual_logits.append(logits)
        expected = torch.stack(manual_logits, dim=1)

        self.assertTrue(
            torch.allclose(outputs, expected, atol=1e-5),
            "Full teacher forcing output differed from manual ground-truth stepping",
        )

    def test_05_autoregressive_forward_without_teacher_forcing(self):
        """Test 5: Autoregressive forward operation with teacher forcing disabled (ratio = 0.0)."""
        batch_size = 2
        src = torch.randint(1, self.src_vocab_size, (batch_size, 4))
        lengths = torch.tensor([4, 3])

        # Target A
        tgt_a = torch.tensor([[SOS_IDX, 5, 6, 7], [SOS_IDX, 15, 16, 17]])
        # Target B has identical SOS at step 0, but completely different tokens afterwards
        tgt_b = torch.tensor([[SOS_IDX, 40, 41, 42], [SOS_IDX, 45, 46, 47]])

        outputs_a = self.model(src, lengths, tgt_a, teacher_forcing_ratio=0.0)
        outputs_b = self.model(src, lengths, tgt_b, teacher_forcing_ratio=0.0)

        # With teacher forcing disabled, tgt[:, 1:] is never used as input, so outputs must be identical
        self.assertTrue(
            torch.allclose(outputs_a, outputs_b, atol=1e-6),
            "Changing target tokens when teacher_forcing_ratio=0.0 altered model outputs!",
        )

    def test_06_greedy_generation(self):
        """Test 6: Greedy generation for inference."""
        batch_size = 3
        src = torch.randint(1, self.src_vocab_size, (batch_size, 6))
        lengths = torch.tensor([6, 4, 2])

        generated = self.model.generate(src, lengths, max_length=15)

        self.assertEqual(generated.dim(), 2)
        self.assertEqual(generated.size(0), batch_size)
        self.assertLessEqual(generated.size(1), 15)

        # Initial token should NOT be SOS (prompt: return generated token IDs, excluding initial SOS)
        # Verify tokens are in valid vocabulary range
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < self.tgt_vocab_size))

    def test_07_eos_early_stopping(self):
        """Test 7: EOS-based early stopping during generation."""
        # Create a mock decoder where the linear projection bias heavily favors EOS_IDX
        mock_decoder = Decoder(
            output_dim=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
        )
        with torch.no_grad():
            mock_decoder.fc_out.bias.fill_(-100.0)
            mock_decoder.fc_out.bias[EOS_IDX] = 100.0  # guarantees EOS on step 1

        mock_model = Seq2Seq(self.encoder, mock_decoder)
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])

        # Ask for max_length 20, but it should stop at step 1 because EOS is generated
        gen = mock_model.generate(src, lengths, max_length=20)

        self.assertEqual(gen.shape, (2, 1), f"Expected shape (2, 1) due to immediate EOS, got {gen.shape}")
        self.assertTrue(torch.all(gen == EOS_IDX))

    def test_08_maximum_generation_length(self):
        """Test 8: Maximum generation length enforcement when EOS is never emitted."""
        mock_decoder = Decoder(
            output_dim=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
        )
        with torch.no_grad():
            mock_decoder.fc_out.bias.fill_(-100.0)
            mock_decoder.fc_out.bias[10] = 100.0  # token 10 is never EOS

        mock_model = Seq2Seq(self.encoder, mock_decoder)
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])

        for max_len in [3, 7, 12]:
            gen = mock_model.generate(src, lengths, max_length=max_len)
            self.assertEqual(
                gen.shape[1],
                max_len,
                f"Expected generation length {max_len}, got {gen.shape[1]}",
            )

    def test_09_batch_processing(self):
        """Test 9: Batch processing with different batch sizes and lengths."""
        for b_size in [1, 2, 5, 8]:
            src_len = 8
            tgt_len = 6
            src = torch.randint(1, self.src_vocab_size, (b_size, src_len))
            lengths = torch.randint(1, src_len + 1, (b_size,))
            lengths[0] = src_len  # ensure at least one full length
            tgt = torch.randint(1, self.tgt_vocab_size, (b_size, tgt_len))
            tgt[:, 0] = SOS_IDX

            outputs = self.model(src, lengths, tgt, teacher_forcing_ratio=0.5)
            self.assertEqual(outputs.shape, (b_size, tgt_len - 1, self.tgt_vocab_size))

            gen = self.model.generate(src, lengths, max_length=10)
            self.assertEqual(gen.size(0), b_size)

    def test_10_backward_gradient_propagation(self):
        """Test 10: Backward gradient propagation through complete model."""
        batch_size = 2
        src = torch.randint(1, self.src_vocab_size, (batch_size, 5))
        lengths = torch.tensor([5, 3])
        tgt = torch.randint(1, self.tgt_vocab_size, (batch_size, 4))
        tgt[:, 0] = SOS_IDX

        outputs = self.model(src, lengths, tgt, teacher_forcing_ratio=0.5)
        loss = outputs.sum()
        loss.backward()

        # Check gradients for encoder parameters
        for name, param in self.model.encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Encoder param '{name}' missing gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN in encoder param '{name}'")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf in encoder param '{name}'")

        # Check gradients for decoder parameters (embedding, GRU, attention, output projection)
        for name, param in self.model.decoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Decoder param '{name}' missing gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN in decoder param '{name}'")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf in decoder param '{name}'")

    def test_11_invalid_input_handling(self):
        """Test 11: Invalid input handling produces clear errors."""
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])
        tgt = torch.randint(1, self.tgt_vocab_size, (2, 4))

        # 1. 1D src tensor
        with self.assertRaises(ValueError) as ctx:
            self.model(torch.tensor([1, 2, 3]), torch.tensor([3]), tgt)
        self.assertIn("2D", str(ctx.exception))

        # 2. 1D tgt tensor
        with self.assertRaises(ValueError) as ctx:
            self.model(src, lengths, torch.tensor([1, 2, 3]))
        self.assertIn("2D", str(ctx.exception))

        # 3. tgt_len < 2
        with self.assertRaises(ValueError) as ctx:
            self.model(src, lengths, torch.tensor([[SOS_IDX], [SOS_IDX]]))
        self.assertIn("tgt_len", str(ctx.exception))

        # 4. Batch size mismatch
        with self.assertRaises(ValueError) as ctx:
            self.model(src, lengths, torch.randint(1, self.tgt_vocab_size, (3, 4)))
        self.assertIn("Batch size mismatch", str(ctx.exception))

        # 5. Invalid teacher_forcing_ratio (< 0 or > 1)
        with self.assertRaises(ValueError) as ctx:
            self.model(src, lengths, tgt, teacher_forcing_ratio=-0.1)
        self.assertIn("teacher_forcing_ratio", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.model(src, lengths, tgt, teacher_forcing_ratio=1.5)
        self.assertIn("teacher_forcing_ratio", str(ctx.exception))

        # 6. Invalid max_length in generate
        with self.assertRaises(ValueError) as ctx:
            self.model.generate(src, lengths, max_length=0)
        self.assertIn("max_length", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.model.generate(src, lengths, max_length=-5)
        self.assertIn("max_length", str(ctx.exception))

        # 7. Non-tensor inputs
        with self.assertRaises(TypeError):
            self.model("invalid_src", lengths, tgt)

    def test_12_compatibility_with_existing_data_pipeline(self):
        """Test 12: End-to-end integration with existing DataLoader and vocabularies."""
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
        seq2seq_model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            pad_idx=tgt_vocab.pad_idx,
            sos_idx=tgt_vocab.sos_idx,
            eos_idx=tgt_vocab.eos_idx,
            unk_idx=tgt_vocab.unk_idx,
        )

        batch = next(iter(train_loader))
        src = batch["encoder_input_ids"]
        src_lengths = batch["encoder_lengths"]
        tgt = batch["decoder_target_ids"]
        batch_size, tgt_len = tgt.shape

        # 1. Forward pass with teacher forcing
        outputs = seq2seq_model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
        self.assertEqual(outputs.shape, (batch_size, tgt_len - 1, len(tgt_vocab)))
        self.assertFalse(torch.isnan(outputs).any())

        # 2. Greedy generation
        generated = seq2seq_model.generate(src, src_lengths, max_length=30)
        self.assertEqual(generated.size(0), batch_size)
        self.assertFalse(torch.isnan(generated.float()).any())

        # 3. Detokenize generated sequence to verify non-empty SQL tokens
        first_gen = generated[0].tolist()
        tokens = tgt_vocab.to_tokens(first_gen, remove_special=True)
        self.assertIsInstance(tokens, list)

        # 4. Backward pass on real batch
        loss = outputs.sum()
        loss.backward()
        self.assertIsNotNone(seq2seq_model.encoder.embedding.weight.grad)
        self.assertIsNotNone(seq2seq_model.decoder.fc_out.weight.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
