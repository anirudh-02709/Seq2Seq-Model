"""Consolidated tests for Encoder, BahdanauAttention, Decoder, and Seq2Seq."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import create_all_dataloaders
from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX
from src.models.attention import BahdanauAttention
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq


class TestModelArchitecture(unittest.TestCase):
    """Verifies tensor contracts, masking, gradients, and integration across model modules."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.src_vocab_size = 50
        self.tgt_vocab_size = 60
        self.encoder = Encoder(
            input_dim=self.src_vocab_size,
            embedding_dim=16,
            hidden_dim=32,
            pad_idx=PAD_IDX,
            dropout=0.0,
        )
        self.attention = BahdanauAttention(hidden_dim=32, attention_dim=16)
        self.decoder = Decoder(
            output_dim=self.tgt_vocab_size,
            embedding_dim=16,
            hidden_dim=32,
            attention_dim=16,
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

    def test_encoder_shapes_packing_and_pad_zero_grad(self) -> None:
        # Unsorted variable-length batch: padded output steps must be zeroed
        src = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 0, 0, 0, 0],
                [30, 31, 32, 33, 0, 0],
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([6, 2, 4], dtype=torch.long)
        outputs, hidden = self.encoder(src, lengths)

        self.assertEqual(outputs.shape, (3, 6, 32))
        self.assertEqual(hidden.shape, (1, 3, 32))
        self.assertTrue(torch.all(outputs[1, 2:] == 0.0))
        self.assertTrue(torch.all(outputs[2, 4:] == 0.0))
        self.assertFalse(torch.all(outputs[1, :2] == 0.0))

        # Padding must not alter final hidden state vs unpadded run
        base = torch.tensor([[15, 25, 35]], dtype=torch.long)
        _, h_unpadded = self.encoder(base, torch.tensor([3]))
        padded = torch.tensor([[15, 25, 35, 0, 0, 0]], dtype=torch.long)
        _, h_padded = self.encoder(padded, torch.tensor([3]))
        self.assertTrue(torch.allclose(h_unpadded, h_padded, atol=1e-6))

        # Custom pad_idx keeps zero vector and accumulates zero gradients
        enc = Encoder(input_dim=20, embedding_dim=8, hidden_dim=16, pad_idx=3)
        self.assertTrue(torch.all(enc.embedding.weight[3] == 0.0))
        out_p, hid_p = enc(torch.tensor([[1, 2, 3, 3]]), torch.tensor([2]))
        (out_p.sum() + hid_p.sum()).backward()
        self.assertTrue(torch.all(enc.embedding.weight.grad[3] == 0.0))
        self.assertFalse(torch.all(enc.embedding.weight.grad[1] == 0.0))

    def test_attention_weights_masking_and_context_math(self) -> None:
        dec_2d = torch.randn(3, 32, requires_grad=True)
        enc_out = torch.randn(3, 6, 32, requires_grad=True)
        lengths = torch.tensor([6, 4, 1], dtype=torch.long)

        ctx, weights = self.attention(dec_2d, enc_out, src_lengths=lengths)
        self.assertEqual(ctx.shape, (3, 32))
        self.assertEqual(weights.shape, (3, 6))

        # Softmax sums to 1 and masks padding strictly to zero
        for i, valid_len in enumerate(lengths.tolist()):
            self.assertAlmostEqual(weights[i].sum().item(), 1.0, places=5)
            self.assertTrue(torch.all(weights[i, :valid_len] > 0.0))
            self.assertTrue(torch.all(weights[i, valid_len:] == 0.0))
        self.assertTrue(torch.allclose(ctx[2], enc_out[2, 0], atol=1e-5))

        # Explicit boolean mask and 3D hidden match src_lengths and 2D hidden
        mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        ctx_m, w_m = self.attention(dec_2d.unsqueeze(0), enc_out, mask=mask)
        self.assertTrue(torch.allclose(weights, w_m, atol=1e-6))
        self.assertTrue(torch.allclose(ctx, ctx_m, atol=1e-6))

        # Weighted sum identity and gradient flow
        manual_ctx = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        self.assertTrue(torch.allclose(ctx, manual_ctx, atol=1e-5))
        ctx.sum().backward()
        self.assertIsNotNone(dec_2d.grad)
        self.assertIsNotNone(enc_out.grad)

    def test_decoder_step_shapes_masking_and_pad_grad(self) -> None:
        for b_size in (1, 4):
            tok_1d = torch.randint(1, self.tgt_vocab_size, (b_size,))
            hidden = torch.randn(1, b_size, 32)
            enc_out = torch.randn(b_size, 5, 32)
            lengths = torch.full((b_size,), 3, dtype=torch.long)

            logits_1d, h1, w1 = self.decoder(tok_1d, hidden, enc_out, src_lengths=lengths)
            logits_2d, _, _ = self.decoder(tok_1d.unsqueeze(1), hidden, enc_out, src_lengths=lengths)
            self.assertEqual(logits_1d.shape, (b_size, self.tgt_vocab_size))
            self.assertEqual(h1.shape, (1, b_size, 32))
            self.assertEqual(w1.shape, (b_size, 5))
            self.assertTrue(torch.allclose(logits_1d, logits_2d, atol=1e-6))
            self.assertTrue(torch.all(w1[:, 3:] == 0.0))

        # Target pad_idx embedding keeps zero gradient
        dec = Decoder(output_dim=25, embedding_dim=8, hidden_dim=16, pad_idx=3)
        logits, new_h, _ = dec(torch.tensor([3, 1]), torch.randn(1, 2, 16), torch.randn(2, 4, 16))
        (logits.sum() + new_h.sum()).backward()
        self.assertTrue(torch.all(dec.embedding.weight.grad[3] == 0.0))
        self.assertFalse(torch.all(dec.embedding.weight.grad[1] == 0.0))

    def test_seq2seq_teacher_forcing_and_autoregressive_forward(self) -> None:
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])
        tgt_a = torch.tensor([[SOS_IDX, 8, 12, 16], [SOS_IDX, 9, 13, 17]])
        tgt_b = torch.tensor([[SOS_IDX, 40, 41, 42], [SOS_IDX, 45, 46, 47]])

        # Full teacher forcing matches manual ground-truth step decoding
        out_tf1 = self.model(src, lengths, tgt_a, teacher_forcing_ratio=1.0)
        self.assertEqual(out_tf1.shape, (2, 3, self.tgt_vocab_size))
        enc_out, hid = self.encoder(src, lengths)
        step_logits = []
        for t in range(tgt_a.size(1) - 1):
            logits, hid, _ = self.decoder(tgt_a[:, t], hid, enc_out, src_lengths=lengths)
            step_logits.append(logits)
        self.assertTrue(torch.allclose(out_tf1, torch.stack(step_logits, dim=1), atol=1e-5))

        # Zero teacher forcing ignores non-SOS target tokens
        out_tf0_a = self.model(src, lengths, tgt_a, teacher_forcing_ratio=0.0)
        out_tf0_b = self.model(src, lengths, tgt_b, teacher_forcing_ratio=0.0)
        self.assertTrue(torch.allclose(out_tf0_a, out_tf0_b, atol=1e-6))

    def test_seq2seq_greedy_eos_and_max_length(self) -> None:
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])

        mock_dec = Decoder(output_dim=self.tgt_vocab_size, embedding_dim=16, hidden_dim=32, attention_dim=16)
        mock_model = Seq2Seq(self.encoder, mock_dec)

        with torch.no_grad():
            mock_dec.fc_out.bias.fill_(-100.0)
            mock_dec.fc_out.bias[EOS_IDX] = 100.0
        gen_eos = mock_model.generate(src, lengths, max_length=20)
        self.assertEqual(gen_eos.shape, (2, 1))
        self.assertTrue(torch.all(gen_eos == EOS_IDX))

        with torch.no_grad():
            mock_dec.fc_out.bias.fill_(-100.0)
            mock_dec.fc_out.bias[10] = 100.0
        gen_cap = mock_model.generate(src, lengths, max_length=7)
        self.assertEqual(gen_cap.shape, (2, 7))

    def test_full_model_gradients_and_device_compatibility(self) -> None:
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])
        tgt = torch.randint(1, self.tgt_vocab_size, (2, 4))
        tgt[:, 0] = SOS_IDX

        outputs = self.model(src, lengths, tgt, teacher_forcing_ratio=0.5)
        outputs.sum().backward()
        for module in (self.model.encoder, self.model.decoder):
            for name, param in module.named_parameters():
                self.assertEqual(param.device.type, "cpu")
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in {name}")

    def test_invalid_inputs_and_dimension_validation(self) -> None:
        src = torch.randint(1, self.src_vocab_size, (2, 5))
        lengths = torch.tensor([5, 3])
        tgt = torch.randint(1, self.tgt_vocab_size, (2, 4))

        # Constructor guards
        with self.assertRaises(ValueError):
            Encoder(input_dim=None, vocab=None)
        with self.assertRaises(ValueError):
            BahdanauAttention(hidden_dim=0)
        with self.assertRaises(ValueError):
            Decoder(output_dim=None, vocab=None)
        with self.assertRaises(ValueError):
            Seq2Seq(self.encoder, Decoder(output_dim=20, embedding_dim=16, hidden_dim=64))

        # Forward/input guards
        bad_calls = [
            lambda: self.encoder(torch.tensor([1, 2, 3]), torch.tensor([3])),
            lambda: self.encoder([[1, 2, 3]], torch.tensor([3])),
            lambda: self.encoder(torch.ones(2, 4, dtype=torch.long), torch.tensor([4, 6])),
            lambda: self.encoder(torch.ones(2, 4, dtype=torch.long), torch.tensor([3, 0])),
            lambda: self.attention(torch.randn(10), torch.randn(2, 5, 32)),
            lambda: self.attention(torch.randn(2, 64), torch.randn(2, 5, 32)),
            lambda: self.decoder(torch.ones(2, 3, dtype=torch.long), torch.randn(1, 2, 32), torch.randn(2, 5, 32)),
            lambda: self.decoder(torch.tensor([1, 2, 3]), torch.randn(1, 2, 32), torch.randn(2, 5, 32)),
            lambda: self.model(src, lengths, torch.tensor([[SOS_IDX], [SOS_IDX]])),
            lambda: self.model(src, lengths, tgt, teacher_forcing_ratio=1.5),
            lambda: self.model.generate(src, lengths, max_length=0),
        ]
        for fn in bad_calls:
            with self.assertRaises((ValueError, TypeError)):
                fn()

    def test_real_dataloader_end_to_end_forward_and_backward(self) -> None:
        train_loader, val_loader, _, src_vocab, tgt_vocab = create_all_dataloaders(
            data_dir=PROJECT_ROOT / "data",
            batch_size=8,
            num_workers=0,
        )
        seq2seq = Seq2Seq(
            encoder=Encoder(vocab=src_vocab, embedding_dim=64, hidden_dim=128),
            decoder=Decoder(vocab=tgt_vocab, embedding_dim=64, hidden_dim=128, attention_dim=64),
            pad_idx=tgt_vocab.pad_idx,
            sos_idx=tgt_vocab.sos_idx,
            eos_idx=tgt_vocab.eos_idx,
            unk_idx=tgt_vocab.unk_idx,
        )

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        outputs = seq2seq(
            train_batch["encoder_input_ids"],
            train_batch["encoder_lengths"],
            train_batch["decoder_target_ids"],
            teacher_forcing_ratio=0.5,
        )
        b_sz, tgt_len = train_batch["decoder_target_ids"].shape
        self.assertEqual(outputs.shape, (b_sz, tgt_len - 1, len(tgt_vocab)))
        self.assertFalse(torch.isnan(outputs).any())

        with torch.no_grad():
            val_out = seq2seq(
                val_batch["encoder_input_ids"],
                val_batch["encoder_lengths"],
                val_batch["decoder_target_ids"],
                teacher_forcing_ratio=1.0,
            )
            self.assertFalse(torch.isnan(val_out).any())

        generated = seq2seq.generate(train_batch["encoder_input_ids"], train_batch["encoder_lengths"], max_length=25)
        self.assertEqual(generated.size(0), b_sz)
        self.assertIsInstance(tgt_vocab.to_tokens(generated[0].tolist(), remove_special=True), list)

        outputs.sum().backward()
        self.assertIsNotNone(seq2seq.encoder.embedding.weight.grad)
        self.assertIsNotNone(seq2seq.decoder.fc_out.weight.grad)


if __name__ == "__main__":
    unittest.main()
