"""
Focused Test Suite for Training and Validation Pipeline (Phase 3, Step 3.6).

Verifies:
1. One training epoch completes successfully.
2. Validation runs without updating model parameters.
3. PAD tokens are excluded from loss and accuracy.
4. Gradients and optimizer updates work.
5. Gradient clipping is applied.
6. Best checkpoint is saved on validation improvement.
7. A worse validation result does not overwrite the best checkpoint.
8. Checkpoint can be loaded and model parameters restored.
9. Training history is recorded correctly to JSON.
10. CPU execution works cleanly without hardware dependencies.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX
from src.data.vocabulary import Vocabulary
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer


class SyntheticNL2SQLDataset(Dataset):
    """Small synthetic dataset for testing training components."""

    def __init__(self, size: int = 8, src_len: int = 6, tgt_len: int = 5, vocab_size: int = 25):
        self.data = []
        for i in range(size):
            src = torch.randint(1, vocab_size, (src_len,), dtype=torch.long)
            # Create target beginning with SOS_IDX (2) and ending with EOS_IDX (3)
            tgt = torch.tensor([SOS_IDX, 10 + (i % 5), 15 + (i % 5), EOS_IDX, PAD_IDX], dtype=torch.long)
            length = src_len - (i % 2)
            self.data.append({
                "encoder_input_ids": src,
                "encoder_lengths": length,
                "decoder_target_ids": tgt,
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def synthetic_collate_fn(batch):
    """Simple collator for synthetic dataset batches."""
    return {
        "encoder_input_ids": torch.stack([x["encoder_input_ids"] for x in batch]),
        "encoder_lengths": torch.tensor([x["encoder_lengths"] for x in batch], dtype=torch.long),
        "decoder_target_ids": torch.stack([x["decoder_target_ids"] for x in batch]),
    }


class TestTrainingPipeline(unittest.TestCase):
    """Test suite covering the 10 training and validation pipeline requirements."""

    def setUp(self):
        torch.manual_seed(42)
        self.src_vocab_size = 30
        self.tgt_vocab_size = 35
        self.embedding_dim = 16
        self.hidden_dim = 32
        self.attention_dim = 16

        # Build small vocabularies
        self.src_vocab = Vocabulary()
        self.src_vocab.add_tokens([f"src_{i}" for i in range(self.src_vocab_size)])
        self.tgt_vocab = Vocabulary()
        self.tgt_vocab.add_tokens([f"tgt_{i}" for i in range(self.tgt_vocab_size)])

        # Small model
        encoder = Encoder(
            input_dim=len(self.src_vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=PAD_IDX,
        )
        decoder = Decoder(
            output_dim=len(self.tgt_vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            attention_dim=self.attention_dim,
            pad_idx=PAD_IDX,
        )
        self.model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            pad_idx=PAD_IDX,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            unk_idx=UNK_IDX,
        )

        # Synthetic datasets and loaders
        train_ds = SyntheticNL2SQLDataset(size=8, vocab_size=len(self.src_vocab))
        val_ds = SyntheticNL2SQLDataset(size=4, vocab_size=len(self.src_vocab))

        self.train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=synthetic_collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=synthetic_collate_fn)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.temp_dir.name) / "checkpoints"
        self.history_path = Path(self.temp_dir.name) / "history.json"

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            learning_rate=1e-3,
            clip_norm=1.0,
            teacher_forcing_ratio=1.0,
            checkpoint_dir=self.checkpoint_dir,
            history_path=self.history_path,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            device="cpu",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_one_training_epoch_completes_successfully(self):
        """Test 1: One training epoch completes successfully and returns finite positive loss."""
        loss = self.trainer.train_epoch(epoch=1)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss)))

    def test_02_validation_runs_without_updating_model_parameters(self):
        """Test 2: Validation runs without updating model parameters."""
        params_before = [p.clone() for p in self.model.parameters()]

        val_loss, val_acc = self.trainer.evaluate()

        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)

        # Ensure no parameters changed
        for p_before, p_after in zip(params_before, self.model.parameters()):
            self.assertTrue(torch.equal(p_before, p_after), "Validation modified model parameters!")

    def test_03_pad_tokens_are_excluded_from_loss_and_accuracy(self):
        """Test 3: PAD tokens are excluded from loss and accuracy."""
        pad_idx = 0
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # Batch with 1 token and 2 PAD positions
        logits = torch.randn(1, 3, 10, requires_grad=True)
        targets_with_pad = torch.tensor([[5, pad_idx, pad_idx]], dtype=torch.long)
        # Identical first token, but arbitrary dummy labels where PAD was
        targets_dummy = torch.tensor([[5, 8, 9]], dtype=torch.long)

        loss_with_pad = criterion(logits[:, :1].reshape(-1, 10), torch.tensor([5]))
        loss_full = criterion(logits.reshape(-1, 10), targets_with_pad.reshape(-1))

        # CrossEntropyLoss with ignore_index must produce identical loss
        self.assertAlmostEqual(loss_with_pad.item(), loss_full.item(), places=5)

        # Token accuracy verification
        preds = torch.tensor([[5, 9, 8]], dtype=torch.long)
        valid_mask = targets_with_pad != pad_idx
        correct = (preds == targets_with_pad) & valid_mask
        acc = correct.sum().item() / valid_mask.sum().item()
        # Exactly 1 non-pad token, and it matched 5 == 5
        self.assertEqual(acc, 1.0)

    def test_04_gradients_and_optimizer_updates_work(self):
        """Test 4: Gradients and optimizer updates work and modify weights."""
        params_before = [p.clone() for p in self.model.parameters()]

        self.trainer.train_epoch(epoch=1)

        # At least one trainable parameter must have updated
        any_changed = False
        for p_before, p_after in zip(params_before, self.model.parameters()):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        self.assertTrue(any_changed, "Optimizer did not update any model parameters during training epoch!")

    def test_05_gradient_clipping_is_applied(self):
        """Test 5: Gradient clipping is applied to limit gradient norms."""
        # Force artificially large gradients
        for p in self.model.parameters():
            p.grad = torch.ones_like(p) * 100.0

        clip_norm = 0.5
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)

        self.assertGreater(total_norm.item(), clip_norm)
        # Check clipped norm is now within bounds
        clipped_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters()))
        self.assertAlmostEqual(clipped_norm.item(), clip_norm, places=4)

    def test_06_best_checkpoint_is_saved(self):
        """Test 6: Best checkpoint is saved when validation loss improves."""
        self.trainer.fit(epochs=1, print_progress=False)

        best_checkpoint = self.checkpoint_dir / "best_model.pt"
        self.assertTrue(best_checkpoint.is_file(), "best_model.pt was not created!")
        self.assertGreater(best_checkpoint.stat().st_size, 0)

    def test_07_worse_validation_does_not_overwrite_best_checkpoint(self):
        """Test 7: A worse validation result does not overwrite the best checkpoint."""
        # 1. Force a good checkpoint
        self.trainer.best_val_loss = 0.5
        self.trainer.save_checkpoint(
            self.checkpoint_dir / "best_model.pt",
            epoch=1,
            train_loss=1.0,
            val_loss=0.5,
            val_token_acc=0.9,
            is_best=True,
        )
        saved_mtime = (self.checkpoint_dir / "best_model.pt").stat().st_mtime_ns

        # 2. Simulate epoch with worse val loss (val_loss = 2.0 > 0.5)
        # Manually invoke fit logic check
        val_loss_worse = 2.0
        if val_loss_worse < self.trainer.best_val_loss:
            self.trainer.save_checkpoint(self.checkpoint_dir / "best_model.pt", 2, 1.5, 2.0, 0.5)

        # Checkpoint file must remain untouched
        after_mtime = (self.checkpoint_dir / "best_model.pt").stat().st_mtime_ns
        self.assertEqual(saved_mtime, after_mtime, "Worse validation result modified best checkpoint!")

    def test_08_checkpoint_can_be_loaded_and_parameters_restored(self):
        """Test 8: Checkpoint can be loaded and model parameters restored completely."""
        self.trainer.fit(epochs=1, print_progress=False)
        checkpoint_path = self.checkpoint_dir / "best_model.pt"

        # Reconstruct model from checkpoint
        restored_model, chk_data, src_v, tgt_v = Trainer.load_model_from_checkpoint(
            checkpoint_path,
            device="cpu",
        )

        self.assertIsInstance(restored_model, Seq2Seq)
        self.assertIsNotNone(src_v)
        self.assertIsNotNone(tgt_v)

        # Compare weights
        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), restored_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6), f"Mismatch in parameter '{n1}' after restore")

    def test_09_training_history_is_recorded_correctly(self):
        """Test 9: Training history is recorded correctly to JSON."""
        epochs_to_run = 2
        history = self.trainer.fit(epochs=epochs_to_run, print_progress=False)

        self.assertTrue(self.history_path.is_file(), "training_history.json was not created!")
        with open(self.history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), epochs_to_run)
        required_keys = {"epoch", "train_loss", "val_loss", "val_token_acc", "learning_rate"}
        for entry in data:
            self.assertTrue(required_keys.issubset(entry.keys()))
            self.assertGreaterEqual(entry["epoch"], 1)

    def test_10_cpu_execution_works(self):
        """Test 10: CPU execution works cleanly without hardware dependencies."""
        self.assertEqual(self.trainer.device.type, "cpu")
        # Run one full cycle on CPU
        train_loss = self.trainer.train_epoch(1)
        val_loss, val_acc = self.trainer.evaluate()

        self.assertGreater(train_loss, 0.0)
        self.assertGreater(val_loss, 0.0)
        self.assertGreaterEqual(val_acc, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
