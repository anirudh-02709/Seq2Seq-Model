"""Consolidated tests for Seq2Seq Trainer, checkpoint selection, and training CLI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from scripts import train as train_script
from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX
from src.data.vocabulary import Vocabulary
from src.inference.pipeline import NL2SQLInferencePipeline
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer


class SyntheticTrainingDataset(Dataset):
    """Deterministic synthetic dataset for fast CPU Trainer tests."""

    def __init__(self, size: int = 8) -> None:
        self.items = [
            {
                "encoder_input_ids": torch.tensor([4, 5, 6 + (i % 3), 7, 8, 9], dtype=torch.long),
                "encoder_lengths": 6,
                "decoder_target_ids": torch.tensor([SOS_IDX, 5, 6 + (i % 3), EOS_IDX, PAD_IDX], dtype=torch.long),
            }
            for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_fn(batch):
    return {
        "encoder_input_ids": torch.stack([b["encoder_input_ids"] for b in batch]),
        "encoder_lengths": torch.tensor([b["encoder_lengths"] for b in batch], dtype=torch.long),
        "decoder_target_ids": torch.stack([b["decoder_target_ids"] for b in batch]),
    }


def build_tiny_model(seed: int = 42):
    torch.manual_seed(seed)
    sv = Vocabulary()
    sv.add_tokens([f"s_{i}" for i in range(16)])
    tv = Vocabulary()
    tv.add_tokens([f"t_{i}" for i in range(16)])
    enc = Encoder(vocab=sv, embedding_dim=12, hidden_dim=16, pad_idx=PAD_IDX)
    dec = Decoder(vocab=tv, embedding_dim=12, hidden_dim=16, attention_dim=12, pad_idx=PAD_IDX)
    return Seq2Seq(enc, dec, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX, unk_idx=UNK_IDX), sv, tv


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.chk_dir = Path(self.tmp_dir.name) / "checkpoints"
        self.hist_path = Path(self.tmp_dir.name) / "history.json"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _make_trainer(
        self,
        teacher_forcing_ratio: float = 0.50,
        size: int = 8,
        shuffle: bool = False,
        **kwargs,
    ) -> Trainer:
        model, sv, tv = build_tiny_model()
        ds = SyntheticTrainingDataset(size)
        tr_loader = DataLoader(ds, batch_size=4, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(SyntheticTrainingDataset(4), batch_size=4, shuffle=False, collate_fn=collate_fn)
        return Trainer(
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            teacher_forcing_ratio=teacher_forcing_ratio,
            checkpoint_dir=self.chk_dir,
            history_path=self.hist_path,
            src_vocab=sv,
            tgt_vocab=tv,
            device="cpu",
            **kwargs,
        )

    def test_cli_defaults_and_teacher_forcing_validation(self) -> None:
        with patch.object(sys, "argv", ["train.py"]):
            args = train_script.parse_args()
        self.assertEqual(args.teacher_forcing_ratio, 0.50)
        self.assertEqual(args.selection_metric, "val_exact_match")
        self.assertEqual(args.early_stopping_patience, 20)
        self.assertEqual(Path(args.checkpoint_dir), Path("checkpoints"))
        self.assertEqual(args.checkpoint_name, "trained_model.pt")
        self.assertEqual(Path(args.history_path), Path("checkpoints/training_history.json"))

        trainer = self._make_trainer()
        self.assertEqual(trainer.teacher_forcing_ratio, 0.50)
        self.assertEqual(trainer.criterion.ignore_index, PAD_IDX)

        for bad_tf, exc in ((-0.1, ValueError), (1.2, ValueError), (True, TypeError)):
            with self.assertRaises(exc):
                Trainer(
                    model=trainer.model,
                    train_loader=trainer.train_loader,
                    val_loader=trainer.val_loader,
                    teacher_forcing_ratio=bad_tf,  # type: ignore[arg-type]
                    device="cpu",
                )

    def test_epoch_updates_weights_clips_gradients_and_eval_preserves_weights(self) -> None:
        trainer = self._make_trainer(clip_norm=0.5)
        params_before = [p.clone() for p in trainer.model.parameters()]

        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as spy_clip:
            train_loss = trainer.train_epoch(epoch=1)
        self.assertGreater(train_loss, 0.0)
        self.assertFalse(torch.isnan(torch.tensor(train_loss)))
        self.assertGreaterEqual(spy_clip.call_count, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(params_before, trainer.model.parameters())))

        after_train = [p.clone() for p in trainer.model.parameters()]
        val_loss, val_acc = trainer.evaluate()
        self.assertGreater(val_loss, 0.0)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)
        for a, b in zip(after_train, trainer.model.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_teacher_forcing_schedule_during_train_and_eval(self) -> None:
        for ratio in (0.50, 1.0):
            trainer = self._make_trainer(teacher_forcing_ratio=ratio, size=4)
            recorded_tfs = []
            orig_forward = trainer.model.forward

            def spy_forward(*args, **kwargs):
                recorded_tfs.append(kwargs.get("teacher_forcing_ratio"))
                return orig_forward(*args, **kwargs)

            with patch.object(trainer.model, "forward", side_effect=spy_forward):
                trainer.train_epoch(epoch=1)
            self.assertEqual(recorded_tfs, [ratio])

            val_tfs = []
            with patch.object(
                trainer.model,
                "forward",
                side_effect=lambda *a, **kw: (val_tfs.append(kw.get("teacher_forcing_ratio")), orig_forward(*a, **kw))[1],
            ):
                trainer.evaluate()
            self.assertEqual(val_tfs, [1.0])

            with patch.object(trainer.model, "generate", wraps=trainer.model.generate) as spy_gen:
                trainer.evaluate_greedy(trainer.val_loader)
            self.assertEqual(spy_gen.call_count, 1)

    def test_seeded_reproducibility_and_train_eval_loader_isolation(self) -> None:
        def run_fit(ratio: float, with_train_eval: bool, shuffle: bool):
            torch.manual_seed(42)
            model, sv, tv = build_tiny_model(seed=42)
            ds = SyntheticTrainingDataset(8)
            tr_loader = DataLoader(ds, batch_size=4, shuffle=shuffle, collate_fn=collate_fn)
            val_loader = DataLoader(SyntheticTrainingDataset(4), batch_size=4, shuffle=False, collate_fn=collate_fn)
            tr_eval = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn) if with_train_eval else None
            tag = f"{ratio}_{with_train_eval}_{shuffle}"
            tr = Trainer(
                model=model,
                train_loader=tr_loader,
                val_loader=val_loader,
                train_eval_loader=tr_eval,
                teacher_forcing_ratio=ratio,
                checkpoint_dir=self.chk_dir / tag,
                history_path=self.chk_dir / f"hist_{tag}.json",
                src_vocab=sv,
                tgt_vocab=tv,
                seed=42,
                device="cpu",
            )
            return tr.fit(epochs=2, print_progress=False)

        self.assertEqual(run_fit(1.0, False, False), run_fit(1.0, False, False))
        self.assertEqual(run_fit(0.50, False, False), run_fit(0.50, False, False))

        h_obs = run_fit(0.50, True, True)
        h_no_obs = run_fit(0.50, False, True)
        for a, b in zip(h_obs, h_no_obs):
            self.assertEqual((a["train_loss"], a["val_loss"], a["val_exact_match"]), (b["train_loss"], b["val_loss"], b["val_exact_match"]))

    def test_checkpoint_selection_priority_and_non_overwrite(self) -> None:
        trainer = self._make_trainer(selection_metric="val_exact_match", min_em_delta=1e-6, min_loss_delta=1e-4)
        self.assertTrue(trainer._is_checkpoint_improvement(val_em=0.10, val_loss=2.0, epoch=1))
        trainer.best_val_em = 0.10
        trainer.best_val_loss = 2.0
        trainer.best_epoch = 1

        # Higher val_em wins; equal val_em breaks ties by lower val_loss; worse metrics rejected
        self.assertTrue(trainer._is_checkpoint_improvement(val_em=0.20, val_loss=2.1, epoch=2))
        self.assertTrue(trainer._is_checkpoint_improvement(val_em=0.10, val_loss=1.8, epoch=3))
        self.assertFalse(trainer._is_checkpoint_improvement(val_em=0.10, val_loss=2.0, epoch=4))
        self.assertFalse(trainer._is_checkpoint_improvement(val_em=0.05, val_loss=0.5, epoch=5))

    def test_early_stopping_restores_best_weights_and_saves_metadata(self) -> None:
        trainer = self._make_trainer(
            checkpoint_name="es_candidate.pt",
            early_stopping_patience=2,
            selection_metric="val_exact_match",
            seed=42,
            batch_size=4,
        )
        val_sequence = [(1.00, 0.50, 0.10), (0.80, 0.70, 0.50), (0.85, 0.65, 0.50), (0.90, 0.60, 0.25)]
        step_weights = {}
        idx_box = [0]

        def fake_train_epoch(epoch: int) -> float:
            with torch.no_grad():
                for p in trainer.model.parameters():
                    p.fill_(float(epoch) * 0.1)
            step_weights[epoch] = [p.clone() for p in trainer.model.parameters()]
            return 1.0 / epoch

        def fake_greedy(_loader):
            idx = idx_box[0]
            idx_box[0] += 1
            return {
                "exact_match": val_sequence[idx][2],
                "strict_exact_match": val_sequence[idx][2],
                "sql_validity_rate": 1.0,
                "schema_compatibility_rate": 1.0,
            }

        with (
            patch.object(trainer, "train_epoch", side_effect=fake_train_epoch),
            patch.object(trainer, "evaluate", side_effect=lambda: val_sequence[idx_box[0]][:2]),
            patch.object(trainer, "evaluate_greedy", side_effect=fake_greedy),
        ):
            history = trainer.fit(epochs=8, print_progress=False)

        self.assertTrue(trainer.early_stopped)
        self.assertEqual((trainer.stopped_epoch, trainer.best_epoch, len(history)), (4, 2, 4))
        for p_curr, p_ep2 in zip(trainer.model.parameters(), step_weights[2]):
            self.assertTrue(torch.allclose(p_curr, p_ep2))

        saved_path = self.chk_dir / "es_candidate.pt"
        chk_data = torch.load(saved_path, map_location="cpu", weights_only=False)
        self.assertEqual(chk_data["best_epoch"], 2)
        self.assertEqual(chk_data["completed_epochs"], 4)
        self.assertTrue(chk_data["early_stopping"]["early_stopped"])
        with open(self.hist_path, "r", encoding="utf-8") as f:
            hist_file = json.load(f)
        self.assertEqual(len(hist_file), 4)
        self.assertEqual(chk_data["training_history"], hist_file)

    def test_protected_checkpoint_overwrite_guard(self) -> None:
        args = SimpleNamespace(
            data_dir="data",
            train_file=None,
            vocab_dir=None,
            checkpoint_dir="checkpoints/candidates",
            checkpoint_name="phase2_clean_best.pt",
            history_path=str(self.hist_path),
            epochs=1,
            batch_size=16,
            embedding_dim=64,
            hidden_dim=128,
            attention_dim=64,
            dropout=0.1,
            lr=1e-3,
            clip_norm=1.0,
            teacher_forcing_ratio=0.50,
            selection_metric="val_exact_match",
            early_stopping_patience=20,
            min_em_delta=1e-6,
            min_loss_delta=1e-4,
            eval_max_length=60,
            seed=42,
            allow_overwrite_baseline=False,
        )
        with self.assertRaises(ValueError):
            train_script.run_training(args)

    def test_checkpoint_roundtrip_into_inference_pipeline(self) -> None:
        trainer = self._make_trainer(checkpoint_name="roundtrip.pt")
        trainer.fit(epochs=1, print_progress=False)

        chk_path = self.chk_dir / "roundtrip.pt"
        restored_model, _, src_v, tgt_v = Trainer.load_model_from_checkpoint(chk_path, device="cpu")
        self.assertEqual(len(src_v), len(trainer.src_vocab))
        self.assertEqual(len(tgt_v), len(trainer.tgt_vocab))
        for p1, p2 in zip(trainer.model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

        pipe = NL2SQLInferencePipeline.from_checkpoint(chk_path, device="cpu")
        with patch.object(pipe.model, "generate", wraps=pipe.model.generate) as spy_gen:
            _ = pipe.generate_sql("s_0 s_1", decoding_mode="greedy")
        self.assertEqual(spy_gen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
