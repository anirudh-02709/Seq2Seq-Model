"""
Training Entry Point for NL2SQL Seq2Seq Model.

Loads:
- Training split (default: data/train.jsonl or explicit --train_file)
- Validation split (data/validation.jsonl)
Strictly isolates test and benchmark splits from training.

Builds:
- Vocabularies exclusively from training data
- Encoder, Decoder, and Seq2Seq with Bahdanau attention
- Trainer with CrossEntropyLoss and Adam optimizer

Saves:
- Best model checkpoint based on validation selection metric
- Epoch-by-epoch training metrics to history JSON
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import NL2SQLDataset, create_dataloader, load_jsonl_records, load_split
from src.data.tokenizer import NL2SQLTokenizer
from src.data.vocabulary import build_vocabularies
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NL2SQL Attention-based Seq2Seq Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset splits")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Optional explicit training JSONL path (defaults to data_dir/train.jsonl)",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default=None,
        help="Optional directory to save src_vocab.json and tgt_vocab.json artifacts",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="trained_model.pt",
        help="Filename for saved candidate checkpoint within checkpoint_dir",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default="checkpoints/training_history.json",
        help="Path to save training history JSON",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for source and target tokens")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden state dimension")
    parser.add_argument("--attention_dim", type=int, default=64, help="Bahdanau attention internal dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        default=0.50,
        help="Teacher forcing ratio during training (default: 0.50; set 1.0 for Phase 1 baseline)",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="val_exact_match",
        choices=["val_exact_match", "val_loss"],
        help="Primary validation metric for checkpoint selection (default: val_exact_match)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Epochs without validation improvement before early stopping (0 disables)",
    )
    parser.add_argument("--min_em_delta", type=float, default=1e-6, help="Minimum exact match improvement delta")
    parser.add_argument("--min_loss_delta", type=float, default=1e-4, help="Minimum val loss improvement delta")
    parser.add_argument("--eval_max_length", type=int, default=60, help="Max decoding length during greedy validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--allow_overwrite_baseline",
        action="store_true",
        help="Explicitly allow overwriting the protected Phase 2 model checkpoint (disabled by default)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)

    target_checkpoint = (Path(args.checkpoint_dir) / args.checkpoint_name).resolve()
    protected_checkpoints = {
        (PROJECT_ROOT / "checkpoints" / "candidates" / "phase2_clean_best.pt").resolve(),
    }
    if target_checkpoint in protected_checkpoints and not getattr(args, "allow_overwrite_baseline", False):
        raise ValueError(
            f"Refusing to overwrite protected checkpoint at '{target_checkpoint}'. "
            "Use a separate candidate path (e.g. checkpoints/trained_model.pt) "
            "or pass --allow_overwrite_baseline explicitly."
        )

    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("NL2SQL Attention-Based Seq2Seq Model Training")
    print("=" * 70)

    # 1. Load train and validation splits (NEVER touching test or benchmark splits)
    print("\n[1/5] Loading dataset splits...")
    train_file = getattr(args, "train_file", None)
    if train_file is not None:
        resolved_train = Path(train_file).resolve()
        forbidden_splits = {
            (data_dir / "validation.jsonl").resolve(),
            (data_dir / "test.jsonl").resolve(),
            (data_dir / "benchmark_eval.jsonl").resolve(),
        }
        if resolved_train in forbidden_splits:
            raise ValueError(f"Refusing to train on held-out split: {resolved_train}")
        train_recs = load_jsonl_records(resolved_train)
        print(f"  Using custom training file: {train_file}")
    else:
        train_recs = load_split("train", data_dir=data_dir)
    val_recs = load_split("validation", data_dir=data_dir)
    print(f"  Loaded Train records: {len(train_recs)}")
    print(f"  Loaded Validation records: {len(val_recs)}")
    print("  Test and benchmark splits strictly excluded from training.")

    # 2. Build vocabularies EXCLUSIVELY from training data
    print("\n[2/5] Building vocabularies from training data...")
    tokenizer = NL2SQLTokenizer()
    src_vocab, tgt_vocab = build_vocabularies(
        train_records=train_recs,
        tokenizer=tokenizer,
    )
    print(f"  Source vocabulary size (encoder input): {len(src_vocab)}")
    print(f"  Target vocabulary size (decoder SQL):  {len(tgt_vocab)}")
    vocab_dir = getattr(args, "vocab_dir", None)
    if vocab_dir is not None:
        vocab_path = Path(vocab_dir)
        vocab_path.mkdir(parents=True, exist_ok=True)
        src_vocab.save(vocab_path / "src_vocab.json")
        tgt_vocab.save(vocab_path / "tgt_vocab.json")
        print(f"  Saved vocabulary artifacts to: {vocab_path.resolve()}")

    # 3. Create DataLoaders
    train_dataset = NL2SQLDataset(
        train_recs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tokenizer=tokenizer,
    )
    val_dataset = NL2SQLDataset(
        val_recs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tokenizer=tokenizer,
    )

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Build Seq2Seq Architecture (unchanged)
    print("\n[3/5] Instantiating Seq2Seq architecture...")
    encoder = Encoder(
        vocab=src_vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    decoder = Decoder(
        vocab=tgt_vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_idx=tgt_vocab.pad_idx,
        sos_idx=tgt_vocab.sos_idx,
        eos_idx=tgt_vocab.eos_idx,
        unk_idx=tgt_vocab.unk_idx,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    config = {
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(tgt_vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "attention_dim": args.attention_dim,
        "dropout": args.dropout,
        "optimizer": "Adam",
        "lr": args.lr,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "clip_norm": args.clip_norm,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "selection_metric": args.selection_metric,
        "early_stopping_patience": patience,
        "min_em_delta": args.min_em_delta,
        "min_loss_delta": args.min_loss_delta,
        "eval_max_length": args.eval_max_length,
        "seed": args.seed,
    }
    if train_file is not None:
        config["train_file"] = str(train_file)

    # 5. Initialize Trainer and run
    print("\n[4/5] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_eval_loader=train_eval_loader,
        learning_rate=args.lr,
        clip_norm=args.clip_norm,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        history_path=args.history_path,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tokenizer=tokenizer,
        selection_metric=args.selection_metric,
        early_stopping_patience=patience,
        min_em_delta=args.min_em_delta,
        min_loss_delta=args.min_loss_delta,
        eval_max_length=args.eval_max_length,
        seed=args.seed,
        batch_size=args.batch_size,
        config=config,
    )
    print(f"  Target Device: {trainer.device}")
    print(f"  Teacher Forcing Ratio: {args.teacher_forcing_ratio:.2f}")
    print(f"  Selection Metric: {args.selection_metric} (tie-breaker: val_loss)")
    print(f"  Early Stopping Patience: {patience if patience is not None else 'disabled'}")
    print(f"  Target Checkpoint: {target_checkpoint}")
    print(f"  History JSON: {Path(args.history_path).resolve()}")

    print("\n[5/5] Commencing training and validation...")
    print("-" * 70)
    history = trainer.fit(epochs=args.epochs, print_progress=True)
    print("-" * 70)

    best_record = next(r for r in history if r["epoch"] == trainer.best_epoch)
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Completed Epochs:     {len(history)} / {args.epochs} (Early Stopped: {trainer.early_stopped})")
    print(f"  Best Epoch:           {trainer.best_epoch}")
    print(f"  Best Val EM:          {best_record['val_exact_match'] * 100:.2f}%")
    print(f"  Best Val Loss:        {best_record['val_loss']:.4f}")
    print(f"  Best Val Token Acc:   {best_record['val_token_acc'] * 100:.2f}%")
    if best_record.get("val_schema_compatibility") is not None:
        print(f"  Best Val Schema Comp: {best_record['val_schema_compatibility'] * 100:.2f}%")
    print(f"  Final Train Loss:     {history[-1]['train_loss']:.4f}")
    print(f"  Saved Checkpoint:     {target_checkpoint}")
    print(f"  Saved History:        {Path(args.history_path)}")
    print("=" * 70)

    return history


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
