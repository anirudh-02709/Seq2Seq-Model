"""
Training Entry Point for NL2SQL Seq2Seq Model (Phase 3, Step 3.6).

Loads:
- Train split (210 records)
- Validation split (45 records)
Strictly reserves test split (45 records) for Step 3.7.

Builds:
- Vocabularies exclusively from training data
- Encoder, Decoder, and Seq2Seq with Bahdanau attention
- Trainer with CrossEntropyLoss and Adam optimizer

Saves:
- Best model checkpoint based on validation loss to checkpoints/best_model.pt
- Epoch-by-epoch metrics to training_history.json
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

from src.data.dataset import NL2SQLDataset, create_dataloader, load_split
from src.data.tokenizer import NL2SQLTokenizer
from src.data.vocabulary import build_vocabularies
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NL2SQL Attention-based Seq2Seq Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset splits")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--history_path", type=str, default="training_history.json", help="Path to save training history JSON")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for source and target tokens")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden state dimension")
    parser.add_argument("--attention_dim", type=int, default=64, help="Bahdanau attention internal dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0, help="Initial teacher forcing ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("NL2SQL Attention-Based Seq2Seq Model Training (Phase 3, Step 3.6)")
    print("=" * 70)

    # 1. Load train and validation splits (NEVER touching test split)
    print("\n[1/5] Loading dataset splits...")
    train_recs = load_split("train", data_dir=data_dir)
    val_recs = load_split("validation", data_dir=data_dir)
    print(f"  Loaded Train records: {len(train_recs)}")
    print(f"  Loaded Validation records: {len(val_recs)}")
    print("  Test split strictly reserved for Step 3.7 evaluation.")

    # 2. Build vocabularies EXCLUSIVELY from training data
    print("\n[2/5] Building vocabularies from training data...")
    tokenizer = NL2SQLTokenizer()
    src_vocab, tgt_vocab = build_vocabularies(
        train_records=train_recs,
        tokenizer=tokenizer,
    )
    print(f"  Source vocabulary size (encoder input): {len(src_vocab)}")
    print(f"  Target vocabulary size (decoder SQL):  {len(tgt_vocab)}")

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
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Build Seq2Seq Architecture
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

    config = {
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(tgt_vocab),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "attention_dim": args.attention_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "clip_norm": args.clip_norm,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "seed": args.seed,
    }

    # 5. Initialize Trainer and run
    print("\n[4/5] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        clip_norm=args.clip_norm,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        checkpoint_dir=args.checkpoint_dir,
        history_path=args.history_path,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=config,
    )
    print(f"  Target Device: {trainer.device}")
    print(f"  Checkpoints Directory: {Path(args.checkpoint_dir).resolve()}")
    print(f"  History JSON: {Path(args.history_path).resolve()}")

    print("\n[5/5] Commencing training and validation...")
    print("-" * 70)
    history = trainer.fit(epochs=args.epochs, print_progress=True)
    print("-" * 70)

    best_record = next(r for r in history if r["epoch"] == trainer.best_epoch)
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Best Epoch:           {trainer.best_epoch} / {args.epochs}")
    print(f"  Best Val Loss:        {best_record['val_loss']:.4f}")
    print(f"  Best Val Token Acc:   {best_record['val_token_acc'] * 100:.2f}%")
    print(f"  Final Train Loss:     {history[-1]['train_loss']:.4f}")
    print(f"  Saved Checkpoint:     {Path(args.checkpoint_dir) / 'best_model.pt'}")
    print(f"  Saved History:        {Path(args.history_path)}")
    print("=" * 70)

    return history


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
