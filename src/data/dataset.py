"""
PyTorch Dataset and DataLoader pipeline for NL2SQL Seq2Seq Preprocessing.

Handles:
1. Loading all three JSONL splits (train, validation, test) without modifying source files.
2. Converting raw records into encoder input sequences and decoder target sequences.
3. Managing variable-length sequences with proper padding and collation.
4. Providing ready-to-use PyTorch DataLoaders for train, val, and test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import (
    PAD_IDX,
    NL2SQLTokenizer,
)
from src.data.vocabulary import (
    Vocabulary,
    build_vocabularies,
)

EXPECTED_SPLIT_COUNTS = {
    "train": 210,
    "validation": 45,
    "test": 45,
}


def load_jsonl_records(filepath: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Defensively loads a JSONL file in read-only mode without modifying the file.
    
    Args:
        filepath: Path to the JSONL file.
        
    Returns:
        List of parsed record dictionaries.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {path.resolve()}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Malformed JSON on line {line_num} in {path.name}: {err}"
                ) from err

    return records


def load_split(split_name: str, data_dir: Union[str, Path] = "data") -> list[dict[str, Any]]:
    """
    Loads a specific split ('train', 'validation', or 'test') from the data directory.
    
    Args:
        split_name: One of 'train', 'validation', 'test'.
        data_dir: Path to directory containing split JSONL files.
        
    Returns:
        List of records.
    """
    cleaned_name = split_name.lower().strip()
    if cleaned_name.endswith(".jsonl"):
        cleaned_name = cleaned_name[:-6]

    file_map = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "val": "validation.jsonl",
        "test": "test.jsonl",
    }

    if cleaned_name not in file_map:
        raise ValueError(
            f"Invalid split name '{split_name}'. Expected one of: train, validation, test."
        )

    filepath = Path(data_dir) / file_map[cleaned_name]
    records = load_jsonl_records(filepath)

    expected_key = "validation" if cleaned_name == "val" else cleaned_name
    expected_count = EXPECTED_SPLIT_COUNTS.get(expected_key)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Split '{cleaned_name}' expected {expected_count} records, but loaded {len(records)}."
        )

    return records


def load_all_splits(
    data_dir: Union[str, Path] = "data",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Loads all three existing JSONL splits: train (210), validation (45), test (45).
    
    Args:
        data_dir: Directory containing train.jsonl, validation.jsonl, test.jsonl.
        
    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    train = load_split("train", data_dir=data_dir)
    val = load_split("validation", data_dir=data_dir)
    test = load_split("test", data_dir=data_dir)
    return train, val, test


class NL2SQLDataset(Dataset):
    """
    PyTorch Dataset for Schema-Aware Seq2Seq NL2SQL.
    
    Maintains separate encoder inputs (schema + question) and decoder targets (SQL query).
    Applies numerical vocabulary lookup and attaches SOS / EOS tokens as configured.
    """

    def __init__(
        self,
        records_or_path: Union[list[dict[str, Any]], str, Path],
        src_vocab: Vocabulary,
        tgt_vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[NL2SQLTokenizer] = None,
        encoder_add_sos: bool = False,
        encoder_add_eos: bool = False,
        decoder_add_sos: bool = True,
        decoder_add_eos: bool = True,
        lowercase_encoder: bool = True,
        data_dir: Union[str, Path] = "data",
    ) -> None:
        """
        Initializes the dataset.
        
        Args:
            records_or_path: In-memory record list, file path, or split name ('train', 'val', 'test').
            src_vocab: Vocabulary for encoder inputs.
            tgt_vocab: Vocabulary for decoder targets (defaults to src_vocab if None).
            tokenizer: NL2SQLTokenizer instance (instantiated if None).
            encoder_add_sos: Whether to prepend SOS token to encoder input sequence.
            encoder_add_eos: Whether to append EOS token to encoder input sequence.
            decoder_add_sos: Whether to prepend SOS token to decoder target sequence.
            decoder_add_eos: Whether to append EOS token to decoder target sequence.
            lowercase_encoder: Whether encoder questions/schema are lowercased.
            data_dir: Directory for loading splits if records_or_path is a split name.
        """
        if isinstance(records_or_path, list):
            self.records = records_or_path
        elif isinstance(records_or_path, (str, Path)):
            path_obj = Path(records_or_path)
            if path_obj.is_file():
                self.records = load_jsonl_records(path_obj)
            else:
                self.records = load_split(str(records_or_path), data_dir=data_dir)
        else:
            raise TypeError(f"Unsupported type for records_or_path: {type(records_or_path)}")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab if tgt_vocab is not None else src_vocab
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else NL2SQLTokenizer(lowercase_encoder=lowercase_encoder)
        )

        self.encoder_add_sos = encoder_add_sos
        self.encoder_add_eos = encoder_add_eos
        self.decoder_add_sos = decoder_add_sos
        self.decoder_add_eos = decoder_add_eos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Retrieves and preprocesses an example by index.
        
        Returns a dictionary containing:
            - Raw fields: id, schema, question, sql, category, table.
            - Token sequences: encoder_tokens, decoder_tokens.
            - Numerical tensor sequences: encoder_input_ids, decoder_target_ids.
            - Sequence lengths: encoder_length, decoder_length.
        """
        record = self.records[idx]
        schema = record["schema"]
        question = record["question"]
        sql = record["sql"]

        # 1. Tokenize encoder input (combined schema + question)
        encoder_tokens = self.tokenizer.tokenize_encoder_record(schema, question)

        # 2. Tokenize decoder target (SQL query)
        decoder_tokens = self.tokenizer.tokenize_sql(sql)

        # 3. Convert to numerical IDs
        encoder_ids = self.src_vocab.to_indices(
            encoder_tokens,
            add_sos=self.encoder_add_sos,
            add_eos=self.encoder_add_eos,
        )
        decoder_ids = self.tgt_vocab.to_indices(
            decoder_tokens,
            add_sos=self.decoder_add_sos,
            add_eos=self.decoder_add_eos,
        )

        return {
            "id": record["id"],
            "schema": schema,
            "question": question,
            "sql": sql,
            "category": record.get("category", ""),
            "table": record.get("table", ""),
            "encoder_tokens": encoder_tokens,
            "decoder_tokens": decoder_tokens,
            "encoder_input_ids": torch.tensor(encoder_ids, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_ids, dtype=torch.long),
            "encoder_length": len(encoder_ids),
            "decoder_length": len(decoder_ids),
        }


class NL2SQLCollator:
    """
    Collation and dynamic padding callable for PyTorch DataLoader.
    Pads variable-length encoder inputs and decoder targets to batch maximum length.
    """

    def __init__(self, pad_idx: int = PAD_IDX, batch_first: bool = True) -> None:
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a list of dataset samples into a padded batch.
        
        Args:
            batch: List of sample dicts from NL2SQLDataset.
            
        Returns:
            Collated batch dictionary with padded tensors, lengths, and masks.
        """
        encoder_seqs = [item["encoder_input_ids"] for item in batch]
        decoder_seqs = [item["decoder_target_ids"] for item in batch]

        encoder_lengths = torch.tensor(
            [item["encoder_length"] for item in batch], dtype=torch.long
        )
        decoder_lengths = torch.tensor(
            [item["decoder_length"] for item in batch], dtype=torch.long
        )

        # Dynamic padding with pad_sequence
        padded_encoder = pad_sequence(
            encoder_seqs,
            batch_first=self.batch_first,
            padding_value=self.pad_idx,
        )
        padded_decoder = pad_sequence(
            decoder_seqs,
            batch_first=self.batch_first,
            padding_value=self.pad_idx,
        )

        # Padding masks: 1 for valid tokens, 0 for PAD
        encoder_mask = (padded_encoder != self.pad_idx).long()
        decoder_mask = (padded_decoder != self.pad_idx).long()

        return {
            "encoder_input_ids": padded_encoder,
            "encoder_lengths": encoder_lengths,
            "encoder_mask": encoder_mask,
            "decoder_target_ids": padded_decoder,
            "decoder_lengths": decoder_lengths,
            "decoder_mask": decoder_mask,
            "ids": [item["id"] for item in batch],
            "schemas": [item["schema"] for item in batch],
            "questions": [item["question"] for item in batch],
            "sqls": [item["sql"] for item in batch],
            "categories": [item["category"] for item in batch],
            "tables": [item["table"] for item in batch],
        }


def create_dataloader(
    dataset: NL2SQLDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    pad_idx: int = PAD_IDX,
    batch_first: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader with dynamic padding collation for an NL2SQLDataset.
    
    Args:
        dataset: NL2SQLDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle examples each epoch.
        pad_idx: Padding index for collation.
        batch_first: Whether batch dimension comes first (batch_size, seq_len).
        num_workers: Number of background worker processes.
        
    Returns:
        Configured PyTorch DataLoader.
    """
    collator = NL2SQLCollator(pad_idx=pad_idx, batch_first=batch_first)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


def create_all_dataloaders(
    data_dir: Union[str, Path] = "data",
    batch_size: int = 16,
    min_freq: int = 1,
    shared_vocab: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    End-to-end preprocessing pipeline loader:
    1. Loads all 3 splits (train, validation, test).
    2. Builds vocabularies EXCLUSIVELY from train_records.
    3. Initializes datasets with proper special tokens and mappings.
    4. Returns DataLoaders for train (shuffled), val (not shuffled), and test (not shuffled).
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, src_vocab, tgt_vocab).
    """
    train_recs, val_recs, test_recs = load_all_splits(data_dir=data_dir)
    tokenizer = NL2SQLTokenizer()

    # Build vocabulary EXCLUSIVELY from training data
    src_vocab, tgt_vocab = build_vocabularies(
        train_records=train_recs,
        tokenizer=tokenizer,
        min_freq=min_freq,
        shared_vocab=shared_vocab,
    )

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
    test_dataset = NL2SQLDataset(
        test_recs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        tokenizer=tokenizer,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
