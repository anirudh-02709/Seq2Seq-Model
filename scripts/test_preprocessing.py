"""
Test Suite for NL2SQL Data Preprocessing Pipeline (Phase 3, Step 3.1).

Verifies:
1. All three splits load with correct counts (train=210, val=45, test=45, total=300).
2. Vocabulary is built exclusively from training data.
3. Unknown tokens map to <UNK>.
4. SOS and EOS are correctly added to decoder targets.
5. Dynamic padding and collation work correctly on variable-length sequences.
6. Encoder and decoder sequences and vocabularies remain strictly separate.
7. A sample batch loads successfully through PyTorch DataLoader with correct shapes and masks.
8. Existing JSONL files remain strictly unmodified (verified via SHA-256 hashes).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.tokenizer import (
    EOS_IDX,
    EOS_TOKEN,
    PAD_IDX,
    PAD_TOKEN,
    SEP_TOKEN,
    SOS_IDX,
    SOS_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    NL2SQLTokenizer,
    combine_schema_and_question,
    detokenize_sql,
    tokenize_encoder,
    tokenize_sql,
)
from src.data.vocabulary import (
    Vocabulary,
    build_vocabularies,
)
from src.data.dataset import (
    EXPECTED_SPLIT_COUNTS,
    NL2SQLCollator,
    NL2SQLDataset,
    create_all_dataloaders,
    create_dataloader,
    load_all_splits,
    load_jsonl_records,
    load_split,
)


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestNL2SQLPreprocessing(unittest.TestCase):
    """Unit and integration tests for data preprocessing pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = PROJECT_ROOT / "data"
        cls.train_path = cls.data_dir / "train.jsonl"
        cls.val_path = cls.data_dir / "validation.jsonl"
        cls.test_path = cls.data_dir / "test.jsonl"

        # Record file hashes prior to running tests
        cls.train_hash_before = compute_sha256(cls.train_path)
        cls.val_hash_before = compute_sha256(cls.val_path)
        cls.test_hash_before = compute_sha256(cls.test_path)

        # Load splits once for reference
        cls.train_records, cls.val_records, cls.test_records = load_all_splits(cls.data_dir)
        cls.tokenizer = NL2SQLTokenizer()
        cls.src_vocab, cls.tgt_vocab = build_vocabularies(
            cls.train_records,
            tokenizer=cls.tokenizer,
        )

    def test_01_all_splits_load_with_correct_counts(self):
        """Test 1: All three splits load with correct counts (210, 45, 45)."""
        train = load_split("train", self.data_dir)
        val = load_split("validation", self.data_dir)
        test = load_split("test", self.data_dir)

        self.assertEqual(len(train), 210, f"Expected 210 train records, got {len(train)}")
        self.assertEqual(len(val), 45, f"Expected 45 validation records, got {len(val)}")
        self.assertEqual(len(test), 45, f"Expected 45 test records, got {len(test)}")
        self.assertEqual(len(train) + len(val) + len(test), 300)

        # Check required schema fields in all records
        required_fields = {"id", "schema", "question", "sql", "category", "table"}
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            for i, rec in enumerate(split_data):
                missing = required_fields - set(rec.keys())
                self.assertFalse(missing, f"Split {split_name} record {i} missing fields: {missing}")

    def test_02_vocabulary_built_only_from_training_data(self):
        """Test 2: Vocabulary is built exclusively from training data."""
        # Tokens known to only exist in validation or test questions/sql
        unseen_in_train_sql = {"72.0", "9.2", "15", "92.0", "8.8"}
        unseen_in_train_words = {"carrying", "profiles", "exceeds", "awarded", "studying"}

        # Check that none of these unseen tokens are in the target vocabulary
        for token in unseen_in_train_sql:
            self.assertNotIn(
                token,
                self.tgt_vocab,
                f"Decoder target vocabulary leaked unseen token: {token}",
            )

        # Check that unseen words from validation/test are not in source vocabulary
        for word in unseen_in_train_words:
            self.assertNotIn(
                word,
                self.src_vocab,
                f"Encoder source vocabulary leaked unseen word: {word}",
            )

        # Check that training tokens ARE in vocabulary
        self.assertIn("SELECT", self.tgt_vocab)
        self.assertIn("FROM", self.tgt_vocab)
        self.assertIn("students", self.tgt_vocab)
        self.assertIn("students", self.src_vocab)
        self.assertIn("cgpa", self.src_vocab)

    def test_03_unknown_tokens_map_to_unk(self):
        """Test 3: Unknown tokens correctly map to <UNK> (ID 1)."""
        # Explicit non-existent token
        dummy_unknown = "completely_nonexistent_token_xyz"
        self.assertNotIn(dummy_unknown, self.src_vocab)
        self.assertNotIn(dummy_unknown, self.tgt_vocab)

        self.assertEqual(self.src_vocab[dummy_unknown], UNK_IDX)
        self.assertEqual(self.tgt_vocab[dummy_unknown], UNK_IDX)
        self.assertEqual(self.src_vocab.lookup_token(dummy_unknown), UNK_IDX)
        self.assertEqual(self.tgt_vocab.lookup_token(dummy_unknown), UNK_IDX)

        # Verify special tokens values and indices
        self.assertEqual(self.src_vocab.pad_idx, PAD_IDX)
        self.assertEqual(self.src_vocab.unk_idx, UNK_IDX)
        self.assertEqual(self.src_vocab.sos_idx, SOS_IDX)
        self.assertEqual(self.src_vocab.eos_idx, EOS_IDX)
        self.assertEqual(PAD_IDX, 0)
        self.assertEqual(UNK_IDX, 1)
        self.assertEqual(SOS_IDX, 2)
        self.assertEqual(EOS_IDX, 3)

        # Map validation example containing unseen tokens to numerical IDs
        val_dataset = NL2SQLDataset(
            self.val_records,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            tokenizer=self.tokenizer,
        )

        # Find a record in validation that has an unseen token (e.g. 72.0 or 9.2)
        found_oov = False
        for item in val_dataset:
            # Check decoder IDs
            dec_ids = item["decoder_target_ids"].tolist()
            if UNK_IDX in dec_ids:
                found_oov = True
                unk_pos = dec_ids.index(UNK_IDX)
                # The token in decoder_tokens at (unk_pos - 1) because of SOS token
                oov_token = item["decoder_tokens"][unk_pos - 1]
                self.assertNotIn(oov_token, self.tgt_vocab)
                break

        self.assertTrue(found_oov, "Expected at least one OOV token in validation SQL to map to UNK")

    def test_04_sos_and_eos_added_to_decoder_targets(self):
        """Test 4: SOS and EOS are correctly added to decoder targets."""
        dataset = NL2SQLDataset(
            self.train_records,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            tokenizer=self.tokenizer,
            decoder_add_sos=True,
            decoder_add_eos=True,
        )

        sample = dataset[0]
        decoder_ids = sample["decoder_target_ids"]

        # Check SOS at position 0
        self.assertEqual(decoder_ids[0].item(), SOS_IDX)
        # Check EOS at final position
        self.assertEqual(decoder_ids[-1].item(), EOS_IDX)

        # Inner tokens match the SQL query
        sql_tokens = sample["decoder_tokens"]
        self.assertEqual(len(decoder_ids), len(sql_tokens) + 2)
        for i, tok in enumerate(sql_tokens):
            expected_id = self.tgt_vocab[tok]
            self.assertEqual(decoder_ids[i + 1].item(), expected_id)

        # Encoder should NOT have SOS/EOS by default
        encoder_ids = sample["encoder_input_ids"]
        self.assertNotEqual(encoder_ids[0].item(), SOS_IDX)
        self.assertNotEqual(encoder_ids[-1].item(), EOS_IDX)

    def test_05_dynamic_padding_and_collation(self):
        """Test 5: Dynamic padding works correctly on variable-length sequences."""
        dataset = NL2SQLDataset(
            self.train_records,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            tokenizer=self.tokenizer,
        )

        # Create a small batch with items of different lengths
        item1 = dataset[0]  # shorter query: SELECT * FROM students
        item2 = dataset[25] # longer query with WHERE, etc.

        collator = NL2SQLCollator(pad_idx=PAD_IDX, batch_first=True)
        batch = collator([item1, item2])

        enc_tensor = batch["encoder_input_ids"]
        dec_tensor = batch["decoder_target_ids"]
        enc_mask = batch["encoder_mask"]
        dec_mask = batch["decoder_mask"]

        max_enc_len = max(item1["encoder_length"], item2["encoder_length"])
        max_dec_len = max(item1["decoder_length"], item2["decoder_length"])

        self.assertEqual(enc_tensor.shape, (2, max_enc_len))
        self.assertEqual(dec_tensor.shape, (2, max_dec_len))
        self.assertEqual(enc_mask.shape, (2, max_enc_len))
        self.assertEqual(dec_mask.shape, (2, max_dec_len))

        # Check that shorter sequence has padding tokens (0) and mask 0 at the end
        if item1["decoder_length"] < max_dec_len:
            shorter_idx = 0
            pad_start = item1["decoder_length"]
            self.assertTrue(torch.all(dec_tensor[shorter_idx, pad_start:] == PAD_IDX))
            self.assertTrue(torch.all(dec_mask[shorter_idx, pad_start:] == 0))
            self.assertTrue(torch.all(dec_mask[shorter_idx, :pad_start] == 1))
        elif item2["decoder_length"] < max_dec_len:
            shorter_idx = 1
            pad_start = item2["decoder_length"]
            self.assertTrue(torch.all(dec_tensor[shorter_idx, pad_start:] == PAD_IDX))
            self.assertTrue(torch.all(dec_mask[shorter_idx, pad_start:] == 0))
            self.assertTrue(torch.all(dec_mask[shorter_idx, :pad_start] == 1))

    def test_06_encoder_and_decoder_sequences_remain_separate(self):
        """Test 6: Encoder inputs and decoder targets remain strictly separate."""
        # 1. Distinct vocabularies
        self.assertIsNot(self.src_vocab, self.tgt_vocab)
        self.assertNotEqual(len(self.src_vocab), len(self.tgt_vocab))

        # 2. Dataset separate fields
        dataset = NL2SQLDataset(
            self.train_records,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            tokenizer=self.tokenizer,
        )
        sample = dataset[5]

        # Encoder input contains schema and question, not pure SQL
        self.assertIn("students", sample["encoder_tokens"])
        self.assertIn(SEP_TOKEN, sample["encoder_tokens"])

        # Decoder target contains SQL tokens
        self.assertIn("SELECT", sample["decoder_tokens"])
        self.assertIn("FROM", sample["decoder_tokens"])

        # Check tensor separation
        self.assertIsInstance(sample["encoder_input_ids"], torch.Tensor)
        self.assertIsInstance(sample["decoder_target_ids"], torch.Tensor)
        self.assertEqual(sample["encoder_input_ids"].dtype, torch.long)
        self.assertEqual(sample["decoder_target_ids"].dtype, torch.long)

    def test_07_sample_batch_loads_through_pytorch_dataloader(self):
        """Test 7: A sample batch loads successfully through PyTorch DataLoader."""
        train_loader, val_loader, test_loader, src_v, tgt_v = create_all_dataloaders(
            data_dir=self.data_dir,
            batch_size=16,
            num_workers=0,
        )

        self.assertEqual(len(train_loader.dataset), 210)
        self.assertEqual(len(val_loader.dataset), 45)
        self.assertEqual(len(test_loader.dataset), 45)

        # Fetch one batch from train_loader
        batch = next(iter(train_loader))

        self.assertIn("encoder_input_ids", batch)
        self.assertIn("decoder_target_ids", batch)
        self.assertIn("encoder_lengths", batch)
        self.assertIn("decoder_lengths", batch)
        self.assertIn("encoder_mask", batch)
        self.assertIn("decoder_mask", batch)

        enc_ids = batch["encoder_input_ids"]
        dec_ids = batch["decoder_target_ids"]

        self.assertEqual(enc_ids.dim(), 2)
        self.assertEqual(dec_ids.dim(), 2)
        self.assertEqual(enc_ids.shape[0], 16)
        self.assertEqual(dec_ids.shape[0], 16)

        # Verify SQL reconstruction via detokenizer
        sample_dec_ids = dec_ids[0].tolist()
        tokens = tgt_v.to_tokens(sample_dec_ids, remove_special=True)
        reconstructed_sql = detokenize_sql(tokens)
        self.assertTrue(reconstructed_sql.startswith("SELECT"))

    def test_08_jsonl_files_never_modified(self):
        """Test 8: Verify existing JSONL split files were never modified."""
        train_hash_after = compute_sha256(self.train_path)
        val_hash_after = compute_sha256(self.val_path)
        test_hash_after = compute_sha256(self.test_path)

        self.assertEqual(self.train_hash_before, train_hash_after, "train.jsonl was modified!")
        self.assertEqual(self.val_hash_before, val_hash_after, "validation.jsonl was modified!")
        self.assertEqual(self.test_hash_before, test_hash_after, "test.jsonl was modified!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
