"""
Vocabulary module for NL2SQL Seq2Seq Preprocessing.

Handles:
1. Special tokens (<PAD>, <UNK>, <SOS>, <EOS>).
2. Numerical mapping (token to ID and ID to token).
3. Mapping unseen tokens to <UNK>.
4. Building vocabularies exclusively from training splits.
5. Saving and loading vocabulary dictionaries.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

from src.data.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
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
)


class Vocabulary:
    """
    Vocabulary class that manages bidirectional mapping between tokens and numerical IDs.
    """

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        sos_token: str = SOS_TOKEN,
        eos_token: str = EOS_TOKEN,
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self.frequencies: Counter[str] = Counter()

        # Initialize special tokens
        if special_tokens is None:
            special_tokens = list(DEFAULT_SPECIAL_TOKENS)

        for token in special_tokens:
            self.add_token(token)

        self.pad_idx: int = self.token2idx.get(self.pad_token, PAD_IDX)
        self.unk_idx: int = self.token2idx.get(self.unk_token, UNK_IDX)
        self.sos_idx: int = self.token2idx.get(self.sos_token, SOS_IDX)
        self.eos_idx: int = self.token2idx.get(self.eos_token, EOS_IDX)

    def add_token(self, token: str) -> int:
        """
        Adds a single token to the vocabulary if not already present.
        
        Args:
            token: Token string.
            
        Returns:
            The integer ID for the token.
        """
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            return idx
        return self.token2idx[token]

    def add_tokens(self, tokens: Iterable[str]) -> list[int]:
        """
        Adds multiple tokens to the vocabulary.
        
        Args:
            tokens: Iterable of token strings.
            
        Returns:
            List of integer IDs.
        """
        return [self.add_token(t) for t in tokens]

    def lookup_token(self, token: str) -> int:
        """
        Looks up the integer ID for a token. Maps unseen tokens to unk_idx.
        
        Args:
            token: Token string.
            
        Returns:
            Integer token ID.
        """
        return self.token2idx.get(token, self.unk_idx)

    def lookup_id(self, idx: int) -> str:
        """
        Looks up the string token for an integer ID.
        
        Args:
            idx: Integer token ID.
            
        Returns:
            String token (or unk_token if index is invalid).
        """
        return self.idx2token.get(idx, self.unk_token)

    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        """Convenience indexing: vocab['token'] -> int ID, vocab[1] -> str token."""
        if isinstance(key, str):
            return self.lookup_token(key)
        elif isinstance(key, int):
            return self.lookup_id(key)
        raise TypeError(f"Key must be str or int, got {type(key)}")

    def __contains__(self, token: str) -> bool:
        """Checks if a token is in the vocabulary."""
        return token in self.token2idx

    def __len__(self) -> int:
        """Returns total number of unique tokens in the vocabulary."""
        return len(self.token2idx)

    def to_indices(
        self,
        tokens: Iterable[str],
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Converts a sequence of tokens into numerical IDs.
        
        Args:
            tokens: Sequence of token strings.
            add_sos: Whether to prepend SOS_IDX.
            add_eos: Whether to append EOS_IDX.
            
        Returns:
            List of integer IDs.
        """
        ids: list[int] = []
        if add_sos:
            ids.append(self.sos_idx)
        for token in tokens:
            ids.append(self.lookup_token(token))
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def to_tokens(
        self,
        indices: Iterable[int],
        remove_special: bool = False,
    ) -> list[str]:
        """
        Converts a sequence of numerical IDs back into tokens.
        
        Args:
            indices: Sequence of token IDs.
            remove_special: Whether to omit PAD, SOS, and EOS tokens from the output.
            
        Returns:
            List of token strings.
        """
        special_to_omit = {self.pad_token, self.sos_token, self.eos_token}
        tokens: list[str] = []
        for idx in indices:
            token = self.lookup_id(idx)
            if remove_special and token in special_to_omit:
                continue
            tokens.append(token)
        return tokens

    @classmethod
    def build_from_tokens(
        cls,
        token_sequences: Iterable[Iterable[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> Vocabulary:
        """
        Constructs a Vocabulary from an iterable of token sequences.
        
        Args:
            token_sequences: Iterable of token lists.
            min_freq: Minimum occurrences required for inclusion.
            max_size: Maximum vocabulary size (including special tokens).
            special_tokens: List of special tokens to initialize.
            
        Returns:
            Initialized Vocabulary instance.
        """
        vocab = cls(special_tokens=special_tokens)
        freq_counter: Counter[str] = Counter()

        for seq in token_sequences:
            freq_counter.update(seq)

        vocab.frequencies = freq_counter

        # Sort candidate tokens by frequency (descending), then alphabetically for determinism
        candidates = [
            token
            for token, count in freq_counter.most_common()
            if count >= min_freq and token not in vocab.token2idx
        ]

        for token in candidates:
            if max_size is not None and len(vocab) >= max_size:
                break
            vocab.add_token(token)

        return vocab

    def to_dict(self) -> dict[str, Any]:
        """Serializes vocabulary state to a dictionary."""
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "sos_idx": self.sos_idx,
            "eos_idx": self.eos_idx,
            "token2idx": self.token2idx,
            "idx2token": {str(k): v for k, v in self.idx2token.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        """Restores a Vocabulary from a serialized dictionary."""
        vocab = cls(
            pad_token=data.get("pad_token", PAD_TOKEN),
            unk_token=data.get("unk_token", UNK_TOKEN),
            sos_token=data.get("sos_token", SOS_TOKEN),
            eos_token=data.get("eos_token", EOS_TOKEN),
        )
        vocab.token2idx = dict(data["token2idx"])
        vocab.idx2token = {int(k): v for k, v in data["idx2token"].items()}
        vocab.pad_idx = data.get("pad_idx", PAD_IDX)
        vocab.unk_idx = data.get("unk_idx", UNK_IDX)
        vocab.sos_idx = data.get("sos_idx", SOS_IDX)
        vocab.eos_idx = data.get("eos_idx", EOS_IDX)
        return vocab

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves vocabulary to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Vocabulary:
        """Loads a Vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_vocabularies(
    train_records: list[dict[str, Any]],
    tokenizer: Optional[NL2SQLTokenizer] = None,
    min_freq: int = 1,
    shared_vocab: bool = False,
    include_sep_token: bool = True,
) -> tuple[Vocabulary, Vocabulary]:
    """
    Builds encoder (source) and decoder (target) vocabularies exclusively from training records.
    
    Args:
        train_records: List of training records (each containing 'schema', 'question', 'sql').
        tokenizer: Optional tokenizer instance.
        min_freq: Minimum frequency threshold for tokens.
        shared_vocab: If True, builds a unified vocabulary across both encoder and decoder.
        include_sep_token: Whether to include SEP_TOKEN in special tokens.
        
    Returns:
        Tuple of (src_vocab, tgt_vocab).
    """
    if tokenizer is None:
        tokenizer = NL2SQLTokenizer()

    special_tokens = list(DEFAULT_SPECIAL_TOKENS)
    if include_sep_token and SEP_TOKEN not in special_tokens:
        special_tokens.append(SEP_TOKEN)

    encoder_token_seqs: list[list[str]] = [
        tokenizer.tokenize_encoder_record(r["schema"], r["question"])
        for r in train_records
    ]
    decoder_token_seqs: list[list[str]] = [
        tokenizer.tokenize_sql(r["sql"])
        for r in train_records
    ]

    if shared_vocab:
        combined_seqs = encoder_token_seqs + decoder_token_seqs
        shared = Vocabulary.build_from_tokens(
            combined_seqs,
            min_freq=min_freq,
            special_tokens=special_tokens,
        )
        return shared, shared

    src_vocab = Vocabulary.build_from_tokens(
        encoder_token_seqs,
        min_freq=min_freq,
        special_tokens=special_tokens,
    )
    tgt_vocab = Vocabulary.build_from_tokens(
        decoder_token_seqs,
        min_freq=min_freq,
        special_tokens=special_tokens,
    )

    return src_vocab, tgt_vocab
