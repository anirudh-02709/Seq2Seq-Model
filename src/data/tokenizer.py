"""
Tokenizer module for NL2SQL Seq2Seq Preprocessing.

Provides tokenization for:
1. Encoder input: Combined database schema and natural-language question.
2. Decoder target: SQL query sequences with specialized SQL tokenization.
3. Special tokens and constants for sequence-to-sequence modeling.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Special Token Constants
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
SOS_TOKEN: str = "<SOS>"
EOS_TOKEN: str = "<EOS>"
SEP_TOKEN: str = "<SEP>"

# Standard Special Token IDs
PAD_IDX: int = 0
UNK_IDX: int = 1
SOS_IDX: int = 2
EOS_IDX: int = 3

DEFAULT_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Canonical student database schema string matching training and validation splits
DEFAULT_STUDENT_SCHEMA: str = (
    "students(student_id, name, age, department, cgpa) | "
    "courses(course_id, course_name, credits) | "
    "enrollments(enrollment_id, student_id, course_id, semester, marks)"
)

# Regex patterns for tokenization
# SQL Tokenizer Pattern:
# - Single-quoted string literals: '[^']*'
# - Multi-character operators: >=, <=, !=
# - Single-character symbols/operators: =, >, <, *, ,, (, )
# - Identifiers, keywords, numbers: [\w.]+
SQL_TOKEN_PATTERN: str = r"'[^']*'|>=|<=|!=|[=><*(),]|[\w.]+"
_SQL_REGEX = re.compile(SQL_TOKEN_PATTERN)

# Encoder Tokenizer Pattern:
# - Special tags: <...>, [...]
# - Quoted literals: '[^']*'
# - Floating point and integer numbers: \d+(?:\.\d+)?
# - Words / identifiers with optional '&' (e.g. AI&DS, student_id): [a-zA-Z0-9_]+(?:&[a-zA-Z0-9_]+)*
# - Multi-char operators: >=, <=, !=
# - Punctuation and symbols: [=><*(),|?.\-']
ENCODER_TOKEN_PATTERN: str = (
    r"<[a-zA-Z0-9_]+>|"
    r"\[[a-zA-Z0-9_]+\]|"
    r"'[^']*'|"
    r"\d+(?:\.\d+)?|"
    r"[a-zA-Z0-9_]+(?:&[a-zA-Z0-9_]+)*|"
    r">=|<=|!=|"
    r"[=><*(),|?.\-']"
)
_ENCODER_REGEX = re.compile(ENCODER_TOKEN_PATTERN)


def combine_schema_and_question(schema: str, question: str, sep: str = SEP_TOKEN) -> str:
    """
    Combines the database schema and natural language question into a single encoder input.
    
    Args:
        schema: Database schema representation.
        question: Natural language question.
        sep: Separator token between schema and question.
        
    Returns:
        Combined string.
    """
    schema_clean = schema.strip()
    question_clean = question.strip()
    return f"{schema_clean} {sep} {question_clean}"


def tokenize_encoder(text: str, lowercase: bool = True) -> list[str]:
    """
    Tokenizes natural language text combined with schema for the encoder.
    
    Args:
        text: Input string (e.g. schema + separator + question).
        lowercase: Whether to lowercase regular tokens (special tokens like <...> or [...] preserved).
        
    Returns:
        List of string tokens.
    """
    raw_tokens = _ENCODER_REGEX.findall(text)
    if not lowercase:
        return raw_tokens

    tokens: list[str] = []
    for token in raw_tokens:
        if (token.startswith("<") and token.endswith(">")) or (token.startswith("[") and token.endswith("]")):
            tokens.append(token)
        else:
            tokens.append(token.lower())
    return tokens


def tokenize_sql(sql_str: str) -> list[str]:
    """
    Tokenizes a SQL query for the decoder target.
    
    Preserves SQL keywords, identifiers, comparison operators, and quoted string literals.
    
    Args:
        sql_str: The raw SQL query string.
        
    Returns:
        List of SQL token strings.
    """
    return _SQL_REGEX.findall(sql_str.strip())


def detokenize_sql(tokens: Iterable[str]) -> str:
    """
    Reconstructs a clean SQL string from a sequence of SQL tokens.
    
    Args:
        tokens: Sequence of SQL token strings.
        
    Returns:
        Formatted SQL string.
    """
    special_no_pre_space = {",", ")"}
    special_no_post_space = {"("}
    
    result: list[str] = []
    for token in tokens:
        # Filter out special non-SQL tokens if present
        if token in {PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN}:
            continue
        if not result:
            result.append(token)
            continue
        
        last = result[-1]
        if token in special_no_pre_space or last in special_no_post_space:
            result.append(token)
        else:
            result.append(" " + token)
            
    return "".join(result)


class NL2SQLTokenizer:
    """
    Unified tokenizer object for NL2SQL Seq2Seq model.
    Maintains distinct pipelines for encoder inputs and decoder targets.
    """

    def __init__(self, lowercase_encoder: bool = True, sep_token: str = SEP_TOKEN) -> None:
        self.lowercase_encoder = lowercase_encoder
        self.sep_token = sep_token

    def combine(self, schema: str, question: str) -> str:
        """Combine schema and question using the configured separator."""
        return combine_schema_and_question(schema, question, sep=self.sep_token)

    def tokenize_encoder(self, text: str) -> list[str]:
        """Tokenize encoder input text."""
        return tokenize_encoder(text, lowercase=self.lowercase_encoder)

    def tokenize_encoder_record(self, schema: str, question: str) -> list[str]:
        """Combine and tokenize schema + question for an encoder input."""
        combined = self.combine(schema, question)
        return self.tokenize_encoder(combined)

    def tokenize_sql(self, sql_str: str) -> list[str]:
        """Tokenize SQL query string for decoder targets."""
        return tokenize_sql(sql_str)

    def detokenize_sql(self, tokens: Iterable[str]) -> str:
        """Reconstruct SQL string from tokens."""
        return detokenize_sql(tokens)
