"""
Schema-Aware Constrained Decoding for NL2SQL Seq2Seq Inference (Phase 2, Step 5).

Provides inference-time schema-constrained beam search over the existing
unmodified GRU + Bahdanau Attention Seq2Seq model without modifying model
weights, architecture, training pipeline, or SQLValidator safety rules.

Key guarantees for single-table DQL generation:
1. Handles SQL syntax where SELECT projection columns precede FROM <table>:
   - Projection columns before FROM are not rejected prematurely before FROM is seen.
   - Multi-column projections must share at least one common owning table.
   - Once FROM is emitted, table selection is restricted to tables owning all
     previously emitted SELECT columns (compatible with both qualified and
     unqualified column references).
2. Enforces single-table ownership after FROM <table>:
   - WHERE and ORDER BY column references (both unqualified and table-qualified)
     are strictly constrained to columns owned by the chosen table.
3. Validates every completed candidate through the unmodified SQLValidator
   (with enforce_model_scope=True), rejects invalid candidates, and ranks
   surviving candidates by model log-probability score.
4. Fails closed on total rejection: if no valid candidate exists, never marks
   any invalid fallback query as schema-valid and preserves explicit error details.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from src.data.tokenizer import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, NL2SQLTokenizer
from src.data.vocabulary import Vocabulary
from src.models.seq2seq import Seq2Seq
from src.validation.validator import SQLValidator, normalize_schema

SQL_KEYWORDS: frozenset[str] = frozenset({
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
})
COMPARISON_OPERATORS: frozenset[str] = frozenset({"=", "!=", "<", "<=", ">", ">="})
SQL_PUNCTUATION: frozenset[str] = frozenset({",", "*", "(", ")"})
_NUMERIC_LITERAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _is_quoted_literal(token: str) -> bool:
    return len(token) >= 2 and token.startswith("'") and token.endswith("'")


def _is_positive_integer_token(token: str) -> bool:
    return token.isdigit() and int(token) > 0


@dataclass
class CandidateEvaluation:
    """Validation and scoring record for a single decoded SQL candidate."""

    sql: str
    tokens: list[str]
    token_ids: list[int]
    raw_score: float
    score: float
    is_valid: bool
    is_syntax_valid: bool
    is_schema_compatible: bool
    validation_errors: list[str] = field(default_factory=list)
    ast_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstrainedDecodingResult:
    """Structured outcome of schema-aware constrained decoding."""

    selected_sql: str
    selected_tokens: list[str]
    selected_token_ids: list[int]
    decoding_mode: str
    beam_width: int
    is_valid_sql: bool
    is_schema_valid: bool
    no_valid_candidate: bool
    fallback_used: bool
    validation_error: Optional[str]
    candidates_examined: int
    candidates_rejected: int
    latency_ms: float
    valid_candidates: list[CandidateEvaluation] = field(default_factory=list)
    rejected_candidates: list[CandidateEvaluation] = field(default_factory=list)


@dataclass
class SchemaVocabProfile:
    """Pre-classified target vocabulary token roles relative to a normalized schema."""

    norm_schema: dict[str, list[str]]
    schema_tables: set[str]
    col_owners: dict[str, set[str]]
    table_token_ids: dict[str, int]
    col_ids_by_table: dict[str, set[int]]
    star_ids_by_table: dict[str, int]
    kw_ids: dict[str, int]
    op_ids: dict[str, int]
    comp_op_ids: set[int]
    literal_ids: set[int]
    pos_int_literal_ids: set[int]
    eos_id: Optional[int]
    token_strings: dict[int, str]


def build_schema_vocab_profile(
    tgt_vocab: Vocabulary,
    schema: Any,
) -> Optional[SchemaVocabProfile]:
    """
    Maps every target vocabulary token to its SQL syntactic and schema role.

    Returns None when the supplied schema cannot be normalized into at least one
    valid table definition.
    """
    norm_schema = normalize_schema(schema)
    if not norm_schema:
        return None

    schema_tables = set(norm_schema.keys())
    schema_cols_by_table: dict[str, set[str]] = {
        tbl: set(cols) for tbl, cols in norm_schema.items()
    }
    col_owners: dict[str, set[str]] = {}
    for tbl, cols in norm_schema.items():
        for col in cols:
            col_owners.setdefault(col, set()).add(tbl)

    table_token_ids: dict[str, int] = {}
    col_ids_by_table: dict[str, set[int]] = {tbl: set() for tbl in schema_tables}
    star_ids_by_table: dict[str, int] = {}
    kw_ids: dict[str, int] = {}
    op_ids: dict[str, int] = {}
    comp_op_ids: set[int] = set()
    literal_ids: set[int] = set()
    pos_int_literal_ids: set[int] = set()
    eos_id: Optional[int] = EOS_IDX if EOS_IDX in tgt_vocab.idx2token else None
    token_strings: dict[int, str] = {}

    for idx, tok in tgt_vocab.idx2token.items():
        token_strings[idx] = tok
        if idx in (PAD_IDX, SOS_IDX, UNK_IDX) or tok in ("<PAD>", "<SOS>", "<UNK>", "<SEP>"):
            continue
        if idx == EOS_IDX or tok == "<EOS>":
            eos_id = idx
            continue

        up = tok.upper()
        low = tok.lower()

        if up in SQL_KEYWORDS:
            kw_ids[up] = idx
            continue

        if tok in COMPARISON_OPERATORS:
            op_ids[tok] = idx
            comp_op_ids.add(idx)
            continue

        if tok in SQL_PUNCTUATION:
            op_ids[tok] = idx
            continue

        if _is_quoted_literal(tok) or _NUMERIC_LITERAL_RE.match(tok):
            literal_ids.add(idx)
            if _is_positive_integer_token(tok):
                pos_int_literal_ids.add(idx)
            continue

        # Qualified column or qualified wildcard, e.g. students.name or students.*
        if "." in low:
            parts = low.split(".", 1)
            t_part, c_part = parts[0], parts[1]
            if t_part in schema_tables:
                if c_part == "*":
                    star_ids_by_table[t_part] = idx
                elif c_part in schema_cols_by_table[t_part]:
                    col_ids_by_table[t_part].add(idx)
            continue

        # Bare table and/or column identifiers
        if low in schema_tables:
            table_token_ids[low] = idx
        if low in col_owners:
            for owner_tbl in col_owners[low]:
                col_ids_by_table[owner_tbl].add(idx)

    return SchemaVocabProfile(
        norm_schema=norm_schema,
        schema_tables=schema_tables,
        col_owners=col_owners,
        table_token_ids=table_token_ids,
        col_ids_by_table=col_ids_by_table,
        star_ids_by_table=star_ids_by_table,
        kw_ids=kw_ids,
        op_ids=op_ids,
        comp_op_ids=comp_op_ids,
        literal_ids=literal_ids,
        pos_int_literal_ids=pos_int_literal_ids,
        eos_id=eos_id,
        token_strings=token_strings,
    )


def _compatible_tables_for_select(
    select_col_tokens: list[str],
    profile: SchemaVocabProfile,
) -> set[str]:
    """
    Computes schema tables owning every projection column emitted in SELECT so far.
    """
    compat = set(profile.schema_tables)
    for tok in select_col_tokens:
        tl = tok.lower()
        if tl == "*":
            continue
        if tl.endswith(".*"):
            t_part = tl[:-2]
            compat &= ({t_part} if t_part in profile.schema_tables else set())
        elif "." in tl:
            t_part, c_part = tl.split(".", 1)
            if t_part in profile.schema_tables and c_part in profile.norm_schema[t_part]:
                compat &= {t_part}
            else:
                compat.clear()
        else:
            compat &= profile.col_owners.get(tl, set())
    return compat


def _is_table_column_token(token: str, table_name: str, profile: SchemaVocabProfile) -> bool:
    """Checks whether `token` is a qualified or unqualified column owned by `table_name`."""
    tl = token.lower()
    valid_cols = set(profile.norm_schema.get(table_name, ()))
    if "." in tl:
        t_part, c_part = tl.split(".", 1)
        return t_part == table_name and c_part in valid_cols
    return tl in valid_cols


def get_allowed_token_ids(
    generated_tokens: list[str],
    profile: SchemaVocabProfile,
) -> set[int]:
    """
    Returns permitted next target vocabulary token IDs given partial SQL tokens
    under single-table schema ownership and clause grammar constraints.
    """
    if not generated_tokens:
        select_id = profile.kw_ids.get("SELECT")
        return {select_id} if select_id is not None else set()

    upper_toks = [t.upper() for t in generated_tokens]
    last_tok = generated_tokens[-1]
    last_up = last_tok.upper()

    has_from = "FROM" in upper_toks
    from_idx = upper_toks.index("FROM") if has_from else -1

    select_end = from_idx if has_from else len(generated_tokens)
    select_proj_tokens = [
        t for t in generated_tokens[1:select_end] if t != ","
    ]
    compat_tables = _compatible_tables_for_select(select_proj_tokens, profile)

    # 1. SELECT clause (before FROM)
    if not has_from:
        already_projected_lower = {
            t.lower().split(".", 1)[-1] for t in select_proj_tokens if t != "*" and not t.endswith(".*")
        }
        if len(generated_tokens) == 1:
            allowed: set[int] = set()
            if "*" in profile.op_ids:
                allowed.add(profile.op_ids["*"])
            allowed.update(profile.star_ids_by_table.values())
            for tbl in compat_tables:
                allowed.update(profile.col_ids_by_table[tbl])
            return allowed

        if last_tok == "*" or last_tok.lower().endswith(".*"):
            from_id = profile.kw_ids.get("FROM")
            return {from_id} if from_id is not None else set()

        if last_tok == ",":
            allowed = set()
            for tbl in compat_tables:
                for cid in profile.col_ids_by_table[tbl]:
                    cname = profile.token_strings[cid].lower().split(".", 1)[-1]
                    if cname not in already_projected_lower:
                        allowed.add(cid)
            return allowed

        # Just emitted a column inside SELECT
        allowed = set()
        if "," in profile.op_ids:
            allowed.add(profile.op_ids[","])
        if "FROM" in profile.kw_ids:
            allowed.add(profile.kw_ids["FROM"])
        return allowed

    # 2. Immediately after FROM -> must emit a single compatible table
    if from_idx == len(generated_tokens) - 1:
        return {
            profile.table_token_ids[tbl]
            for tbl in compat_tables
            if tbl in profile.table_token_ids
        }

    selected_table = generated_tokens[from_idx + 1].lower()
    if selected_table not in compat_tables or selected_table not in profile.schema_tables:
        return set()

    # 3. Immediately after selected FROM table
    if from_idx + 1 == len(generated_tokens) - 1:
        allowed = set()
        for kw in ("WHERE", "ORDER", "LIMIT"):
            if kw in profile.kw_ids:
                allowed.add(profile.kw_ids[kw])
        if profile.eos_id is not None:
            allowed.add(profile.eos_id)
        return allowed

    post_tbl_upper = upper_toks[from_idx + 2:]

    # 4. LIMIT clause
    if "LIMIT" in post_tbl_upper:
        limit_pos = len(generated_tokens) - 1 - list(reversed(upper_toks)).index("LIMIT")
        if limit_pos == len(generated_tokens) - 1:
            return set(profile.pos_int_literal_ids)
        return {profile.eos_id} if profile.eos_id is not None else set()

    # 5. ORDER BY clause
    if "ORDER" in post_tbl_upper:
        order_pos = len(generated_tokens) - 1 - list(reversed(upper_toks)).index("ORDER")
        if order_pos == len(generated_tokens) - 1:
            by_id = profile.kw_ids.get("BY")
            return {by_id} if by_id is not None else set()

        if last_up in ("BY", ","):
            order_toks = generated_tokens[order_pos + 2:]
            used_order_cols = {
                t.lower().split(".", 1)[-1]
                for t in order_toks
                if _is_table_column_token(t, selected_table, profile)
            }
            return {
                cid
                for cid in profile.col_ids_by_table[selected_table]
                if profile.token_strings[cid].lower().split(".", 1)[-1] not in used_order_cols
            }

        if last_up in ("ASC", "DESC"):
            allowed = set()
            if "," in profile.op_ids:
                allowed.add(profile.op_ids[","])
            if "LIMIT" in profile.kw_ids:
                allowed.add(profile.kw_ids["LIMIT"])
            if profile.eos_id is not None:
                allowed.add(profile.eos_id)
            return allowed

        # Just emitted an ORDER BY column
        allowed = set()
        for kw in ("ASC", "DESC", "LIMIT"):
            if kw in profile.kw_ids:
                allowed.add(profile.kw_ids[kw])
        if "," in profile.op_ids:
            allowed.add(profile.op_ids[","])
        if profile.eos_id is not None:
            allowed.add(profile.eos_id)
        return allowed

    # 6. WHERE clause
    where_pos = len(generated_tokens) - 1 - list(reversed(upper_toks)).index("WHERE")
    where_toks = generated_tokens[where_pos + 1:]
    open_parens = sum(1 for t in where_toks if t == "(") - sum(1 for t in where_toks if t == ")")

    if not where_toks or last_up in ("WHERE", "AND", "OR", "("):
        allowed = set(profile.col_ids_by_table[selected_table])
        if "(" in profile.op_ids:
            allowed.add(profile.op_ids["("])
        return allowed

    if last_tok in COMPARISON_OPERATORS:
        return set(profile.literal_ids)

    if _is_table_column_token(last_tok, selected_table, profile):
        return set(profile.comp_op_ids)

    # After a comparison literal or ')'
    allowed = set()
    for kw in ("AND", "OR"):
        if kw in profile.kw_ids:
            allowed.add(profile.kw_ids[kw])
    if open_parens > 0:
        if ")" in profile.op_ids:
            allowed.add(profile.op_ids[")"])
        return allowed

    for kw in ("ORDER", "LIMIT"):
        if kw in profile.kw_ids:
            allowed.add(profile.kw_ids[kw])
    if profile.eos_id is not None:
        allowed.add(profile.eos_id)
    return allowed


class ConstrainedSchemaDecoder:
    """
    Inference-time schema-aware constrained beam decoder with SQLValidator verification.
    """

    def __init__(
        self,
        model: Seq2Seq,
        tgt_vocab: Vocabulary,
        tokenizer: NL2SQLTokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        self.device = device
        self._profile_cache: dict[tuple[Any, ...], Optional[SchemaVocabProfile]] = {}
        self._validator_cache: dict[str, SQLValidator] = {}

    def _get_profile(self, schema: str) -> Optional[SchemaVocabProfile]:
        norm = normalize_schema(schema)
        if not norm:
            return None
        cache_key = tuple(sorted((tbl, tuple(cols)) for tbl, cols in norm.items()))
        if cache_key not in self._profile_cache:
            self._profile_cache[cache_key] = build_schema_vocab_profile(self.tgt_vocab, norm)
        return self._profile_cache[cache_key]

    def _get_validator(self, schema: str) -> SQLValidator:
        if schema not in self._validator_cache:
            self._validator_cache[schema] = SQLValidator(schema=schema, enforce_model_scope=True)
        return self._validator_cache[schema]

    def _tokens_from_ids(self, raw_ids: list[int]) -> Tuple[list[int], list[str]]:
        filtered_ids: list[int] = []
        for tid in raw_ids:
            if tid == EOS_IDX:
                break
            if tid != PAD_IDX:
                filtered_ids.append(tid)
        tokens = self.tgt_vocab.to_tokens(filtered_ids)
        return filtered_ids, tokens

    def _greedy_decode_raw(
        self,
        src_tensor: torch.Tensor,
        src_lengths: torch.Tensor,
        max_length: int,
    ) -> Tuple[list[int], list[str], str, float]:
        self.model.eval()
        with torch.no_grad():
            gen_ids = self.model.generate(
                src=src_tensor,
                src_lengths=src_lengths,
                max_length=max_length,
            )[0].tolist()
        filtered_ids, tokens = self._tokens_from_ids(gen_ids)
        sql = self.tokenizer.detokenize_sql(tokens).strip()
        return filtered_ids, tokens, sql, 0.0

    def _evaluate_candidate(
        self,
        token_ids: list[int],
        tokens: list[str],
        raw_score: float,
        validator: SQLValidator,
    ) -> CandidateEvaluation:
        sql = self.tokenizer.detokenize_sql(tokens).strip()
        val_res = validator.validate(sql)
        return CandidateEvaluation(
            sql=sql,
            tokens=tokens,
            token_ids=token_ids,
            raw_score=raw_score,
            score=raw_score,
            is_valid=val_res.is_valid,
            is_syntax_valid=val_res.is_syntax_valid,
            is_schema_compatible=val_res.is_schema_compatible,
            validation_errors=list(val_res.errors),
            ast_details=dict(val_res.ast_details),
        )

    def decode(
        self,
        src_tensor: torch.Tensor,
        src_lengths: torch.Tensor,
        schema: str,
        beam_width: int = 5,
        max_length: int = 60,
        allow_greedy_fallback: bool = True,
        length_penalty_alpha: float = 0.0,
    ) -> ConstrainedDecodingResult:
        """
        Executes schema-aware constrained beam search, validates all completed
        hypotheses using SQLValidator, and returns the top-ranked valid query.
        """
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
            raise ValueError(f"beam_width must be a positive integer, got {beam_width}.")
        if isinstance(length_penalty_alpha, bool) or not isinstance(length_penalty_alpha, (int, float)) or length_penalty_alpha < 0.0:
            raise ValueError(f"length_penalty_alpha must be a non-negative float, got {length_penalty_alpha}.")
        alpha = float(length_penalty_alpha)

        def _score_seq(raw_score: float, n_tokens: int) -> float:
            if alpha == 0.0:
                return raw_score
            return raw_score / ((n_tokens + 1) ** alpha)

        start_time = time.perf_counter()
        profile = self._get_profile(schema)
        validator = self._get_validator(schema)

        completed: list[Tuple[float, list[int], list[str]]] = []
        if profile is not None:
            self.model.eval()
            with torch.no_grad():
                encoder_outputs, init_hidden = self.model.encoder(src_tensor, src_lengths)
                beams: list[Tuple[float, list[int], list[str], torch.Tensor]] = [
                    (0.0, [], [], init_hidden)
                ]
                max_completed = max(beam_width * 2, 4)

                for _ in range(max_length):
                    if not beams:
                        break
                    next_beams: list[Tuple[float, list[int], list[str], torch.Tensor]] = []
                    for score, tids, tstrs, hid in beams:
                        prev_id = tids[-1] if tids else self.tgt_vocab.sos_idx
                        in_tok = torch.tensor([prev_id], dtype=torch.long, device=self.device)
                        logits, new_hid, _ = self.model.decoder(
                            input_token=in_tok,
                            hidden=hid,
                            encoder_outputs=encoder_outputs,
                            src_lengths=src_lengths,
                        )
                        log_probs = F.log_softmax(logits[0], dim=-1)
                        allowed_ids = get_allowed_token_ids(tstrs, profile)
                        if not allowed_ids:
                            continue

                        masked = torch.full_like(log_probs, float("-inf"))
                        allowed_list = list(allowed_ids)
                        allowed_tensor = torch.tensor(allowed_list, dtype=torch.long, device=log_probs.device)
                        masked[allowed_tensor] = log_probs[allowed_tensor]

                        k = min(beam_width, len(allowed_list))
                        top_vals, top_ids = torch.topk(masked, k=k)
                        for val, tid_t in zip(top_vals.tolist(), top_ids.tolist()):
                            if val == float("-inf"):
                                continue
                            new_score = score + val
                            if profile.eos_id is not None and tid_t == profile.eos_id:
                                completed.append((_score_seq(new_score, len(tids)), list(tids), list(tstrs)))
                            else:
                                tok_str = profile.token_strings[tid_t]
                                next_beams.append((new_score, tids + [tid_t], tstrs + [tok_str], new_hid))

                    next_beams.sort(key=lambda item: (item[0], tuple(item[2])), reverse=True)
                    beams = next_beams[:beam_width]

                    # Only prune early on raw unnormalized cumulative sums (alpha == 0.0),
                    # where extending sequences strictly decreases score; when alpha > 0.0,
                    # longer extensions divide negative log-prob by a larger denominator.
                    if alpha == 0.0 and len(completed) >= max_completed and beams:
                        completed_sorted = sorted(completed, key=lambda item: item[0], reverse=True)
                        if beams[0][0] <= completed_sorted[beam_width - 1][0]:
                            break

                # Retain active prefixes only if nothing completed cleanly (e.g., tight max_length)
                if not completed and beams:
                    for score, tids, tstrs, _ in beams:
                        completed.append((_score_seq(score, len(tids)), tids, tstrs))

        # Deduplicate completed candidates while keeping highest model score
        best_by_token_tuple: dict[tuple[int, ...], Tuple[float, list[int], list[str]]] = {}
        for score, tids, tstrs in completed:
            key = tuple(tids)
            if key not in best_by_token_tuple or score > best_by_token_tuple[key][0]:
                best_by_token_tuple[key] = (score, tids, tstrs)

        valid_candidates: list[CandidateEvaluation] = []
        rejected_candidates: list[CandidateEvaluation] = []
        seen_sqls: set[str] = set()

        for score, tids, tstrs in sorted(best_by_token_tuple.values(), key=lambda x: x[0], reverse=True):
            cand = self._evaluate_candidate(tids, tstrs, score, validator)
            seen_sqls.add(cand.sql)
            if cand.is_valid:
                valid_candidates.append(cand)
            else:
                rejected_candidates.append(cand)

        if valid_candidates:
            valid_candidates.sort(key=lambda c: (c.score, tuple(c.tokens)), reverse=True)
            best = valid_candidates[0]
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConstrainedDecodingResult(
                selected_sql=best.sql,
                selected_tokens=best.tokens,
                selected_token_ids=best.token_ids,
                decoding_mode="schema_aware",
                beam_width=beam_width,
                is_valid_sql=best.is_syntax_valid,
                is_schema_valid=True,
                no_valid_candidate=False,
                fallback_used=False,
                validation_error=None,
                candidates_examined=len(valid_candidates) + len(rejected_candidates),
                candidates_rejected=len(rejected_candidates),
                latency_ms=round(latency_ms, 3),
                valid_candidates=valid_candidates,
                rejected_candidates=rejected_candidates,
            )

        # No schema-valid candidate survived constrained search
        if allow_greedy_fallback:
            g_ids, g_toks, g_sql, g_score = self._greedy_decode_raw(
                src_tensor=src_tensor,
                src_lengths=src_lengths,
                max_length=max_length,
            )
            g_eval = self._evaluate_candidate(g_ids, g_toks, g_score, validator)
            if g_sql not in seen_sqls:
                if g_eval.is_valid:
                    valid_candidates.append(g_eval)
                else:
                    rejected_candidates.append(g_eval)

            total_examined = len(valid_candidates) + len(rejected_candidates)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if g_eval.is_valid:
                return ConstrainedDecodingResult(
                    selected_sql=g_eval.sql,
                    selected_tokens=g_eval.tokens,
                    selected_token_ids=g_eval.token_ids,
                    decoding_mode="schema_aware",
                    beam_width=beam_width,
                    is_valid_sql=g_eval.is_syntax_valid,
                    is_schema_valid=True,
                    no_valid_candidate=False,
                    fallback_used=True,
                    validation_error=None,
                    candidates_examined=total_examined,
                    candidates_rejected=len(rejected_candidates),
                    latency_ms=round(latency_ms, 3),
                    valid_candidates=valid_candidates,
                    rejected_candidates=rejected_candidates,
                )

            primary_err = (
                g_eval.validation_errors[0]
                if g_eval.validation_errors
                else "No schema-compatible candidate satisfied SQLValidator."
            )
            return ConstrainedDecodingResult(
                selected_sql=g_eval.sql,
                selected_tokens=g_eval.tokens,
                selected_token_ids=g_eval.token_ids,
                decoding_mode="schema_aware",
                beam_width=beam_width,
                is_valid_sql=g_eval.is_syntax_valid,
                is_schema_valid=False,
                no_valid_candidate=True,
                fallback_used=True,
                validation_error=f"All {total_examined} candidate(s) rejected; fallback invalid: {primary_err}",
                candidates_examined=total_examined,
                candidates_rejected=len(rejected_candidates),
                latency_ms=round(latency_ms, 3),
                valid_candidates=[],
                rejected_candidates=rejected_candidates,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        primary_err = (
            rejected_candidates[0].validation_errors[0]
            if rejected_candidates and rejected_candidates[0].validation_errors
            else "No schema-compatible candidate satisfied SQLValidator."
        )
        return ConstrainedDecodingResult(
            selected_sql="",
            selected_tokens=[],
            selected_token_ids=[],
            decoding_mode="schema_aware",
            beam_width=beam_width,
            is_valid_sql=False,
            is_schema_valid=False,
            no_valid_candidate=True,
            fallback_used=False,
            validation_error=f"All {len(rejected_candidates)} candidate(s) rejected: {primary_err}",
            candidates_examined=len(rejected_candidates),
            candidates_rejected=len(rejected_candidates),
            latency_ms=round(latency_ms, 3),
            valid_candidates=[],
            rejected_candidates=rejected_candidates,
        )
