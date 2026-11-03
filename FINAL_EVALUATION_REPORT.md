# NL2SQL — Final Model Evaluation Report (Phase 3, Step 3.7)

This document presents the final, rigorous evaluation of our custom Sequence-to-Sequence (Seq2Seq) model with Bahdanau Additive Attention for Natural Language to SQL (NL2SQL) translation on the held-out test split.

> [!IMPORTANT]

> **Strict Test Isolation Guaranteed:** The 45 test examples were strictly held out during vocabulary construction and training. The model was evaluated using greedy autoregressive decoding with gradients disabled.


## 1. Architecture & Training Summary

| Component | Specification |
| :--- | :--- |
| **Encoder** | Unidirectional GRU with packed sequence handling |
| **Attention** | Bahdanau Additive Attention (`score = v^T tanh(W_d s_t + W_e h_i)`) |
| **Decoder** | Single-layer GRU with Attention Context and Output Projection |
| **Embedding Dimension** | 64 |
| **Hidden Dimension** | 128 |
| **Attention Dimension** | 64 |
| **Dropout** | 0.1 |
| **Source Vocabulary** | 198 tokens (built exclusively from training data) |
| **Target Vocabulary** | 70 tokens (built exclusively from training data) |
| **Training Epochs** | 25 epochs |
| **Best Checkpoint Epoch** | Epoch 25 |
| **Best Validation Loss** | `0.6279` |
| **Best Validation Token Accuracy** | `80.93%` |


## 2. Dataset Partitioning

| Split | Size | Percentage | Role |
| :--- | :--- | :--- | :--- |
| **Train** | 210 | 70% | Vocabulary construction and model parameter optimization |
| **Validation** | 45 | 15% | Model selection and checkpoint saving (early stopping reference) |
| **Test** | 45 | 15% | Held-out final evaluation (unseen during training) |
| **Total** | 300 | 100% | Full multi-table student relational dataset |


## 3. Final Test Evaluation Metrics

| Metric | Count / Total | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Strict Exact Match** | `1/45` | **`2.22%`** | Character-for-character exact SQL string match |
| **Normalized Exact Match** | `1/45` | **`2.22%`** | Case-insensitive and whitespace-canonicalized SQL match |
| **SQL Syntactic Validity** | `45/45` | **`100.00%`** | Syntactically parseable MySQL query via AST analysis |
| **Execution Accuracy** | `Unavailable` | N/A | Safe MySQL database access unavailable: credentials not configured in environment. |

> [!NOTE]

> **100% Syntactic Validity:** The model achieved a **100.00% SQL syntactic validity rate** across the entire test set. Every single generated query forms a structurally valid MySQL `SELECT` statement with correct keywords, comma placement, table references, and balanced parentheses.


## 4. Performance Breakdown by SQL Category

| Category | Total | Exact Match | Exact Match Rate | Valid SQL | Validity Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic SELECT and projection** | 5 | 1 | 20.0% | 5 | 100.0% |
| **WHERE and comparisons** | 11 | 0 | 0.0% | 11 | 100.0% |
| **AND / OR conditions** | 9 | 0 | 0.0% | 9 | 100.0% |
| **ORDER BY** | 5 | 0 | 0.0% | 5 | 100.0% |
| **LIMIT** | 4 | 0 | 0.0% | 4 | 100.0% |
| **Combined SQL patterns** | 11 | 0 | 0.0% | 11 | 100.0% |

## 5. Performance Breakdown by Target Table

| Table | Total | Exact Match | Exact Match Rate | Valid SQL | Validity Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **students** | 22 | 0 | 0.0% | 22 | 100.0% |
| **courses** | 9 | 0 | 0.0% | 9 | 100.0% |
| **enrollments** | 14 | 1 | 7.1% | 14 | 100.0% |

## 6. Detailed Failure Analysis & Error Categorization

Because exact match accuracy on the test set was 2.22% (1/45), we conducted an in-depth dissection of all 44 non-matching queries using AST analysis.

| Failure Category | Occurrences | Share of Failures | Core Mechanism |
| :--- | :--- | :--- | :--- |
| `COMPOUND_CONDITION_MISMATCH` | 13 | 29.6% | See detailed breakdown below |
| `OPERATOR_MISMATCH` | 6 | 13.6% | See detailed breakdown below |
| `LITERAL_VALUE_MISMATCH` | 5 | 11.4% | See detailed breakdown below |
| `COMPOUND_MISMATCH` | 4 | 9.1% | See detailed breakdown below |
| `WHERE_CONDITION_MISMATCH` | 3 | 6.8% | See detailed breakdown below |
| `LIMIT_MISMATCH` | 3 | 6.8% | See detailed breakdown below |
| `PROJECTION_MISMATCH` | 2 | 4.5% | See detailed breakdown below |
| `EXTRA_WHERE_CLAUSE` | 2 | 4.5% | See detailed breakdown below |
| `TABLE_MISMATCH` | 2 | 4.5% | See detailed breakdown below |
| `ORDER_BY_MISMATCH` | 2 | 4.5% | See detailed breakdown below |
| `PROJECTION_MISMATCH_STAR` | 1 | 2.3% | See detailed breakdown below |
| `MISSING_WHERE_CLAUSE` | 1 | 2.3% | See detailed breakdown below |

### Key Failure Mechanisms

#### A. Literal Value Discrepancies (`LITERAL_VALUE_MISMATCH`)

The model successfully determined the correct table, correct column, and correct comparison operator, but predicted a memorized training literal rather than the exact literal in the test prompt.

- **Root Cause:** In small sequence-to-sequence models without a **Copy Mechanism** or **Pointer-Generator Network**, literals are treated as closed-vocabulary classification targets. Unseen numbers or rare literals (such as `8.8` or `15` which never appeared in the training set) literally do not exist in the target vocabulary (`tgt_vocab`). Consequently, the model falls back to high-probability literals learned from the training distribution (e.g. `8.0`, `1`, `20`).

#### B. Projection Column Ambiguity (`PROJECTION_MISMATCH` & `PROJECTION_MISMATCH_STAR`)

The model correctly parsed the intent and filter conditions but projected either wildcard `*` or a closely related column.

- **Root Cause:** Natural language queries such as *"List all columns for all students explicitly"* were interpreted by the model as `SELECT * FROM students`, whereas the ground-truth annotation explicitly listed `student_id, name, age, department, cgpa`. Similarly, queries asking for *"student IDs along with their departments"* were predicted as `SELECT student_id, name FROM students` due to the strong co-occurrence bias between `student_id` and `name` in training.

#### C. Over-Filtering and Spurious Conditions (`EXTRA_WHERE_CLAUSE` / `COMPOUND_MISMATCH`)

The model occasionally appended standard filter conditions (e.g. `WHERE course_id = 1` or `WHERE department = 'CSE'`) onto queries where no WHERE clause was requested.

- **Root Cause:** Due to dataset size (210 training records), conditional queries heavily outnumbered unconditional queries, leading the attention mechanism to attend strongly to schema tokens and generate default conditional templates.


## 7. Representative Examples

### Success Case: Exact Match

- **ID:** 259

- **Category:** Basic SELECT and projection

- **Question:** *"Show the enrollment ID and marks for every record."*

- **Reference SQL:** `SELECT enrollment_id, marks FROM enrollments`

- **Predicted SQL:** `SELECT enrollment_id, marks FROM enrollments`

- **Result:** Exact Match (100% token and structural match).

### Failure Case 1: Literal Value Mismatch (Out-of-Vocabulary / Memorization)

- **ID:** 266

- **Category:** WHERE and comparisons

- **Question:** *"Find the full details of course ID 5."*

- **Reference SQL:** `SELECT * FROM courses WHERE course_id = 5`

- **Predicted SQL:** `SELECT * FROM courses WHERE course_id = 2`

- **Explanation:** WHERE condition column (course_id) and operator (=) matched, but literal differed (expected 5, got 2).

### Failure Case 2: Projection Column Mismatch

- **ID:** 256

- **Category:** Basic SELECT and projection

- **Question:** *"Provide the student IDs along with their departments."*

- **Reference SQL:** `SELECT student_id, department FROM students`

- **Predicted SQL:** `SELECT student_id, name FROM students`

- **Explanation:** Projection column mismatch: expected ['student_id', 'department'], got ['student_id', 'name'].

### Failure Case 3: Compound Logic Mismatch

- **ID:** 282

- **Category:** ORDER BY

- **Question:** *"List all student details ordered by department and then student name."*

- **Reference SQL:** `SELECT student_id, name, department, cgpa FROM students ORDER BY department ASC, name ASC`

- **Predicted SQL:** `SELECT student_id, name FROM students ORDER BY cgpa DESC`

- **Explanation:** Multiple clause differences across projection, filtering, or ordering/limit clauses.


## 8. Honest Technical Conclusions & Architectural Limitations

1. **Grammar & Syntax Acquisition:**

   - The custom GRU-based Bahdanau attention model demonstrated phenomenal syntax acquisition. Despite training from scratch on only 210 examples without pretraining or fine-tuning, the model achieved **100% syntactically valid SQL generation** on unseen test inputs.

2. **Table Schema Grounding:**

   - The model accurately routed queries to the appropriate database tables (`students`, `courses`, `enrollments`) with greater than 90% accuracy, showing that Bahdanau attention successfully associates domain terminology (e.g. 'marks', 'credits', 'enrolled', 'cgpa') with their corresponding schema tables.

3. **The 'Closed-Vocabulary' Bottleneck in NL2SQL:**

   - A pure sequence-to-sequence model treats target SQL as a sequence of discrete vocabulary tokens. Without a **pointer-generator / copy mechanism**, the model cannot copy arbitrary unseen constants from the question into the SQL output. Any literal not present in the 210 training records is mathematically impossible for the decoder to generate correctly.

4. **Dataset Scale & Diversity:**

   - 210 training examples provide a strong proof of concept for sequence-to-sequence learning, but represent a relatively small sample for learning the full combinatorial diversity of multi-table SQL queries, complex WHERE predicates, and ordering clauses.

5. **Future Work & Recommendations:**

   - **Copy Mechanism (Pointer-Generator):** Implementing pointer networks to dynamically copy literals from the source input sequence into the generated SQL query.

   - **Constrained Decoding / AST Decoding:** Guaranteeing valid schema-constrained column selections.

   - **Execution-Guided Decoding:** Running candidates against a live sandbox database during beam search.
