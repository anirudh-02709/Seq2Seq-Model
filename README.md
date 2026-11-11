# NL2SQL Assistant — Schema-Aware Seq2Seq System

An end-to-end Natural Language to SQL (NL2SQL) translation and safe execution system combining a PyTorch GRU Sequence-to-Sequence model with Bahdanau Additive Attention, schema-aware constrained beam search decoding, SQLGlot read-only AST validation, least-privilege MySQL execution, a FastAPI backend, and a React + TypeScript web interface.

---

## 1. Purpose & Key Capabilities

- **Seq2Seq Translation (`src/models/`):** 1-layer unidirectional GRU encoder (`emb_dim=64`, `hidden_dim=128`), Bahdanau additive attention (`attn_dim=64`), and 1-layer GRU decoder (`dropout=0.1`).
- **Schema-Aware Constrained Decoding (`src/inference/`):** Supports `schema_aware` constrained beam search (`beam_width=5`, `length_penalty_alpha=0.0`, enforcing `SELECT`-before-`FROM` table compatibility intersection and post-`FROM` table-column ownership before `SQLValidator` verification) and `greedy` baseline decoding.
- **Fail-Closed SQL Validation (`src/validation/`):** Parses candidate SQL under MySQL dialect via SQLGlot and enforces a strict single-table read-only `SELECT` scope (`SELECT`, `FROM`, `WHERE`, `=`, `!=`/`<>`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, `ORDER BY`, `LIMIT`). Multi-statements, SQL comments, DML/DDL mutations, `INTO OUTFILE`, locking clauses, joins, subqueries, aggregates, UDFs, and session/system variables (`@`, `@@`) are rejected.
- **Approved Read-Only MySQL Execution (`src/database/`):** Refuses execution unless both `ValidationResult.is_valid == True` and `approved=True`. Connects exclusively via non-root `MYSQL_EXEC_USER`, enforces `SET SESSION TRANSACTION READ ONLY`, applies `MAX_EXECUTION_TIME`, rolls back on exit, and truncates via `fetchmany(max_rows + 1)`.
- **Dataset Governance & Anti-Leakage Gate (`src/data/anti_leakage.py`):** Programmatically enforces Population A baseline preservation, zero held-out canonical SQL / commutative Boolean / question overlap for Population B, `<= 0.30` AST template coverage caps, live MySQL literal provenance, `'Machine Learning'` exclusion, and two-level projection alignment.
- **Full-Stack Web UI & API (`src/api/`, `frontend/`):** Interactive React UI for database health monitoring, dynamic schema exploration, SQL generation + validation badge review, and explicit human-approved SQL execution.

---

## 2. System Architecture & Component Responsibilities

### End-to-End Pipeline (`src/pipeline/orchestrator.py`)

```text
Natural Language Question + Canonical Schema (<schema> <SEP> <question>)
  │
  ▼  1. Inference (src/inference/pipeline.py + constrained_decoder.py)
     • Tokenizes input via NL2SQLTokenizer & Vocabularies embedded in checkpoint
     • Generates SQL via ConstrainedSchemaDecoder (beam_width=5) or greedy mode
  │
  ▼  2. Validation & Safety Gate (src/validation/validator.py)
     • SQLGlot MySQL AST parse + read-only safety + inspected schema checks
     • Strict model scope enforcement (enforce_model_scope=True)
  │
  ▼  3. Approval & Sandbox Execution (src/database/executor.py)
     • Halts at `execution_pending` when approved=False
     • Executes read-only SELECT as MYSQL_EXEC_USER when approved=True
```

### Package & Module Responsibilities

- `src/models/`: `Encoder`, `BahdanauAttention`, `Decoder`, and `Seq2Seq`.
- `src/data/`: `NL2SQLTokenizer` & `DEFAULT_STUDENT_SCHEMA`, `Vocabulary`, `NL2SQLDataset`/`NL2SQLCollator`, and `AntiLeakageGate` v2.0.
- `src/training/`: `Trainer` supporting configurable teacher forcing (`teacher_forcing_ratio=0.50`), validation EM checkpoint selection (`val_loss` tie-breaker), early stopping, RNG state isolation, and self-contained checkpoint serialization.
- `src/inference/`: `NL2SQLInferencePipeline` and `ConstrainedSchemaDecoder`.
- `src/validation/`: `SQLValidator`, `ValidationResult`, and `validate_sql`.
- `src/database/`: `DatabaseConfig` (credential validation/masking), `DatabaseConnection`, `DatabaseSchemaInspector`, and `SQLExecutor`.
- `src/evaluation/`: `ASTComparator` (commutative `AND`/`OR` & inequality normalization) and `ExecutionComparator`.
- `src/pipeline/`: `NL2SQLPipeline` unifying inference, validation, and gated execution.
- `src/api/`: FastAPI app factory (`app.py`), routes (`routes.py`), and Pydantic contracts (`schemas.py`).

### Technology Stack

- **Backend / ML:** Python 3.11+, PyTorch `>=2.0`, SQLGlot `>=23.0`, FastAPI `>=0.110`, Uvicorn `>=0.28`, Pydantic `>=2.0`, PyMySQL `>=1.1`, Cryptography `>=41.0`.
- **Frontend:** React 18, TypeScript 5.7, Vite 6, Vitest 3 (`@testing-library/react`, `jsdom`).
- **Database:** MySQL 8.0+ (`nl2sql_student_db`).

---

## 3. Repository Structure

```text
├── README.md                                  # Primary project documentation
├── requirements.txt                           # Python dependencies
├── run_server.py                              # FastAPI + Uvicorn backend entry point
├── nl2sql_student_db.sql                      # MySQL schema & seed data
├── .env.example                               # Backend & MySQL environment template
├── .gitignore
├── src/                                       # Core application & ML packages
│   ├── api/                                   # FastAPI app, routes, Pydantic schemas
│   ├── data/                                  # Tokenization, vocabulary, datasets, anti-leakage gate
│   ├── database/                              # Config, connection, read-only executor, schema inspector
│   ├── evaluation/                            # ASTComparator & ExecutionComparator
│   ├── inference/                             # NL2SQLInferencePipeline & ConstrainedSchemaDecoder
│   ├── models/                                # Encoder, BahdanauAttention, Decoder, Seq2Seq
│   ├── pipeline/                              # End-to-end NL2SQLPipeline
│   ├── training/                              # Trainer with early stopping & EM selection
│   └── validation/                            # SQLValidator & ValidationResult
├── scripts/                                   # CLI tools (train, inference, benchmark, validation, DB checks)
├── tests/                                     # Python unit & integration tests (unittest)
├── frontend/                                  # React + TypeScript + Vite frontend
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── src/                                   # Components, API client, types, styles, Vitest tests
├── data/                                      # Official datasets & provenance/gate artifacts
│   ├── train.jsonl                            # Immutable Population A training split (210)
│   ├── validation.jsonl                       # Held-out validation split (45)
│   ├── test.jsonl                             # Held-out test split (45)
│   ├── benchmark_eval.jsonl                   # Independent frozen evaluation benchmark (80)
│   ├── dataset_manifest.json                  # Dataset v1 metadata manifest
│   └── candidates/                            # Official Phase 2 clean corpus, provenance, gate report, policy
└── checkpoints/                               # Model checkpoints & official benchmark results
    └── candidates/
        ├── phase2_clean_best.pt               # Official Phase 2 checkpoint (default application model)
        └── phase2_clean_benchmark_results.json # Official 80-question Phase 2 benchmark results
```

---

## 4. Prerequisites, Installation & Environment Setup

### 1. Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. MySQL Setup & Schema Initialization
1. Load [`nl2sql_student_db.sql`](nl2sql_student_db.sql) into MySQL 8.0+. This creates `nl2sql_student_db` with three tables:
   - `students(student_id, name, age, department, cgpa)` (`20` rows)
   - `courses(course_id, course_name, credits)` (`8` rows)
   - `enrollments(enrollment_id, student_id, course_id, semester, marks)` (`49` rows, FKs to `students` and `courses`)
   - Canonical training schema signature (`DEFAULT_STUDENT_SCHEMA`):
     `students(student_id, name, age, department, cgpa) | courses(course_id, course_name, credits) | enrollments(enrollment_id, student_id, course_id, semester, marks)`
2. Create a dedicated read-only execution user (`nl2sql_exec` with `SELECT` privileges only — `SQLExecutor` strictly rejects `root`):
   ```sql
   CREATE USER 'nl2sql_exec'@'localhost' IDENTIFIED BY '<strong_readonly_password>';
   GRANT SELECT ON nl2sql_student_db.* TO 'nl2sql_exec'@'localhost';
   FLUSH PRIVILEGES;
   ```
3. Copy `.env.example` to `.env` (kept untracked & ignored via `.gitignore`) and configure `MYSQL_*`, `MYSQL_EXEC_*`, `CORS_ORIGINS`, `API_HOST`, and `API_PORT`.
4. Verify MySQL connectivity and schema inspection:
   ```bash
   python scripts/verify_mysql.py --inspect-schema
   python scripts/verify_execution.py          # Dry-run: halts at execution_pending
   python scripts/verify_execution.py --approve # Approved read-only SELECT execution
   ```

### 3. Frontend Setup & Configuration
1. Install Node.js 18+ dependencies inside `frontend/`:
   ```bash
   cd frontend
   npm install
   ```
2. Configure `frontend/.env` from `frontend/.env.example` (`VITE_API_BASE_URL=http://127.0.0.1:8000`).

---

## 5. Running the Application

### Backend Server
```bash
python run_server.py --host 127.0.0.1 --port 8000
```
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Optional `--reload` enables uvicorn hot reload.
- Note: By default, `create_app()`, `NL2SQLPipeline`, `NL2SQLInferencePipeline`, and CLI runners load `checkpoints/candidates/phase2_clean_best.pt`. Pass a custom `checkpoint_path` (or `--checkpoint`) to serve another trained checkpoint.

### Frontend Web Application
In a second terminal:
```bash
cd frontend
npm run dev
```
Open `http://127.0.0.1:5173`. Available frontend npm scripts (`frontend/package.json`):
- `npm run dev` — Starts Vite dev server at `http://127.0.0.1:5173`
- `npm test` — Runs the Vitest suite once (`24` tests across `App.test.tsx` & `components.test.tsx`)
- `npm run build` — Runs `tsc && vite build` to `frontend/dist/` *(gitignored build output)*
- `npm run preview` — Previews the production Vite bundle

**UI Workflow:** On load, the header polls `GET /api/health` and the sidebar populates tables/keys via `GET /api/schema`. Submitting a natural-language question calls `POST /api/generate` (displaying SQL and validation badges without running it). Clicking execute triggers `POST /api/execute` with `approved: true`, rendering tabular records, execution latency, and row counts.

---

## 6. Model Checkpoints & Vocabulary Requirements

Checkpoints are **self-contained**: `src_vocab`, `tgt_vocab`, model hyperparameters, and `training_history` are serialized directly inside each `.pt` file and restored by `Trainer.load_model_from_checkpoint()` and `NL2SQLInferencePipeline.from_checkpoint()`. No separate external vocabulary JSON files are needed at runtime.

| Artifact | Path | Vocab (`src` / `tgt`) | Role |
| :--- | :--- | :---: | :--- |
| **Official Phase 2 Model** | `checkpoints/candidates/phase2_clean_best.pt` | `290` / `96` | Authoritative & default application checkpoint (Epoch 42, `214,816` params) |

---

## 7. API Reference

Base URL: `http://127.0.0.1:8000` (configured CORS origins allow `:5173`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Reports API status (`healthy`/`degraded`), model checkpoint/vocab status, and MySQL connectivity. |
| `GET` | `/api/schema` | Returns inspected `nl2sql_student_db` tables, columns, PKs, FKs, and canonical schema string. |
| `POST` | `/api/generate` | Request: `{ "question": str (1..500), "schema"?: str, "max_length"?: int (1..200) }`. Generates SQL and returns `{ generated_sql, validation, inference_diagnostics, stage_reached, can_execute }`. Never executes SQL. |
| `POST` | `/api/execute` | Request: `{ "sql": str (1..2000), "schema"?: str, "approved": bool, "max_rows"?: int (1..1000, default 100) }`. Re-validates SQL and executes read-only on MySQL only when `approved: true`, returning `{ success, executed_sql, columns, rows, row_count, execution_time_ms, is_truncated, validation }`. |

---

## 8. CLI Workflows (Training, Inference, Evaluation & Governance)

### Standalone Inference
```bash
python scripts/run_inference.py \
  --checkpoint checkpoints/candidates/phase2_clean_best.pt \
  --decoding_mode schema_aware \
  --beam_width 5 \
  --question "Show student ID, name, and CGPA for students with CGPA above 8.5 ordered by CGPA descending"
```

### Independent Benchmark & Split Evaluation
```bash
python scripts/run_benchmark.py \
  --checkpoint checkpoints/candidates/phase2_clean_best.pt \
  --benchmark-file data/benchmark_eval.jsonl \
  --decoding-mode schema_aware \
  --beam-width 5 \
  --length-penalty-alpha 0.0 \
  --output /tmp/benchmark_results.json \
  --report /tmp/benchmark_report.md
```
*(Pass `--benchmark-file data/test.jsonl` or `--test-file data/validation.jsonl` to evaluate held-out splits using the same schema-aware AST and MySQL execution comparator harness.)*

### Model Training & Checkpoint Protection
```bash
python scripts/train.py \
  --train_file data/candidates/phase2_clean_train.jsonl \
  --epochs 100 \
  --teacher_forcing_ratio 0.50 \
  --selection_metric val_exact_match \
  --early_stopping_patience 20 \
  --batch_size 16 \
  --seed 42
```
*Note: `scripts/train.py` writes candidates by default to `checkpoints/trained_model.pt` (history: `checkpoints/training_history.json`) and refuses to overwrite protected checkpoint `checkpoints/candidates/phase2_clean_best.pt` unless `--allow_overwrite_baseline` is explicitly passed.*

### Dataset & Anti-Leakage Gate Verification
```bash
# Phase 2 candidate dataset, literal provenance v2.0, projection policy, and live DB gate
python scripts/validate_candidate_dataset.py \
  --candidate data/candidates/phase2_clean_train.jsonl \
  --provenance-manifest data/candidates/phase2_clean_literal_provenance.json

# Original Dataset v1 split & AST integrity verification (add --static-only to skip MySQL)
python scripts/validate_dataset.py --static-only

# Regenerate original 300-record v1 splits & manifest if needed
python scripts/generate_dataset.py
```

### Automated Test Suites
```bash
python -m unittest discover -s tests -v
npm --prefix frontend test
npm --prefix frontend run build
```

---

## 9. Evaluation Methodology & Official Phase 2 Results

Evaluation uses AST comparison (`ASTComparator`, handling MySQL whitespace/casing and associative/commutative `AND`/`OR` + `!=`/`<>` normalization), `SQLValidator(enforce_model_scope=True)`, and live read-only MySQL row-set comparison (`ExecutionComparator`).

### Independent Benchmark Progression (`data/benchmark_eval.jsonl`, `N = 80`)

| Configuration | Strict EM | AST-Norm EM | MySQL Exec Accuracy | SQL Validity & Schema Compat. | Table / SELECT / WHERE / ORDER / LIMIT |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 Baseline** (`best_model.pt`, `greedy`) | `5/80` (`6.25%`) | `5/80` (`6.25%`) | `5/80` (`6.25%`) | `80/80` (`100%`) / `80/80` (`100%`) | `92.5% / 21.25% / 40.0% / 87.5% / 90.0%` |
| **Phase 2 Step 5** (`phase2_step3_best.pt`, `schema_aware` beam 5) | `11/80` (`13.75%`) | `11/80` (`13.75%`) | `12/80` (`15.00%`) | `80/80` (`100%`) / `80/80` (`100%`) | `96.25% / 26.25% / 55.0% / 92.5% / 95.0%` |
| **Official Phase 2** (`phase2_clean_best.pt`, `schema_aware` beam 5, `alpha=0.0`) | **`14/80` (`17.50%`)** | **`14/80` (`17.50%`)** | **`15/80` (`18.75%`)** | **`80/80` (`100%`) / `80/80` (`100%`)** | **`100.0% / 30.0% / 60.0% / 93.75% / 85.0%`** |

### Clean Corpus & Held-Out Split Summary (`phase2_clean_best.pt`)
- **Clean Training Corpus (`data/candidates/phase2_clean_train.jsonl`, `426` records):** `210` immutable Population A baseline rows + `216` Population B additions; `AntiLeakageGate` v2.0 **`PASSED`** (`0` canonical SQL / Boolean / question overlaps against held-out splits, `<= 2.63%` AST template coverage, `426/426` live rows returned on `nl2sql_student_db`, `'Machine Learning'` excluded). *(An earlier experimental Step 7 dataset was disqualified and removed after forensic audit revealed held-out SQL/literal contamination; its invalid metrics are excluded.)*
- **Validation (`data/validation.jsonl`, `N = 45`):** `19/45` (`42.22%`) Strict & AST EM, `22/45` (`48.89%`) Execution Accuracy, `45/45` (`100%`) Valid SQL (`length_penalty_alpha=0.0` chosen over `>0`, which introduced spurious `WHERE` predicates and higher latency).
- **Held-Out Test (`data/test.jsonl`, `N = 45`):** `14/45` (`31.11%`) Strict EM, `15/45` (`33.33%`) AST-Normalized EM, `17/45` (`37.78%`) Execution Accuracy, `45/45` (`100%`) Valid SQL.
- **Official Benchmark JSON:** Preserved at [`checkpoints/candidates/phase2_clean_benchmark_results.json`](checkpoints/candidates/phase2_clean_benchmark_results.json) (full per-query results, category breakdowns, OOV split: `14/74` (`18.92%`) EM / `15/74` (`20.27%`) Exec on in-vocabulary queries vs. `0/6` on OOV queries, and failure-mode counts: `14` exact matches, `34` compound mismatches, `22` projection mismatches, `4` WHERE mismatches, `3` ORDER BY mismatches, `2` LIMIT mismatches, `1` extra WHERE).

---

## 10. Known Limitations

1. **Single-Table DQL Scope Only:** Joins, subqueries, `GROUP BY`, `HAVING`, aggregates (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`), `DISTINCT`, `LIKE`, `IN`, `BETWEEN`, `NOT`, `IS NULL`, and aliases are unsupported by design and blocked by `SQLValidator`.
2. **Closed-Vocabulary OOV Literals:** Without dynamic copy/placeholder slot binding (`<NUM_i>`/`<STR_i>`, deferred to Phase 3), unseen benchmark numeric values (`8.8`, `9.2`, `72.0`, `92.0`, `94.0`, `14`, `15`) and the excluded course title `'Machine Learning'` (`6/80` benchmark queries) cannot be emitted by the closed `96`-token decoder head.
3. **Small-GRU Capacity & Pattern Trade-Offs:** Zero-shot parenthesized grouping `(A OR B) AND C` (`1029–1031, 1077`), unstated implicit projections (`SELECT *` priors), and short-clause `LIMIT` omissions (`85.0%` vs. `95.0%` in Step 5) remain limitations of the `214k`-parameter unidirectional GRU architecture.
