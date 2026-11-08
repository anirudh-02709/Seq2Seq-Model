# NL2SQL Assistant — React + TypeScript Frontend

This directory contains the single-page web application for the Schema-Aware Natural Language to SQL (NL2SQL) project. It provides an intuitive, developer-focused interface for interacting with the trained Seq2Seq model, inspecting MySQL database schemas, generating SQL queries, and approving executions safely against the MySQL database.

---

## 1. Technology Stack

- **Framework:** React 18 (Functional components + React Hooks)
- **Language:** TypeScript 5.7 (Strict mode)
- **Build Tool:** Vite 6
- **Test Runner:** Vitest 3 with `@testing-library/react` and `jsdom`
- **Styling:** Modern CSS (CSS custom properties, CSS Grid, Flexbox, dark developer-tool theme)

---

## 2. Directory Structure

```
frontend/
├── .env                        # Local environment configuration
├── .env.example                # Template for environment variables
├── index.html                  # HTML5 entry document
├── package.json                # Project dependencies and npm scripts
├── tsconfig.json               # TypeScript compiler configuration (bundler resolution)
├── tsconfig.node.json          # Vite node configuration
├── vite.config.ts              # Vite & Vitest configuration
├── dist/                       # Compiled production build output
└── src/
    ├── App.tsx                 # Main application state coordinator
    ├── main.tsx                # React DOM mount point
    ├── setupTests.ts           # Vitest setup with jest-dom matchers
    ├── vite-env.d.ts           # Vite client type definitions
    ├── components/
    │   ├── Header/             # App title, model status, and DB status indicators
    │   ├── SchemaExplorer/     # Collapsible dynamic MySQL schema explorer
    │   ├── QueryInput/         # Prompt input and example question pills
    │   ├── SQLPreview/         # Generated SQL display, AST badges, and approval gate
    │   ├── ResultsTable/       # Structured tabular results and execution metrics
    │   └── StatusIndicator/    # Reusable status pill badge with pulse animation
    ├── services/
    │   └── api.ts              # Centralized fetch client for FastAPI backend endpoints
    ├── styles/
    │   └── index.css           # Modern dark design system
    ├── types/
    │   └── api.ts              # TypeScript interfaces matching FastAPI Pydantic schemas
    └── __tests__/
        ├── App.test.tsx        # End-to-end integration flow tests
        └── components.test.tsx # Focused component unit tests
```

---

## 3. Environment Configuration

The frontend connects to the FastAPI backend using the `VITE_API_BASE_URL` environment variable.

Create or verify `frontend/.env`:
```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## 4. Setup and Installation

From the project root or the `frontend/` directory:

```bash
cd frontend
npm install
```

---

## 5. Available Scripts

Inside the `frontend/` directory:

| Command | Action |
| :--- | :--- |
| `npm run dev` | Starts Vite development server at `http://127.0.0.1:5173` |
| `npm test` | Runs the Vitest test suite (`24` tests across 2 suites) |
| `npm run build` | Compiles TypeScript and builds production bundle to `dist/` |
| `npm run preview` | Previews the production build locally |

---

## 6. How to Run the Complete System Locally

To run the complete NL2SQL system (Backend + Frontend):

### Step 1: Start the FastAPI Backend
In your first terminal (from project root):
```bash
python run_server.py --port 8000
```
*Backend runs at `http://127.0.0.1:8000` with Swagger docs at `http://127.0.0.1:8000/docs`.*

### Step 2: Start the React Frontend
In a second terminal:
```bash
cd frontend
npm run dev
```
*Frontend runs at `http://127.0.0.1:5173`.*

Open your browser to `http://127.0.0.1:5173`.

---

## 7. User Workflow

1. **Health Verification:** On load, the header queries `GET /api/health` and displays live model and database status.
2. **Schema Inspection:** The sidebar queries `GET /api/schema` to dynamically render tables, columns, primary keys, and foreign keys.
3. **Query Input:** Enter a question in natural language (or click a suggested pill like *"Show all students with CGPA above 8.5"*).
4. **SQL Generation:** Click **Generate SQL** (`POST /api/generate`). The model translates the question and performs AST validation without executing.
5. **Review & Approval:** Review the generated SQL and validation badges.
6. **Execution Gate:** Click **Run Query on MySQL** (`POST /api/execute`). The server re-validates the query and executes it safely using the read-only `nl2sql_exec` account.
7. **Results Display:** Tabular records, column headers, execution time (ms), and row counts appear in the results card.
