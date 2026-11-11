import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import * as api from '../services/api';
import {
  ExecuteResponse,
  GenerateResponse,
  HealthResponse,
  SchemaResponse,
} from '../types/api';

vi.mock('../services/api');

const mockHealthData: HealthResponse = {
  status: 'healthy',
  model: {
    loaded: true,
    checkpoint: 'checkpoints/candidates/phase2_clean_best.pt',
    source_vocab_size: 290,
    target_vocab_size: 96,
    device: 'cpu',
  },
  database: {
    connected: true,
    database: 'nl2sql_student_db',
    host: 'localhost',
    port: 3306,
    exec_user: 'nl2sql_exec',
    mysql_version: '8.0.43',
    error: null,
  },
};

const mockSchemaData: SchemaResponse = {
  database_name: 'nl2sql_student_db',
  canonical_schema: 'students(student_id, name, age, department, cgpa)',
  table_count: 1,
  foreign_key_count: 0,
  tables: {
    students: {
      table_name: 'students',
      columns: [
        {
          name: 'student_id',
          data_type: 'int',
          column_type: 'int',
          is_nullable: false,
          is_primary_key: true,
          column_default: null,
          extra: 'auto_increment',
        },
        {
          name: 'name',
          data_type: 'varchar',
          column_type: 'varchar(100)',
          is_nullable: false,
          is_primary_key: false,
          column_default: null,
          extra: null,
        },
        {
          name: 'cgpa',
          data_type: 'decimal',
          column_type: 'decimal(4,2)',
          is_nullable: true,
          is_primary_key: false,
          column_default: null,
          extra: null,
        },
      ],
      primary_keys: ['student_id'],
      foreign_keys: [],
    },
  },
};

const mockGenerateSuccess: GenerateResponse = {
  question: 'Show all students with CGPA above 8.5',
  schema: 'students(student_id, name, age, department, cgpa)',
  generated_sql: 'SELECT name, cgpa FROM students WHERE cgpa > 8.5',
  validation: {
    is_valid: true,
    is_syntax_valid: true,
    is_permitted_read_only: true,
    is_schema_compatible: true,
    is_in_supported_scope: true,
    is_in_model_scope: true,
    normalized_sql: 'SELECT name, cgpa FROM students WHERE cgpa > 8.5',
    raw_sql: 'SELECT name, cgpa FROM students WHERE cgpa > 8.5',
    errors: [],
    warnings: [],
    ast_details: { statement_type: 'SELECT', tables: ['students'], columns: ['name', 'cgpa'] },
  },
  stage_reached: 'execution_pending',
  can_execute: true,
  error: null,
};

const mockGenerateInvalid: GenerateResponse = {
  question: 'Delete all student records',
  schema: 'students(student_id, name, age, department, cgpa)',
  generated_sql: 'DELETE FROM students',
  validation: {
    is_valid: false,
    is_syntax_valid: true,
    is_permitted_read_only: false,
    is_schema_compatible: true,
    is_in_supported_scope: false,
    is_in_model_scope: false,
    normalized_sql: 'DELETE FROM students',
    raw_sql: 'DELETE FROM students',
    errors: ['Forbidden statement type: expected SELECT, but got DELETE.'],
    warnings: [],
    ast_details: {},
  },
  stage_reached: 'validation',
  can_execute: false,
  error: 'Forbidden statement type: expected SELECT, but got DELETE.',
};

const mockExecuteSuccess: ExecuteResponse = {
  success: true,
  executed_sql: 'SELECT name, cgpa FROM students WHERE cgpa > 8.5',
  columns: ['name', 'cgpa'],
  rows: [
    { name: 'Aarav Sharma', cgpa: 8.92 },
    { name: 'Priya Patel', cgpa: 9.15 },
  ],
  row_count: 2,
  execution_time_ms: 2.15,
  is_truncated: false,
  warning: null,
  error: null,
};

const mockExecuteEmpty: ExecuteResponse = {
  success: true,
  executed_sql: 'SELECT name FROM students WHERE cgpa > 9.99',
  columns: ['name'],
  rows: [],
  row_count: 0,
  execution_time_ms: 1.05,
  is_truncated: false,
  warning: null,
  error: null,
};

describe('NL2SQL Frontend Full Integration Flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getHealth).mockResolvedValue(mockHealthData);
    vi.mocked(api.getSchema).mockResolvedValue(mockSchemaData);
  });

  it('1. Renders application header and initial status indicators', async () => {
    render(<App />);

    expect(screen.getByText('NL2SQL Assistant')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Model Ready (cpu)')).toBeInTheDocument();
      expect(screen.getByText('MySQL Connected (nl2sql_student_db)')).toBeInTheDocument();
    });
  });

  it('2. Fetches and renders database schema dynamically', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('students')).toBeInTheDocument();
      expect(screen.getByText('student_id')).toBeInTheDocument();
      expect(screen.getByText('PK')).toBeInTheDocument();
      expect(screen.getByText('cgpa')).toBeInTheDocument();
    });
  });

  it('3. Displays error banner when schema inspection fails', async () => {
    vi.mocked(api.getSchema).mockRejectedValueOnce(new Error('Database offline'));
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Schema Inspection Failed')).toBeInTheDocument();
      expect(screen.getByText('Database offline')).toBeInTheDocument();
    });
  });

  it('4. Rejects empty or whitespace question on form submit', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: '   ' } });

    // Submit the form containing the input
    const form = input.closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByTestId('input-validation-error')).toBeInTheDocument();
      expect(screen.getByText('Please enter a natural language question.')).toBeInTheDocument();
    });
  });

  it('5. Generates SQL and displays AST validation badges', async () => {
    vi.mocked(api.generateSQL).mockResolvedValueOnce(mockGenerateSuccess);
    render(<App />);

    // Wait for schema to load first
    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: 'Show all students with CGPA above 8.5' } });

    const generateBtn = screen.getByTestId('generate-button');
    fireEvent.click(generateBtn);

    await waitFor(() => {
      expect(api.generateSQL).toHaveBeenCalledWith({
        question: 'Show all students with CGPA above 8.5',
        schema: 'students(student_id, name, age, department, cgpa)',
      });
      expect(screen.getByTestId('generated-sql-code')).toHaveTextContent(
        'SELECT name, cgpa FROM students WHERE cgpa > 8.5'
      );
      expect(screen.getByTestId('validation-status-badge')).toHaveTextContent('✓ Validated');
    });
  });

  it('6. Blocks execution when generated query is invalid (e.g. DELETE query)', async () => {
    vi.mocked(api.generateSQL).mockResolvedValueOnce(mockGenerateInvalid);
    render(<App />);

    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: 'Delete all student records' } });

    fireEvent.click(screen.getByTestId('generate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('validation-status-badge')).toHaveTextContent('✗ Validation Blocked');
      expect(screen.getByTestId('validation-error-banner')).toBeInTheDocument();
      const executeBtn = screen.getByTestId('execute-sql-button');
      expect(executeBtn).toBeDisabled();
    });
  });

  it('7. Approves and executes valid query, displaying structured records', async () => {
    vi.mocked(api.generateSQL).mockResolvedValueOnce(mockGenerateSuccess);
    vi.mocked(api.executeSQL).mockResolvedValueOnce(mockExecuteSuccess);
    render(<App />);

    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: 'Show all students with CGPA above 8.5' } });
    fireEvent.click(screen.getByTestId('generate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('execute-sql-button')).toBeEnabled();
    });

    fireEvent.click(screen.getByTestId('execute-sql-button'));

    await waitFor(() => {
      expect(api.executeSQL).toHaveBeenCalledWith({
        sql: 'SELECT name, cgpa FROM students WHERE cgpa > 8.5',
        schema: 'students(student_id, name, age, department, cgpa)',
        approved: true,
        max_rows: 50,
      });
      expect(screen.getByTestId('results-section')).toBeInTheDocument();
      expect(screen.getByText('Aarav Sharma')).toBeInTheDocument();
      expect(screen.getByText('Priya Patel')).toBeInTheDocument();
      expect(screen.getByText('2.15 ms')).toBeInTheDocument();
    });
  });

  it('8. Handles empty result set with clear informative message', async () => {
    vi.mocked(api.generateSQL).mockResolvedValueOnce(mockGenerateSuccess);
    vi.mocked(api.executeSQL).mockResolvedValueOnce(mockExecuteEmpty);
    render(<App />);

    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: 'Show all students with CGPA above 8.5' } });
    fireEvent.click(screen.getByTestId('generate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('execute-sql-button')).toBeEnabled();
    });

    fireEvent.click(screen.getByTestId('execute-sql-button'));

    await waitFor(() => {
      expect(screen.getByTestId('empty-results-state')).toBeInTheDocument();
      expect(screen.getByText('No records matched this query.')).toBeInTheDocument();
    });
  });

  it('9. Handles execution rejection error clearly without crashing', async () => {
    vi.mocked(api.generateSQL).mockResolvedValueOnce(mockGenerateSuccess);
    vi.mocked(api.executeSQL).mockRejectedValueOnce(
      new Error('Permission denied: user does not have permission to access table.')
    );
    render(<App />);

    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const input = screen.getByTestId('question-input');
    fireEvent.change(input, { target: { value: 'Show all students with CGPA above 8.5' } });
    fireEvent.click(screen.getByTestId('generate-button'));

    await waitFor(() => {
      expect(screen.getByTestId('execute-sql-button')).toBeEnabled();
    });

    fireEvent.click(screen.getByTestId('execute-sql-button'));

    await waitFor(() => {
      expect(screen.getByTestId('execute-error-banner')).toBeInTheDocument();
      expect(
        screen.getByText(/Permission denied: user does not have permission/i)
      ).toBeInTheDocument();
    });
  });

  it('10. Sidebar can be toggled open and closed', async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText('students')).toBeInTheDocument());

    const toggleBtn = screen.getByLabelText('Toggle Schema Explorer');
    expect(screen.getByTestId('schema-explorer')).toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.queryByTestId('schema-explorer')).not.toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.getByTestId('schema-explorer')).toBeInTheDocument();
  });
});
