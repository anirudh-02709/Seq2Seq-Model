import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Header } from '../components/Header/Header';
import { SchemaExplorer } from '../components/SchemaExplorer/SchemaExplorer';
import { QueryInput } from '../components/QueryInput/QueryInput';
import { SQLPreview } from '../components/SQLPreview/SQLPreview';
import { ResultsTable } from '../components/ResultsTable/ResultsTable';
import { ExecuteResponse, GenerateResponse, HealthResponse, SchemaResponse } from '../types/api';

describe('Individual Component Unit Tests', () => {
  // ==========================================================================
  // Header Tests
  // ==========================================================================
  describe('Header Component', () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      model: { loaded: true, checkpoint: 'checkpoints/candidates/phase2_clean_best.pt', source_vocab_size: 290, target_vocab_size: 96, device: 'cuda' },
      database: { connected: true, database: 'test_db', host: 'localhost', port: 3306, exec_user: 'nl2sql_exec', mysql_version: '8.0', error: null },
    };

    it('renders title, model device, and database name', () => {
      render(<Header health={mockHealth} isLoading={false} onRefresh={vi.fn()} />);
      expect(screen.getByText('NL2SQL Assistant')).toBeInTheDocument();
      expect(screen.getByText('Model Ready (cuda)')).toBeInTheDocument();
      expect(screen.getByText('MySQL Connected (test_db)')).toBeInTheDocument();
    });

    it('triggers refresh callback on button click', () => {
      const onRefresh = vi.fn();
      render(<Header health={mockHealth} isLoading={false} onRefresh={onRefresh} />);
      fireEvent.click(screen.getByText('Refresh'));
      expect(onRefresh).toHaveBeenCalledOnce();
    });

    it('triggers sidebar toggle callback when button clicked', () => {
      const onToggle = vi.fn();
      render(<Header health={mockHealth} isLoading={false} onRefresh={vi.fn()} onToggleSidebar={onToggle} isSidebarOpen={true} />);
      fireEvent.click(screen.getByLabelText('Toggle Schema Explorer'));
      expect(onToggle).toHaveBeenCalledOnce();
    });
  });

  // ==========================================================================
  // SchemaExplorer Tests
  // ==========================================================================
  describe('SchemaExplorer Component', () => {
    const mockSchema: SchemaResponse = {
      database_name: 'test_db',
      canonical_schema: 'users(id, email)',
      table_count: 1,
      foreign_key_count: 0,
      tables: {
        users: {
          table_name: 'users',
          columns: [
            { name: 'id', data_type: 'int', column_type: 'int', is_nullable: false, is_primary_key: true, column_default: null, extra: null },
            { name: 'email', data_type: 'varchar', column_type: 'varchar(255)', is_nullable: false, is_primary_key: false, column_default: null, extra: null },
          ],
          primary_keys: ['id'],
          foreign_keys: [],
        },
      },
    };

    it('renders loading spinner when isLoading=true', () => {
      render(<SchemaExplorer schema={null} isLoading={true} error={null} onRetry={vi.fn()} />);
      expect(screen.getByText('Inspecting database schema...')).toBeInTheDocument();
    });

    it('renders error state with retry button', () => {
      const onRetry = vi.fn();
      render(<SchemaExplorer schema={null} isLoading={false} error="Connection timeout" onRetry={onRetry} />);
      expect(screen.getByText('Schema Inspection Failed')).toBeInTheDocument();
      expect(screen.getByText('Connection timeout')).toBeInTheDocument();
      fireEvent.click(screen.getByText('🔄 Retry Inspection'));
      expect(onRetry).toHaveBeenCalledOnce();
    });

    it('collapses and expands table columns on header click', () => {
      render(<SchemaExplorer schema={mockSchema} isLoading={false} error={null} onRetry={vi.fn()} />);
      expect(screen.getByText('email')).toBeInTheDocument();

      // Click to collapse
      fireEvent.click(screen.getByText('users'));
      expect(screen.queryByText('email')).not.toBeInTheDocument();

      // Click to expand
      fireEvent.click(screen.getByText('users'));
      expect(screen.getByText('email')).toBeInTheDocument();
    });
  });

  // ==========================================================================
  // QueryInput Tests
  // ==========================================================================
  describe('QueryInput Component', () => {
    it('submits valid input when form is submitted', () => {
      const onGenerate = vi.fn();
      render(<QueryInput onGenerate={onGenerate} isLoading={false} />);
      const input = screen.getByTestId('question-input');
      fireEvent.change(input, { target: { value: 'Show students' } });
      fireEvent.click(screen.getByTestId('generate-button'));
      expect(onGenerate).toHaveBeenCalledWith('Show students');
    });

    it('clicking an example pill populates input and triggers generate', () => {
      const onGenerate = vi.fn();
      render(<QueryInput onGenerate={onGenerate} isLoading={false} />);
      fireEvent.click(screen.getByText('Find courses with credits equal to 4'));
      expect(onGenerate).toHaveBeenCalledWith('Find courses with credits equal to 4');
    });

    it('disables input and button when isLoading=true', () => {
      render(<QueryInput onGenerate={vi.fn()} isLoading={true} />);
      expect(screen.getByTestId('question-input')).toBeDisabled();
      expect(screen.getByTestId('generate-button')).toBeDisabled();
      expect(screen.getByText('Translating...')).toBeInTheDocument();
    });
  });

  // ==========================================================================
  // SQLPreview Tests
  // ==========================================================================
  describe('SQLPreview Component', () => {
    const mockValidGen: GenerateResponse = {
      question: 'List courses',
      schema: 'courses(id, name)',
      generated_sql: 'SELECT * FROM courses',
      validation: {
        is_valid: true,
        is_syntax_valid: true,
        is_permitted_read_only: true,
        is_schema_compatible: true,
        is_in_supported_scope: true,
        is_in_model_scope: true,
        normalized_sql: 'SELECT * FROM courses',
        raw_sql: 'SELECT * FROM courses',
        errors: [],
        warnings: [],
        ast_details: { statement_type: 'SELECT', tables: ['courses'] },
      },
      stage_reached: 'execution_pending',
      can_execute: true,
      error: null,
    };

    it('renders SQL code and allows execution when can_execute=true', () => {
      const onExecute = vi.fn();
      render(<SQLPreview generation={mockValidGen} onExecute={onExecute} isExecuting={false} />);
      expect(screen.getByTestId('generated-sql-code')).toHaveTextContent('SELECT * FROM courses');
      const runBtn = screen.getByTestId('execute-sql-button');
      expect(runBtn).toBeEnabled();

      fireEvent.click(runBtn);
      expect(onExecute).toHaveBeenCalledWith('SELECT * FROM courses', 50);
    });

    it('disables execution button when can_execute=false and shows error banner', () => {
      const mockInvalidGen: GenerateResponse = {
        ...mockValidGen,
        can_execute: false,
        error: 'Multi-statement injection rejected',
      };
      render(<SQLPreview generation={mockInvalidGen} onExecute={vi.fn()} isExecuting={false} />);
      const runBtn = screen.getByTestId('execute-sql-button');
      expect(runBtn).toBeDisabled();
      expect(screen.getByTestId('validation-error-banner')).toHaveTextContent('Multi-statement injection rejected');
    });
  });

  // ==========================================================================
  // ResultsTable Tests
  // ==========================================================================
  describe('ResultsTable Component', () => {
    it('renders dynamic headers, rows, and NULL cells properly', () => {
      const mockResult: ExecuteResponse = {
        success: true,
        executed_sql: 'SELECT name, age FROM students',
        columns: ['name', 'age'],
        rows: [
          { name: 'Aarav', age: 20 },
          { name: 'Unknown', age: null },
        ],
        row_count: 2,
        execution_time_ms: 1.84,
        is_truncated: false,
        warning: null,
        error: null,
      };

      render(<ResultsTable result={mockResult} />);
      expect(screen.getByText('Aarav')).toBeInTheDocument();
      expect(screen.getByText('20')).toBeInTheDocument();
      expect(screen.getByText('NULL')).toBeInTheDocument();
      expect(screen.getByText('1.84 ms')).toBeInTheDocument();
    });

    it('renders truncation alert when is_truncated=true', () => {
      const mockTruncated: ExecuteResponse = {
        success: true,
        executed_sql: 'SELECT * FROM students',
        columns: ['name'],
        rows: [{ name: 'Aarav' }],
        row_count: 1,
        execution_time_ms: 2.0,
        is_truncated: true,
        warning: 'Query returned more than 1 rows. Truncated to 1.',
        error: null,
      };

      render(<ResultsTable result={mockTruncated} />);
      expect(screen.getByTestId('truncation-warning-banner')).toHaveTextContent('Truncated to 1.');
    });

    it('renders execution error banner on database failure', () => {
      const mockFailed: ExecuteResponse = {
        success: false,
        executed_sql: 'SELECT * FROM students',
        columns: [],
        rows: [],
        row_count: 0,
        execution_time_ms: 0.5,
        is_truncated: false,
        warning: null,
        error: 'Table nl2sql_student_db.students does not exist',
      };

      render(<ResultsTable result={mockFailed} />);
      expect(screen.getByTestId('execution-error-banner')).toHaveTextContent('Table nl2sql_student_db.students does not exist');
    });
  });
});
