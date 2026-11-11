import React, { useState } from 'react';
import { GenerateResponse } from '../../types/api';

interface SQLPreviewProps {
  generation: GenerateResponse;
  onExecute: (sql: string, maxRows: number) => void;
  isExecuting: boolean;
}

export const SQLPreview: React.FC<SQLPreviewProps> = ({
  generation,
  onExecute,
  isExecuting,
}) => {
  const [copied, setCopied] = useState(false);
  const [maxRows, setMaxRows] = useState<number>(50);

  const { generated_sql, validation, can_execute, error } = generation;

  const handleCopy = () => {
    if (generated_sql) {
      navigator.clipboard.writeText(generated_sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRun = () => {
    if (generated_sql && can_execute && !isExecuting) {
      onExecute(generated_sql, maxRows);
    }
  };

  return (
    <section className="preview-card" data-testid="sql-preview-section">
      <div className="preview-card-header">
        <div className="preview-meta">
          <span>⚡</span>
          <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Model Generated SQL</span>
          <span
            className={`val-badge ${can_execute ? 'val-pass' : 'val-fail'}`}
            data-testid="validation-status-badge"
          >
            {can_execute ? '✓ Validated' : '✗ Validation Blocked'}
          </span>
        </div>

        {generated_sql && (
          <button
            type="button"
            className="icon-button"
            onClick={handleCopy}
            title="Copy SQL to Clipboard"
            data-testid="copy-sql-button"
          >
            <span>{copied ? '✓' : '📋'}</span>
            <span>{copied ? 'Copied!' : 'Copy SQL'}</span>
          </button>
        )}
      </div>

      {/* Code Display */}
      <div className="code-container">
        {generated_sql ? (
          <pre className="sql-code-block" data-testid="generated-sql-code">
            <code>{generated_sql}</code>
          </pre>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No SQL generated.
          </div>
        )}
      </div>

      {/* Validation Checks Bar */}
      <div className="validation-badges-bar">
        <span
          className={`val-badge ${validation.is_syntax_valid ? 'val-pass' : 'val-fail'}`}
          title="Syntactically valid MySQL query"
        >
          {validation.is_syntax_valid ? '✓' : '✗'} Syntax
        </span>

        <span
          className={`val-badge ${validation.is_permitted_read_only ? 'val-pass' : 'val-fail'}`}
          title="Strictly read-only single SELECT (no mutations or locking)"
        >
          {validation.is_permitted_read_only ? '✓' : '✗'} Read-Only
        </span>

        <span
          className={`val-badge ${validation.is_schema_compatible ? 'val-pass' : 'val-fail'}`}
          title="All tables and columns exist in active database schema"
        >
          {validation.is_schema_compatible ? '✓' : '✗'} Schema Match
        </span>

        <span
          className={`val-badge ${validation.is_in_model_scope ? 'val-pass' : 'val-fail'}`}
          title="Complies strictly with the trained model's closed target vocabulary"
        >
          {validation.is_in_model_scope ? '✓' : '✗'} Model Scope
        </span>

        {/* AST Tags */}
        {validation.ast_details?.tables && validation.ast_details.tables.length > 0 && (
          <span
            style={{
              fontSize: '0.75rem',
              color: 'var(--text-dim)',
              marginLeft: 'auto',
              fontFamily: 'var(--font-mono)',
            }}
          >
            Tables: {validation.ast_details.tables.join(', ')}
          </span>
        )}
      </div>

      {/* Validation Error Banner */}
      {!can_execute && (
        <div style={{ padding: '0 20px 16px' }}>
          <div className="banner banner-error" data-testid="validation-error-banner">
            <span>⛔</span>
            <div>
              <strong>Execution Guard Triggered: </strong>
              <span>{error || validation.errors[0] || 'Query failed safety verification.'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Execution Approval Action Gate */}
      <div className="approval-action-bar">
        <div className="approval-notice">
          <span>🔒</span>
          <span>
            {can_execute
              ? 'Safety inspection passed. Explicit approval required to execute.'
              : 'Execution is disabled because this query failed safety validation.'}
          </span>
        </div>

        <div className="approval-controls">
          <label
            htmlFor="max-rows-select"
            style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}
          >
            Row Limit:
          </label>
          <select
            id="max-rows-select"
            className="select-limit"
            value={maxRows}
            onChange={(e) => setMaxRows(Number(e.target.value))}
            disabled={!can_execute || isExecuting}
          >
            <option value={10}>10 rows</option>
            <option value={25}>25 rows</option>
            <option value={50}>50 rows</option>
            <option value={100}>100 rows</option>
          </select>

          <button
            type="button"
            className="execute-button"
            onClick={handleRun}
            disabled={!can_execute || isExecuting}
            data-testid="execute-sql-button"
          >
            {isExecuting && <span className="spinner" />}
            <span>{isExecuting ? 'Executing Query...' : 'Run Query on MySQL'}</span>
          </button>
        </div>
      </div>
    </section>
  );
};
