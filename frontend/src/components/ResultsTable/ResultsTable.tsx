import React from 'react';
import { ExecuteResponse } from '../../types/api';

interface ResultsTableProps {
  result: ExecuteResponse;
  onClear?: () => void;
}

export const ResultsTable: React.FC<ResultsTableProps> = ({ result, onClear }) => {
  const {
    success,
    columns,
    rows,
    row_count,
    execution_time_ms,
    is_truncated,
    warning,
    error,
    executed_sql,
  } = result;

  return (
    <section className="results-card" data-testid="results-section">
      <div className="results-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span>📊</span>
          <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Query Execution Results</span>
          <span
            className={`val-badge ${success ? 'val-pass' : 'val-fail'}`}
            data-testid="execution-status-badge"
          >
            {success ? '✓ Query Executed' : '✗ Execution Failed'}
          </span>
        </div>

        <div className="results-metrics">
          <div className="metric-item">
            <span>Execution Time:</span>
            <span className="metric-val">{execution_time_ms.toFixed(2)} ms</span>
          </div>

          <div className="metric-item">
            <span>Rows:</span>
            <span className="metric-val">{row_count}</span>
          </div>

          {onClear && (
            <button
              type="button"
              className="icon-button"
              style={{ fontSize: '0.75rem', padding: '3px 8px' }}
              onClick={onClear}
              title="Clear Results"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Execution Error Banner */}
      {(!success || error) && (
        <div style={{ padding: 16 }}>
          <div className="banner banner-error" data-testid="execution-error-banner">
            <span>❌</span>
            <div>
              <strong>Database Execution Error: </strong>
              <span>{error || 'An unexpected database error occurred.'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Truncation Warning Notice */}
      {is_truncated && (
        <div style={{ padding: '12px 16px 0' }}>
          <div className="banner banner-warning" data-testid="truncation-warning-banner">
            <span>⚠️</span>
            <span>{warning || 'Result set truncated to limit.'}</span>
          </div>
        </div>
      )}

      {/* Empty Result Set State */}
      {success && row_count === 0 && (
        <div className="empty-state" data-testid="empty-results-state">
          <span className="empty-icon">🔍</span>
          <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
            No records matched this query.
          </div>
          <div style={{ fontSize: '0.82rem' }}>
            The query executed successfully on MySQL, but zero rows satisfied the filter criteria.
          </div>
          {executed_sql && (
            <code
              style={{
                marginTop: 8,
                padding: '4px 10px',
                background: 'var(--bg-primary)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
              }}
            >
              {executed_sql}
            </code>
          )}
        </div>
      )}

      {/* Structured Results Data Table */}
      {success && row_count > 0 && columns.length > 0 && (
        <div className="table-scroll-wrapper" data-testid="results-table-wrapper">
          <table className="data-table" data-testid="results-data-table">
            <thead>
              <tr>
                <th style={{ width: 48, textAlign: 'center' }}>#</th>
                {columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex} data-testid={`result-row-${rowIndex}`}>
                  <td
                    style={{
                      textAlign: 'center',
                      color: 'var(--text-dim)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.75rem',
                    }}
                  >
                    {rowIndex + 1}
                  </td>
                  {columns.map((col) => {
                    const value = row[col];
                    const isNull = value === null || value === undefined;

                    return (
                      <td key={col} className={isNull ? 'null-cell' : ''}>
                        {isNull ? 'NULL' : typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
