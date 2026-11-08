import React, { useState } from 'react';
import { SchemaResponse } from '../../types/api';

interface SchemaExplorerProps {
  schema: SchemaResponse | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
  onSelectTable?: (tableName: string) => void;
}

export const SchemaExplorer: React.FC<SchemaExplorerProps> = ({
  schema,
  isLoading,
  error,
  onRetry,
  onSelectTable,
}) => {
  // Keep track of expanded tables
  const [expandedTables, setExpandedTables] = useState<Record<string, boolean>>({});

  const toggleTable = (tableName: string) => {
    setExpandedTables((prev) => ({
      ...prev,
      [tableName]: prev[tableName] === undefined ? false : !prev[tableName],
    }));
    onSelectTable?.(tableName);
  };

  const isTableExpanded = (tableName: string) => {
    // Default to true (expanded) if unset
    return expandedTables[tableName] !== false;
  };

  return (
    <aside className="sidebar-panel" data-testid="schema-explorer">
      <div className="sidebar-header">
        <div className="sidebar-title">
          <span>🗄️</span>
          <span>Database Schema</span>
        </div>
        {schema && (
          <span className="schema-badge" title="Active Inspected Database">
            {schema.database_name}
          </span>
        )}
      </div>

      <div className="sidebar-content">
        {isLoading && (
          <div className="empty-state">
            <div className="spinner" style={{ width: 24, height: 24 }} />
            <span>Inspecting database schema...</span>
          </div>
        )}

        {error && (
          <div className="banner banner-error" style={{ flexDirection: 'column', gap: 8 }}>
            <div style={{ fontWeight: 600 }}>Schema Inspection Failed</div>
            <div style={{ fontSize: '0.8rem' }}>{error}</div>
            <button
              className="icon-button"
              style={{ alignSelf: 'flex-start', marginTop: 4 }}
              onClick={onRetry}
            >
              🔄 Retry Inspection
            </button>
          </div>
        )}

        {!isLoading && !error && (!schema || Object.keys(schema.tables).length === 0) && (
          <div className="empty-state">
            <span className="empty-icon">📂</span>
            <span>No tables found in database.</span>
            <button className="icon-button" onClick={onRetry}>
              🔄 Refresh
            </button>
          </div>
        )}

        {!isLoading && schema && Object.keys(schema.tables).length > 0 && (
          <>
            <div
              style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                display: 'flex',
                justifyContent: 'space-between',
              }}
            >
              <span>{schema.table_count} Tables</span>
              <span>{schema.foreign_key_count} Foreign Keys</span>
            </div>

            {Object.entries(schema.tables).map(([tableName, table]) => {
              const expanded = isTableExpanded(tableName);

              return (
                <div key={tableName} className="table-card" data-testid={`table-card-${tableName}`}>
                  <button
                    className="table-header-btn"
                    onClick={() => toggleTable(tableName)}
                    aria-expanded={expanded}
                  >
                    <div className="table-name-wrapper">
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {expanded ? '▼' : '▶'}
                      </span>
                      <span>{tableName}</span>
                    </div>
                    <span
                      style={{
                        fontSize: '0.72rem',
                        color: 'var(--text-muted)',
                        fontWeight: 400,
                      }}
                    >
                      {table.columns.length} cols
                    </span>
                  </button>

                  {expanded && (
                    <div className="table-columns-list">
                      {table.columns.map((col) => (
                        <div
                          key={col.name}
                          className="column-row"
                          title={`${col.column_type} ${col.is_nullable ? 'NULL' : 'NOT NULL'}`}
                        >
                          <div className="column-name">
                            {col.is_primary_key && (
                              <span className="pk-indicator" title="Primary Key">
                                PK
                              </span>
                            )}
                            {table.foreign_keys.some((fk) => fk.column_name === col.name) && (
                              <span className="fk-indicator" title="Foreign Key">
                                FK
                              </span>
                            )}
                            <span>{col.name}</span>
                          </div>
                          <span className="column-type">{col.column_type}</span>
                        </div>
                      ))}

                      {/* Foreign Keys Details if present */}
                      {table.foreign_keys.length > 0 && (
                        <div
                          style={{
                            marginTop: 6,
                            paddingTop: 6,
                            borderTop: '1px dashed var(--border-subtle)',
                            fontSize: '0.7rem',
                            color: 'var(--text-dim)',
                          }}
                        >
                          {table.foreign_keys.map((fk) => (
                            <div key={fk.constraint_name} style={{ margin: '2px 0' }}>
                              🔗 {fk.column_name} → {fk.referenced_table}({fk.referenced_column})
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}
      </div>
    </aside>
  );
};
