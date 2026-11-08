import React from 'react';
import { HealthResponse } from '../../types/api';
import { StatusIndicator } from '../StatusIndicator/StatusIndicator';

interface HeaderProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onRefresh: () => void;
  onToggleSidebar?: () => void;
  isSidebarOpen?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  isLoading,
  onRefresh,
  onToggleSidebar,
  isSidebarOpen = true,
}) => {
  const modelStatusText = health?.model.loaded
    ? `Model Ready (${health.model.device || 'CPU'})`
    : 'Model Offline';

  const dbStatusText = health?.database.connected
    ? `MySQL Connected (${health.database.database || 'nl2sql_student_db'})`
    : 'Database Offline';

  return (
    <header className="app-header" data-testid="app-header">
      <div className="header-brand">
        {onToggleSidebar && (
          <button
            className="icon-button"
            onClick={onToggleSidebar}
            title={isSidebarOpen ? 'Collapse Schema Explorer' : 'Expand Schema Explorer'}
            aria-label="Toggle Schema Explorer"
          >
            {isSidebarOpen ? '◀' : '▶'}
          </button>
        )}
        <div className="brand-icon">⚡</div>
        <div>
          <div className="brand-title">
            NL2SQL Assistant
            <span className="brand-badge">Milestone 5</span>
          </div>
        </div>
      </div>

      <div className="header-status-group">
        {/* Model Status */}
        <StatusIndicator
          status={health?.model.loaded ? 'healthy' : 'unhealthy'}
          label={modelStatusText}
          title={`Checkpoint: ${health?.model.checkpoint || 'N/A'}`}
        />

        {/* Database Status */}
        <StatusIndicator
          status={health?.database.connected ? 'healthy' : 'unhealthy'}
          label={dbStatusText}
          title={`Host: ${health?.database.host || 'localhost'}:${health?.database.port || 3306} | User: ${health?.database.exec_user || 'nl2sql_exec'}`}
        />

        {/* Refresh Button */}
        <button
          className="icon-button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh Backend Health Status"
        >
          <span className={isLoading ? 'spinner' : ''}>{isLoading ? '' : '🔄'}</span>
          <span>Refresh</span>
        </button>
      </div>
    </header>
  );
};
