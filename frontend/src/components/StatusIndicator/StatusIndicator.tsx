import React from 'react';

interface StatusIndicatorProps {
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  label: string;
  pulsing?: boolean;
  title?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  pulsing = false,
  title,
}) => {
  const normalizedStatus =
    status === 'healthy' || status === 'degraded' || status === 'unhealthy'
      ? status
      : 'unhealthy';

  return (
    <div
      className={`status-pill status-${normalizedStatus}`}
      title={title || `Status: ${status}`}
      data-testid="status-indicator"
    >
      <span className={`status-dot ${pulsing ? 'pulsing' : ''}`} />
      <span>{label}</span>
    </div>
  );
};
