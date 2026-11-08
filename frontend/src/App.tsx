import React, { useEffect, useState } from 'react';
import { Header } from './components/Header/Header';
import { SchemaExplorer } from './components/SchemaExplorer/SchemaExplorer';
import { QueryInput } from './components/QueryInput/QueryInput';
import { SQLPreview } from './components/SQLPreview/SQLPreview';
import { ResultsTable } from './components/ResultsTable/ResultsTable';
import {
  ExecuteResponse,
  GenerateResponse,
  HealthResponse,
  SchemaResponse,
} from './types/api';
import { executeSQL, generateSQL, getHealth, getSchema } from './services/api';

export const App: React.FC = () => {
  // Service health & schema state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthLoading, setIsHealthLoading] = useState<boolean>(true);

  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [isSchemaLoading, setIsSchemaLoading] = useState<boolean>(true);
  const [schemaError, setSchemaError] = useState<string | null>(null);

  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);

  // Query generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generation, setGeneration] = useState<GenerateResponse | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  // Execution state
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [execution, setExecution] = useState<ExecuteResponse | null>(null);
  const [executeError, setExecuteError] = useState<string | null>(null);

  // Load initial backend health and schema metadata on mount
  const loadInitialData = async () => {
    setIsHealthLoading(true);
    try {
      const healthData = await getHealth();
      setHealth(healthData);
    } catch (err: any) {
      setHealth({
        status: 'unhealthy',
        model: {
          loaded: false,
          checkpoint: null,
          source_vocab_size: null,
          target_vocab_size: null,
          device: null,
        },
        database: {
          connected: false,
          database: null,
          host: null,
          port: null,
          exec_user: null,
          mysql_version: null,
          error: err.message || 'Cannot reach FastAPI backend.',
        },
      });
    } finally {
      setIsHealthLoading(false);
    }

    setIsSchemaLoading(true);
    setSchemaError(null);
    try {
      const schemaData = await getSchema();
      setSchema(schemaData);
    } catch (err: any) {
      setSchemaError(err.message || 'Failed to inspect database schema.');
    } finally {
      setIsSchemaLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Handle Natural Language Query Generation
  const handleGenerate = async (question: string) => {
    setIsGenerating(true);
    setGenerateError(null);
    setGeneration(null);
    setExecution(null);
    setExecuteError(null);

    try {
      const genResponse = await generateSQL({
        question,
        schema: schema?.canonical_schema || null,
      });
      setGeneration(genResponse);
    } catch (err: any) {
      setGenerateError(err.message || 'SQL translation failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  // Handle Approved SQL Execution
  const handleExecute = async (sql: string, maxRows: number) => {
    setIsExecuting(true);
    setExecuteError(null);

    try {
      const execResponse = await executeSQL({
        sql,
        schema: schema?.canonical_schema || null,
        approved: true,
        max_rows: maxRows,
      });
      setExecution(execResponse);
    } catch (err: any) {
      setExecuteError(err.message || 'Database execution rejected.');
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="app-layout" data-testid="app-root">
      {/* Top Application Header */}
      <Header
        health={health}
        isLoading={isHealthLoading}
        onRefresh={loadInitialData}
        onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
        isSidebarOpen={isSidebarOpen}
      />

      <div className="main-container">
        {/* Collapsible Database Schema Explorer */}
        {isSidebarOpen && (
          <SchemaExplorer
            schema={schema}
            isLoading={isSchemaLoading}
            error={schemaError}
            onRetry={loadInitialData}
          />
        )}

        {/* Central Content Area */}
        <main className="content-area">
          {/* Natural Language Prompt Input */}
          <QueryInput
            onGenerate={handleGenerate}
            isLoading={isGenerating}
            disabled={health?.status === 'unhealthy'}
          />

          {/* Error Banner for Generation Failure */}
          {generateError && (
            <div className="banner banner-error" data-testid="generate-error-banner">
              <span>⛔</span>
              <div>
                <strong>SQL Translation Error: </strong>
                <span>{generateError}</span>
              </div>
            </div>
          )}

          {/* Model Generated SQL Preview & Execution Gate */}
          {generation && (
            <SQLPreview
              generation={generation}
              onExecute={handleExecute}
              isExecuting={isExecuting}
            />
          )}

          {/* Error Banner for Execution Failure */}
          {executeError && (
            <div className="banner banner-error" data-testid="execute-error-banner">
              <span>❌</span>
              <div>
                <strong>Execution Rejection: </strong>
                <span>{executeError}</span>
              </div>
            </div>
          )}

          {/* Query Results Table */}
          {execution && (
            <ResultsTable
              result={execution}
              onClear={() => setExecution(null)}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
