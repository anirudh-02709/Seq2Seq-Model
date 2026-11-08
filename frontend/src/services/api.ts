/**
 * Centralized API Service for communicating with the FastAPI backend.
 */

import {
  ExecuteRequest,
  ExecuteResponse,
  GenerateRequest,
  GenerateResponse,
  HealthResponse,
  SchemaResponse,
} from '../types/api';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || 'http://127.0.0.1:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        if (typeof errorJson.detail === 'string') {
          errorMessage = errorJson.detail;
        } else if (Array.isArray(errorJson.detail)) {
          // Pydantic validation error format
          errorMessage = errorJson.detail
            .map((err: any) => `${err.loc?.join('.') || 'field'}: ${err.msg}`)
            .join('; ');
        }
      }
    } catch {
      // If response body is not JSON, retain default status text
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Retrieves the operational health status of the model and database.
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });
  return handleResponse<HealthResponse>(response);
}

/**
 * Retrieves database schema metadata via live inspection.
 */
export async function getSchema(): Promise<SchemaResponse> {
  const response = await fetch(`${API_BASE_URL}/api/schema`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });
  return handleResponse<SchemaResponse>(response);
}

/**
 * Converts a natural language question into candidate SQL and AST validation metadata.
 * Does NOT execute against MySQL.
 */
export async function generateSQL(req: GenerateRequest): Promise<GenerateResponse> {
  const response = await fetch(`${API_BASE_URL}/api/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      question: req.question,
      schema: req.schema || null,
      max_length: req.max_length || 60,
    }),
  });
  return handleResponse<GenerateResponse>(response);
}

/**
 * Re-validates candidate SQL and executes against MySQL with explicit approval.
 */
export async function executeSQL(req: ExecuteRequest): Promise<ExecuteResponse> {
  const response = await fetch(`${API_BASE_URL}/api/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      sql: req.sql,
      schema: req.schema || null,
      approved: req.approved,
      max_rows: req.max_rows || 100,
    }),
  });
  return handleResponse<ExecuteResponse>(response);
}
