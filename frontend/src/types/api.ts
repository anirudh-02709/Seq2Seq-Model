/**
 * TypeScript data models matching the FastAPI backend schemas.
 */

export interface ModelHealthInfo {
  loaded: boolean;
  checkpoint: string | null;
  source_vocab_size: number | null;
  target_vocab_size: number | null;
  device: string | null;
}

export interface DatabaseHealthInfo {
  connected: boolean;
  database: string | null;
  host: string | null;
  port: number | null;
  exec_user: string | null;
  mysql_version: string | null;
  error: string | null;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  model: ModelHealthInfo;
  database: DatabaseHealthInfo;
}

export interface ColumnInfo {
  name: string;
  data_type: string;
  column_type: string;
  is_nullable: boolean;
  is_primary_key: boolean;
  column_default: string | null;
  extra: string | null;
}

export interface ForeignKeyInfo {
  constraint_name: string;
  column_name: string;
  referenced_table: string;
  referenced_column: string;
  update_rule: string | null;
  delete_rule: string | null;
}

export interface TableInfo {
  table_name: string;
  columns: ColumnInfo[];
  primary_keys: string[];
  foreign_keys: ForeignKeyInfo[];
}

export interface SchemaResponse {
  database_name: string;
  canonical_schema: string;
  table_count: number;
  foreign_key_count: number;
  tables: Record<string, TableInfo>;
}

export interface AstDetails {
  statement_type?: string;
  tables?: string[];
  columns?: string[];
  aliases?: Record<string, string>;
  has_where?: boolean;
  has_order_by?: boolean;
  has_limit?: boolean;
  has_joins?: boolean;
  has_aggregations?: boolean;
  has_group_by?: boolean;
  has_having?: boolean;
  has_subqueries?: boolean;
  has_locks?: boolean;
  has_into?: boolean;
  [key: string]: any;
}

export interface ValidationSummary {
  is_valid: boolean;
  is_syntax_valid: boolean;
  is_permitted_read_only: boolean;
  is_schema_compatible: boolean;
  is_in_supported_scope: boolean;
  is_in_model_scope: boolean;
  normalized_sql: string | null;
  raw_sql: string | null;
  errors: string[];
  warnings: string[];
  ast_details: AstDetails;
}

export interface InferenceDiagnostics {
  tokens: string[];
  source_tokens: string[];
  source_unk_count: number;
  target_unk_count: number;
}

export interface GenerateRequest {
  question: string;
  schema?: string | null;
  max_length?: number;
}

export interface GenerateResponse {
  question: string;
  schema: string;
  generated_sql: string | null;
  validation: ValidationSummary;
  inference_diagnostics?: InferenceDiagnostics | null;
  stage_reached: string;
  can_execute: boolean;
  error: string | null;
}

export interface ExecuteRequest {
  sql: string;
  schema?: string | null;
  approved: boolean;
  max_rows?: number;
}

export interface ExecuteResponse {
  success: boolean;
  executed_sql: string | null;
  columns: string[];
  rows: Record<string, any>[];
  row_count: number;
  execution_time_ms: number;
  is_truncated: boolean;
  warning: string | null;
  error: string | null;
  validation?: ValidationSummary | null;
}
