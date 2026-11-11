"""
Database package for MySQL connectivity and schema inspection (Milestone 4, Task 4.2).
"""

from src.database.config import (
    DatabaseConfig,
    DatabaseConfigurationError,
    DatabaseError,
    parse_dotenv_file,
)
from src.database.connection import (
    DatabaseAuthenticationError,
    DatabaseConnection,
    DatabaseConnectionError,
    DatabaseNotFoundError,
    sanitize_db_error,
)
from src.database.executor import ExecutionResult, SQLExecutor
from src.database.schema import (
    ColumnMetadata,
    DatabaseSchemaInspector,
    DatabaseSchemaMetadata,
    ForeignKeyMetadata,
    TableMetadata,
)

__all__ = [
    "DatabaseConfig",
    "parse_dotenv_file",
    "DatabaseConnection",
    "DatabaseError",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseAuthenticationError",
    "DatabaseNotFoundError",
    "sanitize_db_error",
    "ExecutionResult",
    "SQLExecutor",
    "ColumnMetadata",
    "ForeignKeyMetadata",
    "TableMetadata",
    "DatabaseSchemaMetadata",
    "DatabaseSchemaInspector",
]
