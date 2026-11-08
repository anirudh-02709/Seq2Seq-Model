"""
Database Schema Inspection Module for NL2SQL Project (Milestone 4, Task 4.2).

Inspects MySQL metadata using INFORMATION_SCHEMA to retrieve database tables,
columns, data types, primary keys, and foreign keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from src.database.config import DatabaseConfig
from src.database.connection import DatabaseConnection


@dataclass
class ColumnMetadata:
    """Metadata describing a single table column."""
    name: str
    data_type: str
    column_type: str
    is_nullable: bool
    is_primary_key: bool = False
    column_default: Optional[str] = None
    extra: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "column_type": self.column_type,
            "is_nullable": self.is_nullable,
            "is_primary_key": self.is_primary_key,
            "column_default": self.column_default,
            "extra": self.extra,
        }


@dataclass
class ForeignKeyMetadata:
    """Metadata describing a foreign key constraint."""
    constraint_name: str
    column_name: str
    referenced_table: str
    referenced_column: str
    update_rule: Optional[str] = None
    delete_rule: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "column_name": self.column_name,
            "referenced_table": self.referenced_table,
            "referenced_column": self.referenced_column,
            "update_rule": self.update_rule,
            "delete_rule": self.delete_rule,
        }


@dataclass
class TableMetadata:
    """Metadata describing a table within the database."""
    name: str
    columns: list[ColumnMetadata] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyMetadata] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
        }

    def format_signature(self) -> str:
        """Returns string signature, e.g., 'students(student_id, name, age, department, cgpa)'."""
        cols_str = ", ".join(self.column_names)
        return f"{self.name}({cols_str})"


@dataclass
class DatabaseSchemaMetadata:
    """Comprehensive metadata for the entire inspected database."""
    database_name: str
    tables: dict[str, TableMetadata] = field(default_factory=dict)

    @property
    def table_names(self) -> list[str]:
        return sorted(list(self.tables.keys()))

    @property
    def total_tables(self) -> int:
        return len(self.tables)

    @property
    def total_foreign_keys(self) -> int:
        return sum(len(t.foreign_keys) for t in self.tables.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_name": self.database_name,
            "table_count": self.total_tables,
            "foreign_key_count": self.total_foreign_keys,
            "tables": {t_name: t_meta.to_dict() for t_name, t_meta in self.tables.items()},
        }

    def format_schema_summary(self) -> str:
        """
        Formats all tables into the project's canonical schema format:
        table1(col1, col2) | table2(...) | table3(...)
        """
        sigs = [self.tables[t].format_signature() for t in self.table_names]
        return " | ".join(sigs)


class DatabaseSchemaInspector:
    """
    Inspector class that queries MySQL INFORMATION_SCHEMA to extract structured schema metadata.
    """

    def __init__(
        self,
        connection: Optional[DatabaseConnection] = None,
        config: Optional[DatabaseConfig] = None,
    ) -> None:
        """
        Initializes the schema inspector.

        Args:
            connection: Optional DatabaseConnection instance.
            config: Optional DatabaseConfig instance.
        """
        if connection is not None:
            self.connection = connection
        elif config is not None:
            self.connection = DatabaseConnection(config=config)
        else:
            self.connection = DatabaseConnection.from_env()

        self.database_name = self.connection.config.database

    def get_table_names(self) -> list[str]:
        """
        Retrieves names of all BASE TABLEs in the target database.

        Returns:
            Alphabetically sorted list of table names.
        """
        query = """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s
              AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME ASC;
        """
        rows = self.connection.execute_read_query(query, (self.database_name,))
        return [row["TABLE_NAME"] for row in rows]

    def get_columns(self, table_name: str) -> list[ColumnMetadata]:
        """
        Retrieves column metadata for a specific table.

        Args:
            table_name: Target table name.

        Returns:
            List of ColumnMetadata instances ordered by ordinal position.
        """
        pk_columns = set(self.get_primary_keys(table_name))

        query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                COLUMN_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                EXTRA
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION ASC;
        """
        rows = self.connection.execute_read_query(query, (self.database_name, table_name))

        columns: list[ColumnMetadata] = []
        for row in rows:
            col_name = row["COLUMN_NAME"]
            is_pk = col_name in pk_columns
            columns.append(
                ColumnMetadata(
                    name=col_name,
                    data_type=row["DATA_TYPE"].lower(),
                    column_type=row["COLUMN_TYPE"],
                    is_nullable=(row["IS_NULLABLE"].upper() == "YES"),
                    is_primary_key=is_pk,
                    column_default=row.get("COLUMN_DEFAULT"),
                    extra=row.get("EXTRA"),
                )
            )
        return columns

    def get_primary_keys(self, table_name: str) -> list[str]:
        """
        Retrieves primary key column names for a table.

        Args:
            table_name: Target table name.

        Returns:
            List of primary key column names.
        """
        query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY ORDINAL_POSITION ASC;
        """
        rows = self.connection.execute_read_query(query, (self.database_name, table_name))
        return [row["COLUMN_NAME"] for row in rows]

    def get_foreign_keys(self, table_name: str) -> list[ForeignKeyMetadata]:
        """
        Retrieves foreign key constraints defined on a table.

        Args:
            table_name: Target table name.

        Returns:
            List of ForeignKeyMetadata objects.
        """
        query = """
            SELECT 
                kcu.CONSTRAINT_NAME,
                kcu.COLUMN_NAME,
                kcu.REFERENCED_TABLE_NAME,
                kcu.REFERENCED_COLUMN_NAME,
                rc.UPDATE_RULE,
                rc.DELETE_RULE
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            LEFT JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
              ON kcu.CONSTRAINT_SCHEMA = rc.CONSTRAINT_SCHEMA
             AND kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            WHERE kcu.TABLE_SCHEMA = %s
              AND kcu.TABLE_NAME = %s
              AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY kcu.ORDINAL_POSITION ASC;
        """
        rows = self.connection.execute_read_query(query, (self.database_name, table_name))

        fks: list[ForeignKeyMetadata] = []
        for row in rows:
            fks.append(
                ForeignKeyMetadata(
                    constraint_name=row["CONSTRAINT_NAME"],
                    column_name=row["COLUMN_NAME"],
                    referenced_table=row["REFERENCED_TABLE_NAME"],
                    referenced_column=row["REFERENCED_COLUMN_NAME"],
                    update_rule=row.get("UPDATE_RULE"),
                    delete_rule=row.get("DELETE_RULE"),
                )
            )
        return fks

    def inspect_table(self, table_name: str) -> TableMetadata:
        """
        Performs complete inspection of a single table.

        Args:
            table_name: Table name.

        Returns:
            Populated TableMetadata instance.
        """
        pks = self.get_primary_keys(table_name)
        fks = self.get_foreign_keys(table_name)
        cols = self.get_columns(table_name)
        return TableMetadata(
            name=table_name,
            columns=cols,
            primary_keys=pks,
            foreign_keys=fks,
        )

    def inspect_database(self) -> DatabaseSchemaMetadata:
        """
        Inspects all tables, columns, and constraints in the database.

        Returns:
            Complete DatabaseSchemaMetadata tree.
        """
        table_names = self.get_table_names()
        tables: dict[str, TableMetadata] = {}

        for t_name in table_names:
            tables[t_name] = self.inspect_table(t_name)

        return DatabaseSchemaMetadata(
            database_name=self.database_name,
            tables=tables,
        )
