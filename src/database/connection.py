"""
Database Connection Module for NL2SQL Project (Milestone 4, Task 4.2).

Manages MySQL database connections using PyMySQL with explicit error handling,
context management for resource cleanup, and credential protection.
"""

from __future__ import annotations

from typing import Any, Optional, Type, Union

import pymysql
import pymysql.cursors

from src.database.config import DatabaseConfig, DatabaseError


class DatabaseConnectionError(DatabaseError):
    """Raised when connecting to MySQL server fails."""
    pass


class DatabaseAuthenticationError(DatabaseConnectionError):
    """Raised when authentication (credentials/user/password) fails."""
    pass


class DatabaseNotFoundError(DatabaseConnectionError):
    """Raised when the specified database does not exist."""
    pass


def sanitize_db_error(exc: Exception, raw_password: Optional[str] = None) -> str:
    """
    Strips raw passwords or sensitive credentials from error messages.

    Args:
        exc: Exception instance.
        raw_password: Password string to scrub from message if present.

    Returns:
        Sanitized error message string.
    """
    msg = str(exc)
    if raw_password and raw_password in msg:
        msg = msg.replace(raw_password, "********")
    return msg


class DatabaseConnection:
    """
    Database connection manager providing lifecycle control, verification,
    and automatic resource cleanup via context management.

    Example usage:
        >>> with DatabaseConnection.from_env() as conn:
        ...     with conn.cursor() as cur:
        ...         cur.execute("SELECT 1")
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """
        Initializes connection manager with configuration.

        Args:
            config: DatabaseConfig instance (reads from environment if None).
        """
        self.config = config if config is not None else DatabaseConfig.from_env()
        self._connection: Optional[pymysql.connections.Connection] = None

    @classmethod
    def from_env(
        cls,
        env_file: Optional[str] = None,
        **overrides: Any,
    ) -> DatabaseConnection:
        """Constructs DatabaseConnection loading configuration from environment variables."""
        config = DatabaseConfig.from_env(env_file=env_file, **overrides)
        return cls(config=config)

    @classmethod
    def from_exec_env(
        cls,
        env_file: Optional[str] = None,
        **overrides: Any,
    ) -> DatabaseConnection:
        """Constructs DatabaseConnection loading read-only execution configuration from environment variables."""
        config = DatabaseConfig.from_exec_env(env_file=env_file, **overrides)
        return cls(config=config)

    def connect(self) -> pymysql.connections.Connection:
        """
        Establishes a connection to the MySQL database if not already connected.

        Returns:
            Active pymysql Connection object.

        Raises:
            DatabaseAuthenticationError: If username or password is rejected.
            DatabaseNotFoundError: If the target database does not exist.
            DatabaseConnectionError: If network, host, or server is unreachable.
        """
        if self._connection is not None and self._connection.open:
            return self._connection

        try:
            self._connection = pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                connect_timeout=self.config.connect_timeout,
                charset=self.config.charset,
                autocommit=self.config.autocommit,
                cursorclass=pymysql.cursors.DictCursor,
            )
            return self._connection

        except pymysql.err.OperationalError as err:
            err_code = err.args[0]
            clean_msg = sanitize_db_error(err, self.config.password)

            if err_code == 1045:
                # ER_ACCESS_DENIED_ERROR
                raise DatabaseAuthenticationError(
                    f"Authentication failed for user '{self.config.user}'@'{self.config.host}': {clean_msg}"
                ) from err
            elif err_code == 1049:
                # ER_BAD_DB_ERROR
                raise DatabaseNotFoundError(
                    f"Database '{self.config.database}' not found on MySQL server at '{self.config.host}:{self.config.port}': {clean_msg}"
                ) from err
            elif err_code in (2003, 2005, 2013):
                # Cannot connect to MySQL server / Unknown MySQL server host / Lost connection
                raise DatabaseConnectionError(
                    f"Could not connect to MySQL server at '{self.config.host}:{self.config.port}': {clean_msg}"
                ) from err
            else:
                raise DatabaseConnectionError(
                    f"MySQL OperationalError ({err_code}) connecting to '{self.config.host}:{self.config.port}': {clean_msg}"
                ) from err

        except Exception as err:
            clean_msg = sanitize_db_error(err, self.config.password)
            raise DatabaseConnectionError(
                f"Unexpected error establishing MySQL connection to '{self.config.host}:{self.config.port}': {clean_msg}"
            ) from err

    def close(self) -> None:
        """Safely closes the connection and cleans up resources."""
        if self._connection is not None:
            try:
                if self._connection.open:
                    self._connection.close()
            except Exception:
                pass
            finally:
                self._connection = None

    def is_connected(self) -> bool:
        """Returns True if the connection is currently open and valid."""
        return self._connection is not None and bool(self._connection.open)

    def cursor(self, cursor_type: Optional[Type[pymysql.cursors.Cursor]] = None) -> Any:
        """
        Returns a cursor for executing queries. Ensures connection is open.
        """
        conn = self.connect()
        return conn.cursor(cursor_type) if cursor_type else conn.cursor()

    def execute_read_query(
        self,
        query: str,
        params: Optional[Union[tuple, list, dict]] = None,
    ) -> list[dict[str, Any]]:
        """
        Safely executes a read-only query and returns all result rows as a list of dictionaries.

        Args:
            query: SQL query string.
            params: Optional query parameters for parameterized execution.

        Returns:
            List of row dictionaries.
        """
        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def verify_connection(self) -> dict[str, Any]:
        """
        Executes a harmless verification query to test connectivity,
        database selection, and MySQL version.

        Returns:
            Verification result dictionary with status and metadata.
        """
        try:
            with self:
                with self.cursor() as cur:
                    cur.execute("SELECT 1 AS alive, DATABASE() AS current_db, VERSION() AS version;")
                    row = cur.fetchone()

            return {
                "status": "success",
                "connected": True,
                "alive": bool(row and row.get("alive") == 1),
                "database": row.get("current_db") if row else self.config.database,
                "mysql_version": row.get("version") if row else "unknown",
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.user,
                "error": None,
            }
        except DatabaseError as err:
            return {
                "status": "failure",
                "connected": False,
                "alive": False,
                "database": self.config.database,
                "mysql_version": None,
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.user,
                "error": str(err),
            }

    def __enter__(self) -> DatabaseConnection:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
