"""
Database Configuration Module for NL2SQL Project (Milestone 4, Task 4.2).

Reads database connection settings from environment variables and optional .env files,
ensuring credentials are never hardcoded and are masked in logging and string representations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union


class DatabaseError(Exception):
    """Base exception for all database connectivity errors."""
    pass


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is invalid or missing."""
    pass


def parse_dotenv_file(filepath: Union[str, Path]) -> dict[str, str]:
    """
    Parses a simple .env file without external dependencies.
    Ignores comment lines starting with '#' and blank lines.
    Strips single and double quotes from values.

    Args:
        filepath: Path to the .env file.

    Returns:
        Dictionary of key-value environment pairs.
    """
    path = Path(filepath)
    if not path.is_file():
        return {}

    env_vars: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, _, val = stripped.partition("=")
                key = key.strip()
                val = val.strip()
                # Strip wrapping quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env_vars[key] = val
    return env_vars


@dataclass
class DatabaseConfig:
    """
    Structured database connection configuration.
    Masks passwords in string representations to prevent accidental credential leakage.
    """
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "nl2sql_student_db"
    connect_timeout: int = 5
    charset: str = "utf8mb4"
    autocommit: bool = False

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(f"Database port must be an integer between 1 and 65535, got {self.port}.")
        if not self.host or not isinstance(self.host, str):
            raise ValueError("Database host cannot be empty.")
        if not self.user or not isinstance(self.user, str):
            raise ValueError("Database user cannot be empty.")
        if not self.database or not isinstance(self.database, str):
            raise ValueError("Database name cannot be empty.")

    @property
    def masked_password(self) -> str:
        """Returns masked password string."""
        return "********" if self.password else ""

    def to_dict(self, mask_password: bool = True) -> dict[str, Any]:
        """
        Converts configuration to a dictionary.

        Args:
            mask_password: If True, masks the password field.

        Returns:
            Dictionary of configuration options.
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.masked_password if mask_password else self.password,
            "database": self.database,
            "connect_timeout": self.connect_timeout,
            "charset": self.charset,
            "autocommit": self.autocommit,
        }

    def __repr__(self) -> str:
        """Masked string representation."""
        return (
            f"DatabaseConfig(host='{self.host}', port={self.port}, "
            f"user='{self.user}', password='{self.masked_password}', "
            f"database='{self.database}', charset='{self.charset}')"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_env(
        cls,
        env_file: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> DatabaseConfig:
        """
        Constructs DatabaseConfig reading environment variables with fallback to .env file.

        Lookup order:
        1. Explicit overrides passed as kwargs.
        2. os.environ environment variables.
        3. Key-values in env_file or .env in project root.
        4. Default values.

        Supported Environment Variables:
        - MYSQL_HOST
        - MYSQL_PORT
        - MYSQL_USER
        - MYSQL_PASSWORD
        - MYSQL_DATABASE
        - MYSQL_CONNECT_TIMEOUT
        - MYSQL_CHARSET
        """
        # Load dotenv file if available
        file_vars: dict[str, str] = {}
        target_env = env_file or Path(__file__).resolve().parent.parent.parent / ".env"
        if target_env and Path(target_env).is_file():
            file_vars = parse_dotenv_file(target_env)

        def get_val(key: str, default: Any) -> Any:
            if key in overrides and overrides[key] is not None:
                return overrides[key]
            if key in os.environ:
                return os.environ[key]
            if key in file_vars:
                return file_vars[key]
            return default

        raw_port = get_val("MYSQL_PORT", 3306)
        try:
            port = int(raw_port)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid MYSQL_PORT '{raw_port}': must be an integer.") from err

        raw_timeout = get_val("MYSQL_CONNECT_TIMEOUT", 5)
        try:
            connect_timeout = int(raw_timeout)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid MYSQL_CONNECT_TIMEOUT '{raw_timeout}': must be an integer.") from err

        return cls(
            host=str(get_val("MYSQL_HOST", "localhost")),
            port=port,
            user=str(get_val("MYSQL_USER", "root")),
            password=str(get_val("MYSQL_PASSWORD", "")),
            database=str(get_val("MYSQL_DATABASE", "nl2sql_student_db")),
            connect_timeout=connect_timeout,
            charset=str(get_val("MYSQL_CHARSET", "utf8mb4")),
            autocommit=bool(get_val("MYSQL_AUTOCOMMIT", False)),
        )

    @classmethod
    def from_exec_env(
        cls,
        env_file: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> DatabaseConfig:
        """
        Constructs DatabaseConfig for dedicated read-only SQL execution (Milestone 4, Task 4.4).

        Reads MYSQL_EXEC_* environment variables with fallback to .env file.
        Strictly enforces least-privilege security:
        1. A dedicated execution user (MYSQL_EXEC_USER) is configured.
        2. The user cannot be 'root' (rejects root execution for safety).
        3. A non-empty password is provided (MYSQL_EXEC_PASSWORD).

        Supported Environment Variables:
        - MYSQL_EXEC_HOST (default: localhost)
        - MYSQL_EXEC_PORT (default: 3306)
        - MYSQL_EXEC_USER (mandatory, non-root)
        - MYSQL_EXEC_PASSWORD (mandatory)
        - MYSQL_EXEC_DATABASE (default: nl2sql_student_db)
        - MYSQL_EXEC_CONNECT_TIMEOUT (default: 5)
        - MYSQL_EXEC_CHARSET (default: utf8mb4)
        - MYSQL_EXEC_AUTOCOMMIT (default: False)

        Raises:
            DatabaseConfigurationError: If execution user or password is missing or user is 'root'.
        """
        file_vars: dict[str, str] = {}
        target_env = env_file or Path(__file__).resolve().parent.parent.parent / ".env"
        if target_env and Path(target_env).is_file():
            file_vars = parse_dotenv_file(target_env)

        def get_val(key: str, alt_key: str, default: Any) -> Any:
            if key in overrides and overrides[key] is not None:
                return overrides[key]
            if alt_key in overrides and overrides[alt_key] is not None:
                return overrides[alt_key]
            if key in os.environ:
                return os.environ[key]
            if key in file_vars:
                return file_vars[key]
            return default

        raw_user = get_val("MYSQL_EXEC_USER", "user", "")
        user = str(raw_user).strip() if raw_user is not None else ""
        if not user:
            raise DatabaseConfigurationError(
                "Execution user (MYSQL_EXEC_USER) is not configured. "
                "A dedicated read-only database account is required."
            )
        if user.lower() == "root":
            raise DatabaseConfigurationError(
                "Execution user cannot be 'root'. "
                "A dedicated non-root read-only account must be used for query execution."
            )

        raw_pass = get_val("MYSQL_EXEC_PASSWORD", "password", None)
        password = str(raw_pass) if raw_pass is not None else ""
        if not password:
            raise DatabaseConfigurationError(
                f"Execution password (MYSQL_EXEC_PASSWORD) for user '{user}' is not configured."
            )

        raw_port = get_val("MYSQL_EXEC_PORT", "port", 3306)
        try:
            port = int(raw_port)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid MYSQL_EXEC_PORT '{raw_port}': must be an integer.") from err

        raw_timeout = get_val("MYSQL_EXEC_CONNECT_TIMEOUT", "connect_timeout", 5)
        try:
            connect_timeout = int(raw_timeout)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid MYSQL_EXEC_CONNECT_TIMEOUT '{raw_timeout}': must be an integer.") from err

        return cls(
            host=str(get_val("MYSQL_EXEC_HOST", "host", "localhost")),
            port=port,
            user=user,
            password=password,
            database=str(get_val("MYSQL_EXEC_DATABASE", "database", "nl2sql_student_db")),
            connect_timeout=connect_timeout,
            charset=str(get_val("MYSQL_EXEC_CHARSET", "charset", "utf8mb4")),
            autocommit=bool(get_val("MYSQL_EXEC_AUTOCOMMIT", "autocommit", False)),
        )
