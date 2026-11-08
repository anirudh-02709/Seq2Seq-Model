"""
FastAPI Application Factory and Lifecycle Management (Milestone 5.2).

Features:
- Application lifespan handler that initializes the Seq2Seq NL2SQL model once on startup.
- Configurable CORS middleware for React + TypeScript Vite frontend integration.
- Exposes OpenAPI / Swagger UI documentation at /docs.
- Clean separation between server initialization, routing, and core model logic.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.pipeline.orchestrator import NL2SQLPipeline

logger = logging.getLogger("src.api.app")

DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "best_model.pt"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def resolve_cors_origins(origins_input: Optional[Union[list[str], str]] = None) -> list[str]:
    """Resolves and parses allowed CORS origins from argument or environment variable."""
    if origins_input is not None:
        if isinstance(origins_input, str):
            return [o.strip() for o in origins_input.split(",") if o.strip()]
        return list(origins_input)

    env_origins = os.getenv("CORS_ORIGINS")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]

    return list(DEFAULT_CORS_ORIGINS)


def create_app(
    pipeline: Optional[NL2SQLPipeline] = None,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
    cors_origins: Optional[Union[list[str], str]] = None,
) -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Args:
        pipeline: Optional pre-initialized NL2SQLPipeline instance (useful for unit tests).
        checkpoint_path: Path to PyTorch model checkpoint.
        cors_origins: Optional list or comma-separated string of allowed CORS origins.

    Returns:
        Configured FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # If pipeline was not explicitly injected, load once from checkpoint on startup
        if getattr(app.state, "pipeline", None) is None:
            chk_path = Path(checkpoint_path)
            if chk_path.is_file():
                try:
                    logger.info(f"Loading NL2SQLPipeline from checkpoint: {chk_path.resolve()}")
                    app.state.pipeline = NL2SQLPipeline.from_checkpoint(
                        checkpoint_path=chk_path,
                    )
                    logger.info("NL2SQLPipeline successfully initialized.")
                except Exception as err:
                    logger.error(f"Failed to initialize NL2SQLPipeline on startup: {err}")
                    app.state.pipeline = None
            else:
                logger.warning(f"Checkpoint file not found at: {chk_path.resolve()}")
                app.state.pipeline = None
        else:
            logger.info("Using pre-injected NL2SQLPipeline instance.")

        yield

        # Teardown logic
        app.state.pipeline = None
        logger.info("Application shutdown complete.")

    app = FastAPI(
        title="Schema-Aware NL2SQL API",
        description=(
            "Production-grade REST API exposing the Attention-based Seq2Seq "
            "Natural Language to SQL translation model, SQLGlot AST validation, "
            "and least-privilege MySQL 8.0 query execution."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Injected pipeline if provided (e.g. during test setup)
    if pipeline is not None:
        app.state.pipeline = pipeline

    # Configure CORS
    allowed_origins = resolve_cors_origins(cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(router)

    return app


# Default application instance for Uvicorn
app = create_app()
