"""
CLI Entrypoint Script to Launch the NL2SQL FastAPI Server (Milestone 5.2).

Usage:
    python run_server.py
    python run_server.py --port 8000 --host 127.0.0.1
    python run_server.py --reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Schema-Aware NL2SQL FastAPI server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Network interface to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", 8000)),
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Schema-Aware NL2SQL FastAPI Server (Milestone 5.2)")
    print("=" * 70)
    print(f"  Host:         {args.host}")
    print(f"  Port:         {args.port}")
    print(f"  API Docs:     http://{args.host}:{args.port}/docs")
    print(f"  ReDoc:        http://{args.host}:{args.port}/redoc")
    print(f"  Health Check: http://{args.host}:{args.port}/api/health")
    print(f"  Schema API:   http://{args.host}:{args.port}/api/schema")
    print("=" * 70)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
