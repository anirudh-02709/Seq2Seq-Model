"""
Pipeline package for end-to-end NL2SQL orchestration (Milestone 4, Task 4.4).
"""

from src.pipeline.orchestrator import NL2SQLPipeline, PipelineResult

__all__ = [
    "NL2SQLPipeline",
    "PipelineResult",
]
