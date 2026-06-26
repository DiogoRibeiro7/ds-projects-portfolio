from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem locations needed by the reproducible analysis pipeline."""

    project_root: Path
    notebook_py: Path
    notebook_ipynb: Path
    summary_script: Path


@dataclass(frozen=True)
class PipelineStep:
    """One executable pipeline step."""

    name: str
    command: tuple[str, ...]
    cwd: Path


def default_project_root() -> Path:
    """Resolve the project root from the package location."""
    return Path(__file__).resolve().parents[2]


def build_pipeline_paths(project_root: Path | None = None) -> PipelinePaths:
    """Return canonical project paths for the analysis pipeline."""
    root = project_root.resolve() if project_root is not None else default_project_root()
    return PipelinePaths(
        project_root=root,
        notebook_py=root / "notebooks" / "01_salary_gamma_distribution_portugal.py",
        notebook_ipynb=root / "notebooks" / "01_salary_gamma_distribution_portugal.ipynb",
        summary_script=root / "scripts" / "summarize_notebook_outputs.py",
    )


def build_pipeline_steps(
    paths: PipelinePaths,
    python_executable: str | None = None,
    notebook_timeout_seconds: int = 1800,
    include_summary: bool = True,
) -> list[PipelineStep]:
    """Build the standard full-analysis pipeline steps."""
    python_cmd = python_executable or sys.executable
    steps = [
        PipelineStep(
            name="sync_notebook",
            command=(
                python_cmd,
                "-m",
                "jupytext",
                "--to",
                "ipynb",
                "--output",
                str(paths.notebook_ipynb),
                str(paths.notebook_py),
            ),
            cwd=paths.project_root,
        ),
        PipelineStep(
            name="execute_notebook",
            command=(
                python_cmd,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                f"--ExecutePreprocessor.timeout={notebook_timeout_seconds}",
                str(paths.notebook_ipynb),
            ),
            cwd=paths.project_root,
        ),
    ]
    if include_summary:
        steps.append(
            PipelineStep(
                name="build_summary",
                command=(python_cmd, str(paths.summary_script)),
                cwd=paths.project_root,
            )
        )
    return steps


def validate_pipeline_paths(paths: PipelinePaths) -> None:
    """Fail early if the expected project files are missing."""
    missing = [
        str(path)
        for path in (paths.notebook_py, paths.notebook_ipynb, paths.summary_script)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required pipeline file(s): {missing}")


def run_pipeline(
    project_root: Path | None = None,
    python_executable: str | None = None,
    notebook_timeout_seconds: int = 1800,
    include_summary: bool = True,
) -> list[PipelineStep]:
    """Execute the full notebook-driven analysis pipeline."""
    paths = build_pipeline_paths(project_root)
    validate_pipeline_paths(paths)
    steps = build_pipeline_steps(
        paths=paths,
        python_executable=python_executable,
        notebook_timeout_seconds=notebook_timeout_seconds,
        include_summary=include_summary,
    )

    for step in steps:
        subprocess.run(step.command, cwd=step.cwd, check=True)

    return steps


def steps_to_frame(steps: Sequence[PipelineStep]) -> list[dict[str, str]]:
    """Return a simple serializable description of the pipeline steps."""
    return [
        {
            "name": step.name,
            "cwd": str(step.cwd),
            "command": " ".join(step.command),
        }
        for step in steps
    ]
