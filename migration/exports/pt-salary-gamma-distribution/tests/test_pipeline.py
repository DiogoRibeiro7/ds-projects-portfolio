from __future__ import annotations

from pathlib import Path

from pt_salary_gamma_distribution.pipeline import (
    build_pipeline_paths,
    build_pipeline_steps,
    steps_to_frame,
    validate_pipeline_paths,
)


def test_build_pipeline_paths_points_to_expected_project_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = build_pipeline_paths(project_root)

    assert paths.notebook_py == project_root / "notebooks" / "01_salary_gamma_distribution_portugal.py"
    assert paths.notebook_ipynb == project_root / "notebooks" / "01_salary_gamma_distribution_portugal.ipynb"
    assert paths.summary_script == project_root / "scripts" / "summarize_notebook_outputs.py"


def test_build_pipeline_steps_includes_sync_execute_and_summary() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = build_pipeline_paths(project_root)
    steps = build_pipeline_steps(paths, python_executable="python", notebook_timeout_seconds=123, include_summary=True)

    assert [step.name for step in steps] == ["sync_notebook", "execute_notebook", "build_summary"]
    assert steps[0].command[:4] == ("python", "-m", "jupytext", "--to")
    assert "--output" in steps[0].command
    assert "--ExecutePreprocessor.timeout=123" in steps[1].command
    assert steps[2].command == ("python", str(paths.summary_script))


def test_validate_pipeline_paths_accepts_existing_repo_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = build_pipeline_paths(project_root)
    validate_pipeline_paths(paths)


def test_steps_to_frame_returns_serializable_rows() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = build_pipeline_paths(project_root)
    steps = build_pipeline_steps(paths, python_executable="python", include_summary=False)
    frame = steps_to_frame(steps)

    assert frame[0]["name"] == "sync_notebook"
    assert "jupytext" in frame[0]["command"]
