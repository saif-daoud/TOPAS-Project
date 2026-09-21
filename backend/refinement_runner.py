from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import settings
from .database import add_event, get_project, update_project, utc_now
from .extraction_runner import resolve_deployment
from .provider import run_azure_chat


executor = ThreadPoolExecutor(max_workers=1)
active_jobs: dict[str, Any] = {}
active_jobs_lock = threading.Lock()
provider_patch_lock = threading.Lock()


def _merged_paths(project: dict[str, Any]) -> dict[str, str]:
    raw = project.get("output_path")
    if not raw:
        raise ValueError("Complete extraction before starting refinement.")
    root = Path(str(raw)).resolve()
    system = root / "system" / "merged_components"
    if not (system / "micro_actions.json").exists() or not (system / "conversation_states.json").exists():
        raise ValueError("Refinement requires extracted micro actions and conversation states.")
    paths = {"system": str(system)}
    user = root / "user" / "merged_components"
    if user.exists():
        paths["user"] = str(user)
    return paths


def _macro_count(system_path: str) -> int:
    try:
        value = json.loads((Path(system_path) / "micro_actions.json").read_text(encoding="utf-8"))
        return max(1, len(value) if isinstance(value, list) else 1)
    except (OSError, json.JSONDecodeError):
        return 1


def run_refinement(project_id: str) -> None:
    project = get_project(project_id)
    if not project:
        return
    update_project(
        project_id,
        refinement_status="running",
        refinement_progress=3,
        refinement_step="Preparing extracted components",
        refinement_error=None,
    )
    add_event(project_id, "refinement_started", "Refinement started", {"model": project["model"]})
    try:
        paths = _merged_paths(project)
        total = _macro_count(paths["system"])
        completed = 0

        import topa.refinement.refinement as refinement_module

        deployment = resolve_deployment(project["model"])

        def tracked_query(prompt: str, _: str) -> tuple[str, int, int]:
            nonlocal completed
            output, input_tokens, output_tokens, _elapsed = run_azure_chat(prompt, deployment)
            completed += 1
            progress = min(92, 8 + round(min(completed, total) / total * 84))
            message = f"Reviewed action group {min(completed, total)} of {total}"
            update_project(project_id, refinement_progress=progress, refinement_step=message)
            add_event(
                project_id,
                "refinement_progress",
                message,
                {"completed": min(completed, total), "total": total},
            )
            return output, input_tokens, output_tokens

        update_project(project_id, refinement_progress=8, refinement_step="Auditing conversation-state coverage")
        with provider_patch_lock:
            original_query = refinement_module.run_llm_query
            refinement_module.run_llm_query = tracked_query
            try:
                phase = refinement_module.RefinementPhase(
                    domain=project["domain"],
                    api_key=settings.azure_api_key,
                )
                refined_path = Path(phase.run(paths)).resolve()
            finally:
                refinement_module.run_llm_query = original_query

        refined_root = refined_path if refined_path.is_dir() else refined_path.parent
        update_project(
            project_id,
            refinement_status="completed",
            refinement_progress=100,
            refinement_step="Refinement complete",
            refined_path=str(refined_root),
            refined_at=utc_now(),
        )
        add_event(project_id, "refinement_completed", "Refinement completed")
    except Exception as exc:  # pragma: no cover - provider and pipeline failures
        message = f"{type(exc).__name__}: {exc}"
        update_project(
            project_id,
            refinement_status="failed",
            refinement_step="Refinement stopped",
            refinement_error=message,
        )
        add_event(project_id, "refinement_failed", "Refinement stopped", {"error": message})
    finally:
        with active_jobs_lock:
            active_jobs.pop(project_id, None)


def start_refinement(project_id: str) -> None:
    with active_jobs_lock:
        current = active_jobs.get(project_id)
        if current and not current.done():
            raise RuntimeError("Refinement is already running for this project.")
        active_jobs[project_id] = executor.submit(run_refinement, project_id)


def read_refinement_report(project: dict[str, Any], report: str) -> Any:
    names = {
        "summary": "refinement_new_dimensions_summary.json",
        "details": "refinement_report.json",
    }
    if report not in names or not project.get("refined_path"):
        raise FileNotFoundError
    root = Path(str(project["refined_path"])).resolve()
    path = (root / names[report]).resolve()
    if root not in path.parents or not path.exists():
        raise FileNotFoundError
    return json.loads(path.read_text(encoding="utf-8"))
