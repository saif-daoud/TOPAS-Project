from __future__ import annotations

import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings
from .database import add_event, get_project, update_project, utc_now
from .provider import run_azure_chat


SYSTEM_COMPONENTS = [
    "macro_actions",
    "micro_actions",
    "conversation_states",
    "knowledge_graph",
    "cautions",
]
USER_COMPONENTS = ["user_profile"]
COMPONENTS = {"system": SYSTEM_COMPONENTS, "user": USER_COMPONENTS}

executor = ThreadPoolExecutor(max_workers=int(os.getenv("TOPAS_EXTRACTION_WORKERS", "2")))
active_jobs: dict[str, Any] = {}
active_jobs_lock = threading.Lock()


def resolve_deployment(model_label: str) -> str:
    try:
        return settings.models[model_label]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_label}") from exc


def build_extractor(project: dict[str, Any], upload_dir: Path, output_base: Path):
    """Create TOPA late fusion lazily so ordinary API startup remains fast.

    All extraction, conversion, component, and fusion prompt methods are inherited
    unchanged from TOPAOurExtractor. Only the provider call is routed through the
    OpenAI-compatible Azure AI Foundry endpoint from simulations/llm.py.
    """
    from topa.extraction.topa_ours import TOPAOurExtractor

    class AzureTOPALateFusionExtractor(TOPAOurExtractor):
        def __init__(self, *args: Any, deployment: str, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.deployment = deployment

        def run_llm_query(self, prompt: str):  # type: ignore[override]
            output, input_tokens, output_tokens, elapsed = run_azure_chat(prompt, self.deployment)
            self._record_usage(input_tokens, output_tokens, elapsed, tag="llm")
            return output, input_tokens, output_tokens

    return AzureTOPALateFusionExtractor(
        project["domain"],
        project["system_name"],
        project["user_name"],
        project["interaction_unit"],
        str(upload_dir),
        str(output_base),
        settings.azure_api_key,
        "dialogue",
        deployment=resolve_deployment(project["model"]),
    )


def _artifact_root(project: dict[str, Any]) -> Path | None:
    raw = project.get("output_path")
    return Path(raw).resolve() if raw else None


def _discover_files(role_root: Path) -> dict[Path, tuple[str, dict[str, Any], str]]:
    found: dict[Path, tuple[str, dict[str, Any], str]] = {}
    for path in role_root.glob("summaries_book_*.json"):
        match = re.search(r"summaries_book_(\d+)\.json$", path.name)
        if match:
            index = int(match.group(1))
            found[path] = (
                "summary_ready",
                {"role": role_root.name, "book_index": index},
                f"{role_root.name.title()} textbook {index + 1} summary is ready",
            )
    for path in role_root.glob("book_*/*.json"):
        match = re.search(r"book_(\d+)$", path.parent.name)
        if match:
            index = int(match.group(1))
            found[path] = (
                "book_component_ready",
                {"role": role_root.name, "book_index": index, "component": path.stem},
                f"Extracted {path.stem.replace('_', ' ')} from {role_root.name} textbook {index + 1}",
            )
    for path in (role_root / "merged_components").glob("*.json"):
        if path.name == "usage.json":
            continue
        found[path] = (
            "component_ready",
            {"role": role_root.name, "component": path.stem},
            f"Late-fused {path.stem.replace('_', ' ')} is ready",
        )
    return found


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def artifact_manifest(project: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_root(project)
    manifest: dict[str, Any] = {
        "roles": {
            "system": {"summaries": [], "book_components": {}, "merged_components": [], "refined_components": []},
            "user": {"summaries": [], "book_components": {}, "merged_components": [], "refined_components": []},
        },
        "refinement": {"report_available": False, "summary_available": False},
    }
    if not root or not root.exists():
        return manifest
    for role in ("system", "user"):
        role_root = root / role
        if not role_root.exists():
            continue
        role_manifest = manifest["roles"][role]
        for path in sorted(role_root.glob("summaries_book_*.json")):
            match = re.search(r"(\d+)", path.stem)
            if not match:
                continue
            data = _safe_json(path)
            role_manifest["summaries"].append(
                {
                    "book_index": int(match.group(1)),
                    "chapter_count": len(data) if isinstance(data, list) else 0,
                }
            )
        for directory in sorted(role_root.glob("book_*")):
            if not directory.is_dir():
                continue
            match = re.search(r"(\d+)", directory.name)
            if match:
                role_manifest["book_components"][match.group(1)] = [
                    path.stem for path in sorted(directory.glob("*.json"))
                ]
        merged = role_root / "merged_components"
        if merged.exists():
            role_manifest["merged_components"] = [
                component
                for component in COMPONENTS[role]
                if (merged / f"{component}.json").exists()
            ]
    refined_raw = project.get("refined_path")
    if refined_raw:
        refined_root = Path(str(refined_raw)).resolve()
        if refined_root.exists():
            for role in ("system", "user"):
                manifest["roles"][role]["refined_components"] = [
                    component
                    for component in COMPONENTS[role]
                    if (refined_root / f"{component}.json").exists()
                ]
            manifest["refinement"] = {
                "report_available": (refined_root / "refinement_report.json").exists(),
                "summary_available": (refined_root / "refinement_new_dimensions_summary.json").exists(),
            }
    return manifest


def read_summary(project: dict[str, Any], role: str, book_index: int) -> Any:
    root = _artifact_root(project)
    if not root:
        raise FileNotFoundError
    path = (root / role / f"summaries_book_{book_index}.json").resolve()
    if root not in path.parents or not path.exists():
        raise FileNotFoundError
    return json.loads(path.read_text(encoding="utf-8"))


def read_component(project: dict[str, Any], role: str, component: str) -> Any:
    root = _artifact_root(project)
    if not root or component not in COMPONENTS[role]:
        raise FileNotFoundError
    path = (root / role / "merged_components" / f"{component}.json").resolve()
    if root not in path.parents or not path.exists():
        raise FileNotFoundError
    return json.loads(path.read_text(encoding="utf-8"))


def _progress_total(project: dict[str, Any]) -> int:
    counts = {"system": 0, "user": 0}
    for book in project["books"]:
        counts[book["role"]] += 1
    return max(
        1,
        sum(counts[role] * (1 + len(COMPONENTS[role])) + len(COMPONENTS[role]) for role in counts if counts[role]),
    )


def _monitor_role(
    project_id: str,
    role_root: Path,
    extraction_thread: threading.Thread,
    seen: set[Path],
    completed_units: list[int],
    total_units: int,
) -> None:
    while extraction_thread.is_alive():
        _scan_role(project_id, role_root, seen, completed_units, total_units)
        extraction_thread.join(timeout=0.75)
    _scan_role(project_id, role_root, seen, completed_units, total_units)


def _scan_role(
    project_id: str,
    role_root: Path,
    seen: set[Path],
    completed_units: list[int],
    total_units: int,
) -> None:
    for path, (kind, payload, message) in _discover_files(role_root).items():
        resolved = path.resolve()
        if resolved in seen or path.stat().st_size == 0:
            continue
        seen.add(resolved)
        completed_units[0] += 1
        progress = min(96, max(2, round(completed_units[0] / total_units * 96)))
        add_event(project_id, kind, message, payload)
        update_project(project_id, progress=progress, current_step=message)


def _run_role(project: dict[str, Any], role: str, output_base: Path, total: int, completed: list[int]) -> None:
    books = [book for book in project["books"] if book["role"] == role]
    if not books:
        return
    upload_dir = Path(books[0]["stored_path"]).parent
    extractor = build_extractor(project, upload_dir, output_base)
    extractor.components[role] = list(COMPONENTS[role])
    add_event(
        project["id"],
        "role_started",
        f"Reading {len(books)} {role} textbook{'s' if len(books) != 1 else ''}",
        {"role": role, "book_count": len(books)},
    )
    update_project(project["id"], current_step=f"Reading {role} textbooks")

    result: dict[str, Any] = {}

    def extract() -> None:
        try:
            result["path"] = extractor.extract(role=role, fusion_type="Late")
        except Exception as exc:
            result["error"] = exc
            result["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=extract, name=f"topas-{project['id']}-{role}")
    thread.start()
    role_root = output_base / project["domain"] / "TOPAOurExtractor_LateFusion" / "extracted_components" / role
    _monitor_role(project["id"], role_root, thread, set(), completed, total)
    if result.get("error"):
        raise result["error"]
    extractor.save_usage(str(result["path"]))
    add_event(
        project["id"],
        "role_completed",
        f"{role.title()} extraction is complete",
        {"role": role},
    )


def run_project(project_id: str) -> None:
    project = get_project(project_id)
    if not project:
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_base = settings.data_dir / project["user_id"] / project_id / "runs" / run_id
    artifact_root = output_base / project["domain"] / "TOPAOurExtractor_LateFusion" / "extracted_components"
    artifact_root.mkdir(parents=True, exist_ok=True)
    update_project(
        project_id,
        status="running",
        progress=1,
        current_step="Preparing extraction",
        run_id=run_id,
        output_path=str(artifact_root),
        error=None,
        completed_at=None,
        refinement_status="not_started",
        refinement_progress=0,
        refinement_step="Ready after extraction",
        refined_path=None,
        refinement_error=None,
        refined_at=None,
    )
    add_event(project_id, "started", "Late-fusion extraction started", {"model": project["model"]})
    try:
        total = _progress_total(project)
        completed = [0]
        for role in ("system", "user"):
            _run_role(project, role, output_base, total, completed)
        update_project(
            project_id,
            status="completed",
            progress=100,
            current_step="Extraction complete",
            completed_at=utc_now(),
        )
        add_event(project_id, "completed", "Late-fusion extraction completed")
    except Exception as exc:  # pragma: no cover - exercised with provider failures
        message = f"{type(exc).__name__}: {exc}"
        update_project(
            project_id,
            status="failed",
            current_step="Extraction stopped",
            error=message,
        )
        add_event(project_id, "failed", "Extraction stopped", {"error": message})
    finally:
        with active_jobs_lock:
            active_jobs.pop(project_id, None)


def start_project(project_id: str) -> None:
    with active_jobs_lock:
        current = active_jobs.get(project_id)
        if current and not current.done():
            raise RuntimeError("This project is already running.")
        active_jobs[project_id] = executor.submit(run_project, project_id)
