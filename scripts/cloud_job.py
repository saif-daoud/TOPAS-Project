from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


API_URL = os.environ["TOPAS_API_URL"].rstrip("/")
JOB_TOKEN = os.environ["TOPAS_JOB_TOKEN"]
PROJECT_ID = os.environ["TOPAS_PROJECT_ID"]
JOB_TYPE = os.environ.get("TOPAS_JOB_TYPE", "extraction")
HEADERS = {"X-TOPAS-JOB-TOKEN": JOB_TOKEN}
SYSTEM_COMPONENTS = ["macro_actions", "micro_actions", "conversation_states", "knowledge_graph", "cautions"]
USER_COMPONENTS = ["user_profile"]
COMPONENTS = {"system": SYSTEM_COMPONENTS, "user": USER_COMPONENTS}


def client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True)


def api(method: str, path: str, *, json_body: Any = None, content: bytes | str | None = None) -> httpx.Response:
    with client() as http:
        result = http.request(method, f"{API_URL}{path}", headers=HEADERS, json=json_body, content=content)
        result.raise_for_status()
        return result


def progress(
    percent: int,
    step: str,
    *,
    status: str = "running",
    kind: str | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    api(
        "POST",
        f"/internal/projects/{PROJECT_ID}/progress",
        json_body={
            "job_type": JOB_TYPE,
            "status": status,
            "progress": percent,
            "step": step,
            "kind": kind,
            "message": message,
            "payload": payload or {},
            "error": error,
        },
    )


def put_artifact(
    path: Path,
    *,
    phase: str,
    role: str = "",
    kind: str,
    book_index: int = -1,
    component: str = "",
    report: str = "",
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    params = urlencode(
        {
            "phase": phase,
            "role": role,
            "kind": kind,
            "book_index": book_index,
            "component": component,
            "report": report,
            "chapter_count": len(data) if kind == "summary" and isinstance(data, list) else 0,
        }
    )
    api("PUT", f"/internal/artifacts/{PROJECT_ID}?{params}", content=json.dumps(data, ensure_ascii=False).encode("utf-8"))


def classify(path: Path, role_root: Path) -> tuple[dict[str, Any], str] | None:
    relative = path.relative_to(role_root).as_posix()
    summary = re.fullmatch(r"summaries_book_(\d+)\.json", relative)
    if summary:
        index = int(summary.group(1))
        return ({"phase": "extraction", "role": role_root.name, "kind": "summary", "book_index": index}, f"{role_root.name.title()} textbook {index + 1} summary is ready")
    per_book = re.fullmatch(r"book_(\d+)/([^/]+)\.json", relative)
    if per_book:
        index, component = int(per_book.group(1)), per_book.group(2)
        return ({"phase": "extraction", "role": role_root.name, "kind": "book_component", "book_index": index, "component": component}, f"Extracted {component.replace('_', ' ')} from {role_root.name} textbook {index + 1}")
    merged = re.fullmatch(r"merged_components/([^/]+)\.json", relative)
    if merged and merged.group(1) in COMPONENTS[role_root.name]:
        component = merged.group(1)
        return ({"phase": "extraction", "role": role_root.name, "kind": "component", "component": component}, f"Late-fused {component.replace('_', ' ')} is ready")
    return None


def build_extractor(project: dict[str, Any], upload_dir: Path, output_base: Path):
    from backend.provider import run_azure_chat
    from topa.extraction.topa_ours import TOPAOurExtractor

    deployments = {
        "gpt-5.1": os.getenv("TOPAS_MODEL_GPT_5_1", "gpt-5.1"),
        "gpt-4.1": os.getenv("TOPAS_MODEL_GPT_4_1", "gpt-4.1"),
        "DeepSeek-V4-Pro": os.getenv("TOPAS_MODEL_DEEPSEEK_V4_PRO", "DeepSeek-V4-Pro"),
    }

    class CloudExtractor(TOPAOurExtractor):
        def run_llm_query(self, prompt: str):  # type: ignore[override]
            output, input_tokens, output_tokens, elapsed = run_azure_chat(prompt, deployments[project["model"]])
            self._record_usage(input_tokens, output_tokens, elapsed, tag="llm")
            return output, input_tokens, output_tokens

    return CloudExtractor(
        project["domain"],
        project["system_name"],
        project["user_name"],
        project["interaction_unit"],
        str(upload_dir),
        str(output_base),
        os.environ["AZURE_OPENAI_API_KEY"],
        "dialogue",
    )


def extraction(job: dict[str, Any], root: Path) -> None:
    project, books = job["project"], job["books"]
    source_root, output_root = root / "sources", root / "outputs"
    counts = {"system": 0, "user": 0}
    for role in counts:
        role_dir = source_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        role_books = [book for book in books if book["role"] == role]
        counts[role] = len(role_books)
        for index, book in enumerate(role_books):
            data = api("GET", f"/internal/books/{book['id']}").content
            (role_dir / f"{index:04d}-{book['id'][:8]}.pdf").write_bytes(data)

    total = max(1, sum(counts[role] * (1 + len(COMPONENTS[role])) + len(COMPONENTS[role]) for role in counts if counts[role]))
    completed = 0
    seen: set[Path] = set()
    progress(2, "Preparing uploaded textbooks", kind="extraction_started", message="Late-fusion extraction started", payload={"model": project["model"]})

    for role in ("system", "user"):
        if not counts[role]:
            continue
        progress(max(2, round(completed / total * 96)), f"Reading {role} textbooks", kind="role_started", message=f"Reading {counts[role]} {role} textbook{'s' if counts[role] != 1 else ''}", payload={"role": role, "book_count": counts[role]})
        extractor = build_extractor(project, source_root / role, output_root)
        extractor.components[role] = list(COMPONENTS[role])
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["path"] = extractor.extract(role=role, fusion_type="Late")
            except Exception as caught:  # noqa: BLE001
                result["error"] = caught

        thread = threading.Thread(target=run, daemon=False)
        thread.start()
        role_root = output_root / project["domain"] / "TOPAOurExtractor_LateFusion" / "extracted_components" / role
        while thread.is_alive():
            if role_root.exists():
                for path in sorted(role_root.rglob("*.json")):
                    resolved = path.resolve()
                    classified = classify(path, role_root)
                    if resolved in seen or not classified or path.stat().st_size == 0:
                        continue
                    values, message = classified
                    put_artifact(path, **values)
                    seen.add(resolved)
                    completed += 1
                    progress(min(96, max(2, round(completed / total * 96))), message, kind=values["kind"] + "_ready", message=message, payload={key: value for key, value in values.items() if key not in {"phase", "kind"}})
            thread.join(timeout=2)
        if "error" in result:
            raise result["error"]
        if role_root.exists():
            for path in sorted(role_root.rglob("*.json")):
                if path.resolve() not in seen and (classified := classify(path, role_root)):
                    values, message = classified
                    put_artifact(path, **values)
                    seen.add(path.resolve())
                    completed += 1
                    progress(min(96, max(2, round(completed / total * 96))), message, kind=values["kind"] + "_ready", message=message, payload={key: value for key, value in values.items() if key not in {"phase", "kind"}})

    progress(100, "Extraction complete", status="completed", kind="completed", message="Late-fusion extraction completed")


def refinement(job: dict[str, Any], root: Path) -> None:
    from backend.provider import run_azure_chat
    import topa.refinement.refinement as refinement_module

    project, artifacts = job["project"], job["artifacts"]
    extracted = root / "extracted_components"
    paths: dict[str, str] = {}
    for role in ("system", "user"):
        destination = extracted / role / "merged_components"
        destination.mkdir(parents=True, exist_ok=True)
        role_artifacts = [item for item in artifacts if item["phase"] == "extraction" and item["kind"] == "component" and item["role"] == role]
        for artifact in role_artifacts:
            data = api("GET", f"/internal/artifacts/{PROJECT_ID}?artifact_id={artifact['id']}").content
            (destination / f"{artifact['component']}.json").write_bytes(data)
        if role_artifacts:
            paths[role] = str(destination)
    if "system" not in paths:
        raise RuntimeError("Refinement requires extracted system components.")

    deployment = {
        "gpt-5.1": os.getenv("TOPAS_MODEL_GPT_5_1", "gpt-5.1"),
        "gpt-4.1": os.getenv("TOPAS_MODEL_GPT_4_1", "gpt-4.1"),
        "DeepSeek-V4-Pro": os.getenv("TOPAS_MODEL_DEEPSEEK_V4_PRO", "DeepSeek-V4-Pro"),
    }[project["model"]]
    actions = json.loads((Path(paths["system"]) / "micro_actions.json").read_text(encoding="utf-8"))
    total, completed = max(1, len(actions) if isinstance(actions, list) else 1), 0

    def tracked_query(prompt: str, _: str):
        nonlocal completed
        output, input_tokens, output_tokens, _elapsed = run_azure_chat(prompt, deployment)
        completed += 1
        percent = min(92, 8 + round(min(completed, total) / total * 84))
        progress(percent, f"Reviewed action group {min(completed, total)} of {total}", kind="refinement_progress", message=f"Reviewed action group {min(completed, total)} of {total}", payload={"completed": min(completed, total), "total": total})
        return output, input_tokens, output_tokens

    progress(8, "Auditing conversation-state coverage", kind="refinement_started", message="Refinement started", payload={"model": project["model"]})
    original = refinement_module.run_llm_query
    refinement_module.run_llm_query = tracked_query
    try:
        refined_path = Path(refinement_module.RefinementPhase(project["domain"], os.environ["AZURE_OPENAI_API_KEY"]).run(paths))
    finally:
        refinement_module.run_llm_query = original
    if refined_path.is_file():
        refined_path = refined_path.parent
    for component in set(SYSTEM_COMPONENTS + USER_COMPONENTS):
        path = refined_path / f"{component}.json"
        if path.exists():
            put_artifact(path, phase="refinement", role="user" if component == "user_profile" else "system", kind="component", component=component)
    for report, filename in (("summary", "refinement_new_dimensions_summary.json"), ("details", "refinement_report.json")):
        path = refined_path / filename
        if path.exists():
            put_artifact(path, phase="refinement", kind="report", report=report)
    progress(100, "Refinement complete", status="completed", kind="refinement_completed", message="Refinement completed")


def main() -> None:
    job = api("GET", f"/internal/projects/{PROJECT_ID}/job").json()
    with tempfile.TemporaryDirectory(prefix="topas-cloud-") as directory:
        if JOB_TYPE == "refinement":
            refinement(job, Path(directory))
        else:
            extraction(job, Path(directory))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        try:
            progress(0, "Refinement stopped" if JOB_TYPE == "refinement" else "Extraction stopped", status="failed", kind=f"{JOB_TYPE}_failed", message=f"{JOB_TYPE.title()} stopped", payload={"error": message}, error=message)
        finally:
            traceback.print_exc()
        raise
