from __future__ import annotations

import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .auth import create_token, current_user, validate_email, verify_access_code
from .config import DEMO_DATA_DIR, settings
from .component_store import read_component_data, write_component_data
from .database import (
    add_book,
    add_event,
    create_project,
    delete_book,
    ensure_demo_project,
    get_events,
    get_project,
    init_db,
    list_projects,
    update_project,
    upsert_user,
)
from .extraction_runner import (
    COMPONENTS,
    artifact_manifest,
    read_summary,
    start_project,
)
from .refinement_runner import read_refinement_report, start_refinement


DEMO_ROOT = (
    DEMO_DATA_DIR
    / "cbt_showcase"
    / "extracted_components"
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.env == "production" and settings.token_secret == "topas-local-dev-secret-change-me":
        raise RuntimeError("Set TOPAS_TOKEN_SECRET before starting the production API.")
    init_db()
    yield


app = FastAPI(
    title="TOPAS Studio API",
    version="0.1.0",
    description="Persistent extraction workspace for TOPAS late fusion.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    access_code: str = Field(min_length=1, max_length=256)


class ProjectRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    domain: str = Field(min_length=2, max_length=100)
    system_name: str = Field(min_length=2, max_length=60)
    user_name: str = Field(min_length=2, max_length=60)
    interaction_unit: Literal["session", "conversation"]
    model: Literal["gpt-5.1", "gpt-4.1", "DeepSeek-V4-Pro"]

    @field_validator("name", "domain", "system_name", "user_name")
    @classmethod
    def safe_text(cls, value: str) -> str:
        clean = " ".join(value.strip().split())
        if any(character in clean for character in ("/", "\\", "\0")) or ".." in clean:
            raise ValueError("Path characters are not allowed.")
        return clean


class ComponentUpdateRequest(BaseModel):
    data: Any
    phase: Literal["extraction", "refinement"] = "extraction"


def owned_project(project_id: str, user: dict = Depends(current_user)) -> dict:
    project = get_project(project_id, user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def public_project(project: dict | None) -> dict | None:
    if not project:
        return None
    clean = {
        key: value
        for key, value in project.items()
        if key not in {"user_id", "output_path", "refined_path", "run_id"}
    }
    clean["books"] = [
        {key: value for key, value in book.items() if key != "stored_path"}
        for book in project.get("books", [])
    ]
    return clean


def public_project_list(projects: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in project.items()
            if key not in {"user_id", "output_path", "refined_path", "run_id"}
        }
        for project in projects
    ]


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "TOPAS Studio",
        "models": list(settings.models),
        "azure_configured": bool(settings.azure_api_key and settings.azure_endpoint),
    }


@app.post("/api/auth/login")
def login(body: LoginRequest) -> dict:
    if not verify_access_code(body.access_code):
        raise HTTPException(status_code=401, detail="That access code is not valid.")
    email = validate_email(body.email)
    user = upsert_user(email)
    ensure_demo_project(user["id"], DEMO_ROOT)
    return {
        "token": create_token(user),
        "user": {"id": user["id"], "email": user["email"]},
        "returning": user["created_at"] != user["last_seen_at"],
    }


@app.get("/api/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"id": user["id"], "email": user["email"]}


@app.get("/api/projects")
def projects(user: dict = Depends(current_user)) -> dict:
    return {"projects": public_project_list(list_projects(user["id"]))}


@app.post("/api/projects", status_code=201)
def new_project(body: ProjectRequest, user: dict = Depends(current_user)) -> dict:
    project = create_project(user["id"], body.model_dump())
    add_event(project["id"], "created", "Project workspace created")
    return {"project": public_project(project)}


@app.get("/api/projects/{project_id}")
def project_detail(project: dict = Depends(owned_project)) -> dict:
    return {"project": public_project(project), "manifest": artifact_manifest(project)}


@app.get("/api/projects/{project_id}/events")
def project_events(
    after: int = 0,
    project: dict = Depends(owned_project),
) -> dict:
    return {"events": get_events(project["id"], max(0, after)), "project": public_project(get_project(project["id"]))}


@app.post("/api/projects/{project_id}/books", status_code=201)
async def upload_books(
    role: Annotated[Literal["system", "user"], Form()],
    files: Annotated[list[UploadFile], File()],
    project: dict = Depends(owned_project),
) -> dict:
    if project["is_demo"]:
        raise HTTPException(status_code=409, detail="The showcase project is read-only.")
    if project["status"] == "running":
        raise HTTPException(status_code=409, detail="Wait for extraction to finish before adding books.")
    if not files:
        raise HTTPException(status_code=422, detail="Choose at least one PDF.")

    upload_dir = settings.data_dir / project["user_id"] / project["id"] / "uploads" / role
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    existing_role_count = sum(1 for book in project["books"] if book["role"] == role)
    for offset, upload in enumerate(files, start=1):
        original = Path(upload.filename or "textbook.pdf").name
        if Path(original).suffix.lower() != ".pdf":
            raise HTTPException(status_code=415, detail=f"{original} is not a PDF.")
        safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(original).stem).strip(".-") or "textbook"
        sequence = existing_role_count + offset
        destination = upload_dir / f"{sequence:04d}-{uuid.uuid4().hex[:8]}-{safe_stem}.pdf"
        size = 0
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    target.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{original} exceeds the {settings.max_upload_mb} MB upload limit.",
                    )
                target.write(chunk)
        try:
            import pymupdf

            with pymupdf.open(destination) as document:
                pages = document.page_count
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"{original} is not a readable PDF.") from exc
        saved.append(add_book(project["id"], role, original, destination, size, pages))
        add_event(
            project["id"],
            "book_uploaded",
            f"Added {original} to {role} sources",
            {"role": role, "name": original, "pages": pages},
        )
    update_project(
        project["id"],
        status="draft",
        progress=0,
        current_step="Ready to extract",
        refinement_status="not_started",
        refinement_progress=0,
        refinement_step="Ready after extraction",
        refined_path=None,
        refinement_error=None,
        refined_at=None,
    )
    return {
        "books": [{key: value for key, value in book.items() if key != "stored_path"} for book in saved],
        "project": public_project(get_project(project["id"])),
    }


@app.delete("/api/projects/{project_id}/books/{book_id}")
def remove_book(book_id: str, project: dict = Depends(owned_project)) -> dict:
    if project["is_demo"] or project["status"] == "running":
        raise HTTPException(status_code=409, detail="This book cannot be removed right now.")
    book = delete_book(book_id, project["id"])
    if not book:
        raise HTTPException(status_code=404, detail="Book not found.")
    path = Path(book["stored_path"])
    if path.exists() and settings.data_dir.resolve() in path.resolve().parents:
        path.unlink()
    add_event(project["id"], "book_removed", f"Removed {book['original_name']}")
    update_project(
        project["id"],
        refinement_status="not_started",
        refinement_progress=0,
        refinement_step="Ready after extraction",
        refined_path=None,
        refinement_error=None,
        refined_at=None,
    )
    return {"ok": True, "project": public_project(get_project(project["id"]))}


@app.post("/api/projects/{project_id}/run", status_code=202)
def run_extraction(project: dict = Depends(owned_project)) -> dict:
    if project["is_demo"]:
        raise HTTPException(status_code=409, detail="The showcase run is already complete.")
    if not project["books"]:
        raise HTTPException(status_code=422, detail="Upload at least one system or user textbook.")
    if not settings.azure_api_key:
        raise HTTPException(
            status_code=503,
            detail="Set AZURE_OPENAI_API_KEY in website/backend/.env before starting extraction.",
        )
    try:
        start_project(project["id"])
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True, "project_id": project["id"]}


@app.get("/api/projects/{project_id}/summaries/{role}/{book_index}")
def summary_artifact(
    role: Literal["system", "user"],
    book_index: int,
    project: dict = Depends(owned_project),
) -> dict:
    try:
        return {"data": read_summary(project, role, book_index)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="That summary is not ready yet.") from exc


@app.get("/api/projects/{project_id}/components/{role}/{component}")
def component_artifact(
    role: Literal["system", "user"],
    component: str,
    phase: Literal["extraction", "refinement"] = "extraction",
    project: dict = Depends(owned_project),
) -> dict:
    if component not in COMPONENTS[role]:
        raise HTTPException(status_code=404, detail="Component not found.")
    try:
        return {"data": read_component_data(project, role, component, phase)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="That component is not ready yet.") from exc


@app.put("/api/projects/{project_id}/components/{role}/{component}")
def update_component_artifact(
    role: Literal["system", "user"],
    component: str,
    body: ComponentUpdateRequest,
    project: dict = Depends(owned_project),
) -> dict:
    if component not in COMPONENTS[role]:
        raise HTTPException(status_code=404, detail="Component not found.")
    if project["status"] == "running" or project.get("refinement_status") == "running":
        raise HTTPException(status_code=409, detail="Wait for the active pipeline step to finish before editing.")
    try:
        write_component_data(project, role, component, body.phase, body.data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="That component is not ready yet.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.phase == "extraction" and project.get("refinement_status") == "completed":
        update_project(
            project["id"],
            refinement_status="stale",
            refinement_step="Extracted components changed",
        )
    add_event(
        project["id"],
        "component_edited",
        f"Updated {component.replace('_', ' ')}",
        {"role": role, "component": component, "phase": body.phase},
    )
    return {"saved": True, "data": body.data, "project": public_project(get_project(project["id"]))}


@app.post("/api/projects/{project_id}/refine", status_code=202)
def run_refinement(project: dict = Depends(owned_project)) -> dict:
    if project["status"] != "completed":
        raise HTTPException(status_code=409, detail="Complete extraction before starting refinement.")
    if not settings.azure_api_key:
        raise HTTPException(status_code=503, detail="Azure model credentials are not configured.")
    try:
        start_refinement(project["id"])
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True, "project_id": project["id"]}


@app.get("/api/projects/{project_id}/refinement/{report}")
def refinement_report(
    report: Literal["summary", "details"],
    project: dict = Depends(owned_project),
) -> dict:
    try:
        return {"data": read_refinement_report(project, report)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="That refinement report is not ready yet.") from exc


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
