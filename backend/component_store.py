from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import settings
from .extraction_runner import COMPONENTS


Phase = Literal["extraction", "refinement"]
MAX_COMPONENT_BYTES = 8 * 1024 * 1024


def component_path(project: dict[str, Any], role: str, component: str, phase: Phase) -> Path:
    if role not in COMPONENTS or component not in COMPONENTS[role]:
        raise FileNotFoundError
    raw_root = project.get("refined_path") if phase == "refinement" else project.get("output_path")
    if not raw_root:
        raise FileNotFoundError
    root = Path(str(raw_root)).resolve()
    path = (
        root / f"{component}.json"
        if phase == "refinement"
        else root / role / "merged_components" / f"{component}.json"
    ).resolve()
    if root != path.parent and root not in path.parents:
        raise FileNotFoundError
    return path


def read_component_data(project: dict[str, Any], role: str, component: str, phase: Phase) -> Any:
    path = component_path(project, role, component, phase)
    if not path.exists():
        raise FileNotFoundError
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_component(component: str, data: Any) -> None:
    if component == "macro_actions":
        valid = isinstance(data, dict) and isinstance(data.get("macro_actions"), list)
    elif component in {"micro_actions", "conversation_states", "cautions"}:
        valid = isinstance(data, list)
    elif component == "knowledge_graph":
        valid = (
            isinstance(data, dict)
            and isinstance(data.get("nodes"), list)
            and isinstance(data.get("edges"), list)
        )
    elif component == "user_profile":
        valid = (
            isinstance(data, dict)
            and isinstance(data.get("static_dimensions"), list)
            and isinstance(data.get("dynamic_dimensions"), list)
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"The {component.replace('_', ' ')} structure is not valid.")


def write_component_data(
    project: dict[str, Any],
    role: str,
    component: str,
    phase: Phase,
    data: Any,
) -> Path:
    _validate_component(component, data)
    path = component_path(project, role, component, phase)
    data_root = settings.data_dir.resolve()
    if data_root != path.parent and data_root not in path.parents:
        raise PermissionError("Components outside the profile workspace cannot be edited.")
    if not path.exists():
        raise FileNotFoundError

    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) > MAX_COMPONENT_BYTES:
        raise ValueError("This component is too large to save from the browser.")

    history = path.parent / ".history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    shutil.copy2(path, history / f"{component}-{stamp}.json")

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
