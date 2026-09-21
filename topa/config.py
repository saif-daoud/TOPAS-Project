import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


def _maybe_load_dotenv(search_from: Optional[Path] = None) -> None:
    """Load .env if present (cwd or near a provided path)."""
    try:
        from dotenv import load_dotenv, find_dotenv
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)

    if search_from is None:
        return

    start = Path(search_from).resolve()
    if start.is_file():
        start = start.parent
    for candidate_dir in [start, *start.parents]:
        candidate = candidate_dir / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break


def _resolve_api_key(cfg: Dict[str, Any]) -> str:
    """Resolve API key without ever hardcoding secrets.

    Supported patterns:
      - api_key_env: TOPA_API_KEY
      - api_key: TOPA_API_KEY  (treated as env var name if present)
      - api_key: sk-...        (direct value, discouraged but supported)
      - api_key: ENV:TOPA_API_KEY
    """
    api_key_env = cfg.get("api_key_env")
    api_key_raw = str(cfg.get("api_key", "") or "").strip()

    # Explicit env var wins
    if api_key_env:
        v = os.getenv(str(api_key_env).strip(), "").strip()
        if v:
            return v

    # ENV:NAME syntax
    if api_key_raw.upper().startswith("ENV:"):
        env_name = api_key_raw.split(":", 1)[1].strip()
        v = os.getenv(env_name, "").strip()
        if v:
            return v
        return ""

    # If the value looks like an env var name and exists, use it.
    if api_key_raw and api_key_raw.isidentifier():
        v = os.getenv(api_key_raw, "").strip()
        if v:
            return v

    # Otherwise, treat it as a literal key.
    return api_key_raw


@dataclass
class Config:
    domain: str
    mode: str
    textbooks_path: Path
    data_path: Path
    output_path: Path
    api_key: str

    extraction_params: Dict[str, Any]
    annotations_params: Dict[str, Any]
    rl_params: Dict[str, Any]
    simulation_params: Dict[str, Any]
    domain_params: Dict[str, Any]

    config_path: Path
    seed: Optional[int] = None

    @staticmethod
    def load(path: str) -> "Config":
        path = Path(path)
        _maybe_load_dotenv(path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        
        api_key = _resolve_api_key(cfg)
        if not api_key:
            raise ValueError(
                "Missing API key. Add api_key_env: TOPA_API_KEY in YAML and set TOPA_API_KEY in .env, "
            )

        out = Path(cfg["paths"]["output"]).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        return Config(
            domain=cfg["domain"],
            mode=cfg["mode"],
            textbooks_path=Path(cfg["paths"]["textbooks"]).expanduser().resolve(),
            data_path=Path(cfg["paths"]["data"]).expanduser().resolve(),
            output_path=out,
            api_key=api_key,
            extraction_params=cfg["extraction_params"],
            annotations_params=cfg["annotations_params"],
            rl_params=cfg["rl"],
            simulation_params=cfg["simulation"],
            domain_params=cfg["domain_params"],
            seed=cfg["seed"],
            config_path=path.resolve(),
        )

    def get_domain_role(self, role: str):
        return self.domain_params.get(role)

    def get_adj(self):
        return self.domain_params.get("adj")

    def get_interaction_unit(self):
        return self.domain_params.get("interaction_unit")

    def get_role_components(self, role: str):
        return self.extraction_params["role_components"].get(role, [])