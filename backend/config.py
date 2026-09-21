from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("TOPAS_DATA_DIR", str(WEBSITE_DIR / "data"))).expanduser().resolve()
DEMO_DATA_DIR = WEBSITE_DIR / "demo_data"

load_dotenv(BACKEND_DIR / ".env")

APP_ENV = os.getenv("TOPAS_ENV", "development").lower()
DEFAULT_ACCESS_CODE = "" if APP_ENV == "production" else "topas-preview"


@dataclass(frozen=True)
class Settings:
    env: str = APP_ENV
    data_dir: Path = DATA_DIR
    database_path: Path = DATA_DIR / "topas.db"
    token_secret: str = os.getenv("TOPAS_TOKEN_SECRET", "topas-local-dev-secret-change-me")
    token_ttl_days: int = int(os.getenv("TOPAS_TOKEN_TTL_DAYS", "30"))
    access_code: str = os.getenv("TOPAS_ACCESS_CODE", DEFAULT_ACCESS_CODE)
    access_code_hashes: str = os.getenv("TOPAS_ACCESS_CODE_HASHES", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    azure_endpoint: str = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://qcri-sakina.services.ai.azure.com/openai/v1",
    ).rstrip("/")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    max_upload_mb: int = int(os.getenv("TOPAS_MAX_UPLOAD_MB", "250"))
    allowed_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "TOPAS_ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if item.strip()
    )

    @property
    def models(self) -> dict[str, str]:
        return {
            "gpt-5.1": os.getenv("TOPAS_MODEL_GPT_5_1", "gpt-5.1"),
            "gpt-4.1": os.getenv("TOPAS_MODEL_GPT_4_1", "gpt-4.1"),
            "DeepSeek-V4-Pro": os.getenv(
                "TOPAS_MODEL_DEEPSEEK_V4_PRO", "DeepSeek-V4-Pro"
            ),
        }


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
