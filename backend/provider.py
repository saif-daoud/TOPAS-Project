from __future__ import annotations

import os
import ssl
import time
from pathlib import Path
from typing import Any

import httpx
import truststore

from .config import settings


def _http_client() -> httpx.Client:
    """Use the native OS trust store, including organization-installed CAs."""
    ca_bundle = os.getenv("TOPAS_CA_BUNDLE", "").strip()
    if ca_bundle:
        bundle_path = Path(ca_bundle).expanduser().resolve()
        if not bundle_path.is_file():
            raise RuntimeError(f"TOPAS_CA_BUNDLE does not exist: {bundle_path}")
        verify: ssl.SSLContext | str | bool = str(bundle_path)
    elif os.getenv("TOPAS_USE_SYSTEM_CA", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        verify = True

    return httpx.Client(
        verify=verify,
        timeout=httpx.Timeout(180.0, connect=30.0),
    )


def run_azure_chat(prompt: str, deployment: str) -> tuple[str, int, int, float]:
    """Run one TOPAS prompt against the configured Azure endpoint."""
    from openai import AzureOpenAI, OpenAI

    endpoint = settings.azure_endpoint
    retries = int(os.getenv("TOPAS_LLM_RETRIES", "5"))
    started = time.perf_counter()
    last_error: Exception | None = None
    with _http_client() as http_client:
        if "/openai/v1" in endpoint:
            client: Any = OpenAI(
                base_url=endpoint,
                api_key=settings.azure_api_key,
                http_client=http_client,
            )
        else:
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=settings.azure_api_key,
                api_version=settings.azure_api_version,
                http_client=http_client,
            )

        for attempt in range(1, retries + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = response.choices[0].message.content or ""
                usage = response.usage
                return (
                    output,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                    time.perf_counter() - started,
                )
            except Exception as exc:  # pragma: no cover - requires provider failure
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 12))
    raise RuntimeError(
        f"Azure model request failed after {retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
