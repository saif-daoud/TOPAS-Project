from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any

from fastapi import Header, HTTPException

from .config import settings
from .database import get_user


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def validate_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 254 or not EMAIL_RE.match(normalized):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return normalized


def verify_access_code(code: str) -> bool:
    candidate = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
    configured = {
        item.strip().lower() for item in settings.access_code_hashes.split(",") if item.strip()
    }
    if settings.access_code:
        configured.add(hashlib.sha256(settings.access_code.encode("utf-8")).hexdigest())
    if not configured and settings.env == "production":
        raise HTTPException(status_code=503, detail="Access codes are not configured.")
    return any(hmac.compare_digest(candidate, expected) for expected in configured)


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_token(user: dict[str, Any]) -> str:
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "exp": int(time.time()) + settings.token_ttl_days * 86400,
    }
    encoded = _encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        settings.token_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{_encode(signature)}"


def read_token(token: str) -> dict[str, Any]:
    try:
        encoded, supplied = token.split(".", 1)
        expected = hmac.new(
            settings.token_secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _decode(supplied)):
            raise ValueError("signature")
        payload = json.loads(_decode(encoded))
        if int(payload["exp"]) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.") from exc


def current_user(authorization: str = Header(default="")) -> dict[str, Any]:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    payload = read_token(authorization[7:].strip())
    user = get_user(str(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="This account no longer exists.")
    return user

