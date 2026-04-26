from __future__ import annotations

from typing import Any, Optional
from fastapi import Header, HTTPException

import httpx

from app.settings import settings


def _extract_email(user: dict[str, Any]) -> str:
    for key in ("email", "email_address", "emailAddress"):
        val = user.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "Unknown"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _allowed_auth_emails() -> set[str]:
    return {
        _normalize_email(email)
        for email in settings.allowed_auth_emails.split(",")
        if email.strip()
    }


def _require_allowed_user(user: dict[str, Any]) -> str:
    email = _extract_email(user)
    allowed_emails = _allowed_auth_emails()
    if allowed_emails and _normalize_email(email) not in allowed_emails:
        raise HTTPException(status_code=403, detail="Email is not authorized for this application")
    return email


def _verify_refresh_token(token: str) -> dict[str, Any]:
    if not settings.instantdb_app_id:
        raise HTTPException(status_code=500, detail="Server missing BLAST_INSTANTDB_APP_ID")
    try:
        r = httpx.post(
            f"{settings.instantdb_api_uri.rstrip('/')}/runtime/auth/verify_refresh_token",
            json={"app-id": settings.instantdb_app_id, "refresh-token": token},
            timeout=10.0,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=401, detail="Invalid token")
        data = r.json()
        if not isinstance(data, dict) or "user" not in data:
            raise HTTPException(status_code=401, detail="Invalid token")
        return data
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_auth(authorization: Optional[str] = Header(default=None)) -> str:
    """
    InstantDB auth dependency.

    Frontend should send InstantDB `user.refresh_token`:
      Authorization: Bearer <refresh_token>

    We verify it via InstantDB runtime endpoint:
      POST /runtime/auth/verify_refresh_token
    """
    if not settings.require_auth:
        return "local"
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Expected Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    data = _verify_refresh_token(token)
    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    _require_allowed_user(user)
    return token


def require_user(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    if not settings.require_auth:
        return {"token": "local", "email": "Local", "user": {"email": "Local"}}
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Expected Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    data = _verify_refresh_token(token)
    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    return {"token": token, "email": _require_allowed_user(user), "user": user}

