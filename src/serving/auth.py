from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_MIN_TOKEN_LENGTH = 32
_bearer = HTTPBearer()


def validate_token_at_startup() -> str:
    """Read API_TOKEN from environment and assert it meets minimum requirements.

    Called once at application startup. Raises RuntimeError on failure.
    """
    token = os.environ.get("API_TOKEN", "")
    if len(token) < _MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"API_TOKEN must be at least {_MIN_TOKEN_LENGTH} characters; "
            f"got {len(token)}. Set API_TOKEN in your environment."
        )
    return token


def require_auth(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> str:
    """FastAPI dependency that validates the bearer token on every request."""
    expected = os.environ.get("API_TOKEN", "")
    if credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    return credentials.credentials
