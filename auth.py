#!/usr/bin/env python3
"""
auth.py - JWT Authentication Helpers for E-Commerce AI Studio

Provides:
  - Password hashing / verification (bcrypt directly)
  - JWT token creation and decoding (python-jose)
  - FastAPI dependency for extracting the current authenticated user
"""

import os
import time
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.environ.get("STUDIO_SECRET_KEY", "change-me-in-production-super-secret-key-2024")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS: int = int(os.environ.get("TOKEN_EXPIRE_SECONDS", str(60 * 60 * 24 * 7)))  # 7 days

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ---------------------------------------------------------------------------
# Password Hashing (bcrypt direct)
# ---------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    """Returns a bcrypt hash of the given plain-text password."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT Token Utilities
# ---------------------------------------------------------------------------
def create_access_token(user_id: int, username: str, email: str) -> str:
    """Creates a signed JWT access token containing user identity."""
    payload = {
        "sub": str(user_id),
        "username": username,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + ACCESS_TOKEN_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates a JWT token.
    Returns the payload dict on success, or None if invalid / expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

verify_token = decode_access_token



# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency that extracts and validates the JWT from the
    Authorization header.  Returns a dict with user identity fields.
    Raises HTTP 401 if the token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    username = payload.get("username")
    if not user_id or not username:
        raise credentials_exception

    return {
        "id": int(user_id),
        "username": username,
        "email": payload.get("email", ""),
    }
