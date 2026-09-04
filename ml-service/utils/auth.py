import os
from typing import Optional

from fastapi import Header
from jose import JWTError, jwt

from .errors import MLServiceError


class UnauthorizedError(MLServiceError):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)


ALGORITHM = "HS256"


def _bearer(authorization: Optional[str]) -> str:
    """Extract the bearer credential from an Authorization header."""
    if not authorization:
        raise UnauthorizedError("Missing Authorization header")

    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        raise UnauthorizedError("Expected an Authorization: Bearer <token> header")

    return credential


async def verify_token(authorization: Optional[str] = Header(default=None)) -> str:
    """
    Authenticate a caller of the ML service.
    
    """
    token = _bearer(authorization)

    service_token = os.getenv("ML_SERVICE_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    if not service_token and not access_token_secret:
        raise UnauthorizedError(
            "ML service is not configured for authentication: set ACCESS_TOKEN_SECRET "
            "or ML_SERVICE_TOKEN"
        )

    if service_token and token == service_token:
        return "service"

    if not access_token_secret:
        raise UnauthorizedError("Invalid credential")

    try:
        payload = jwt.decode(token, access_token_secret, algorithms=[ALGORITHM])
    except JWTError:
        raise UnauthorizedError("Invalid or expired access token")

    subject = payload.get("userId") or payload.get("sub")
    if not subject:
        raise UnauthorizedError("Access token carries no subject")

    return str(subject)
