"""
Authenticated encryption for the biometric template store.

Generate a key with:

    python -m utils.crypto

"""

import base64
import json
import os
from typing import Any, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import MLServiceError

KEY_ENV = "FACE_ENCODING_KEY"
KEY_BYTES = 32
NONCE_BYTES = 12
ENVELOPE_VERSION = 1
ALGORITHM = "AES-256-GCM"

# Bound into every ciphertext as associated data, so an envelope cannot be
# replayed under a different version or algorithm label.
AAD = b"trace-face-encodings-v1"


class EncryptionKeyError(MLServiceError):
    """The encryption key is absent or malformed."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class DecryptionError(MLServiceError):
    """The stored template store could not be decrypted."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


def generate_key() -> str:
    """Return a fresh base64-encoded 256-bit key."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def key_is_configured() -> bool:
    """Report whether an encryption key is present in the environment."""
    return bool(os.getenv(KEY_ENV))


def load_key() -> bytes:
    """
    Read and validate the encryption key.

    """
    raw = os.getenv(KEY_ENV)
    if not raw:
        raise EncryptionKeyError(
            f"{KEY_ENV} is not set. Biometric templates are encrypted at rest and "
            f"cannot be read or written without it. Generate one with "
            f"'python -m utils.crypto'."
        )

    try:
        key = base64.b64decode(raw, validate=True)
    except Exception:
        raise EncryptionKeyError(
            f"{KEY_ENV} is not valid base64. Generate one with "
            f"'python -m utils.crypto'."
        )

    if len(key) != KEY_BYTES:
        raise EncryptionKeyError(
            f"{KEY_ENV} decodes to {len(key)} bytes; AES-256-GCM requires "
            f"{KEY_BYTES}. Generate one with 'python -m utils.crypto'."
        )

    return key


def is_encrypted(payload: Any) -> bool:
    """
    Report whether a parsed JSON document is an encryption envelope.

    """
    return isinstance(payload, dict) and "ciphertext" in payload and "nonce" in payload


def encrypt(document: Dict[str, Any]) -> Dict[str, Any]:
    """Encrypt a JSON-serialisable document into an on-disk envelope."""
    key = load_key()
    nonce = os.urandom(NONCE_BYTES)
    plaintext = json.dumps(document).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, AAD)

    return {
        "version": ENVELOPE_VERSION,
        "algorithm": ALGORITHM,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt an on-disk envelope back into its document."""
    version = envelope.get("version")
    if version != ENVELOPE_VERSION:
        raise DecryptionError(
            f"Unsupported template store version {version!r}; expected "
            f"{ENVELOPE_VERSION}."
        )

    key = load_key()

    try:
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
    except Exception:
        raise DecryptionError("Template store envelope is malformed.")

    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, AAD)
    except InvalidTag:
        raise DecryptionError(
            f"Template store failed authentication. Either {KEY_ENV} is not the key "
            f"the store was written with, or the file has been altered."
        )

    return json.loads(plaintext.decode("utf-8"))


if __name__ == "__main__":
    print(generate_key())
