"""Key handling helpers (SIMULATION). No key material is ever stored in code or config.

Design decisions
  * Keys come from the operating system's CSPRNG (``secrets``) or from the environment / a secret store
    supplied by the deployer. They are never derived from a seed (a seeded PRNG is reproducible, hence
    guessable by anyone who knows the seed - acceptable for unit-test fixtures only, never for keys).
  * Error messages never contain key material or environment values.
  * One key per purpose (message sender, baseline store, ...): a key used for two purposes lets a MAC
    produced in one context be replayed in the other. See ``baseline_integrity`` for the domain label.
"""
from __future__ import annotations

import os
import secrets

MIN_KEY_BYTES = 16          # same floor the existing MessageValidator enforces
RECOMMENDED_KEY_BYTES = 32  # HMAC-SHA256 block-size-independent strength target


class KeyConfigError(ValueError):
    """A key is missing or malformed. The message never includes the key or the variable's value."""


def generate_key(n_bytes: int = RECOMMENDED_KEY_BYTES) -> bytes:
    """Fresh random key from the OS CSPRNG."""
    if not isinstance(n_bytes, int) or isinstance(n_bytes, bool) or n_bytes < MIN_KEY_BYTES:
        raise KeyConfigError(f"key length must be an int >= {MIN_KEY_BYTES}")
    return secrets.token_bytes(n_bytes)


def load_key_from_env(var: str, min_bytes: int = RECOMMENDED_KEY_BYTES) -> bytes:
    """Read a hex-encoded key from an environment variable. Fails closed if absent or too short."""
    raw = os.environ.get(var)
    if not raw:
        raise KeyConfigError(f"environment variable {var!r} is not set")
    try:
        key = bytes.fromhex(raw.strip())
    except ValueError:
        raise KeyConfigError(f"environment variable {var!r} is not valid hex") from None
    if len(key) < min_bytes:
        raise KeyConfigError(f"environment variable {var!r} holds a key shorter than {min_bytes} bytes")
    return key


def check_key(key: object, what: str = "key") -> bytes:
    """Validate a key supplied programmatically; returns it unchanged."""
    if not isinstance(key, (bytes, bytearray)) or len(key) < MIN_KEY_BYTES:
        raise KeyConfigError(f"{what} must be bytes of at least {MIN_KEY_BYTES} bytes")
    return bytes(key)
