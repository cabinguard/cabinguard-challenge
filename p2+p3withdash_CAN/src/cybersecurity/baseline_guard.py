"""Authenticated Driver-Digital-Twin baseline files (SIMULATION of local-storage protection).

Self-contained: operates on the baseline *file format* (``{"content": {...}, "sha256": "..."}``, as written by
the Digital Twin's ``Baseline.save``) and returns the verified ``content`` dict. It does not import or
modify Person 1's ``Baseline`` class; the caller builds that object from the verified content.

What is protected, and what is not
  * HMAC-SHA256 over a domain label + canonical JSON of {content, sha256} with a device key kept OUTSIDE the
    file. An attacker who edits the file cannot recompute the MAC (unlike the plain SHA-256, which anyone can).
  * The inner SHA-256 of ``content`` is verified at *sign* time and again at *load* time, so a malformed
    blob cannot be authenticated in the first place.
  * The MAC is verified in constant time BEFORE any field of the content is interpreted, over bytes read ONCE
    (no second open between check and use).
  * Duplicate JSON object keys are rejected (stdlib ``json.loads`` would otherwise keep the last value).
  * The signed envelope schema is exact: ``content``, ``sha256``, ``hmac_sha256`` and nothing else.
  * The domain label separates baseline MACs from message MACs, but a distinct key per purpose is still
    required (``keys.load_key_from_env``). Keys are never hardcoded.
  * A signature is an authenticity statement, not a validity one: the loader still requires the baseline's own
    status to be in ``accepted_statuses`` (so a signed CONTAMINATED/INSUFFICIENT baseline is refused).
  * The baseline must belong to the expected driver (substituting another driver's validly signed file fails).
  * Writes are atomic (temp file + fsync + replace), so a crash cannot leave a half-written signed file.
  * Same-version uniqueness: at most one authenticated content fingerprint may occupy a given
    ``(driver_id, baseline_version)``. A *newer* version with different content is allowed (that is an
    update). A *different* file at the *same* version is a conflict, not an update.
  * Version decisions go through ``TrustedVersionSource.accept()`` (atomic check-and-advance). Separate
    ``floor()`` + ``commit()`` is not used for load, because that is a check-then-act race.
  * HMAC provides integrity/authenticity ONLY. It gives NO CONFIDENTIALITY: the file stays readable.
    Encryption at rest is not designed here.
  * HMAC does NOT prevent baseline poisoning through the legitimate update path. A VALID, correctly
    signed baseline whose feature values were influenced by untrusted observations will still load.
    Stopping that requires an update policy (trusted observations, quality gates, controlled adaptation),
    which is outside this module.
  * Rollback protection is only as strong as ``TrustedVersionSource``. An attacker who can restore an old
    *file* can usually also reset a counter stored next to it; the floor must live in storage the attacker
    cannot write (secure element / TPM monotonic counter / OS-protected store). ``InMemoryVersionSource`` is a
    test double and provides NO protection across restarts and is NOT persistent trusted storage.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from .audit_log import register_event
from .audit_safe import as_safe
from .keys import check_key

register_event("BASELINE_REJECTED")
register_event("BASELINE_ACCEPTED")

MAC_FIELD = "hmac_sha256"
DOMAIN = b"cabinguard-twin/baseline-mac/v1\x00"
MAX_FILE_BYTES = 1 << 20
_ENVELOPE_UNSIGNED = frozenset({"content", "sha256"})
_ENVELOPE_SIGNED = frozenset({"content", "sha256", MAC_FIELD})
_HEX64 = frozenset("0123456789abcdef")


class BaselineIntegrityError(ValueError):
    """Unreadable, unsigned, tampered, wrong-driver or malformed baseline. Never contains key/MAC/content."""


class BaselineRollbackError(BaselineIntegrityError):
    """Authentic baseline older than the trusted minimum version."""


class BaselineVersionConflictError(BaselineIntegrityError):
    """Authentic baseline reuses a version already bound to different content."""


class TrustedVersionSource(Protocol):
    """Trusted monotonic store for per-driver baseline versions.

    ``accept(driver_id, version, fingerprint)`` MUST be atomic: check rollback, bind
    ``(driver_id, version) -> fingerprint``, and advance the floor as one operation.

    ``InMemoryVersionSource`` is a test double. A real source must survive process restart
    in storage the attacker who can rewrite the baseline *file* cannot also rewrite.
    """

    def floor(self, driver_id: str) -> int: ...
    def accept(self, driver_id: str, version: int, fingerprint: str) -> None: ...


class InMemoryVersionSource:
    """TEST DOUBLE. Monotonic within a process only; NOT a trusted source.

    Provides no persistence, no crash recovery, and no protection across restarts.
    Do not describe this as rollback protection in a deployment.
    """

    def __init__(self, floors: dict | None = None):
        self._f = dict(floors or {})
        self._fp: dict[str, str] = {}
        self._lock = threading.Lock()

    def floor(self, driver_id: str) -> int:
        with self._lock:
            return int(self._f.get(driver_id, 0))

    def commit(self, driver_id: str, version: int) -> None:
        """Legacy helper. Load no longer uses this (it is not atomic with conflict checks)."""
        with self._lock:
            self._f[driver_id] = max(int(self._f.get(driver_id, 0)), int(version))

    def accept(self, driver_id: str, version: int, fingerprint: str) -> None:
        if not isinstance(driver_id, str) or not driver_id:
            raise BaselineIntegrityError("invalid driver id for version source")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise BaselineIntegrityError("invalid baseline version for version source")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise BaselineIntegrityError("invalid content fingerprint for version source")
        with self._lock:
            current = int(self._f.get(driver_id, 0))
            bound = self._fp.get(driver_id)
            if version < current:
                raise BaselineRollbackError("baseline rollback refused")
            if version == current:
                if bound is None:
                    # Floor was seeded without a fingerprint (tests). Bind it.
                    self._fp[driver_id] = fingerprint
                    return
                if bound != fingerprint:
                    raise BaselineVersionConflictError(
                        "conflicting baseline at the same version"
                    )
                return
            self._f[driver_id] = version
            self._fp[driver_id] = fingerprint


@dataclass(frozen=True)
class VerifiedBaseline:
    content: dict
    version: int


def _mac(key: bytes, content: dict, sha: str) -> str:
    msg = DOMAIN + json.dumps(
        {"content": content, "sha256": sha},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _inner_sha(content: dict) -> str:
    # Same recipe as the Digital Twin's Baseline.checksum(): sha256(json.dumps(content, sort_keys=True))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


def _reject_non_finite(node) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _reject_non_finite(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _reject_non_finite(v)
    elif isinstance(node, float) and not math.isfinite(node):
        raise BaselineIntegrityError("baseline contains a non-finite number")


def _strict_json_loads(raw: bytes):
    def _pairs(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    return json.loads(raw, object_pairs_hook=_pairs)


def _envelope_unsigned_ok(blob) -> bool:
    return (
        isinstance(blob, dict)
        and set(blob) == _ENVELOPE_UNSIGNED
        and isinstance(blob.get("content"), dict)
        and isinstance(blob.get("sha256"), str)
    )


def sign_baseline_blob(blob: dict, key: bytes) -> dict:
    key = check_key(key, "baseline key")
    if not isinstance(blob, dict):
        raise BaselineIntegrityError("baseline blob must be {'content': dict, 'sha256': str}")
    extra = set(blob) - _ENVELOPE_UNSIGNED
    if extra:
        raise BaselineIntegrityError("baseline blob has extra envelope keys")
    if not _envelope_unsigned_ok(blob):
        raise BaselineIntegrityError("baseline blob must be {'content': dict, 'sha256': str}")
    content = blob["content"]
    sha = blob["sha256"]
    _reject_non_finite(content)
    try:
        computed = _inner_sha(content)
    except (TypeError, ValueError) as e:
        raise BaselineIntegrityError("baseline content is not serialisable") from e
    if not isinstance(sha, str) or not hmac.compare_digest(
        sha.encode("utf-8", "replace"), computed.encode("utf-8")
    ):
        raise BaselineIntegrityError("baseline internal checksum mismatch")
    try:
        mac = _mac(key, content, computed)
    except (TypeError, ValueError) as e:
        raise BaselineIntegrityError("baseline content is not serialisable") from e
    return {"content": content, "sha256": computed, MAC_FIELD: mac}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".baseline-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_signed_baseline(path: str | Path, blob: dict, key: bytes) -> None:
    signed = sign_baseline_blob(blob, key)
    data = json.dumps(signed, indent=1).encode()
    if len(data) > MAX_FILE_BYTES:
        raise BaselineIntegrityError("baseline file too large")
    _atomic_write(Path(path), data)


def sign_baseline_file(path: str | Path, key: bytes) -> None:
    """Sign a file already written by ``Baseline.save`` (replaces any previous MAC), atomically."""
    p = Path(path)
    try:
        blob = _strict_json_loads(p.read_bytes())
    except (OSError, ValueError) as e:
        raise BaselineIntegrityError(f"baseline file unreadable: {type(e).__name__}") from None
    if not isinstance(blob, dict):
        raise BaselineIntegrityError("baseline file unsigned or malformed")
    blob.pop(MAC_FIELD, None)
    write_signed_baseline(p, blob, key)


def load_authenticated_baseline(
    path: str | Path,
    key: bytes,
    *,
    expected_driver_id: str,
    version_source: TrustedVersionSource,
    accepted_statuses: Iterable[str],
    required_features: Iterable[str] | None = None,
    verify_inner_sha256: bool = True,
    audit=None,
) -> VerifiedBaseline:
    """Verify then return the baseline content.

    Any failure raises ``BaselineIntegrityError`` (rollback: ``BaselineRollbackError``;
    same-version clash: ``BaselineVersionConflictError``). The caller must then treat the
    baseline as unavailable - never fall back to an unverified copy.

    ``accepted_statuses`` (e.g. the Digital Twin's VALID marker, owned by Person 1) is
    REQUIRED: a valid signature proves who signed the file, not that the baseline is usable.

    On success the trusted version store atomically binds this content fingerprint to
    ``baseline_version`` (or accepts an identical replay of the already-bound file).
    """
    key = check_key(key, "baseline key")
    a = as_safe(audit)

    def fail(exc_cls, public: str, code: str):
        if a:
            a.record("BASELINE_REJECTED", "baseline", "<local>", code)
        raise exc_cls(public)

    try:
        with open(path, "rb") as fh:
            raw = fh.read(MAX_FILE_BYTES + 1)
    except OSError:
        fail(BaselineIntegrityError, "baseline file unreadable", "unreadable")
    if len(raw) > MAX_FILE_BYTES:
        fail(BaselineIntegrityError, "baseline file too large", "too_large")
    try:
        blob = _strict_json_loads(raw)
    except ValueError as e:
        if "duplicate" in str(e).lower():
            fail(BaselineIntegrityError, "baseline file has duplicate JSON keys", "duplicate")
        fail(BaselineIntegrityError, "baseline file is not valid JSON", "not_json")
    except RecursionError:
        fail(BaselineIntegrityError, "baseline file is not valid JSON", "not_json")

    if not isinstance(blob, dict):
        fail(BaselineIntegrityError, "baseline file unsigned or malformed", "unsigned")
    keys = set(blob)
    if not _ENVELOPE_SIGNED <= keys:
        fail(BaselineIntegrityError, "baseline file unsigned or malformed", "unsigned")
    if keys != _ENVELOPE_SIGNED:
        fail(BaselineIntegrityError, "baseline file unsigned or malformed", "malformed")
    if not (
        isinstance(blob.get("content"), dict)
        and isinstance(blob.get("sha256"), str)
        and isinstance(blob.get(MAC_FIELD), str)
    ):
        fail(BaselineIntegrityError, "baseline file unsigned or malformed", "unsigned")

    content = blob["content"]
    sha = blob["sha256"]
    try:
        expected = _mac(key, content, sha)
    except (TypeError, ValueError):
        fail(BaselineIntegrityError, "baseline file malformed", "malformed")
    got = blob[MAC_FIELD]
    if not _is_hex64(got) or not hmac.compare_digest(
        got.encode("utf-8", "replace"), expected.encode()
    ):
        fail(BaselineIntegrityError, "baseline authentication failed", "auth_failed")
    try:
        _reject_non_finite(content)
    except BaselineIntegrityError:
        fail(BaselineIntegrityError, "baseline contains a non-finite number", "non_finite")

    if verify_inner_sha256:
        try:
            computed = _inner_sha(content)
        except (TypeError, ValueError):
            fail(BaselineIntegrityError, "baseline internal checksum mismatch", "checksum")
        if not hmac.compare_digest(computed.encode(), sha.encode("utf-8", "replace")):
            fail(BaselineIntegrityError, "baseline internal checksum mismatch", "checksum")

    if not isinstance(expected_driver_id, str) or content.get("driver_id") != expected_driver_id:
        fail(BaselineIntegrityError, "baseline belongs to a different driver", "driver_mismatch")
    accepted = frozenset(accepted_statuses)
    if not accepted or content.get("status") not in accepted:
        fail(BaselineIntegrityError, "baseline status not acceptable", "bad_status")
    ver = content.get("baseline_version")
    if isinstance(ver, bool) or not isinstance(ver, int) or ver < 1:
        fail(BaselineIntegrityError, "baseline version missing or invalid", "bad_version")
    if required_features is not None:
        feats = content.get("features")
        if not isinstance(feats, dict) or any(f not in feats for f in required_features):
            fail(BaselineIntegrityError, "baseline lacks required features", "missing_features")

    try:
        fingerprint = _inner_sha(content)
    except (TypeError, ValueError):
        fail(BaselineIntegrityError, "baseline file malformed", "malformed")
    try:
        version_source.accept(expected_driver_id, ver, fingerprint)
    except BaselineRollbackError:
        fail(BaselineRollbackError, "baseline rollback refused", "rollback")
    except BaselineVersionConflictError:
        fail(BaselineVersionConflictError, "conflicting baseline at the same version", "same_version")
    except Exception:
        fail(BaselineIntegrityError, "trusted version source failed", "version source")

    if a:
        a.record("BASELINE_ACCEPTED", "baseline", "<local>", "verified", f"ver={ver}")
    return VerifiedBaseline(content, ver)