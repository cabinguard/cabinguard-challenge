"""SIMULATED secure message layer (HMAC-SHA256 from the Python standard library).

Checks, in this order: schema -> authentication (known sender) -> integrity (tag format + MAC; with
per-sender keys a valid MAC also authenticates the sender) -> freshness (past age AND future clock
skew) -> replay (read-only check) -> authorization (role permits message type) -> COMMIT of replay
state.

Person-3 audit changes, round 1 (see reports/security/audit_report.md):
  * D1  replay state is committed only AFTER every check has passed, so a rejected packet can never
        advance the accepted sequence counter.
  * D2-D5 strict schema: finite numeric timestamp, non-negative non-bool integer sequence, canonically
        serialisable payload; malformed input is *rejected*, never raised as an exception.
  * D6  separate, configurable future clock-skew tolerance.
  * D7  constructor-time validation of max_age_s / max_future_skew_s (finite, non-negative); a
        misconfigured validator fails fast at construction.
  * D8  audit-log failures are isolated: counted in stats["audit_failures"], never turn a rejection
        into an acceptance, never crash the caller.
  * D9  tag format checked (64 lowercase hex chars) immediately before the MAC comparison, both
        grouped under "integrity", AFTER authentication - preserving the documented stage order.
  * D10 validate()/reset_sender() are internally synchronised so concurrent calls cannot race on
        last_seq (check-then-commit is atomic).
  * D11 tag-format validation happens strictly after the authentication lookup (fixes a stage-order
        regression that existed briefly between D9 and this fix).
  * D12 the lock protects ONLY in-memory state mutation (last_seq + stats). All audit-log I/O happens
        strictly after the lock is released, so a slow/raising audit sink can never serialise or
        block unrelated validate() calls.
  * explicit, authorised sender resynchronisation (``reset_sender``) - no silent counter resets.
  * injectable clock and redacting audit log.

Person-3 audit changes, round 2 ("CABINGUARD-TWIN" cybersecurity foundation hardening pass):
  * E1  Configuration hardening: ``registry``/``roles`` are structurally validated and then
        defensively copied into read-only (``MappingProxyType``) structures at construction time.
        Mutating the caller's original dict/bytearray/set after construction can no longer change
        live authentication keys or permissions.
  * E2  Payload TOCTOU fix: ``validate()`` takes an internal deep-copy snapshot of the payload
        *before* performing any check, runs every subsequent check (canonicalisation, MAC, freshness,
        replay, authorization) against that frozen snapshot, and - on success - returns that exact
        snapshot via ``ValidationResult.payload``. The bytes that were authenticated are the bytes
        the caller receives; there is no window for another thread/component to swap the payload.
        ``trusted_payload()`` is kept for backward compatibility but is documented as weaker
        (non-atomic) relative to ``ValidationResult.payload``.
  * E3  ``verify_tag()`` validates its own argument types first and therefore truly never raises,
        for any input shape.
  * E4  ``sign_message()`` now runs the same schema validation the receiver performs, and validates
        the signing key, before signing - raising ``SecureMessageError`` instead of silently signing
        a message no validator could ever accept, or leaking a bare stdlib exception.
  * E5  Standardised, centralised audit event/stage/reason CODES (see ``audit_log.py``) are used for
        everything written to the audit sink. ``ValidationResult.reason`` keeps the original
        human-readable, debugging-friendly text; only the safe code (e.g. ``Reason.MAC_MISMATCH``)
        is ever sent to the audit log.

Limitations (unchanged): symmetric pre-shared test keys, no key provisioning/rotation, no
confidentiality (payload not encrypted), no hardware root of trust, JSON canonical form is a
simulation convenience. NOT a claim that a real network is secured. No custom cryptography is
implemented.

Residual limitations callers MUST account for (by design, not "bugs"):
  * Concurrency assumption: if the *same* ``SecureMessage`` object is mutated by another thread while
    ``validate()`` is actively running on it, the result reflects whatever the deep-copy snapshot
    managed to observe at that instant; Python's GIL prevents memory corruption but not a
    torn/partial view of a concurrently-mutated nested structure. Do not share a mutable
    ``SecureMessage`` across threads while it is being validated. Once ``validate()`` returns,
    ``ValidationResult.payload`` is a fully independent object nothing else can affect.
  * ``reset_sender(sender_id, authorized=True, ...)`` is a bare boolean gate, not an authentication
    mechanism. It MUST only be called from trusted administrative code that has independently
    verified the request out-of-band, and MUST NOT be exposed to a network-facing API driven by an
    unauthenticated or merely-authenticated-as-sender client. ``last_seq`` is in-memory only and is
    lost on restart; a real deployment needs persisted replay state or a session/key-rotation scheme.
  * A single process-wide lock serialises state mutation across ALL senders. For this simulation's
    throughput that's a non-issue (the critical section is a handful of dict operations, no I/O); a
    high-throughput multi-sender deployment might prefer per-sender locks.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable

from .audit_log import SecurityAuditLog, Event, Stage, Reason

MAX_SEQ = 2 ** 63 - 1
MIN_KEY_BYTES = 16
_HEX_DIGITS = frozenset("0123456789abcdef")


class SecureMessageError(ValueError):
    """Raised when a message cannot be constructed/signed (e.g. non-canonical payload)."""


@dataclass(frozen=True)
class SecureMessage:
    sender_id: str
    seq: int
    ts: float
    msg_type: str
    payload: dict
    tag: str = ""


def trusted_payload(m: SecureMessage) -> dict:
    """Return a deep copy of ``m.payload``.

    PREFER ``ValidationResult.payload`` from ``MessageValidator.validate()``: that snapshot is taken
    atomically with MAC verification itself (E2), so there is no window for another
    thread/component to swap the payload between "verified" and "copied". This helper simply
    deep-copies whatever ``m.payload`` looks like *at the moment it is called* - if called separately
    after ``validate()`` has already returned, a concurrent mutation of ``m.payload`` in that gap is
    NOT protected against. Kept for convenience/back-compat in single-threaded call sites.
    """
    return deepcopy(m.payload)


def _canon(m: SecureMessage) -> bytes:
    """Deterministic serialisation covering ALL metadata (sender, seq, timestamp, type) + payload."""
    return json.dumps({"s": m.sender_id, "n": m.seq, "t": m.ts, "y": m.msg_type, "p": m.payload},
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _mac_hex(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _valid_tag_format(tag: str) -> bool:
    """A SHA-256 HMAC hex digest is exactly 64 lowercase hex characters."""
    return isinstance(tag, str) and len(tag) == 64 and all(c in _HEX_DIGITS for c in tag)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _schema_error(m) -> str | None:
    """Shared schema check used by both the sender (``sign_message``) and the receiver
    (``MessageValidator``), so a message can never be signed in a shape the receiver would reject."""
    if not isinstance(m, SecureMessage):
        return "not a SecureMessage"
    if not isinstance(m.sender_id, str) or not m.sender_id:
        return "sender_id must be a non-empty string"
    if not isinstance(m.seq, int) or isinstance(m.seq, bool) or not (0 <= m.seq <= MAX_SEQ):
        return "seq must be a non-negative integer"
    if not _is_num(m.ts) or not math.isfinite(m.ts):
        return "ts must be a finite number"
    if not isinstance(m.msg_type, str) or not m.msg_type:
        return "msg_type must be a non-empty string"
    if not isinstance(m.payload, dict):
        return "payload must be a dict"
    if not isinstance(m.tag, str):
        return "tag must be a string"
    return None


def sign_message(m: SecureMessage, key: bytes) -> SecureMessage:
    """Sign ``m`` with HMAC-SHA256 over its canonical serialisation.

    Rejects malformed messages and malformed/unusable keys with ``SecureMessageError`` instead of
    silently signing something no validator could ever accept, or leaking a raw stdlib exception
    (e.g. a ``TypeError`` from ``hmac.new``) to the caller (E4).
    """
    err = _schema_error(m)
    if err:
        raise SecureMessageError(f"cannot sign malformed message: {err}")
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise SecureMessageError("invalid signing key: must be non-empty bytes")
    try:
        data = _canon(m)
    except (TypeError, ValueError) as e:
        raise SecureMessageError(f"message not canonically serialisable: {e}") from e
    try:
        tag = _mac_hex(bytes(key), data)
    except TypeError as e:
        raise SecureMessageError(f"invalid signing key: {e}") from e
    return replace(m, tag=tag)


def verify_tag(m: SecureMessage, key: bytes) -> bool:
    """Constant-time tag comparison. NEVER raises: any malformed/unsupported input verifies False (E3)."""
    try:
        if not isinstance(m, SecureMessage):
            return False
        if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
            return False
        if not isinstance(m.tag, str):
            return False
        expected = _mac_hex(bytes(key), _canon(m)).encode("utf-8")
        got = m.tag.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, AttributeError):
        return False
    return hmac.compare_digest(got, expected)


@dataclass
class ValidationResult:
    ok: bool
    stage: str
    reason: str
    # Populated ONLY when ok=True: an independent deep-copy snapshot of the payload that was
    # actually MAC-verified (E2). Downstream code should consume THIS, not ``message.payload``.
    payload: dict | None = None


def _validate_registry(registry) -> dict:
    if not isinstance(registry, dict):
        raise ValueError("registry must be a dict")
    out = {}
    for sid, ent in registry.items():
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"registry key {sid!r} must be a non-empty string")
        if not isinstance(ent, dict):
            raise ValueError(f"registry[{sid!r}] must be a dict")
        key = ent.get("key")
        if not isinstance(key, (bytes, bytearray)) or len(key) < MIN_KEY_BYTES:
            raise ValueError(f"registry[{sid!r}]: key must be bytes of at least {MIN_KEY_BYTES} bytes")
        role = ent.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"registry[{sid!r}]: role must be a non-empty string")
        # Freeze a NEW entry: bytes(key) copies bytearray content, MappingProxyType blocks further
        # mutation of this entry through the validator's own registry reference (E1).
        out[sid] = MappingProxyType({"key": bytes(key), "role": role})
    return out


def _validate_roles(roles) -> dict:
    if not isinstance(roles, dict):
        raise ValueError("roles must be a dict")
    out = {}
    for role, perms in roles.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"role key {role!r} must be a non-empty string")
        if isinstance(perms, (str, bytes)) or not isinstance(perms, (set, frozenset, list, tuple)):
            raise ValueError(f"roles[{role!r}] must be a set/list/tuple of message-type strings")
        clean = set()
        for p in perms:
            if not isinstance(p, str) or not p:
                raise ValueError(f"roles[{role!r}] contains an invalid message type {p!r}")
            clean.add(p)
        out[role] = frozenset(clean)
    return out


class MessageValidator:
    """Per-sender-key validator with replay state. ``registry``: sender_id -> {"key": bytes, "role": str}."""

    def __init__(self, registry: dict, roles: dict, max_age_s: float, max_future_skew_s: float | None = None,
                 clock: Callable[[], float] | None = None, audit: SecurityAuditLog | None = None):
        # E1: structural validation + defensive copy BEFORE anything is stored. After this point,
        # mutating the caller's original `registry`/`roles` objects has zero effect on this instance.
        validated_registry = _validate_registry(registry)
        validated_roles = _validate_roles(roles)

        # D7: fail fast on a misconfigured validator instead of silently weakening freshness checks.
        if not _is_num(max_age_s) or not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("max_age_s must be a finite, non-negative number")
        if max_future_skew_s is not None:
            if (
                not _is_num(max_future_skew_s)
                or not math.isfinite(max_future_skew_s)
                or max_future_skew_s < 0
            ):
                raise ValueError("max_future_skew_s must be a finite, non-negative number")

        self.registry = MappingProxyType(validated_registry)
        self.roles = MappingProxyType(validated_roles)
        self.max_age = float(max_age_s)
        self.max_future = self.max_age if max_future_skew_s is None else float(max_future_skew_s)
        self.clock, self.audit = clock, audit
        self.last_seq: dict[str, int] = {}
        self.stats: dict[str, int] = {}
        # D10/D12: protects last_seq and stats ONLY (pure in-memory ops). Never held during audit I/O.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _label(self, m) -> str:
        """Returns sender_id ONLY if it is a known/registered identifier; otherwise a fixed,
        non-attacker-controlled placeholder. First line of defence against log injection via a
        forged sender_id (audit_log.py sanitises again, in depth)."""
        sid = getattr(m, "sender_id", None)
        return sid if isinstance(sid, str) and sid in self.registry else "<unregistered>"

    def _bump_stat(self, stage: str) -> None:
        with self._lock:
            self.stats[stage] = self.stats.get(stage, 0) + 1

    def _safe_audit(self, event: str, stage: str, sender_label: str, reason_code: str, detail: str = "") -> None:
        """Call ``self.audit.record`` without letting a broken/raising sink affect the caller or the
        verdict (D8). MUST be called with no lock held - audit I/O may be slow (D12). Only
        standardised codes (``Event``/``Stage``/``Reason``) are ever sent here - never free text,
        payloads, keys or tags (E5)."""
        if self.audit is None:
            return
        try:
            self.audit.record(event, stage, sender_label, reason_code, detail)
        except Exception:
            self._bump_stat("audit_failures")

    def _seq_detail(self, m) -> str:
        seq = getattr(m, "seq", None)
        return f"seq={seq}" if isinstance(seq, int) and not isinstance(seq, bool) else ""

    def _rej(self, m, stage: str, reason: str, code: str) -> ValidationResult:
        """Used by every check stage that runs OUTSIDE the last_seq/commit critical section (schema,
        authentication, integrity, freshness). ``reason`` is the full, descriptive text kept on the
        returned ``ValidationResult`` for debugging; ``code`` is the standardised audit-log code."""
        self._bump_stat(stage)
        self._safe_audit(Event.MESSAGE_REJECTED, stage, self._label(m), code, self._seq_detail(m))
        return ValidationResult(False, stage, reason)

    # ------------------------------------------------------------------ public API
    def validate(self, m: SecureMessage, now: float | None = None) -> ValidationResult:
        if now is None:
            now = self.clock() if self.clock is not None else time.time()
        if not _is_num(now) or not math.isfinite(now):
            raise ValueError("now must be a finite number")           # programmer error, not attacker input

        # ---- schema ----
        err = _schema_error(m)
        if err:
            return self._rej(m, Stage.SCHEMA, f"malformed message: {err}", Reason.MALFORMED_MESSAGE)

        # ---- E2: take an immutable, deep-copied snapshot of the payload NOW, before any further
        # check runs. Every subsequent step (canonicalisation for the MAC, freshness, replay,
        # authorization, and the payload returned to the caller) operates on THIS snapshot, so the
        # bytes that get authenticated are exactly the bytes the caller eventually receives - there
        # is no time-of-check/time-of-use gap.
        try:
            snap = replace(m, payload=deepcopy(m.payload))
        except Exception:
            return self._rej(m, Stage.SCHEMA, "malformed message: payload not copyable", Reason.MALFORMED_MESSAGE)

        try:
            data = _canon(snap)
        except (TypeError, ValueError):
            return self._rej(snap, Stage.SCHEMA, "malformed message: payload not canonically serialisable",
                              Reason.MALFORMED_MESSAGE)

        # ---- authentication ---- (D11: BEFORE any tag inspection, per the documented stage order)
        ent = self.registry.get(snap.sender_id)
        if ent is None:
            return self._rej(snap, Stage.AUTHENTICATION, "unknown sender", Reason.UNKNOWN_SENDER)

        # ---- integrity ---- (format check, D9, then MAC compare; both after authentication)
        if not _valid_tag_format(snap.tag):
            return self._rej(snap, Stage.INTEGRITY, "invalid HMAC tag format", Reason.INVALID_TAG_FORMAT)
        try:
            tag_ok = hmac.compare_digest(snap.tag.encode("utf-8"), _mac_hex(ent["key"], data).encode())
        except UnicodeError:
            tag_ok = False
        if not tag_ok:
            return self._rej(snap, Stage.INTEGRITY, "MAC mismatch (tampered or forged)", Reason.MAC_MISMATCH)

        # ---- freshness ----
        age = now - snap.ts
        if age > self.max_age:
            return self._rej(snap, Stage.FRESHNESS, f"timestamp older than {self.max_age}s", Reason.TOO_OLD)
        if -age > self.max_future:
            return self._rej(snap, Stage.FRESHNESS, f"timestamp more than {self.max_future}s in the future",
                              Reason.TOO_FAR_FUTURE)

        # ---- replay (check) + authorization (check) + COMMIT ----
        # D12: the lock wraps ONLY this fast, in-memory, no-I/O section. Any audit logging for a
        # rejection here happens strictly AFTER the lock is released.
        with self._lock:
            if snap.seq <= self.last_seq.get(snap.sender_id, -1):
                stage = Stage.REPLAY
                reason = "sequence number not increasing (replay/duplicate)"
                code = Reason.REPLAY_OR_DUPLICATE
            elif snap.msg_type not in self.roles.get(ent["role"], frozenset()):
                stage = Stage.AUTHORIZATION
                reason = f"role {ent['role']!r} not permitted to send {snap.msg_type}"
                code = Reason.NOT_AUTHORIZED
            else:
                self.last_seq[snap.sender_id] = snap.seq    # COMMIT only after every check passed (D1)
                stage, reason, code = Stage.ACCEPTED, "all checks passed", Reason.ALL_CHECKS_PASSED
            self.stats[stage] = self.stats.get(stage, 0) + 1

        if stage != Stage.ACCEPTED:
            self._safe_audit(Event.MESSAGE_REJECTED, stage, self._label(snap), code, self._seq_detail(snap))
            return ValidationResult(False, stage, reason)
        return ValidationResult(True, Stage.ACCEPTED, reason, payload=snap.payload)

    def reset_sender(self, sender_id: str, authorized: bool, reason: str = "authorised resynchronisation") -> None:
        """Explicit replay-state reset for a legitimate reconnect/reboot.

        ``authorized`` is a bare boolean gate, NOT an authentication mechanism. This method MUST
        remain internal to a trusted administrative component and MUST NOT be exposed through an API
        that lets a network client supply ``authorized`` directly. ``last_seq`` is in-memory only.

        ``reason`` is accepted for the CALLER's own bookkeeping/logging but is deliberately NOT
        forwarded verbatim into the shared security audit log (E5): only the standardised
        ``Reason.ADMIN_RESYNC`` code is recorded there.
        """
        if not authorized:
            raise PermissionError("reset refused: not authorised")
        if sender_id not in self.registry:
            raise KeyError("unknown sender")
        with self._lock:
            self.last_seq.pop(sender_id, None)
        self._safe_audit(Event.SENDER_RESET, Stage.RESYNC, sender_id, Reason.ADMIN_RESYNC)