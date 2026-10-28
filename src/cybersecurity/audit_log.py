"""Bounded, redacting security audit log (SIMULATION).

Design rule: a security log entry may contain *what happened and where* (event type, stage,
pseudonymous sender label, standardized reason code, coarse detail code). It must NEVER contain
payload contents, authentication tags, keys, raw sensor values or driver identity.

Privacy-by-construction, not truncation-by-convention:
  * ``event``, ``stage``, and ``reason`` must belong to a small, centralised, extensible controlled
    vocabulary (``Event`` / ``Stage`` / ``Reason`` below, plus ``register_event`` / ``register_stage``
    / ``register_reason`` for future modules such as telemetry.py or sensor_trust.py). Anything not
    a recognised code is replaced by a safe generic placeholder - never logged verbatim, because
    truncating an attacker-controlled string does NOT stop control-character/newline log-injection,
    it just shortens it.
  * ``sender`` and ``detail_code`` are strictly format-checked (bounded length, a small safe
    character set, no newlines/control characters/ANSI escapes). Unregistered or malformed senders
    are logged as ``"<unregistered>"``/``"<invalid-sender>"`` rather than copied verbatim.

Thread safety: the sequence counter and ring buffer are protected by a single lock, so concurrent
``record()`` calls can never produce duplicate sequence numbers or a corrupted buffer; ``entries()``,
``to_dicts()`` and ``__len__()`` take a consistent snapshot under the same lock.

Storage: a fixed-size ``collections.deque`` ring buffer. ``seq`` keeps increasing monotonically even
once older entries are evicted. This is an in-process, in-memory SIMULATION audit trail: it is NOT
persistent (lost on process exit) and NOT tamper-proof (any code in the same process can read or
clear it) - a real deployment would need an append-only / write-once store, ideally outside this
component's own trust domain.
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import asdict, dataclass

_ALLOWED_FIELDS = ("seq", "event", "stage", "sender", "reason", "detail_code")


# --------------------------------------------------------------------------- controlled vocabulary
class Event:
    MESSAGE_REJECTED = "MESSAGE_REJECTED"
    MESSAGE_ACCEPTED = "MESSAGE_ACCEPTED"
    SENDER_RESET = "SENDER_RESET"
    CONFIG_ERROR = "CONFIG_ERROR"


class Stage:
    SCHEMA = "schema"
    AUTHENTICATION = "authentication"
    INTEGRITY = "integrity"
    FRESHNESS = "freshness"
    REPLAY = "replay"
    AUTHORIZATION = "authorization"
    ACCEPTED = "accepted"
    RESYNC = "resync"
    CONFIG = "config"


class Reason:
    MALFORMED_MESSAGE = "MALFORMED_MESSAGE"
    UNKNOWN_SENDER = "UNKNOWN_SENDER"
    INVALID_TAG_FORMAT = "INVALID_TAG_FORMAT"
    MAC_MISMATCH = "MAC_MISMATCH"
    TOO_OLD = "TOO_OLD"
    TOO_FAR_FUTURE = "TOO_FAR_FUTURE"
    REPLAY_OR_DUPLICATE = "REPLAY_OR_DUPLICATE"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    ALL_CHECKS_PASSED = "ALL_CHECKS_PASSED"
    ADMIN_RESYNC = "ADMIN_RESYNC"
    CONFIG_INVALID = "CONFIG_INVALID"
    UNSPECIFIED = "UNSPECIFIED"


def _collect(cls) -> frozenset:
    return frozenset(v for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, str))


_EVENT_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_STAGE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SENDER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")
_DETAIL_RE = re.compile(r"^[A-Za-z0-9_=.-]{0,24}$")
_SPECIAL_SENDER_LABELS = frozenset({"<unregistered>"})

_vocab_lock = threading.Lock()
_EVENTS = set(_collect(Event))
_STAGES = set(_collect(Stage))
_REASONS = set(_collect(Reason))


def register_event(name: str) -> None:
    """Allow future cybersecurity modules to extend the event vocabulary. Rejects anything that
    doesn't look like a fixed code (uppercase, digits, underscore) - never free text."""
    if not isinstance(name, str) or not _EVENT_CODE_RE.match(name):
        raise ValueError("event name must match ^[A-Z][A-Z0-9_]{0,63}$")
    with _vocab_lock:
        _EVENTS.add(name)


def register_stage(name: str) -> None:
    if not isinstance(name, str) or not _STAGE_CODE_RE.match(name):
        raise ValueError("stage name must match ^[a-z][a-z0-9_]{0,63}$")
    with _vocab_lock:
        _STAGES.add(name)


def register_reason(code: str) -> None:
    if not isinstance(code, str) or not _EVENT_CODE_RE.match(code):
        raise ValueError("reason code must match ^[A-Z][A-Z0-9_]{0,63}$")
    with _vocab_lock:
        _REASONS.add(code)


def _sanitize_event(event: object) -> str:
    with _vocab_lock:
        ok = isinstance(event, str) and event in _EVENTS
    return event if ok else Event.MESSAGE_REJECTED


def _sanitize_stage(stage: object) -> str:
    with _vocab_lock:
        ok = isinstance(stage, str) and stage in _STAGES
    return stage if ok else Stage.SCHEMA


def _sanitize_reason(reason: object) -> str:
    with _vocab_lock:
        ok = isinstance(reason, str) and reason in _REASONS
    return reason if ok else Reason.UNSPECIFIED


def _sanitize_sender(sender: object) -> str:
    if sender in _SPECIAL_SENDER_LABELS:
        return sender
    if isinstance(sender, str) and _SENDER_RE.match(sender):
        return sender
    return "<invalid-sender>"


def _sanitize_detail_code(detail_code: object) -> str:
    if detail_code == "" or detail_code is None:
        return ""
    if isinstance(detail_code, str) and _DETAIL_RE.match(detail_code):
        return detail_code
    return "<invalid-detail>"


# --------------------------------------------------------------------------- entry + log
@dataclass(frozen=True)
class AuditEntry:
    seq: int                 # log-local monotonically increasing index (never reused, never reset)
    event: str                 # controlled vocabulary, e.g. Event.MESSAGE_REJECTED
    stage: str                  # controlled vocabulary, e.g. Stage.INTEGRITY
    sender: str                  # registered sender label, "<unregistered>" or "<invalid-sender>"
    reason: str                   # controlled vocabulary code, e.g. Reason.MAC_MISMATCH
    detail_code: str = ""          # optional short, format-restricted code, e.g. "seq=12"


class SecurityAuditLog:
    """Bounded, thread-safe, privacy-sanitising audit trail. NOT persistent, NOT tamper-proof."""

    def __init__(self, maxlen: int = 10_000):
        if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        self._buf: deque[AuditEntry] = deque(maxlen=maxlen)
        self._n = 0
        self._lock = threading.Lock()

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen

    def record(self, event: str, stage: str, sender: str, reason: str, detail_code: str = "") -> AuditEntry:
        """Append a sanitised audit entry. Never raises on malformed/attacker-controlled input:
        unrecognised event/stage/reason values fall back to safe generic codes, and sender/detail_code
        are strictly format-checked (no newlines, no control characters, bounded length) rather than
        merely truncated."""
        safe_event = _sanitize_event(event)
        safe_stage = _sanitize_stage(stage)
        safe_reason = _sanitize_reason(reason)
        safe_sender = _sanitize_sender(sender)
        safe_detail = _sanitize_detail_code(detail_code)
        with self._lock:
            self._n += 1
            e = AuditEntry(self._n, safe_event, safe_stage, safe_sender, safe_reason, safe_detail)
            self._buf.append(e)      # seq keeps climbing even once old entries are evicted
        return e

    def entries(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._buf)

    def to_dicts(self) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._buf]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)