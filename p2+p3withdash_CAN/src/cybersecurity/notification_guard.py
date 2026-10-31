"""External-notification privacy guard (TB3, SIMULATION; nothing is transmitted).

  * Strict allow-list of keys AND values. The payload carries exactly the six permitted fields; every value
    comes from a closed vocabulary or a fixed format. Free text, nested objects, identities and raw signals
    can not pass. Domain vocabularies (event categories, Guardian states) are injected by their owners.
  * Event ids are random and unlinkable to the driver.
  * Timestamp mitigation (partial, tested): the reference signs the EXACT transmission time in the envelope,
    defeating the 10 s coarsening of the payload. Here the envelope ``ts`` is rounded DOWN to the bucket
    (default 10 s) and the receiving validator for this channel uses ``max_age = bucket + slack``.
    Trade-off: the replay-freshness window of this channel grows from ~2 s to ~12 s; strict sequence numbers
    still reject re-delivery of an already accepted notification, but a *delayed* first delivery (suppress,
    then release within the window) can not be distinguished from a slow one.
    Residual: network arrival time, the sequence counter and any transport metadata still reveal timing;
    this is a reduction of what the signed fields disclose, NOT a solution to timing privacy.
"""
from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass, field

from .keys import check_key
from .secure_message import MessageValidator, SecureMessage, ValidationResult, sign_message

NOTIFICATION_MSG_TYPE = "EMERGENCY_NOTIFICATION"
ALLOWED_KEYS = frozenset({"event_id", "category", "risk_level", "confidence_level", "timestamp", "vehicle_state"})
_EVENT_ID = re.compile(r"[0-9a-f]{12}")


class NotificationPrivacyError(ValueError):
    pass


@dataclass(frozen=True)
class NotificationSchema:
    categories: frozenset                 # supplied by the AI/feature owner
    vehicle_states: frozenset             # supplied by the safety owner
    risk_levels: frozenset = frozenset({"LOW", "MODERATE", "HIGH", "CRITICAL"})
    confidence_levels: frozenset = frozenset({"LOW", "MEDIUM", "HIGH"})
    bucket_s: int = 10

    def __post_init__(self):
        if not self.categories or not self.vehicle_states:
            raise ValueError("categories and vehicle_states must be non-empty")
        if isinstance(self.bucket_s, bool) or not isinstance(self.bucket_s, int) or self.bucket_s < 1:
            raise ValueError("bucket_s must be a positive integer")


def _in(name: str, v, allowed) -> None:
    if not isinstance(v, str) or v not in allowed:
        raise NotificationPrivacyError(f"field {name!r} is outside its allowed vocabulary")


def validate_notification_payload(payload, schema: NotificationSchema) -> None:
    if not isinstance(payload, dict):
        raise NotificationPrivacyError("notification payload must be a dict")
    if any(not isinstance(k, str) for k in payload):
        raise NotificationPrivacyError("non-string field names are not allowed")
    if set(payload) != ALLOWED_KEYS:
        raise NotificationPrivacyError("payload must contain exactly the allowed fields")
    if not (isinstance(payload["event_id"], str) and _EVENT_ID.fullmatch(payload["event_id"])):
        raise NotificationPrivacyError("event_id must be 12 lowercase hex characters")
    _in("category", payload["category"], schema.categories)
    _in("risk_level", payload["risk_level"], schema.risk_levels)
    _in("confidence_level", payload["confidence_level"], schema.confidence_levels)
    _in("vehicle_state", payload["vehicle_state"], schema.vehicle_states)
    ts = payload["timestamp"]
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0 or ts % schema.bucket_s:
        raise NotificationPrivacyError("timestamp must be a non-negative integer on the coarse bucket")


class NotificationSender:
    def __init__(self, sender_id: str, key: bytes, schema: NotificationSchema, first_seq: int = 1):
        if not isinstance(sender_id, str) or not sender_id:
            raise ValueError("sender_id must be a non-empty string")
        self.sender_id, self.schema, self._seq = sender_id, schema, first_seq
        self._key = check_key(key, f"key for {sender_id!r}")

    def build(self, category: str, confidence: float, risk_level: str, t: float, vehicle_state: str,
              event_id: str | None = None) -> SecureMessage:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise NotificationPrivacyError("confidence must be a finite number")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise NotificationPrivacyError("t must be a finite non-negative number")
        b = self.schema.bucket_s
        coarse = int(t // b * b)
        payload = {"event_id": event_id or secrets.token_hex(6), "category": category, "risk_level": risk_level,
                   "confidence_level": "HIGH" if confidence >= 0.6 else "MEDIUM" if confidence >= 0.35 else "LOW",
                   "timestamp": coarse, "vehicle_state": vehicle_state}
        validate_notification_payload(payload, self.schema)            # fail BEFORE signing
        msg = sign_message(SecureMessage(self.sender_id, self._seq, float(coarse), NOTIFICATION_MSG_TYPE, payload),
                           self._key)                                  # envelope ts == coarse bucket start
        self._seq += 1
        return msg

    def __repr__(self) -> str:
        return f"NotificationSender(sender_id={self.sender_id!r}, next_seq={self._seq})"


def make_notification_validator(registry: dict, roles: dict, schema: NotificationSchema, slack_s: float = 2.0,
                                clock=None, audit=None) -> MessageValidator:
    """Dedicated validator for this channel: freshness window = bucket + slack (see module docstring)."""
    return MessageValidator(registry, roles, max_age_s=schema.bucket_s + slack_s, max_future_skew_s=slack_s,
                            clock=clock, audit=audit)


def accept_notification(validator: MessageValidator, msg: SecureMessage, now: float,
                        schema: NotificationSchema) -> ValidationResult:
    """Validator checks first (its replay state is committed only if everything passed there), then the
    privacy schema on the receiving side."""
    r = validator.validate(msg, now)
    if not r.ok:
        return r
    if msg.msg_type != NOTIFICATION_MSG_TYPE:
        return ValidationResult(False, "payload", "not a notification")
    try:
        validate_notification_payload(msg.payload, schema)
    except NotificationPrivacyError:
        if validator.audit is not None:
            validator.audit.record("NOTIFICATION_REJECTED", "payload", msg.sender_id, "payload violates privacy schema")
        return ValidationResult(False, "payload", "payload violates privacy schema")
    if msg.ts != float(msg.payload["timestamp"]):
        return ValidationResult(False, "payload", "envelope timestamp does not match coarse payload timestamp")
    return r
