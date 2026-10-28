"""Log-injection-safe, redacting front end for ``SecurityAuditLog`` (SIMULATION).

The existing ``SecurityAuditLog.record`` truncates strings but does not neutralise control characters
or CR/LF, so any caller that passes attacker-influenced text could forge log lines. ``SafeAudit`` is a
drop-in object (same ``record`` signature, so it can also be handed to ``MessageValidator(audit=...)``)
that applies, before delegating to the real log:

  * event names restricted to a closed vocabulary;
  * sender labels restricted to *registered* senders or fixed internal labels - anything else, e.g. a
    forged ``sender_id``, is logged as ``<unregistered>``;
  * every free-text field stripped of control characters / non-ASCII, redacted if it looks like a MAC tag,
    key or other long encoded blob, then truncated;
  * ``detail_code`` accepted only in the strict form ``name=value``.

It does not make the log tamper-proof or confidential; it prevents *this module's callers* from writing
attacker-controlled or secret-looking text into it.
"""
from __future__ import annotations

import re
from typing import Iterable

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_LONG_HEX = re.compile(r"[0-9a-fA-F]{24,}")
_LONG_B64 = re.compile(r"[A-Za-z0-9+/_\-]{40,}={0,2}")
_DETAIL = re.compile(r"[a-z_]{1,12}=-?[a-z0-9_]{1,20}")

INTERNAL_LABELS = frozenset({"<unregistered>", "<local>", "<internal>"})
EVENTS = frozenset({
    "MESSAGE_REJECTED", "COMMAND_REJECTED", "COMMAND_ACCEPTED", "SENDER_RESET", "PAYLOAD_REJECTED",
    "SENSOR_IMPLAUSIBLE", "BASELINE_REJECTED", "BASELINE_ACCEPTED", "NOTIFICATION_REJECTED",
})


def sanitize(value: object, maxlen: int = 120) -> str:
    """Neutralise control characters / non-ASCII, redact key- or tag-looking blobs, truncate."""
    s = value if isinstance(value, str) else str(value)
    s = _CONTROL.sub("?", s)
    s = s.encode("ascii", "replace").decode("ascii")
    s = _LONG_HEX.sub("[redacted]", s)
    s = _LONG_B64.sub("[redacted]", s)
    return s[:maxlen]


class SafeAudit:
    def __init__(self, sink, known_senders: Iterable[str] = ()):
        self.sink = sink
        self.known = frozenset(s for s in known_senders if isinstance(s, str))

    def with_senders(self, senders: Iterable[str]) -> "SafeAudit":
        return SafeAudit(self.sink, self.known | frozenset(senders))

    def record(self, event: str, stage: str, sender: str, reason: str, detail_code: str = ""):
        if self.sink is None:
            return None
        ev = event if isinstance(event, str) and event in EVENTS else "UNKNOWN_EVENT"
        snd = sender if isinstance(sender, str) and (sender in self.known or sender in INTERNAL_LABELS) \
            else "<unregistered>"
        det = detail_code if isinstance(detail_code, str) and _DETAIL.fullmatch(detail_code) else ""
        return self.sink.record(ev, sanitize(stage, 24), snd, sanitize(reason, 120), det)


def as_safe(audit, known_senders: Iterable[str] = ()) -> SafeAudit | None:
    """Wrap any audit sink (or pass an existing SafeAudit through, extending its sender set)."""
    if audit is None:
        return None
    if isinstance(audit, SafeAudit):
        return audit.with_senders(known_senders)
    return SafeAudit(audit, known_senders)
