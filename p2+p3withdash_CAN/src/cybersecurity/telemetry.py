"""Sensor-side secure telemetry construction (TB1, SIMULATION - no BLE, no radio).

Reuses the existing ``SecureMessage`` / ``sign_message`` primitives (HMAC-SHA256); no cryptography is
implemented here. The canonical serialisation is whatever ``secure_message._canon`` produces, so messages
built here validate with the existing ``MessageValidator`` unchanged.

Payload shape (consumed by ``SensorTrustManager``)::

    {"obs": {modality: {"q": quality in [0,1], "f": {feature: float | None}} | None}}

``None`` for a modality, or for a feature value, means "not measured / unavailable". It is NEVER encoded
as 0 or as an extreme value, so absence can not be mistaken for a physiological reading.

Hardening history (Person 3, cybersecurity foundation review):
  * T1  ``SensorEndpoint.emit()`` now deep-copies ``obs`` into an independent snapshot BEFORE
        constructing/signing the ``SecureMessage``. Previously the signed message's payload held a
        reference to the caller's own ``obs`` dict; a mutation of that dict by the caller (or any
        other code holding the same reference) after ``emit()`` returned would silently change
        ``message.payload`` without touching ``message.tag``, producing an internally-inconsistent
        message that a receiving ``MessageValidator`` would (correctly) reject at the integrity
        stage. This was an availability/integrity risk, not an authentication bypass - but it meant
        a purely accidental, non-malicious mutation on the sender side could manifest as a mysterious
        "MAC mismatch" far away at the receiver, with no exploitation required.
  * T2  Sequence-number allocation and message construction/signing are now performed under a
        per-endpoint lock, so concurrent ``emit()`` calls on the SAME ``SensorEndpoint`` instance can
        never be issued the same sequence number (which a receiving ``MessageValidator`` would
        otherwise accept exactly once and reject every other concurrent caller as a replay - not a
        security hole by itself, since forgeries still can't pass authentication, but a correctness
        bug that would silently drop legitimate concurrent telemetry).
  * T3  ``SensorEndpoint`` now enforces ``secure_message.MAX_SEQ`` as a hard upper bound, both at
        construction (``first_seq`` must be in ``[0, MAX_SEQ]``) and at ``emit()`` time (raises
        ``SequenceLimitError`` instead of silently producing a sequence number a receiver would
        reject as malformed under the schema's own ``0 <= seq <= MAX_SEQ`` rule).
  * T4  ``make_obs()`` now rejects a modality that is simultaneously declared available (a non-null
        entry in ``qualities``) and unavailable (present in ``unavailable``), instead of silently
        discarding the available measurement by overwriting it with ``None``. This is the same
        "reject contradictory input" principle already applied throughout ``secure_message.py``.
  * T5  ``make_obs()`` now rejects ``unavailable`` being passed as a bare ``str``/``bytes`` (which is
        technically ``Iterable`` and would otherwise be silently iterated character-by-character),
        mirroring the equivalent guard already added to ``secure_message._validate_roles``.

Concurrency note (T2): the per-endpoint lock is held for the full duration of sequence-number
allocation AND signing, which serialises ``emit()`` calls on a single endpoint instance. This mirrors
the design already used in ``MessageValidator`` (a single lock around a small, no-I/O critical
section) and is an acceptable trade-off for this simulation's throughput. It does NOT serialise
calls across different ``SensorEndpoint`` instances - each endpoint has its own independent lock.

Residual limitations (unchanged / out of scope here):
  * ``_clean()`` still accepts any value ``float()`` can coerce, including numeric strings
    (``_clean("85")`` -> ``85.0``); if your acquisition layer never hands this module strings,
    that's fine, but it is not currently enforced. See the open review item in the project notes.
  * The sequence counter is in-memory only; see ``SensorEndpoint``'s own docstring for the
    persistence/resynchronisation caveat, which is unchanged by this pass.
  * ``keys.py``'s ``check_key()`` contract (what it validates, what it raises) is assumed, not
    re-verified here, since that file was not available for review at the time of this change.
"""
from __future__ import annotations

import math
import threading
from copy import deepcopy
from typing import Iterable, Mapping

from .keys import check_key
from .secure_message import MAX_SEQ, SecureMessage, sign_message


class SequenceLimitError(RuntimeError):
    """Raised when a ``SensorEndpoint`` has exhausted its available sequence-number space.

    This is NOT a malformed-input error (hence not a ``ValueError``/``SecureMessageError``): it is a
    lifecycle/exhaustion condition. Recovering from it requires retiring this sender identity (new
    sender_id + new key, issued through whatever provisioning process the deployment uses) - it must
    NOT be "fixed" by silently wrapping the counter back to 0, which would reopen replay protection
    for an attacker who recorded the sender's earlier traffic.
    """


def _clean(v):
    """Feature value -> float, or None if missing/non-finite (JSON has no NaN). Booleans are rejected."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("feature values must be numbers, not booleans")
    f = float(v)
    return f if math.isfinite(f) else None


def make_obs(feature_values: Mapping, qualities: Mapping, features_by_modality: Mapping,
             unavailable: Iterable[str] = ()) -> dict:
    """Build the ``obs`` dict for the modalities in ``qualities`` (+ explicit ``unavailable`` ones).

    ``features_by_modality`` is supplied by the caller (owned by the feature/AI side), so this module
    carries no copy of Person 1's feature specification. A quality of ``None`` marks the modality
    unavailable.

    Raises ``ValueError`` if a modality is declared BOTH available (a non-null entry in ``qualities``)
    AND unavailable (present in ``unavailable``) - this is treated as contradictory caller input and
    rejected outright (T4), rather than silently discarding the available measurement by letting the
    ``unavailable`` pass overwrite it with ``None``.
    """
    if isinstance(unavailable, (str, bytes)):
        # A bare string is technically Iterable[str] (of characters) - without this guard,
        # unavailable="phys" would silently be treated as unavailable=['p','h','y','s'] (T5).
        raise TypeError("unavailable must be an iterable of modality-name strings, not a single str/bytes")

    # Materialised once: `unavailable` may be a one-shot iterator/generator, and it needs to be
    # inspected twice below (the conflict check, then the dedicated per-modality loop).
    unavailable_list = list(unavailable)
    for m in unavailable_list:
        if not isinstance(m, str) or not m:
            raise ValueError(f"unavailable modality name must be a non-empty string, got {m!r}")

    unavailable_set = set(unavailable_list)
    conflicts = sorted(m for m in qualities if m in unavailable_set and qualities[m] is not None)
    if conflicts:
        raise ValueError(
            f"modality(ies) {conflicts} declared both available (non-null quality in `qualities`) "
            "and unavailable (present in `unavailable`) - contradictory input is rejected rather "
            "than silently discarding the available measurement"
        )

    out: dict = {}
    for m, q in qualities.items():
        if m not in features_by_modality:
            raise KeyError(f"unknown modality {m!r}")
        if q is None:
            out[m] = None
            continue
        if (isinstance(q, bool) or not isinstance(q, (int, float))
                or not math.isfinite(float(q)) or not 0.0 <= float(q) <= 1.0):
            raise ValueError(f"quality for {m!r} must be a finite number in [0, 1]")
        out[m] = {"q": float(q), "f": {f: _clean(feature_values.get(f)) for f in features_by_modality[m]}}
    for m in unavailable_list:
        if m not in features_by_modality:
            raise KeyError(f"unknown modality {m!r}")
        out[m] = None
    return out


class SensorEndpoint:
    """Signs telemetry with this endpoint's own key and a strictly increasing sequence number.

    Per-sender identity: one endpoint = one ``sender_id`` = one key. The role/permissions of that
    identity are decided by the *receiver's* registry, never by the sender.

    Thread safety (T2): sequence-number allocation and message signing are performed atomically
    under a per-instance lock, so concurrent ``emit()`` calls on the SAME endpoint can never be
    issued duplicate sequence numbers. The lock is per-endpoint-instance: concurrent ``emit()`` calls
    on two DIFFERENT ``SensorEndpoint`` instances never contend with each other.

    Sequence space (T3): ``first_seq`` and every value ``emit()`` issues are bounded to
    ``[0, secure_message.MAX_SEQ]``, matching the receiver's own schema limit exactly. Once
    exhausted, ``emit()`` raises ``SequenceLimitError`` instead of producing a sequence number a
    validator would reject as malformed.

    Persistence: the sequence counter lives in memory. After a reboot the endpoint must resume above
    the last accepted value (``first_seq``) or the receiver will - correctly - reject it as replay
    until an authorised resynchronisation (``MessageValidator.reset_sender``). Where that value is
    stored is not designed here (see docs/security_architecture_p3.md).
    """

    def __init__(self, sender_id: str, key: bytes, msg_type: str = "SENSOR_TELEMETRY", first_seq: int = 1):
        if not isinstance(sender_id, str) or not sender_id:
            raise ValueError("sender_id must be a non-empty string")
        if not isinstance(msg_type, str) or not msg_type:
            raise ValueError("msg_type must be a non-empty string")
        if isinstance(first_seq, bool) or not isinstance(first_seq, int) or not (0 <= first_seq <= MAX_SEQ):
            raise ValueError(f"first_seq must be a non-negative integer no greater than MAX_SEQ ({MAX_SEQ})")
        self.sender_id, self.msg_type = sender_id, msg_type
        self._key = check_key(key, f"key for {sender_id!r}")
        self._seq = first_seq
        self._lock = threading.Lock()   # T2: guards _seq allocation + the signing that uses it

    @property
    def next_seq(self) -> int:
        """Informational only: a plain, unsynchronised read of the current counter value. Not
        intended for coordinating external logic (another thread may allocate a seq the instant
        after this returns) - only ``emit()`` itself performs the atomic allocate-and-sign."""
        return self._seq

    def emit(self, ts: float, obs: dict) -> SecureMessage:
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError("ts must be a finite number")
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dict")

        # T1: independent snapshot taken BEFORE signing. The dict (and any nested dict/list inside
        # it) that the caller continues to hold a reference to after emit() returns must never be
        # able to retroactively change what was actually signed. Taken outside the lock: it only
        # touches the caller's local `obs`, not any endpoint state, so there's nothing to serialise.
        obs_snapshot = deepcopy(obs)

        with self._lock:
            if self._seq > MAX_SEQ:
                raise SequenceLimitError(
                    f"sender {self.sender_id!r} has exhausted its sequence-number space "
                    f"(MAX_SEQ={MAX_SEQ}); this identity must be retired and a new sender_id/key "
                    "issued - do not reset/wrap the counter, which would reopen replay protection "
                    "for anyone who recorded this sender's earlier traffic"
                )
            seq = self._seq
            # raises SecureMessageError -> seq NOT consumed (exception propagates before the
            # `self._seq = seq + 1` line below ever runs; the `with` block still releases the lock)
            msg = sign_message(
                SecureMessage(self.sender_id, seq, float(ts), self.msg_type, {"obs": obs_snapshot}),
                self._key,
            )
            self._seq = seq + 1
        return msg

    def __repr__(self) -> str:                             # never expose key material
        return f"SensorEndpoint(sender_id={self.sender_id!r}, msg_type={self.msg_type!r}, next_seq={self._seq})"