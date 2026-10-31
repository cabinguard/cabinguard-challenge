"""Sensor trust management: the boundary between the secure message layer and the AI/fusion pipeline
(SIMULATION).

Four things are kept deliberately separate; nothing in this module converts one into another:

  1. Authentication / integrity   - did a registered sender produce these exact bytes, fresh, once?
                                    (``MessageValidator``; outcome: accepted or rejected)
  2. Physical plausibility        - is each value inside sanity limits derived from the feature's definition?
                                    A valid HMAC says nothing about whether the sensor is accurate, and bounds
                                    can NOT detect forged *in-range* values from a compromised key.
  3. Signal quality               - the sensor's own declared quality (and, downstream, the fusion's gating)
  4. AI confidence / safety       - not touched here. A trust failure never becomes "abnormal evidence":
                                    untrusted modalities become *unavailable* to the consumer, which can only
                                    reduce evidence, never create an emergency signal.

Observation semantics
  * A packet rejected by the validator is never parsed and never stored: it cannot overwrite what was
    previously accepted. It only increments counters. An integrity hold is started on the *claimed*
    source only for stages in ``INTEGRITY_HOLD_STAGES`` (schema / authentication / integrity) - those
    cast doubt on the whole channel, so the hold applies to EVERY modality that source reports.
    Replay is counted but does NOT start a hold: a previously-seen sequence is "not new", not a
    broken channel. Holding on replay would (1) let a sniffed packet DoS every modality of that
    source without a key, and (2) hide a good observation after out-of-order delivery. Store
    protection for that case is ``_last_stored_seq`` (and the validator's own replay window).
  * An *authenticated* packet whose payload is malformed or implausible IS recorded as the latest state of
    its modalities (status CORRUPTED / IMPLAUSIBLE). Rationale: a sender reporting impossible values must
    not be masked by an older good reading. It is never exposed as a usable observation.
  * MODALITY-LEVEL ISOLATION (MT1): within a single authenticated, correctly-shaped packet, a malformed or
    implausible value in ONE modality does NOT affect any OTHER modality reported in the same packet. An
    implausible PPG heart rate does not invalidate an independently plausible IMU reading sent alongside it.
    Packet-wide rejection is reserved for: (a) the sender/role is not authorised at all, or (b) the payload's
    own top-level shape is wrong (not a dict, or references a modality the sender isn't permitted to report) -
    both of which are genuine packet-level problems with no sensible per-modality attribution. Authentication
    and integrity failures are caught earlier, in ``ingest()``, before any modality is ever parsed, and
    likewise remain packet (sender)-wide.
  * Store ordering: only a strictly newer ``msg.seq`` than the last sequence committed for that sender
    may mutate ``_obs``. An older (or equal) validated packet is counted as ``ordering`` and left out of
    sensor state, so concurrent ``ingest()`` cannot let a stale writer win the lock.
  * The only way to read a measurement is ``usable_observation(modality, now)``, which returns it only if
    the modality is currently TRUSTED or DEGRADED. There is deliberately no accessor that returns stale,
    implausible, corrupted or unavailable data.

Interface to Person 1 (feature owner): a ``FeatureCatalog`` (modality -> feature names, feature -> bounds).
No import of the project's ``schemas`` module is needed; ``FeatureCatalog.from_feature_specs`` adapts it.

Hardening history (Person 3, cybersecurity foundation review):
  * ST1 - ``ingest()`` passes the ``MessageValidator``'s own authenticated payload snapshot
        (``ValidationResult.payload``, produced atomically with MAC verification - see
        ``secure_message.py``'s E2) into ``_accept()``, instead of letting ``_accept()``
        independently re-read ``msg.payload`` after ``validate()`` had already returned. Closes a
        TOCTOU window where a concurrent mutation of the shared ``msg`` object between validation
        and storage could be silently treated as authenticated. Preserved, unmodified, by MT1/TH1.
  * MT1 - removed the cross-modality "taint" logic that previously marked every OTHER modality in a
        packet as untrusted (``R_COTAINT``) whenever any one modality failed plausibility or parsing.
        Each modality's stored state now depends only on its own parse outcome. See "Observation
        semantics" above for exactly which failures remain packet-wide vs. modality-isolated.
        ``R_COTAINT`` is kept defined (not removed) for backward compatibility with any external
        code that may reference the constant, but it is no longer produced by this module.
  * TH1 - all shared mutable state (``_obs``, ``_sec_reject_t``, ``_global_reject_t``, ``_pending``,
        ``stats``, ``_last_stored_seq``) is now protected by a single internal lock (``self._lock``),
        held only around the actual state mutation/read in ``ingest()``/``_accept()``/``assess()``/
        ``usable_observation()``. ``self.v.validate(...)`` and ``self.audit.record(...)`` are NEVER
        called while holding this lock, specifically to avoid a lock-ordering deadlock with the
        validator's own internal lock (``secure_message.py`` D10/D12) or a slow/blocking audit sink.
        Immutable, construction-time-only state (``self.sources``, ``self.catalog``, ``self._src_of``,
        ``self.telemetry_types``, the numeric thresholds) is read without the lock, since nothing
        ever mutates it after ``__init__``.
  * CT1 - ``FeatureCatalog.__post_init__`` now builds independent copies of the caller's
        ``features_by_modality``/``bounds`` mappings (feature lists -> tuples, bound pairs -> tuples)
        and freezes both with ``MappingProxyType`` via ``object.__setattr__`` (required because the
        dataclass itself is ``frozen=True``, which only blocks attribute *reassignment*, not mutation
        of a mutable object already assigned to a field). Mutating the caller's original dict/list
        after constructing a ``FeatureCatalog`` - or attempting to write through
        ``catalog.features_by_modality``/``catalog.bounds`` directly - now has no effect / raises
        ``TypeError``.
  * RH1 - integrity hold is no longer started for every ``SECURITY_STAGES`` rejection. Replay is
        counted and still never stored, but it is excluded from ``INTEGRITY_HOLD_STAGES``. A
        previously-seen sequence is not evidence the channel is compromised; treating it as such
        made out-of-order delivery (and a sniffed old packet) look like a source-wide integrity
        failure and hid a good observation behind ``CORRUPTED``.

Residual limitations, NOT addressed by this pass (unchanged from before):
  * ``self.stats`` remains a plain ``dict`` PUBLIC attribute (unchanged interface). Writes to it are
    now protected by ``self._lock``, so concurrent increments are never lost. However, code OUTSIDE
    this class that iterates over ``manager.stats`` directly (e.g. ``for k, v in manager.stats.items()``)
    while ingestion is concurrently happening on another thread could still hit Python's usual
    "dictionary changed size during iteration" hazard, because reading it is not itself serialised
    against the lock. To read it safely under concurrency, take a snapshot via ``dict(manager.stats)``
    at a quiescent point, or rely on ``TrustSnapshot.rejected`` (returned by ``assess()``), which IS a
    safe, already-copied per-interval snapshot.
  * The in-memory, single-process nature of ``_obs``/``_sec_reject_t`` is unchanged: no persistence,
    no cross-process sharing.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .audit_safe import as_safe
from .secure_message import MessageValidator, SecureMessage, ValidationResult

DEFAULT_TELEMETRY_TYPES = ("WEARABLE_TELEMETRY", "SENSOR_TELEMETRY")
SECURITY_STAGES = ("schema", "authentication", "integrity", "replay")
# Replay is a "not new" verdict, not channel compromise. Holding on replay would
# (1) let a sniffed packet DoS every modality of that source and (2) hide a good
# observation after out-of-order delivery. Store protection for that case is
# `_last_stored_seq` (and the validator's own replay window).
INTEGRITY_HOLD_STAGES = frozenset({"schema", "authentication", "integrity"})

# Fixed-vocabulary reasons: no attacker-controlled text ever reaches a reason string or the audit log.
R_UNAVAILABLE = "sensor reported modality unavailable"
R_MALFORMED = "malformed authenticated payload"
R_BAD_QUALITY = "quality outside [0,1]"
R_BAD_FEATURE = "unexpected feature name"
R_NONNUMERIC = "non-numeric feature value"
R_BAD_MODALITY = "modality not permitted for this source"
R_IMPLAUSIBLE = "value outside physical bounds"
R_COTAINT = "co-reported modality failed validation"   # retained for compatibility; no longer emitted (MT1)


class TrustStatus(str, Enum):
    TRUSTED = "TRUSTED"                  # authentic, fresh, plausible, quality >= q_ok
    DEGRADED = "DEGRADED"                # authentic, fresh, plausible, q_min <= quality < q_ok
    LOW_QUALITY = "LOW_QUALITY"          # authentic, fresh, plausible, quality < q_min (not usable)
    MISSING = "MISSING"                  # nothing authentic received, or sensor declared it unavailable
    STALE = "STALE"                      # last authentic observation older than the maximum data age
    CORRUPTED = "CORRUPTED"              # security rejection for this source recently / malformed payload
    UNAUTHENTICATED = "UNAUTHENTICATED"  # reserved: only candidate data came from an unknown sender
    IMPLAUSIBLE = "IMPLAUSIBLE"          # authentic but outside physical sanity bounds


USABLE = frozenset({TrustStatus.TRUSTED, TrustStatus.DEGRADED})
BLOCKED = frozenset({TrustStatus.MISSING, TrustStatus.STALE, TrustStatus.CORRUPTED,
                     TrustStatus.UNAUTHENTICATED, TrustStatus.IMPLAUSIBLE})


@dataclass(frozen=True)
class FeatureCatalog:
    """Minimal contract with the feature owner: which features exist per modality and their sanity bounds.

    CT1: the mappings passed to the constructor are copied and frozen (``MappingProxyType``) during
    ``__post_init__``; mutating the caller's original ``features_by_modality``/``bounds`` objects (or
    any list/tuple nested inside them) after construction has no effect on this instance.
    """
    features_by_modality: Mapping[str, tuple]
    bounds: Mapping[str, tuple]

    def __post_init__(self):
        if not isinstance(self.features_by_modality, Mapping):
            raise TypeError("features_by_modality must be a mapping of modality -> iterable of feature names")
        if not isinstance(self.bounds, Mapping):
            raise TypeError("bounds must be a mapping of feature -> (lo, hi)")

        # CT1: copy BEFORE validating/freezing, so the object this instance ends up holding is
        # already fully independent of whatever the caller passed in.
        frozen_fbm = {}
        for m, feats in self.features_by_modality.items():
            if isinstance(feats, (str, bytes)) or not isinstance(feats, Iterable):
                raise TypeError(
                    f"features_by_modality[{m!r}] must be an iterable of feature-name strings, "
                    f"not {type(feats).__name__}"
                )
            frozen_fbm[m] = tuple(feats)

        frozen_bounds = {f: tuple(b) for f, b in self.bounds.items()}

        seen = set()
        for m, feats in frozen_fbm.items():
            for f in feats:
                if f in seen:
                    raise ValueError(f"feature {f!r} listed under more than one modality")
                seen.add(f)
                if f not in frozen_bounds:
                    raise ValueError(f"no plausibility bounds for feature {f!r}")
        for f, bound in frozen_bounds.items():
            if len(bound) != 2:
                raise ValueError(f"bounds[{f!r}] must be a 2-tuple (lo, hi)")
            lo, hi = bound
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"invalid bounds for {f!r}")

        object.__setattr__(self, "features_by_modality", MappingProxyType(frozen_fbm))
        object.__setattr__(self, "bounds", MappingProxyType(frozen_bounds))

    @classmethod
    def from_feature_specs(cls, specs: Mapping, bounds: Mapping) -> "FeatureCatalog":
        """Adapter for ``schemas.FEATURE_SPECS`` ({feature: (modality, kind)}); read-only use."""
        by_mod: dict = {}
        for f, (m, _kind) in specs.items():
            by_mod.setdefault(m, []).append(f)
        return cls({m: tuple(v) for m, v in by_mod.items()}, dict(bounds))

    @property
    def modalities(self) -> tuple:
        return tuple(self.features_by_modality)


@dataclass(frozen=True)
class ModalityTrust:
    modality: str
    status: TrustStatus
    quality: float | None
    age_s: float | None
    reason: str

    @property
    def usable(self) -> bool:
        """Trusted enough AND good enough quality to contribute evidence."""
        return self.status in USABLE

    @property
    def secure(self) -> bool:
        """Passes the security/availability layer (quality gating is a separate, downstream concern)."""
        return self.status not in BLOCKED


@dataclass(frozen=True)
class TrustedObservation:
    """A measurement that was usable at the moment it was requested. Immutable; NaN = feature not measured."""
    modality: str
    features: Mapping[str, float]
    quality: float
    age_s: float
    status: TrustStatus


@dataclass
class TrustSnapshot:
    t: float
    modalities: dict
    integrity_ok: bool                                  # no security rejection within the integrity hold
    data_fresh: bool                                    # no held observation has gone STALE
    rejected: dict = field(default_factory=dict)        # stage -> count since the previous snapshot

    def status(self, modality: str) -> TrustStatus:
        return self.modalities[modality].status

    def usable_modalities(self) -> list:
        return [m for m, v in self.modalities.items() if v.usable]

    def blocked_modalities(self) -> list:
        return [m for m, v in self.modalities.items() if not v.secure]


@dataclass(frozen=True)
class _Obs:
    ts: float
    quality: float
    features: Mapping
    plausible: bool
    reason: str = ""
    malformed: bool = False
    unavailable: bool = False


class SensorTrustManager:
    """One instance per session. Consumes signed telemetry, tracks freshness/availability per modality.

    Thread safety (TH1): ``ingest()``, ``assess()``, and ``usable_observation()`` are all safe to call
    concurrently from multiple threads (including concurrently with each other) on the same instance.
    A single internal lock protects ``_obs``/``_sec_reject_t``/``_global_reject_t``/``_pending``/``stats``/
    ``_last_stored_seq``; it is never held while calling out to the validator or the audit sink
    (see module docstring TH1).
    """

    def __init__(self, validator: MessageValidator, catalog: FeatureCatalog, sources: Mapping,
                 max_data_age_s: float, q_min: float, q_ok: float, integrity_hold_s: float = 1.0,
                 telemetry_types: Iterable[str] = DEFAULT_TELEMETRY_TYPES, audit=None):
        if not (max_data_age_s > 0 and 0.0 <= q_min <= q_ok <= 1.0 and integrity_hold_s >= 0):
            raise ValueError("require max_data_age_s > 0, 0 <= q_min <= q_ok <= 1, integrity_hold_s >= 0")
        self._src_of: dict = {}
        for sid, mods in sources.items():
            for m in mods:
                if m not in catalog.features_by_modality:
                    raise ValueError(f"source {sid!r} references unknown modality {m!r}")
                if m in self._src_of:
                    raise ValueError(f"modality {m!r} is claimed by more than one source")
                self._src_of[m] = sid
        self.v, self.catalog = validator, catalog
        self.sources = {k: tuple(v) for k, v in sources.items()}
        self.max_age, self.q_min, self.q_ok, self.hold = float(max_data_age_s), q_min, q_ok, float(integrity_hold_s)
        self.telemetry_types = frozenset(telemetry_types)
        self.audit = as_safe(audit if audit is not None else validator.audit, validator.registry.keys())
        self._obs: dict = {}
        self._sec_reject_t: dict = {}
        self._global_reject_t = -math.inf
        self._pending: dict = {}
        self.stats: dict = {}
        # Highest sequence number committed to sensor state, per sender.
        self._last_stored_seq: dict[str, int] = {}
        # TH1: guards the mutable-state containers above ONLY
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg, validator: MessageValidator, catalog: FeatureCatalog, max_data_age_s: float,
                    q_min: float, q_ok: float, audit=None) -> "SensorTrustManager":
        """``cfg``: a ``security_config.SecurityConfig``. Thresholds q_min/q_ok/max_data_age_s come from the
        project's quality/safety configuration (owned by Persons 1-2), not from security configuration."""
        return cls(validator, catalog, cfg.sources, max_data_age_s, q_min, q_ok, cfg.integrity_hold_s,
                   cfg.telemetry_types, audit)

    # ------------------------------------------------------------------ bookkeeping
    def _count(self, stage: str) -> None:
        """MUST be called only while holding ``self._lock``."""
        self._pending[stage] = self._pending.get(stage, 0) + 1
        self.stats[stage] = self.stats.get(stage, 0) + 1

    def _commit_if_newer(self, sender_id: str, seq: int) -> bool:
        """MUST be called only while holding ``self._lock``.

        Returns True iff ``seq`` is strictly newer than the last sequence committed for this
        sender (and records it). Returns False and counts ``ordering`` otherwise; caller must
        not mutate ``_obs``.
        """
        last_seq = self._last_stored_seq.get(sender_id, -1)
        if seq <= last_seq:
            self._count("ordering")
            return False
        self._last_stored_seq[sender_id] = seq
        return True

    # ------------------------------------------------------------------ ingestion
    def ingest(self, msg: SecureMessage, now: float) -> ValidationResult:
        """Validate, then store. The only entry point that can change held observations."""
        # TH1: deliberately OUTSIDE self._lock. The validator manages its own internal locking
        # (secure_message.py D10/D12); calling it while holding our lock would risk a lock-ordering
        # deadlock if either side ever gained a call-back into the other.
        r = self.v.validate(msg, now)
        if not r.ok:
            with self._lock:
                self._count(r.stage)
                if r.stage in INTEGRITY_HOLD_STAGES:
                    self._global_reject_t = now
                    sid = getattr(msg, "sender_id", None)
                    if isinstance(sid, str) and sid in self.sources:
                        self._sec_reject_t[sid] = now
            return r
        # ST1 (preserved): use the validator's own atomically-authenticated snapshot, never
        # msg.payload - see module docstring ST1 and secure_message.py's E2.
        return self._accept(msg, r.payload)

    def _accept(self, msg: SecureMessage, payload: Mapping) -> ValidationResult:
        mods = self.sources.get(msg.sender_id)                      # immutable after __init__: no lock needed
        if mods is None or msg.msg_type not in self.telemetry_types:
            with self._lock:
                self._count("authorization")
            if self.audit:                                          # TH1: audit call OUTSIDE the lock
                self.audit.record("MESSAGE_REJECTED", "authorization", msg.sender_id,
                                  "sender is not a registered sensor source")
            return ValidationResult(False, "authorization", "sender is not a registered sensor source")

        obs = payload.get("obs")                                    # ST1: payload, never msg.payload
        if not isinstance(obs, dict) or any(k not in mods for k in obs):
            # Genuine packet-level structural failure: the payload's own shape is wrong, or it
            # references a modality this sender isn't permitted to report. No per-modality
            # attribution is possible here, so this remains packet-wide (unlike MT1 below).
            reason = R_MALFORMED if not isinstance(obs, dict) else R_BAD_MODALITY
            with self._lock:
                if not self._commit_if_newer(msg.sender_id, msg.seq):
                    return ValidationResult(False, "ordering", "older validated message")
                for m in mods:
                    self._obs[m] = _Obs(
                        msg.ts,
                        0.0,
                        MappingProxyType({}),
                        False,
                        reason,
                        malformed=True,
                    )
                self._count("payload")
            if self.audit:
                self.audit.record("PAYLOAD_REJECTED", "payload", msg.sender_id, reason, f"seq={msg.seq}")
            return ValidationResult(False, "payload", reason)

        # MT1: parse each modality independently. _parse() only reads self.catalog/self.q_min
        # (immutable after __init__), so this is deliberately done OUTSIDE the lock to keep the
        # critical section as short as possible.
        parsed = {m: (_Obs(msg.ts, 0.0, MappingProxyType({}), True, R_UNAVAILABLE, unavailable=True)
                      if obs.get(m) is None else self._parse(m, obs[m], msg.ts)) for m in mods}

        with self._lock:
            if not self._commit_if_newer(msg.sender_id, msg.seq):
                return ValidationResult(False, "ordering", "older validated message")
            self._obs.update(parsed)   # single atomic batch update: no other thread can observe a
                                        # half-updated state for this packet's modalities
            bad = {m: o for m, o in parsed.items() if o.malformed or not o.plausible}
            if bad:
                stage = "payload" if any(o.malformed for o in bad.values()) else "plausibility"
                reason = next(iter(bad.values())).reason
                self._count(stage)
            else:
                stage = reason = None

        if stage is None:
            return ValidationResult(True, "accepted", "stored")
        if self.audit:
            self.audit.record("PAYLOAD_REJECTED" if stage == "payload" else "SENSOR_IMPLAUSIBLE", stage,
                              msg.sender_id, reason, f"seq={msg.seq}")
        return ValidationResult(False, stage, reason)

    def _parse(self, mod: str, entry, ts: float) -> _Obs:
        def mal(why):
            return _Obs(ts, 0.0, MappingProxyType({}), False, why, malformed=True)
        if not isinstance(entry, dict) or set(entry) - {"q", "f"} or not isinstance(entry.get("f"), dict):
            return mal(R_MALFORMED)
        q = entry.get("q")
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0.0 <= q <= 1.0:
            return mal(R_BAD_QUALITY)
        allowed = self.catalog.features_by_modality[mod]
        feats = {f: math.nan for f in allowed}
        plausible, why = True, ""
        for f, v in entry["f"].items():
            if f not in allowed:
                return mal(R_BAD_FEATURE)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return mal(R_NONNUMERIC)
            lo, hi = self.catalog.bounds[f]
            # Out-of-range values from a modality that itself declares unusable quality (q < q_min) are sensor
            # noise and stay with quality gating; plausibility targets values presented with credible quality.
            if not lo <= v <= hi and q >= self.q_min:
                plausible, why = False, R_IMPLAUSIBLE
            feats[f] = float(v)
        return _Obs(ts, float(q), MappingProxyType(feats), plausible, why)

    # ------------------------------------------------------------------ assessment
    def _modality_trust(self, m: str, now: float) -> ModalityTrust:
        """Reads ``self._obs``/``self._sec_reject_t``. Caller MUST already hold ``self._lock``."""
        o = self._obs.get(m)
        if o is None:
            return ModalityTrust(m, TrustStatus.MISSING, None, None, "no authentic observation received")
        age = max(0.0, now - o.ts)
        held = self._sec_reject_t.get(self._src_of.get(m, ""), -math.inf)
        if age > self.max_age:
            st, why = TrustStatus.STALE, "no fresh authentic observation"
        elif held > now - self.hold:
            st, why = TrustStatus.CORRUPTED, "security rejection for this source in the current interval"
        elif o.unavailable:
            st, why = TrustStatus.MISSING, o.reason
        elif o.malformed:
            st, why = TrustStatus.CORRUPTED, o.reason
        elif not o.plausible:
            st, why = TrustStatus.IMPLAUSIBLE, o.reason
        elif o.quality < self.q_min:
            st, why = TrustStatus.LOW_QUALITY, "quality below q_min"
        elif o.quality < self.q_ok:
            st, why = TrustStatus.DEGRADED, "quality below q_ok"
        else:
            st, why = TrustStatus.TRUSTED, "authentic, fresh, plausible, good quality"
        return ModalityTrust(m, st, o.quality, age, why)

    @staticmethod
    def _check_now(now: float) -> None:
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("now must be a finite number")

    def assess(self, now: float) -> TrustSnapshot:
        self._check_now(now)
        with self._lock:
            out = {m: self._modality_trust(m, now) for m in self.catalog.modalities}
            stale_any = any(v.status is TrustStatus.STALE for v in out.values())
            integrity_ok = not (self._global_reject_t > now - self.hold)
            pending = dict(self._pending)
            self._pending = {}
        return TrustSnapshot(now, out, integrity_ok, not stale_any, pending)

    def usable_observation(self, modality: str, now: float) -> TrustedObservation | None:
        """The ONLY way to read a measurement: returned iff the modality is TRUSTED/DEGRADED right now."""
        self._check_now(now)
        if modality not in self.catalog.features_by_modality:
            raise KeyError(f"unknown modality {modality!r}")
        with self._lock:
            t = self._modality_trust(modality, now)
            if not t.usable:
                return None
            o = self._obs[modality]
            return TrustedObservation(modality, o.features, o.quality, t.age_s, t.status)