"""Attack-simulation tests for CABINGUARD-TWIN's telemetry security boundary.

Builds each attack directly on confirmed, already-tested primitives (SecureMessage, sign_message,
MessageValidator, SensorTrustManager) rather than through telemetry.py's SensorEndpoint, whose
internal implementation (sequence allocation, locking) was not available for review in this
session. This keeps every assertion traceable to source actually inspected, per this task's
"do not invent APIs" constraint.

Four attacks, matching Step 8's table:
  1. Sensor identity spoofing   - an unregistered sender_id attempts to inject telemetry.
  2. Telemetry tampering        - an authenticated message's payload is altered after signing.
  3. Replay attack              - a previously accepted, authentic message is resubmitted.
  4. Stale telemetry            - an authentic message's timestamp is outside the freshness window.

For each: a legitimate baseline message is shown to be accepted and usable, the attack message
is shown to be rejected at the correct stage with a specific reason, and the attack's poisoned/
replayed/stale value is shown to never surface via usable_observation() - the only read path to
the Digital Twin / fusion boundary.

Fixtures intentionally mirror test_sensor_trust.py's real, passing fixtures (same KEY, same
catalog, same sources/validator construction) to avoid introducing a parallel, unverified setup.
Each test function gets its own fresh manager/validator (ordinary pytest function-scoped
fixtures - no shared mutable state between tests).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.cybersecurity.secure_message import MessageValidator, SecureMessage, sign_message
from src.cybersecurity.sensor_trust import FeatureCatalog, SensorTrustManager

KEY = b"W" * 32
ATTACKER_KEY = b"X" * 32
SENDER = "wearable-01"
MSG_TYPE = "SENSOR_TELEMETRY"


@pytest.fixture
def catalog():
    return FeatureCatalog(
        features_by_modality={"ppg": ("hr", "hrv"), "imu": ("accel",)},
        bounds={"hr": (30.0, 220.0), "hrv": (0.0, 300.0), "accel": (-20.0, 20.0)},
    )


@pytest.fixture
def sources():
    return {SENDER: ("ppg", "imu")}


@pytest.fixture
def validator(sources):
    registry = {SENDER: {"key": KEY, "role": "wearable"}}
    roles = {"wearable": {MSG_TYPE}}
    return MessageValidator(registry, roles, max_age_s=30.0, clock=lambda: 100.0)


@pytest.fixture
def manager(validator, catalog, sources):
    return SensorTrustManager(
        validator, catalog, sources, max_data_age_s=30.0,
        q_min=0.3, q_ok=0.7, integrity_hold_s=1.0,
        telemetry_types=(MSG_TYPE,), audit=None,
    )


def _signed(sender_id: str, seq: int, ts: float, obs: dict, key: bytes = KEY) -> SecureMessage:
    msg = SecureMessage(sender_id=sender_id, seq=seq, ts=ts, msg_type=MSG_TYPE, payload={"obs": obs})
    return sign_message(msg, key)


# =================================================================================== ATTACK 1: SENSOR IDENTITY SPOOFING

class TestSensorIdentitySpoofing:
    """An attacker with no valid registered identity attempts to inject telemetry under a
    sender_id the validator has never heard of. Expected: rejected at "authentication", before
    any modality is ever parsed or stored."""

    def test_legitimate_sender_is_accepted(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        r = manager.ingest(msg, now)
        assert r.ok
        obs = manager.usable_observation("ppg", now)
        assert obs is not None and obs.features["hr"] == 70.0

    def test_unregistered_sender_rejected_at_authentication(self, manager):
        now = 100.0
        forged = _signed("attacker-99", seq=1, ts=now,
                         obs={"ppg": {"q": 0.9, "f": {"hr": 999.0}}}, key=ATTACKER_KEY)
        r = manager.ingest(forged, now)
        assert not r.ok
        assert r.stage == "authentication"
        assert "unknown sender" in r.reason

    def test_spoofed_identity_never_becomes_a_usable_observation(self, manager):
        now = 100.0
        forged = _signed("attacker-99", seq=1, ts=now,
                         obs={"ppg": {"q": 0.9, "f": {"hr": 999.0}}}, key=ATTACKER_KEY)
        manager.ingest(forged, now)
        assert manager.usable_observation("ppg", now) is None


# =================================================================================== ATTACK 2: TELEMETRY TAMPERING

class TestTelemetryTampering:
    """A legitimate message is intercepted and its payload altered after signing (the attacker
    does not have the sender's key, so the tag can no longer verify). Expected: rejected at
    "integrity" with a MAC mismatch, and the forged value never overwrites the last authentic
    observation."""

    def test_legitimate_message_establishes_trusted_baseline(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(msg, now).ok
        obs = manager.usable_observation("ppg", now)
        assert obs is not None and obs.features["hr"] == 70.0

    def test_tampered_payload_rejected_at_integrity(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=2, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 72.0}}})
        # Payload swapped AFTER signing: the tag still matches the ORIGINAL bytes, not these.
        tampered = replace(msg, payload={"obs": {"ppg": {"q": 0.9, "f": {"hr": 180.0}}}})
        r = manager.ingest(tampered, now)
        assert not r.ok
        assert r.stage == "integrity"
        assert "MAC mismatch" in r.reason

    def test_tampered_value_never_overwrites_the_last_authentic_reading(self, manager):
        now = 100.0
        legit = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(legit, now).ok

        msg2 = _signed(SENDER, seq=2, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 72.0}}})
        tampered = replace(msg2, payload={"obs": {"ppg": {"q": 0.9, "f": {"hr": 180.0}}}})
        r = manager.ingest(tampered, now)
        assert not r.ok and r.stage == "integrity"

        # Immediately after a MAC failure, the WHOLE source enters a security hold (RH1):
        # every modality it reports becomes unusable for `integrity_hold_s` seconds, even one
        # with a perfectly good prior reading. This is intentional - a failed MAC casts doubt
        # on the whole channel, not just the one forged message. See sensor_trust.py's
        # INTEGRITY_HOLD_STAGES and the module's RH1 design note.
        assert manager.usable_observation("ppg", now) is None

        # Once the hold interval has fully elapsed, the modality must become usable again -
        # and must return the LAST AUTHENTIC value (70.0 from seq=1), never the forged 180.0.
        # The tampered message was rejected by validate() BEFORE _accept() ever ran, so its
        # content was never parsed or stored anywhere in the manager - there is nothing of it
        # left to leak out once the hold clears.
        later = now + manager.hold + 0.01
        obs = manager.usable_observation("ppg", later)
        assert obs is not None
        assert obs.features["hr"] == 70.0


# =================================================================================== ATTACK 3: REPLAY ATTACK

class TestReplayAttack:
    """A genuinely authentic, previously accepted message is captured and resubmitted verbatim.
    Expected: rejected at "replay" on the second submission; the stored observation is left
    exactly as it was from the first, legitimate acceptance."""

    def test_legitimate_message_accepted_once(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(msg, now).ok

    def test_identical_replayed_message_rejected_at_replay(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(msg, now).ok

        replay = manager.ingest(msg, now)
        assert not replay.ok
        assert replay.stage == "replay"
        assert "replay" in replay.reason or "duplicate" in replay.reason

    def test_replay_does_not_alter_the_stored_observation(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        manager.ingest(msg, now)
        manager.ingest(msg, now)  # replay attempt

        obs = manager.usable_observation("ppg", now)
        assert obs is not None
        assert obs.features["hr"] == 70.0


# =================================================================================== ATTACK 4: STALE TELEMETRY

class TestStaleTelemetryAttack:
    """A genuinely authentic message, signed with a timestamp outside the configured freshness
    window (max_age_s=30.0 on the validator fixture), is submitted as if fresh. Expected:
    rejected at "freshness" before replay/authorization are ever evaluated, and never stored."""

    def test_legitimate_fresh_message_accepted(self, manager):
        now = 100.0
        msg = _signed(SENDER, seq=1, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(msg, now).ok

    def test_expired_timestamp_rejected_at_freshness(self, manager):
        now = 200.0
        stale = _signed(SENDER, seq=1, ts=now - 31.0, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        r = manager.ingest(stale, now)
        assert not r.ok
        assert r.stage == "freshness"
        assert "older than" in r.reason

    def test_stale_message_never_becomes_a_usable_observation(self, manager):
        now = 200.0
        stale = _signed(SENDER, seq=1, ts=now - 31.0, obs={"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        manager.ingest(stale, now)
        assert manager.usable_observation("ppg", now) is None