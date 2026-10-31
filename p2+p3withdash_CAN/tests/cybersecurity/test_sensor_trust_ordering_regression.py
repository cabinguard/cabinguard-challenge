"""Regression tests for SensorTrustManager's store-ordering invariant (SQ1).

These tests reproduce a race that cannot happen through validate()+ingest() alone, because
MessageValidator's own replay protection (secure_message.py) already refuses a non-increasing
seq at the authentication layer. The scenario under test is narrower and happens one level
deeper: two packets are BOTH validated successfully, in increasing seq order (a legitimate,
realistic sequence of events), but their completion into SensorTrustManager's stored state
(_accept()) is reversed -- e.g. by thread scheduling delaying the earlier packet's write.

Per the project's testing guidance, this is the one documented, sanctioned case for calling
the internal _accept() directly instead of the public ingest(): there is no way to produce
this exact "validated in order, stored out of order" condition through the public API alone,
since ingest() always validates and stores back-to-back for a single call.

What is being protected against: SensorTrustManager._commit_if_newer() must prevent an older,
already-validated packet from overwriting newer sensor state, for BOTH:
  (a) a normal, well-formed observation, and
  (b) a malformed authenticated payload (R_MALFORMED) -- which must be counted and rejected
      at stage "ordering", NOT silently treated as a fresh "payload" rejection that would
      otherwise clobber the already-stored newer-and-good observation.

This file is additive only: it does not modify or replace tests/cybersecurity/test_sensor_trust.py.
"""
from __future__ import annotations

import pytest

from src.cybersecurity.secure_message import MessageValidator, SecureMessage, sign_message
from src.cybersecurity.sensor_trust import FeatureCatalog, SensorTrustManager

SENDER = "wearable-1"
KEY = b"0123456789abcdef"          # 16 bytes, meets MessageValidator's MIN_KEY_BYTES
MSG_TYPE = "WEARABLE_TELEMETRY"    # must be in SensorTrustManager.DEFAULT_TELEMETRY_TYPES


@pytest.fixture
def validator() -> MessageValidator:
    registry = {SENDER: {"key": KEY, "role": "wearable"}}
    roles = {"wearable": {MSG_TYPE}}
    return MessageValidator(registry, roles, max_age_s=60.0)


@pytest.fixture
def catalog() -> FeatureCatalog:
    return FeatureCatalog(
        features_by_modality={"ppg": ("hr",)},
        bounds={"hr": (30.0, 220.0)},
    )


@pytest.fixture
def manager(validator, catalog) -> SensorTrustManager:
    sources = {SENDER: ("ppg",)}
    return SensorTrustManager(
        validator, catalog, sources,
        max_data_age_s=60.0, q_min=0.3, q_ok=0.7, integrity_hold_s=1.0,
    )


def _signed(seq: int, ts: float, obs) -> SecureMessage:
    msg = SecureMessage(sender_id=SENDER, seq=seq, ts=ts, msg_type=MSG_TYPE, payload={"obs": obs})
    return sign_message(msg, KEY)


class TestReversedCompletionOrder:
    """Both packets authenticate successfully, in increasing seq order; storage is reversed."""

    def test_normal_payload_newer_survives_a_late_older_commit(self, manager):
        now = 1_000.0
        msg_a = _signed(seq=10, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70}}})
        msg_b = _signed(seq=11, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 85}}})

        # Both pass authentication, in the correct (increasing) order.
        r_a = manager.v.validate(msg_a, now)
        r_b = manager.v.validate(msg_b, now)
        assert r_a.ok and r_b.ok

        # Simulate reversed completion: B (newer) is stored first.
        res_b = manager._accept(msg_b, r_b.payload)
        assert res_b.ok

        # A (older) completes afterward and must be rejected without touching stored state.
        res_a = manager._accept(msg_a, r_a.payload)
        assert not res_a.ok
        assert res_a.stage == "ordering"

        obs = manager.usable_observation("ppg", now)
        assert obs is not None
        assert obs.features["hr"] == 85.0   # B's value; never overwritten by stale A

    def test_malformed_older_packet_does_not_overwrite_newer_good_state(self, manager):
        now = 1_000.0
        msg_a = _signed(seq=10, ts=now, obs="not-a-dict")  # malformed: obs must be a dict
        msg_b = _signed(seq=11, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 85}}})

        r_a = manager.v.validate(msg_a, now)
        r_b = manager.v.validate(msg_b, now)
        assert r_a.ok and r_b.ok

        res_b = manager._accept(msg_b, r_b.payload)
        assert res_b.ok

        stats_before = dict(manager.stats)
        res_a = manager._accept(msg_a, r_a.payload)

        # Must be classified as a stale commit, not a fresh malformed-payload rejection.
        assert not res_a.ok
        assert res_a.stage == "ordering"

        # "payload" must NOT have been incremented by the stale malformed packet.
        assert manager.stats.get("payload", 0) == stats_before.get("payload", 0)
        assert manager.stats.get("ordering", 0) == stats_before.get("ordering", 0) + 1

        # B's good observation must still be intact and usable -- not clobbered by stale A.
        obs = manager.usable_observation("ppg", now)
        assert obs is not None
        assert obs.features["hr"] == 85.0

    def test_forward_order_sanity_check(self, manager):
        """Baseline: when completion happens in the SAME order as validation, both still work
        as expected. Not a regression test by itself -- confirms the fixtures are correct."""
        now = 1_000.0
        msg_a = _signed(seq=10, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 70}}})
        msg_b = _signed(seq=11, ts=now, obs={"ppg": {"q": 0.9, "f": {"hr": 85}}})

        r_a = manager.v.validate(msg_a, now)
        r_b = manager.v.validate(msg_b, now)
        assert r_a.ok and r_b.ok

        res_a = manager._accept(msg_a, r_a.payload)
        assert res_a.ok
        res_b = manager._accept(msg_b, r_b.payload)
        assert res_b.ok

        obs = manager.usable_observation("ppg", now)
        assert obs is not None
        assert obs.features["hr"] == 85.0