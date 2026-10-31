"""
tests/cybersecurity/test_telemetry_integration.py

Integration tests for the FULL signed-telemetry -> trusted-observation pipeline:

    SensorEndpoint.emit()  ->  MessageValidator (inside SensorTrustManager.ingest())
                            ->  SensorTrustManager storage / trust state
                            ->  SensorTrustManager.usable_observation()

Unlike test_telemetry.py (which validates SensorEndpoint/make_obs output directly against a bare
MessageValidator and never touches SensorTrustManager), every test here goes through
SensorTrustManager.ingest()/assess()/usable_observation() using REAL objects - no mocks/fakes for
SensorEndpoint, MessageValidator, or SensorTrustManager.

Self-contained by design, same rationale as test_telemetry.py: local fixtures, not a shared
conftest.py, since that file was not available for review at the time this suite was written.

NOTE: adjust the import lines below if this repo's actual package root is `cabinguard.cybersecurity`
rather than `src.cybersecurity` - not yet confirmed (same caveat as test_telemetry.py).

Scope / what this file deliberately does NOT cover:
  * The fusion pipeline / AI confidence layer downstream of SensorTrustManager - out of scope for
    the cybersecurity boundary (see sensor_trust.py's module docstring, point 4).
  * Re-testing SensorEndpoint/make_obs in isolation - that's test_telemetry.py's job; duplicating it
    here would just be redundant.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from src.cybersecurity.secure_message import MessageValidator
from src.cybersecurity.telemetry import SensorEndpoint, make_obs
from src.cybersecurity.sensor_trust import (
    FeatureCatalog,
    SensorTrustManager,
    TrustStatus,
)

# --------------------------------------------------------------------------------- local fixtures

FEATS = {
    "phys": ("hr", "hrv"),
    "face": ("blink",),
}
BOUNDS = {
    "hr": (30.0, 220.0),
    "hrv": (0.0, 200.0),
    "blink": (0.0, 10.0),
}


@pytest.fixture
def keys():
    return {"wearable-01": b"W" * 32}


@pytest.fixture
def wearable(keys):
    return SensorEndpoint("wearable-01", keys["wearable-01"])


@pytest.fixture
def validator(keys):
    registry = {"wearable-01": {"key": keys["wearable-01"], "role": "wearable"}}
    roles = {"wearable": {"SENSOR_TELEMETRY"}}
    return MessageValidator(registry, roles, max_age_s=30.0, clock=lambda: 100.0)


@pytest.fixture
def catalog():
    return FeatureCatalog(FEATS, BOUNDS)


@pytest.fixture
def trust_manager(validator, catalog):
    sources = {"wearable-01": ("phys", "face")}
    return SensorTrustManager(
        validator, catalog, sources,
        max_data_age_s=15.0, q_min=0.3, q_ok=0.7, integrity_hold_s=1.0,
    )


# =================================================================================== E2E HAPPY PATH

class TestFullPipelineHappyPath:

    def test_signed_message_becomes_a_trusted_observation(self, wearable, trust_manager):
        obs = make_obs({"hr": 72.0, "hrv": 55.0, "blink": 2}, {"phys": 0.95, "face": 0.9}, FEATS)
        msg = wearable.emit(100.0, obs)

        r = trust_manager.ingest(msg, 100.0)
        assert r.ok

        snap = trust_manager.assess(100.0)
        assert snap.status("phys") is TrustStatus.TRUSTED
        assert snap.status("face") is TrustStatus.TRUSTED

        phys = trust_manager.usable_observation("phys", 100.0)
        assert phys is not None
        assert phys.features["hr"] == 72.0
        assert phys.features["hrv"] == 55.0
        assert phys.quality == 0.95
        assert phys.status is TrustStatus.TRUSTED

    def test_low_quality_reading_is_stored_but_not_usable(self, wearable, trust_manager):
        obs = make_obs({"hr": 72.0, "hrv": 55.0}, {"phys": 0.1}, FEATS, unavailable=["face"])
        msg = wearable.emit(100.0, obs)
        assert trust_manager.ingest(msg, 100.0).ok

        snap = trust_manager.assess(100.0)
        assert snap.status("phys") is TrustStatus.LOW_QUALITY
        assert trust_manager.usable_observation("phys", 100.0) is None   # below q_min: not usable

    def test_unavailable_modality_reads_missing(self, wearable, trust_manager):
        obs = make_obs({}, {}, FEATS, unavailable=["phys", "face"])
        msg = wearable.emit(100.0, obs)
        assert trust_manager.ingest(msg, 100.0).ok

        snap = trust_manager.assess(100.0)
        assert snap.status("phys") is TrustStatus.MISSING
        assert snap.status("face") is TrustStatus.MISSING

    def test_observation_goes_stale_after_max_data_age(self, wearable, trust_manager):
        obs = make_obs({"hr": 72.0, "hrv": 55.0}, {"phys": 0.95}, FEATS, unavailable=["face"])
        msg = wearable.emit(100.0, obs)
        assert trust_manager.ingest(msg, 100.0).ok
        assert trust_manager.usable_observation("phys", 100.0) is not None

        later = 100.0 + trust_manager.max_age + 0.01
        snap = trust_manager.assess(later)
        assert snap.status("phys") is TrustStatus.STALE
        assert trust_manager.usable_observation("phys", later) is None


# =================================================================================== TAMPERING / INTEGRITY

class TestTamperingEndToEnd:

    def test_tampered_message_rejected_and_never_stored(self, wearable, trust_manager):
        obs = make_obs({"hr": 72.0, "hrv": 55.0}, {"phys": 0.95}, FEATS, unavailable=["face"])
        msg = wearable.emit(100.0, obs)
        forged = replace(msg, payload={"obs": {"phys": {"q": 0.95, "f": {"hr": 999.0, "hrv": 55.0}}}})

        r = trust_manager.ingest(forged, 100.0)
        assert not r.ok and r.stage == "integrity"

        # Rejected-at-validation packets are never parsed/stored at all (per module docstring):
        # the modality must read as if nothing was ever received, NOT as IMPLAUSIBLE/CORRUPTED
        # from the forged content.
        snap = trust_manager.assess(100.0)
        assert snap.status("phys") is TrustStatus.MISSING
        assert trust_manager.usable_observation("phys", 100.0) is None

    def test_integrity_rejection_holds_every_modality_of_that_sender(self, wearable, trust_manager):
        """An authentication/integrity failure casts doubt on the WHOLE claimed channel: every
        modality that sender reports must be held CORRUPTED, even a modality the forged packet
        never touched. This is distinct from, and must not be confused with, MT1 (modality
        isolation WITHIN a single successfully-authenticated packet, tested separately below)."""
        good = make_obs({"hr": 70.0, "hrv": 40.0, "blink": 1}, {"phys": 0.9, "face": 0.9}, FEATS)
        assert trust_manager.ingest(wearable.emit(100.0, good), 100.0).ok

        bad = make_obs({"hr": 75.0, "hrv": 42.0, "blink": 1}, {"phys": 0.9, "face": 0.9}, FEATS)
        msg = wearable.emit(100.1, bad)
        forged = replace(msg, payload={"obs": {"phys": {"q": 0.9, "f": {"hr": 999.0, "hrv": 42.0}}}})
        r = trust_manager.ingest(forged, 100.1)
        assert not r.ok and r.stage == "integrity"

        snap = trust_manager.assess(100.1)
        assert snap.status("phys") is TrustStatus.CORRUPTED
        assert snap.status("face") is TrustStatus.CORRUPTED      # untouched by the forged packet
        assert trust_manager.usable_observation("face", 100.1) is None

        # Once the hold window elapses, the earlier good `face` reading (never overwritten)
        # governs again.
        later = 100.1 + trust_manager.hold + 0.01
        snap2 = trust_manager.assess(later)
        assert snap2.status("face") is TrustStatus.TRUSTED


# =================================================================================== MT1: MODALITY ISOLATION

class TestModalityIsolationEndToEnd:

    def test_implausible_modality_does_not_affect_sibling_modality_in_same_packet(self, wearable, trust_manager):
        obs = {
            "phys": {"q": 0.9, "f": {"hr": 999.0, "hrv": 40.0}},   # hr out of bounds -> implausible
            "face": {"q": 0.9, "f": {"blink": 2}},                 # independently fine
        }
        msg = wearable.emit(100.0, obs)
        r = trust_manager.ingest(msg, 100.0)
        assert not r.ok and r.stage == "plausibility"

        snap = trust_manager.assess(100.0)
        assert snap.status("phys") is TrustStatus.IMPLAUSIBLE
        assert snap.status("face") is TrustStatus.TRUSTED

        assert trust_manager.usable_observation("phys", 100.0) is None
        face = trust_manager.usable_observation("face", 100.0)
        assert face is not None and face.features["blink"] == 2.0


# =================================================================================== ARRIVAL-ORDER RACE
#
# The existing test_concurrent_emit_messages_all_independently_validate (test_telemetry.py)
# deliberately SORTS messages by seq before validating - correct for what it's testing (that
# concurrent *generation* doesn't corrupt an individual message), but it never exercises what
# happens when authenticated packets reach SensorTrustManager OUT OF the order they were
# generated in (e.g. network reordering). That is exercised here, end-to-end, against
# SensorTrustManager.ingest() -> usable_observation(), not just the validator.
#
# Invariant under test: regardless of delivery order (sequential-but-reversed, or genuinely
# concurrent-and-reversed), SensorTrustManager must never expose feature values from an OLDER
# (lower-seq) message once a NEWER (higher-seq) one has been accepted. A stale regression here
# would mean an attacker - or just an unlucky network - could resurrect old sensor readings.

class TestReversedArrivalOrderIntegration:

    def _emit_increasing_series(self, wearable, n):
        return [
            wearable.emit(100.0 + i, make_obs({"hr": 60.0 + i, "hrv": 40.0}, {"phys": 0.9}, FEATS,
                                               unavailable=["face"]))
            for i in range(n)
        ]

    def test_reversed_sequential_ingest_never_exposes_stale_data(self, wearable, trust_manager):
        n = 10
        messages = self._emit_increasing_series(wearable, n)
        now = 100.0 + n

        for m in reversed(messages):                      # highest seq (freshest) delivered FIRST
            trust_manager.ingest(m, now)

        to = trust_manager.usable_observation("phys", now)
        assert to is not None
        assert to.features["hr"] == 60.0 + (n - 1), (
            "stored observation regressed to an older (lower-seq) reading after a newer one "
            "had already been delivered - this is the arrival-order race"
        )

    def test_reversed_concurrent_ingest_never_exposes_stale_data(self, wearable, trust_manager):
        """Same invariant, but the reversed list is handed to ingest() from multiple threads at
        once, so the race is on actual wall-clock concurrency, not just call ordering."""
        n = 10
        messages = self._emit_increasing_series(wearable, n)
        now = 100.0 + n
        reversed_messages = list(reversed(messages))

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda m: trust_manager.ingest(m, now), reversed_messages))

        to = trust_manager.usable_observation("phys", now)
        assert to is not None
        assert to.features["hr"] == 60.0 + (n - 1), (
            "concurrent, reversed-order delivery allowed an older reading to win the race and "
            "overwrite the freshest accepted one"
        )