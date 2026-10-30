"""
Test suite for sensor_trust.py.

Covers: the preserved ST1 authenticated-payload-snapshot fix, MT1 modality-level isolation, TH1
thread-safety, and CT1 FeatureCatalog configuration immutability.

Self-contained: defines its own fixtures locally rather than depending on a project conftest.py
that was not available for review. Catalog uses two modalities on one source ("ppg": hr/hrv,
"imu": accel) specifically to test cross-modality isolation within a single packet.
"""
from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from src.cybersecurity.secure_message import MessageValidator, ValidationResult
from src.cybersecurity.sensor_trust import FeatureCatalog, SensorTrustManager, TrustStatus
from src.cybersecurity.telemetry import SensorEndpoint

KEY = b"W" * 32


@pytest.fixture
def catalog():
    return FeatureCatalog(
        features_by_modality={"ppg": ("hr", "hrv"), "imu": ("accel",)},
        bounds={"hr": (30.0, 220.0), "hrv": (0.0, 300.0), "accel": (-20.0, 20.0)},
    )


@pytest.fixture
def sources():
    return {"wearable-01": ("ppg", "imu")}


@pytest.fixture
def validator(sources):
    registry = {"wearable-01": {"key": KEY, "role": "wearable"}}
    roles = {"wearable": {"SENSOR_TELEMETRY"}}
    return MessageValidator(registry, roles, max_age_s=30.0, clock=lambda: 100.0)


@pytest.fixture
def manager(validator, catalog, sources):
    return SensorTrustManager(validator, catalog, sources, max_data_age_s=30.0,
                              q_min=0.3, q_ok=0.7, integrity_hold_s=1.0,
                              telemetry_types=("SENSOR_TELEMETRY",), audit=None)


@pytest.fixture
def wearable():
    return SensorEndpoint("wearable-01", KEY, msg_type="SENSOR_TELEMETRY")


# =================================================================================== ST1: PAYLOAD SNAPSHOT (preserved)

class _RacyValidator:
    """Wraps a real MessageValidator and deterministically injects a mutation of the ORIGINAL
    message's payload immediately after validate() returns ok=True - simulating an attacker or
    concurrent component racing the validate()-to-_accept() handoff. No threading/timing dependency:
    the mutation happens synchronously, exactly at the point the vulnerability lived."""
    def __init__(self, inner, mutate):
        self._inner = inner
        self._mutate = mutate

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def validate(self, m, now=None):
        r = self._inner.validate(m, now)
        if r.ok:
            self._mutate(m)
        return r


class TestPayloadSnapshotSafety:
    """ST1 regression coverage, preserved unmodified through the MT1/TH1 changes."""

    def test_mutation_in_the_validate_to_accept_window_does_not_leak_into_stored_observation(
        self, validator, catalog, sources, wearable
    ):
        def attacker_mutation(m):
            m.payload["obs"]["ppg"]["f"]["hr"] = 999.0

        racy = _RacyValidator(validator, attacker_mutation)
        manager = SensorTrustManager(racy, catalog, sources, max_data_age_s=30.0,
                                     q_min=0.3, q_ok=0.7, integrity_hold_s=1.0,
                                     telemetry_types=("SENSOR_TELEMETRY",), audit=None)

        msg = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0, "hrv": 40.0}}})
        r = manager.ingest(msg, 100.0)
        assert r.ok

        obs = manager.usable_observation("ppg", 100.0)
        assert obs is not None
        assert obs.features["hr"] == 70.0, (
            "stored observation must reflect what was actually authenticated (70.0), not the "
            "mutated value (999.0) injected after validate() returned"
        )

    def test_mutating_original_message_after_ingest_returns_does_not_affect_stored_observation(
        self, manager, wearable
    ):
        msg = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        r = manager.ingest(msg, 100.0)
        assert r.ok

        msg.payload["obs"]["ppg"]["f"]["hr"] = 999.0

        obs = manager.usable_observation("ppg", 100.0)
        assert obs.features["hr"] == 70.0

    def test_stored_features_mapping_is_independent_of_the_message_object(self, manager, wearable):
        msg = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0}}})
        assert manager.ingest(msg, 100.0).ok

        obs = manager.usable_observation("ppg", 100.0)
        assert obs.features is not msg.payload["obs"]["ppg"]["f"]

    def test_baseline_ingest_and_usable_observation_still_work(self, manager, wearable):
        msg = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0, "hrv": 40.0}}})
        r = manager.ingest(msg, 100.0)
        assert r.ok and r.stage == "accepted"

        obs = manager.usable_observation("ppg", 100.0)
        assert obs is not None
        assert obs.features["hr"] == 70.0 and obs.features["hrv"] == 40.0
        assert obs.quality == 0.9


# =================================================================================== MT1: MODALITY ISOLATION

class TestModalityIsolation:
    def test_implausible_ppg_does_not_invalidate_plausible_imu_same_packet(self, manager, wearable):
        obs = {"ppg": {"q": 0.9, "f": {"hr": 300.0}},    # implausible: bounds are (30, 220)
              "imu": {"q": 0.9, "f": {"accel": 5.0}}}    # plausible: bounds are (-20, 20)
        msg = wearable.emit(100.0, obs)
        r = manager.ingest(msg, 100.0)
        assert not r.ok and r.stage == "plausibility"

        snap = manager.assess(100.0)
        assert snap.status("ppg") == TrustStatus.IMPLAUSIBLE
        assert snap.status("imu") == TrustStatus.TRUSTED

        assert manager.usable_observation("ppg", 100.0) is None
        imu_obs = manager.usable_observation("imu", 100.0)
        assert imu_obs is not None and imu_obs.features["accel"] == 5.0

    def test_malformed_single_modality_does_not_invalidate_sibling_modality(self, manager, wearable):
        """A structurally malformed single-modality entry is likewise isolated to that modality -
        only the sender-level authentication/integrity check, or a broken top-level payload shape,
        is packet-wide."""
        obs = {"ppg": {"q": 0.9, "f": "not-a-dict"},     # malformed shape within one modality
              "imu": {"q": 0.9, "f": {"accel": 5.0}}}    # otherwise fine
        msg = wearable.emit(100.0, obs)
        r = manager.ingest(msg, 100.0)
        assert not r.ok and r.stage == "payload"

        assert manager.usable_observation("ppg", 100.0) is None
        imu_obs = manager.usable_observation("imu", 100.0)
        assert imu_obs is not None and imu_obs.features["accel"] == 5.0

    def test_authentication_failure_still_blocks_the_entire_sender_not_one_modality(self, manager, wearable):
        """Contrast case: an authentication/integrity failure remains packet (sender)-wide, unlike
        a single modality's plausibility/shape problem - BOTH modalities become unusable, even
        though only the tag was forged, because the whole channel is now in doubt."""
        good = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0}},
                                     "imu": {"q": 0.9, "f": {"accel": 1.0}}})
        assert manager.ingest(good, 100.0).ok
        assert manager.usable_observation("ppg", 100.0) is not None
        assert manager.usable_observation("imu", 100.0) is not None

        forged = replace(wearable.emit(100.1, {}), tag="f" * 64)
        r = manager.ingest(forged, 100.1)
        assert not r.ok and r.stage == "integrity"

        assert manager.usable_observation("ppg", 100.1) is None
        assert manager.usable_observation("imu", 100.1) is None
        snap = manager.assess(100.1)
        assert snap.status("ppg") == TrustStatus.CORRUPTED
        assert snap.status("imu") == TrustStatus.CORRUPTED


# =================================================================================== TH1: THREAD SAFETY

class TestThreadSafety:
    def test_concurrent_identical_message_ingested_many_times_exactly_one_accepted(self, manager, wearable):
        """N threads ingest the SAME signed message. Exactly one accepted; rest replay.

        RH1: replay is "not new", not channel compromise — no integrity hold.
        """
        msg = wearable.emit(100.0, {"ppg": {"q": 0.9, "f": {"hr": 70.0}},
                                    "imu": {"q": 0.9, "f": {"accel": 1.0}}})
        n = 32
        results = []
        results_lock = threading.Lock()

        def worker():
            r = manager.ingest(msg, 100.0)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert all(not t.is_alive() for t in threads), "a thread did not finish - possible deadlock"

        accepted = [r for r in results if r.ok]
        replayed = [r for r in results if not r.ok and r.stage == "replay"]
        assert len(accepted) == 1
        assert len(replayed) == n - 1

        snap = manager.assess(100.0)
        assert snap.status("ppg") is TrustStatus.TRUSTED
        assert snap.status("imu") is TrustStatus.TRUSTED
        assert snap.integrity_ok is True
        phys = manager.usable_observation("ppg", 100.0)
        assert phys is not None and phys.features["hr"] == 70.0

    def test_concurrent_ingest_and_assess_and_usable_observation_no_deadlock(self, manager, wearable):
        """Mixed ingest + assess/usable_observation. No hang, no exception, counters not lost."""
        n = 30
        results = []
        results_lock = threading.Lock()
        errors = []

        def emit_and_ingest(i):
            try:
                msg = wearable.emit(100.0 + i * 0.001,
                                    {"ppg": {"q": 0.9, "f": {"hr": 70.0}},
                                     "imu": {"q": 0.9, "f": {"accel": 1.0}}})
                r = manager.ingest(msg, 100.0 + i * 0.001)
                with results_lock:
                    results.append(r)
            except Exception as e:  # pragma: no cover
                with results_lock:
                    errors.append(e)

        def spin_assess():
            try:
                for _ in range(50):
                    manager.assess(200.0)
                    manager.usable_observation("ppg", 200.0)
                    manager.usable_observation("imu", 200.0)
            except Exception as e:  # pragma: no cover
                with results_lock:
                    errors.append(e)

        threads = [threading.Thread(target=emit_and_ingest, args=(i,)) for i in range(n)]
        threads += [threading.Thread(target=spin_assess) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(not t.is_alive() for t in threads), "a thread did not finish - possible deadlock"
        assert not errors, f"unexpected exception(s) under concurrent load: {errors}"

        accepted = sum(1 for r in results if r.ok)
        replayed = sum(1 for r in results if not r.ok and r.stage == "replay")
        assert accepted + replayed == n
        assert manager.stats.get("replay", 0) == replayed

    def test_usable_observation_raises_on_unknown_modality_even_under_lock(self, manager):
        with pytest.raises(KeyError):
            manager.usable_observation("nonexistent-modality", 100.0)
# =================================================================================== CT1: FEATURECATALOG IMMUTABILITY

class TestFeatureCatalogImmutability:
    def test_mutating_original_features_by_modality_dict_after_construction_has_no_effect(self):
        fbm = {"ppg": ["hr"]}
        bounds = {"hr": (30.0, 220.0)}
        cat = FeatureCatalog(fbm, bounds)

        fbm["ppg"].append("new_feature")     # mutate the original list
        fbm["imu"] = ["accel"]               # attempted injection of a new modality

        assert cat.features_by_modality == {"ppg": ("hr",)}
        assert "imu" not in cat.features_by_modality

    def test_mutating_original_bounds_dict_after_construction_has_no_effect(self):
        fbm = {"ppg": ["hr"]}
        bounds = {"hr": [30.0, 220.0]}
        cat = FeatureCatalog(fbm, bounds)

        bounds["hr"][0] = -999.0
        bounds["hr"] = [0.0, 1.0]

        assert cat.bounds["hr"] == (30.0, 220.0)

    def test_features_by_modality_is_read_only(self):
        cat = FeatureCatalog({"ppg": ("hr",)}, {"hr": (30.0, 220.0)})
        with pytest.raises(TypeError):
            cat.features_by_modality["ppg"] = ("hr", "extra")

    def test_bounds_is_read_only(self):
        cat = FeatureCatalog({"ppg": ("hr",)}, {"hr": (30.0, 220.0)})
        with pytest.raises(TypeError):
            cat.bounds["hr"] = (0.0, 1.0)

    def test_feature_list_passed_as_list_is_normalised_to_tuple(self):
        cat = FeatureCatalog({"ppg": ["hr", "hrv"]}, {"hr": (30.0, 220.0), "hrv": (0.0, 300.0)})
        assert cat.features_by_modality["ppg"] == ("hr", "hrv")
        assert isinstance(cat.features_by_modality["ppg"], tuple)

    def test_bounds_passed_as_list_is_normalised_to_tuple(self):
        cat = FeatureCatalog({"ppg": ("hr",)}, {"hr": [30.0, 220.0]})
        assert cat.bounds["hr"] == (30.0, 220.0)
        assert isinstance(cat.bounds["hr"], tuple)

    def test_from_feature_specs_also_produces_an_immutable_independent_catalog(self):
        specs = {"hr": ("ppg", "numeric")}
        bounds = {"hr": (30.0, 220.0)}
        cat = FeatureCatalog.from_feature_specs(specs, bounds)

        specs["hr"] = ("ppg", "changed")
        bounds["hr"] = (0.0, 1.0)

        assert cat.bounds["hr"] == (30.0, 220.0)
        with pytest.raises(TypeError):
            cat.bounds["hr"] = (1.0, 2.0)

    @pytest.mark.parametrize("bad_fbm", [None, [], "ppg", {"ppg": "hr"}, {"ppg": 123}])
    def test_malformed_features_by_modality_rejected(self, bad_fbm):
        with pytest.raises((TypeError, ValueError)):
            FeatureCatalog(bad_fbm, {"hr": (30.0, 220.0)})

    @pytest.mark.parametrize("bad_bounds", [None, [], {"hr": (220.0, 30.0)}, {"hr": (1.0,)}])
    def test_malformed_bounds_rejected(self, bad_bounds):
        with pytest.raises((TypeError, ValueError)):
            FeatureCatalog({"ppg": ("hr",)}, bad_bounds)