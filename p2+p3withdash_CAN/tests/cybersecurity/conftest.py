"""Shared TEST FIXTURES for the Person 3 suites. Features/bounds/states below are placeholders chosen for
testing the security logic; they are NOT Person 1's feature spec nor Person 2's Guardian states."""
import pytest

from src.cybersecurity.audit_log import SecurityAuditLog
from src.cybersecurity.keys import generate_key
from src.cybersecurity.secure_message import MessageValidator
from .p3_support import BOUNDS, FEATS, ROLES, SOURCES

@pytest.fixture
def keys():
    return {s: generate_key() for s in ("wearable-01", "camera-01", "guardian-01", "detector-01")}


@pytest.fixture
def registry(keys):
    role = {"wearable-01": "wearable", "camera-01": "camera", "guardian-01": "guardian", "detector-01": "detector"}
    return {s: {"key": keys[s], "role": role[s]} for s in keys}


@pytest.fixture
def audit():
    return SecurityAuditLog()


@pytest.fixture
def validator(registry, audit):
    return MessageValidator(registry, ROLES, max_age_s=2.0, audit=audit)


@pytest.fixture
def catalog():
    from src.cybersecurity.sensor_trust import FeatureCatalog      # lazy: keeps unrelated suites runnable
    return FeatureCatalog(FEATS, BOUNDS)


@pytest.fixture
def mgr(validator, catalog):
    from src.cybersecurity.sensor_trust import SensorTrustManager
    return SensorTrustManager(validator, catalog, SOURCES, max_data_age_s=2.0, q_min=0.3, q_ok=0.6,
                                integrity_hold_s=1.0)


@pytest.fixture
def wearable(keys):
    from src.cybersecurity.telemetry import SensorEndpoint
    return SensorEndpoint("wearable-01", keys["wearable-01"])


@pytest.fixture
def camera(keys):
    from src.cybersecurity.telemetry import SensorEndpoint
    return SensorEndpoint("camera-01", keys["camera-01"])
