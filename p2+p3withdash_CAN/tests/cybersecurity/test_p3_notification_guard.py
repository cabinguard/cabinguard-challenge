from dataclasses import replace
import pytest

from src.cybersecurity.notification_guard import (ALLOWED_KEYS, NotificationPrivacyError, NotificationSchema,
                                                         NotificationSender, accept_notification,
                                                         make_notification_validator, validate_notification_payload)
from src.cybersecurity.secure_message import SecureMessage, sign_message

SCHEMA = NotificationSchema(categories=frozenset({"CAT_A", "CAT_B"}), vehicle_states=frozenset({"ST_1", "ST_2"}))
ROLES_N = {"guardian": {"EMERGENCY_NOTIFICATION"}}


@pytest.fixture
def chan(keys):
    reg = {"guardian-01": {"key": keys["guardian-01"], "role": "guardian"}}
    return NotificationSender("guardian-01", keys["guardian-01"], SCHEMA), make_notification_validator(reg, ROLES_N, SCHEMA)


def good(**kw):
    p = {"event_id": "0123456789ab", "category": "CAT_A", "risk_level": "HIGH", "confidence_level": "LOW",
         "timestamp": 1230, "vehicle_state": "ST_1"}
    return p | kw


def test_valid_payload_passes():
    validate_notification_payload(good(), SCHEMA)


@pytest.mark.parametrize("bad", [
    good(driver_name="X"), {k: v for k, v in good().items() if k != "category"},
    good(category="free text about the driver"), good(category="cat_a"), good(risk_level="EXTREME"),
    good(vehicle_state="ST_9"), good(event_id="driver-123456"), good(event_id="0123456789AB"),
    good(timestamp=1234), good(timestamp=1230.0), good(timestamp=True), good(timestamp=-10),
    good(category=["CAT_A"]), good(category={"x": 1}), good(confidence_level=0.9),
])
def test_allow_list_rejects_extra_missing_and_out_of_vocabulary(bad):
    with pytest.raises(NotificationPrivacyError):
        validate_notification_payload(bad, SCHEMA)


def test_non_dict_and_non_string_keys():
    with pytest.raises(NotificationPrivacyError):
        validate_notification_payload([], SCHEMA)
    with pytest.raises(NotificationPrivacyError):
        validate_notification_payload({1: 2}, SCHEMA)


def test_sender_builds_only_allowed_fields_and_rejects_bad_category(chan):
    s, _ = chan
    m = s.build("CAT_A", 0.9, "HIGH", 1234.5, "ST_1")
    assert set(m.payload) == ALLOWED_KEYS and m.payload["confidence_level"] == "HIGH"
    with pytest.raises(NotificationPrivacyError):
        s.build("Patient John Doe has arrhythmia", 0.9, "HIGH", 1234.5, "ST_1")
    assert s.build("CAT_B", 0.1, "LOW", 10.0, "ST_2").seq == 2      # failed build did not consume a seq


def test_event_ids_random_and_unlinkable(chan):
    s, _ = chan
    ids = {s.build("CAT_A", 0.5, "LOW", 100.0, "ST_1").payload["event_id"] for _ in range(20)}
    assert len(ids) == 20


# ---- timestamp mitigation ----------------------------------------------------------------------
def test_exact_time_not_present_in_any_signed_field(chan):
    s, _ = chan
    t = 1234.567
    m = s.build("CAT_A", 0.5, "LOW", t, "ST_1")
    assert m.ts == 1230.0 and m.payload["timestamp"] == 1230
    assert str(t) not in repr(m) and m.ts != t


def test_many_times_in_one_bucket_are_indistinguishable(chan):
    s, _ = chan
    assert {(s.build("CAT_A", 0.5, "LOW", t, "ST_1").ts) for t in (1230.0, 1233.3, 1239.99)} == {1230.0}


def test_accepted_within_window_rejected_after(keys):
    reg = {"guardian-01": {"key": keys["guardian-01"], "role": "guardian"}}
    v = make_notification_validator(reg, ROLES_N, SCHEMA)
    s = NotificationSender("guardian-01", keys["guardian-01"], SCHEMA)
    m = s.build("CAT_A", 0.5, "LOW", 1239.9, "ST_1")                # ts=1230
    assert accept_notification(v, m, 1239.9, SCHEMA).ok
    s2 = NotificationSender("guardian-01", keys["guardian-01"], SCHEMA, first_seq=10)
    m2 = s2.build("CAT_A", 0.5, "LOW", 1230.0, "ST_1")
    assert accept_notification(v, m2, 1242.0, SCHEMA).ok            # age == bucket+slack: boundary accepted
    m3 = s2.build("CAT_A", 0.5, "LOW", 1230.0, "ST_1")
    assert accept_notification(v, m3, 1242.5, SCHEMA).stage == "freshness"


def test_replay_still_rejected_inside_the_wider_window(chan):
    s, v = chan
    m = s.build("CAT_A", 0.5, "LOW", 1235.0, "ST_1")
    assert accept_notification(v, m, 1236.0, SCHEMA).ok
    assert accept_notification(v, m, 1237.0, SCHEMA).stage == "replay"


def test_receiver_rejects_envelope_payload_timestamp_mismatch_and_bad_payload(keys):
    reg = {"guardian-01": {"key": keys["guardian-01"], "role": "guardian"}}
    v = make_notification_validator(reg, ROLES_N, SCHEMA)
    k = keys["guardian-01"]
    exact_env = sign_message(SecureMessage("guardian-01", 1, 1234.5, "EMERGENCY_NOTIFICATION", good(timestamp=1230)), k)
    assert accept_notification(v, exact_env, 1235.0, SCHEMA).stage == "payload"   # exact envelope ts not accepted
    leaky = sign_message(SecureMessage("guardian-01", 2, 1230.0, "EMERGENCY_NOTIFICATION", good(driver_name="X")), k)
    assert accept_notification(v, leaky, 1231.0, SCHEMA).stage == "payload"


def test_schema_validation():
    with pytest.raises(ValueError):
        NotificationSchema(frozenset(), frozenset({"a"}))
    with pytest.raises(ValueError):
        NotificationSchema(frozenset({"a"}), frozenset({"b"}), bucket_s=0)
