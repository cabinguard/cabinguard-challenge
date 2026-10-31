"""
tests/cybersecurity/test_message_integrity.py

Test suite for secure_message.py (SecureMessage / MessageValidator).

Organised to mirror the documented validation pipeline:
    schema -> authentication -> integrity -> freshness -> replay -> authorization -> commit

Each class below targets one stage, plus dedicated classes for specific audit findings
(D1 commit-after-all-checks, D7 config validation, D8 audit isolation, D10/D12 concurrency,
E1 config hardening, E2 payload-snapshot safety, E3/E4 sign/verify hardening) since those are
regression-prone: it would be easy for a future refactor to silently reintroduce the bugs they fixed.

NOTE: adjust the import path below to match your actual package layout.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.cybersecurity.secure_message import (
    MIN_KEY_BYTES,
    MessageValidator,
    SecureMessage,
    SecureMessageError,
    ValidationResult,
    sign_message,
    trusted_payload,
    verify_tag,
)
from src.cybersecurity.audit_log import SecurityAuditLog, Event, Stage, Reason


# --------------------------------------------------------------------------------- test doubles

class FakeAuditLog:
    """Records calls instead of writing anywhere real. Can simulate a broken sink (D8)."""

    def __init__(self, raise_on_record: bool = False):
        self.records: list[tuple] = []
        self.raise_on_record = raise_on_record

    def record(self, event, stage, sender, reason, extra=""):
        if self.raise_on_record:
            raise RuntimeError("simulated audit sink failure")
        self.records.append((event, stage, sender, reason, extra))


# --------------------------------------------------------------------------------- fixtures

KEY_A = b"A" * MIN_KEY_BYTES
KEY_B = b"B" * MIN_KEY_BYTES
FIXED_NOW = 1_000_000.0


@pytest.fixture
def registry():
    return {
        "device-a": {"key": KEY_A, "role": "sensor"},
        "device-b": {"key": KEY_B, "role": "controller"},
    }


@pytest.fixture
def roles():
    return {
        "sensor": {"TELEMETRY"},
        "controller": {"TELEMETRY", "COMMAND"},
    }


@pytest.fixture
def validator(registry, roles):
    return MessageValidator(
        registry=registry,
        roles=roles,
        max_age_s=30.0,
        max_future_skew_s=5.0,
        clock=lambda: FIXED_NOW,
    )


def signed(sender_id, seq, ts, msg_type, payload, key=KEY_A):
    m = SecureMessage(sender_id=sender_id, seq=seq, ts=ts, msg_type=msg_type, payload=payload)
    return sign_message(m, key)


def valid_msg(seq=1, ts=FIXED_NOW, msg_type="TELEMETRY", payload=None, sender="device-a", key=KEY_A):
    return signed(sender, seq, ts, msg_type, payload or {"reading": 42}, key)


# =================================================================================== SCHEMA (D2-D5)

class TestSchemaValidation:
    """validate() must REJECT malformed input, never raise, and label it stage='schema'."""

    def test_valid_message_passes_schema(self, validator):
        result = validator.validate(valid_msg())
        assert result.ok

    @pytest.mark.parametrize("bad_input", [None, "not a message", 42, {"sender_id": "x"}])
    def test_non_securemessage_rejected_not_raised(self, validator, bad_input):
        result = validator.validate(bad_input)
        assert result == ValidationResult(False, "schema", result.reason)
        assert "not a SecureMessage" in result.reason

    def test_empty_sender_id_rejected(self, validator):
        # NOTE (E4 regression fix): sign_message() now schema-validates its input before signing,
        # so it would correctly REFUSE to sign a message with an empty sender_id. To test that the
        # VALIDATOR's own schema stage independently catches this (defence in depth, not reliance on
        # the sender ever being honest), we build the forged message directly rather than through
        # sign_message - mirroring the same technique test_invalid_seq_rejected already uses below.
        base = valid_msg()
        forged = base.__class__("", base.seq, base.ts, base.msg_type, base.payload, base.tag)
        result = validator.validate(forged)
        assert not result.ok and result.stage == "schema"

    @pytest.mark.parametrize("bad_seq", [-1, -100, True, False, 1.5, "1", None])
    def test_invalid_seq_rejected(self, validator, bad_seq):
        # Sign a schema-valid message first to obtain a well-formed tag, then rebuild the message
        # directly (bypassing sign_message's own validation) substituting the bad seq - this proves
        # the VALIDATOR's schema stage independently rejects it.
        base = sign_message(SecureMessage("device-a", 0, FIXED_NOW, "TELEMETRY", {}), KEY_A)
        m = base.__class__("device-a", bad_seq, FIXED_NOW, "TELEMETRY", {}, base.tag)
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    @pytest.mark.parametrize("bad_ts", [float("nan"), float("inf"), float("-inf"), "now", None])
    def test_invalid_timestamp_rejected(self, validator, bad_ts):
        m = SecureMessage("device-a", 1, bad_ts, "TELEMETRY", {})
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    def test_empty_msg_type_rejected(self, validator):
        m = SecureMessage("device-a", 1, FIXED_NOW, "", {})
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    @pytest.mark.parametrize("bad_payload", [None, [], "payload", 42, {"a": {1, 2, 3}}])
    def test_invalid_payload_rejected(self, validator, bad_payload):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", bad_payload)
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    def test_non_canonically_serialisable_payload_rejected(self, validator):
        """A dict that passes isinstance(dict) but contains something json.dumps rejects."""
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"bad": {1, 2}})  # a set
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"
        assert "serialisable" in result.reason

    def test_nan_inside_payload_rejected(self, validator):
        """allow_nan=False means NaN/Infinity values embedded in payload must be rejected,
        not silently serialised (which would break interoperability / signature stability)."""
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"x": float("nan")})
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    def test_non_string_tag_rejected(self, validator):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}, tag=12345)
        result = validator.validate(m)
        assert not result.ok and result.stage == "schema"

    def test_now_nan_raises_programmer_error(self, validator):
        """`now` is an internal/programmer-supplied value, not attacker input - a bad clock
        reading is a bug in the calling code and SHOULD raise, unlike attacker-controlled m.ts."""
        with pytest.raises(ValueError):
            validator.validate(valid_msg(), now=float("nan"))


# =================================================================================== AUTHENTICATION

class TestAuthentication:
    def test_unknown_sender_rejected(self, validator):
        m = signed("device-x", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A)
        result = validator.validate(m)
        assert not result.ok and result.stage == "authentication"
        assert "unknown sender" in result.reason

    def test_registry_rejects_short_keys_at_construction(self, roles):
        with pytest.raises(ValueError):
            MessageValidator(
                registry={"device-a": {"key": b"short", "role": "sensor"}},
                roles=roles,
                max_age_s=30.0,
            )

    def test_registry_rejects_non_bytes_key_at_construction(self, roles):
        with pytest.raises(ValueError):
            MessageValidator(
                registry={"device-a": {"key": "not-bytes-at-all-16ch", "role": "sensor"}},
                roles=roles,
                max_age_s=30.0,
            )


# =================================================================================== INTEGRITY (D9, D11)

class TestIntegrity:
    def test_correct_tag_passes(self, validator):
        assert validator.validate(valid_msg()).ok

    def test_wrong_key_fails_mac(self, validator):
        """Signed with the wrong key: well-formed tag, but it won't match."""
        m = signed("device-a", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_B)
        result = validator.validate(m)
        assert not result.ok and result.stage == "integrity"
        assert "MAC mismatch" in result.reason

    @pytest.mark.parametrize("bad_tag", [
        "",                      # empty
        "deadbeef",              # too short
        "Z" * 64,                # invalid hex chars
        "A" * 64,                # uppercase hex (format requires lowercase)
        "0" * 63,                # off by one length
        "0" * 65,
    ])
    def test_malformed_tag_format_rejected(self, validator, bad_tag):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}, tag=bad_tag)
        result = validator.validate(m)
        assert not result.ok and result.stage == "integrity"
        assert "format" in result.reason

    def test_unknown_sender_with_malformed_tag_reports_authentication_not_integrity(self, validator):
        """Regression test for D11: stage order must be authentication BEFORE integrity.
        An unknown sender with a garbage tag must be reported as 'authentication', not
        'integrity' - downstream alerting/triage may treat these very differently."""
        m = SecureMessage("totally-unknown-device", 1, FIXED_NOW, "TELEMETRY", {}, tag="not-hex-at-all")
        result = validator.validate(m)
        assert result.stage == "authentication"

    @pytest.mark.parametrize("field,new_value", [
        ("sender_id", "device-b"),
        ("seq", 999),
        ("ts", FIXED_NOW + 1),
        ("msg_type", "COMMAND"),
        ("payload", {"tampered": True}),
    ])
    def test_tampering_any_field_invalidates_tag(self, validator, field, new_value):
        m = valid_msg()
        tampered = m.__class__(**{**m.__dict__, field: new_value})
        result = validator.validate(tampered)
        assert not result.ok and result.stage in ("integrity", "authentication", "schema")
        # sender_id tampering may surface as 'authentication' (unknown sender) rather than
        # 'integrity' depending on the new value - both are correct rejections.


# =================================================================================== FRESHNESS (D6, D7)

class TestFreshness:
    def test_message_too_old_rejected(self, validator):
        m = valid_msg(ts=FIXED_NOW - 31)  # max_age=30
        result = validator.validate(m)
        assert not result.ok and result.stage == "freshness"
        assert "older" in result.reason

    def test_message_at_exact_max_age_boundary_passes(self, validator):
        m = valid_msg(ts=FIXED_NOW - 30)  # exactly at boundary: age > max_age is strict
        result = validator.validate(m)
        assert result.ok

    def test_message_too_far_in_future_rejected(self, validator):
        m = valid_msg(ts=FIXED_NOW + 6)  # max_future_skew=5
        result = validator.validate(m)
        assert not result.ok and result.stage == "freshness"
        assert "future" in result.reason

    def test_message_at_exact_future_boundary_passes(self, validator):
        m = valid_msg(ts=FIXED_NOW + 5)
        result = validator.validate(m)
        assert result.ok

    def test_future_skew_defaults_to_max_age_when_not_specified(self, registry, roles):
        v = MessageValidator(registry=registry, roles=roles, max_age_s=10.0, clock=lambda: FIXED_NOW)
        assert v.max_future == v.max_age == 10.0
        too_far = valid_msg(ts=FIXED_NOW + 11)
        assert not v.validate(too_far).ok


class TestFreshnessConfigValidation:
    """D7: a misconfigured validator must fail fast at construction, not at message time."""

    @pytest.mark.parametrize("bad_max_age", [-1, -0.001, float("nan"), float("inf"), "30"])
    def test_invalid_max_age_rejected_at_construction(self, registry, roles, bad_max_age):
        with pytest.raises(ValueError):
            MessageValidator(registry=registry, roles=roles, max_age_s=bad_max_age)

    @pytest.mark.parametrize("bad_skew", [-1, float("nan"), float("inf")])
    def test_invalid_future_skew_rejected_at_construction(self, registry, roles, bad_skew):
        with pytest.raises(ValueError):
            MessageValidator(registry=registry, roles=roles, max_age_s=30.0, max_future_skew_s=bad_skew)

    def test_zero_is_a_valid_tolerance(self, registry, roles):
        """Zero tolerance is strict but legitimate - must not be rejected as 'falsy'."""
        v = MessageValidator(registry=registry, roles=roles, max_age_s=0.0, max_future_skew_s=0.0)
        assert v.max_age == 0.0 and v.max_future == 0.0


# =================================================================================== REPLAY + D1 COMMIT ORDERING

class TestReplayAndCommitOrdering:
    def test_duplicate_seq_rejected(self, validator):
        assert validator.validate(valid_msg(seq=1)).ok
        result = validator.validate(valid_msg(seq=1))
        assert not result.ok and result.stage == "replay"

    def test_lower_seq_after_higher_rejected(self, validator):
        assert validator.validate(valid_msg(seq=5)).ok
        result = validator.validate(valid_msg(seq=3))
        assert not result.ok and result.stage == "replay"

    def test_strictly_increasing_seq_accepted(self, validator):
        for seq in (1, 2, 3, 10, 11):
            assert validator.validate(valid_msg(seq=seq)).ok

    def test_D1_authorization_failure_does_not_advance_replay_counter(self, validator):
        """Regression test for D1. 'device-a' has role 'sensor', which is only authorised for
        TELEMETRY. Sending an unauthorised COMMAND at seq=1 must be rejected WITHOUT
        advancing last_seq - otherwise the sender's legitimate seq=1 TELEMETRY message would
        later be wrongly rejected as a replay, permanently blocking that sender."""
        unauthorised = valid_msg(seq=1, msg_type="COMMAND")  # sensor can't send COMMAND
        result = validator.validate(unauthorised)
        assert not result.ok and result.stage == "authorization"

        legitimate = valid_msg(seq=1, msg_type="TELEMETRY")
        result2 = validator.validate(legitimate)
        assert result2.ok, (
            "D1 regression: an authorised-but-rejected message must not have advanced "
            "the replay counter and blocked the sender's legitimate retry."
        )

    def test_D1_schema_failure_does_not_advance_counter(self, validator):
        bad = SecureMessage("device-a", 1, float("nan"), "TELEMETRY", {})
        assert validator.validate(bad).stage == "schema"
        assert validator.validate(valid_msg(seq=1)).ok

    def test_D1_freshness_failure_does_not_advance_counter(self, validator):
        stale = valid_msg(seq=1, ts=FIXED_NOW - 1000)
        assert validator.validate(stale).stage == "freshness"
        assert validator.validate(valid_msg(seq=1)).ok


class TestReplayUnaffectedByIntegrityFailure:
    """Completes D1 coverage: an invalid MAC must likewise never advance replay state."""

    def test_D1_integrity_failure_does_not_advance_counter(self, validator):
        forged = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}, tag="f" * 64)
        assert validator.validate(forged).stage == "integrity"
        assert validator.validate(valid_msg(seq=1)).ok


# =================================================================================== AUTHORIZATION

class TestAuthorization:
    def test_role_not_permitted_rejected(self, validator):
        m = valid_msg(msg_type="COMMAND")  # device-a is 'sensor', can't send COMMAND
        result = validator.validate(m)
        assert not result.ok and result.stage == "authorization"

    def test_role_permitted_accepted(self, validator):
        m = signed("device-b", 1, FIXED_NOW, "COMMAND", {}, key=KEY_B)  # controller
        assert validator.validate(m).ok

    def test_unknown_role_in_registry_denies_everything(self, registry, roles):
        registry["device-c"] = {"key": b"C" * MIN_KEY_BYTES, "role": "ghost-role"}
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0, clock=lambda: FIXED_NOW)
        m = signed("device-c", 1, FIXED_NOW, "TELEMETRY", {}, key=b"C" * MIN_KEY_BYTES)
        result = v.validate(m)
        assert not result.ok and result.stage == "authorization"


# =================================================================================== RESET_SENDER

class TestResetSender:
    def test_unauthorised_reset_refused(self, validator):
        with pytest.raises(PermissionError):
            validator.reset_sender("device-a", authorized=False)

    def test_reset_unknown_sender_raises_keyerror(self, validator):
        with pytest.raises(KeyError):
            validator.reset_sender("nope", authorized=True)

    def test_authorised_reset_allows_sequence_restart(self, validator):
        assert validator.validate(valid_msg(seq=5)).ok
        assert not validator.validate(valid_msg(seq=1)).ok  # would be replay

        validator.reset_sender("device-a", authorized=True)

        assert validator.validate(valid_msg(seq=1)).ok  # now accepted post-reset

    def test_reset_only_affects_the_named_sender(self, validator):
        validator.validate(valid_msg(seq=5))
        validator.validate(signed("device-b", 5, FIXED_NOW, "TELEMETRY", {}, key=KEY_B))

        validator.reset_sender("device-a", authorized=True)

        # device-b's counter must be untouched
        result = validator.validate(signed("device-b", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_B))
        assert not result.ok and result.stage == "replay"


# =================================================================================== AUDIT LOGGING (D8)

class TestAuditIsolation:
    def test_broken_audit_sink_does_not_crash_validation(self, registry, roles):
        audit = FakeAuditLog(raise_on_record=True)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        m = signed("device-x", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A)  # unknown sender -> rejection

        result = v.validate(m)  # must not raise

        assert not result.ok
        assert v.stats.get("audit_failures", 0) == 1

    def test_broken_audit_sink_cannot_turn_rejection_into_acceptance(self, registry, roles):
        audit = FakeAuditLog(raise_on_record=True)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        m = valid_msg(msg_type="COMMAND")  # authorization failure
        result = v.validate(m)
        assert result.stage == "authorization" and not result.ok

    def test_working_audit_sink_receives_rejection_events(self, registry, roles):
        audit = FakeAuditLog()
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        v.validate(signed("device-x", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A))

        assert len(audit.records) == 1
        event, stage, sender, reason, extra = audit.records[0]
        assert event == "MESSAGE_REJECTED" and stage == "authentication"

    def test_audit_log_never_contains_payload_or_key_material(self, registry, roles):
        audit = FakeAuditLog()
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        secret_payload_marker = "TOP-SECRET-PAYLOAD-VALUE-xyz123"
        m = valid_msg(msg_type="COMMAND", payload={"note": secret_payload_marker})
        v.validate(m)  # rejected (authorization) -> should be logged

        assert audit.records, "expected a rejection to be logged"
        for record in audit.records:
            flattened = " ".join(str(f) for f in record)
            assert secret_payload_marker not in flattened
            assert KEY_A.decode(errors="ignore") not in flattened
            assert m.tag not in flattened

    def test_successful_sender_reset_is_logged(self, validator):
        audit = FakeAuditLog()
        validator.audit = audit
        validator.reset_sender("device-a", authorized=True, reason="test resync")
        assert any(r[0] == "SENDER_RESET" for r in audit.records)

    def test_reset_audit_failure_is_isolated_too(self, validator):
        validator.audit = FakeAuditLog(raise_on_record=True)
        # must not raise despite the audit sink failing
        validator.reset_sender("device-a", authorized=True)
        assert validator.stats.get("audit_failures", 0) == 1


# =================================================================================== REAL AUDIT LOG INTEGRATION

class TestRealAuditLogIntegration:
    """FakeAuditLog (above) proves crash/isolation safety; this class exercises the actual
    SecurityAuditLog/Event/Stage/Reason sanitising sink end-to-end."""

    def test_rejection_is_recorded_with_a_standardised_reason_code(self, registry, roles):
        audit = SecurityAuditLog(maxlen=50)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        v.validate(signed("device-x", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A))

        entries = audit.to_dicts()
        assert len(entries) == 1
        assert entries[0]["event"] == Event.MESSAGE_REJECTED
        assert entries[0]["stage"] == Stage.AUTHENTICATION
        assert entries[0]["reason"] == Reason.UNKNOWN_SENDER

    def test_no_payload_key_or_tag_leakage_through_the_real_sink(self, registry, roles):
        audit = SecurityAuditLog(maxlen=50)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        secret_marker = "TOP-SECRET-PAYLOAD-VALUE-xyz123"
        m = valid_msg(msg_type="COMMAND", payload={"note": secret_marker})
        v.validate(m)

        blob = str(audit.to_dicts())
        assert secret_marker not in blob
        assert KEY_A.decode(errors="ignore") not in blob
        assert m.tag not in blob

    def test_forged_sender_id_cannot_inject_log_lines(self, registry, roles):
        audit = SecurityAuditLog(maxlen=50)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        malicious_sender = "device-a\nFAKE LOG LINE: ADMIN OVERRIDE\x1b[31m"
        v.validate(signed(malicious_sender, 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A))

        entries = audit.to_dicts()
        assert len(entries) == 1
        assert "\n" not in entries[0]["sender"]
        assert entries[0]["sender"] == "<unregistered>"

    def test_audit_log_injection_via_direct_record_call_is_neutralised(self):
        """Defence in depth below MessageValidator._label()'s own reduction."""
        audit = SecurityAuditLog(maxlen=50)
        entry = audit.record(Event.MESSAGE_REJECTED, Stage.SCHEMA,
                              "evil\nFAKE: ADMIN\x1b[31m", Reason.MALFORMED_MESSAGE, "x\ny")
        assert "\n" not in entry.sender
        assert entry.sender == "<invalid-sender>"
        assert entry.detail_code == "<invalid-detail>"

    def test_audit_buffer_eviction_preserves_monotonic_sequence(self):
        audit = SecurityAuditLog(maxlen=3)
        for i in range(10):
            audit.record(Event.MESSAGE_REJECTED, Stage.SCHEMA, "<unregistered>",
                         Reason.MALFORMED_MESSAGE, f"i={i}")
        entries = audit.entries()
        assert len(entries) == 3
        seqs = [e.seq for e in entries]
        assert seqs == sorted(seqs)
        assert seqs[-1] == 10          # seq keeps climbing even though only 3 entries remain

    def test_unrecognised_reason_code_falls_back_to_unspecified_not_arbitrary_text(self):
        audit = SecurityAuditLog(maxlen=10)
        entry = audit.record(Event.MESSAGE_REJECTED, Stage.INTEGRITY, "device-a",
                              "role 'sensor' not permitted to send COMMAND")  # free text, not a code
        assert entry.reason == Reason.UNSPECIFIED

    @pytest.mark.parametrize("bad_maxlen", [0, -1, -100, 1.5, "10", None, True, False])
    def test_invalid_maxlen_rejected(self, bad_maxlen):
        with pytest.raises(ValueError):
            SecurityAuditLog(maxlen=bad_maxlen)

    def test_concurrent_writes_to_real_audit_log_produce_unique_sequence_numbers(self, registry, roles):
        audit = SecurityAuditLog(maxlen=1000)
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=audit)
        msgs = [signed(f"unknown-{i}", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A) for i in range(32)]

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(v.validate, msgs))

        seqs = [e.seq for e in audit.entries()]
        assert len(seqs) == 32 and len(set(seqs)) == 32


# =================================================================================== CONFIGURATION HARDENING (E1)

class TestConfigurationHardening:
    """E1: registry/roles are structurally validated and defensively copied at construction time,
    so mutating the caller's original objects afterwards can never change live auth keys or
    permissions."""

    def test_mutating_original_registry_dict_after_init_has_no_effect(self, registry, roles):
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0, clock=lambda: FIXED_NOW)

        registry["device-a"]["role"] = "controller"                      # attempted escalation
        registry["device-c"] = {"key": b"C" * MIN_KEY_BYTES, "role": "sensor"}  # attempted injection

        still_sensor = v.validate(valid_msg(msg_type="COMMAND"))
        assert not still_sensor.ok and still_sensor.stage == "authorization"

        forged = signed("device-c", 1, FIXED_NOW, "TELEMETRY", {}, key=b"C" * MIN_KEY_BYTES)
        result = v.validate(forged)
        assert not result.ok and result.stage == "authentication"

    def test_mutating_original_registry_key_bytearray_after_init_has_no_effect(self, roles):
        key = bytearray(KEY_A)
        reg = {"device-a": {"key": key, "role": "sensor"}}
        v = MessageValidator(registry=reg, roles=roles, max_age_s=30.0, clock=lambda: FIXED_NOW)

        m = signed("device-a", 1, FIXED_NOW, "TELEMETRY", {}, key=bytes(key))
        key[:] = b"Z" * MIN_KEY_BYTES   # mutate key material in place after construction

        assert v.validate(m).ok, "validator must keep using the key captured at construction time"

    def test_mutating_original_roles_dict_after_init_has_no_effect(self, registry, roles):
        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0, clock=lambda: FIXED_NOW)
        roles["sensor"].add("COMMAND")   # attempted post-hoc permission grant

        result = v.validate(valid_msg(msg_type="COMMAND"))   # device-a is 'sensor'
        assert not result.ok and result.stage == "authorization"

    def test_registry_is_read_only_on_the_validator(self, validator):
        with pytest.raises(TypeError):
            validator.registry["device-a"] = {"key": KEY_A, "role": "controller"}
        with pytest.raises(TypeError):
            validator.registry["device-a"]["role"] = "controller"

    def test_roles_is_read_only_on_the_validator(self, validator):
        with pytest.raises(TypeError):
            validator.roles["sensor"] = frozenset({"COMMAND"})

    @pytest.mark.parametrize("bad_registry", [
        None, [], "not-a-dict",
        {123: {"key": KEY_A, "role": "sensor"}},
        {"": {"key": KEY_A, "role": "sensor"}},
        {"device-a": "not-a-dict"},
        {"device-a": {"key": KEY_A}},                 # missing role
        {"device-a": {"role": "sensor"}},              # missing key
        {"device-a": {"key": KEY_A, "role": ""}},
        {"device-a": {"key": KEY_A, "role": 123}},
        {"device-a": {"key": b"", "role": "sensor"}},   # zero-length key
        {"device-a": {"key": 12345, "role": "sensor"}},  # key not bytes-like
    ])
    def test_malformed_registry_rejected_at_construction(self, roles, bad_registry):
        with pytest.raises(ValueError):
            MessageValidator(registry=bad_registry, roles=roles, max_age_s=30.0)

    @pytest.mark.parametrize("bad_roles", [
        None, [], "not-a-dict",
        {123: {"TELEMETRY"}},
        {"": {"TELEMETRY"}},
        {"sensor": "TELEMETRY"},        # string is iterable char-by-char - must be rejected
        {"sensor": {"", "TELEMETRY"}},
        {"sensor": {1, 2}},
    ])
    def test_malformed_roles_rejected_at_construction(self, registry, bad_roles):
        with pytest.raises(ValueError):
            MessageValidator(registry=registry, roles=bad_roles, max_age_s=30.0)

    def test_bytearray_registry_key_is_accepted(self, roles):
        reg = {"device-a": {"key": bytearray(KEY_A), "role": "sensor"}}
        v = MessageValidator(registry=reg, roles=roles, max_age_s=30.0, clock=lambda: FIXED_NOW)
        assert v.validate(signed("device-a", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A)).ok


# =================================================================================== PAYLOAD MUTATION / SNAPSHOT SAFETY (E2)

class TestPayloadMutationBoundary:
    def test_frozen_dataclass_does_not_freeze_payload_dict(self, validator):
        """Characterisation test documenting a known, intentional limitation: `frozen=True`
        prevents reassigning m.payload, but NOT mutating the dict it points to. Downstream
        code must use ValidationResult.payload (or trusted_payload()), not the original message,
        after validation."""
        m = valid_msg(payload={"value": 1})
        assert validator.validate(m).ok

        m.payload["value"] = 999  # succeeds - this is the documented gap, not a bug
        assert m.payload["value"] == 999

    def test_trusted_payload_returns_independent_copy(self, validator):
        m = valid_msg(payload={"nested": {"value": 1}})
        assert validator.validate(m).ok

        safe = trusted_payload(m)
        safe["nested"]["value"] = 999

        assert m.payload["nested"]["value"] == 1, "mutating the copy must not affect the original"

    def test_trusted_payload_equals_original_at_time_of_call(self, validator):
        m = valid_msg(payload={"a": 1, "b": [1, 2, 3]})
        assert trusted_payload(m) == m.payload
        assert trusted_payload(m) is not m.payload


class TestPayloadSnapshotSafety:
    """E2: ValidationResult.payload (ok=True only) is an independent deep-copy snapshot taken
    BEFORE any check runs, so the bytes that were MAC-verified are exactly the bytes returned to
    the caller - no window for another thread/component to swap the payload in between."""

    def test_accepted_result_carries_an_independent_payload_snapshot(self, validator):
        m = valid_msg(payload={"reading": 42})
        result = validator.validate(m)
        assert result.ok
        assert result.payload == {"reading": 42}
        assert result.payload is not m.payload

    def test_rejected_result_has_no_payload(self, validator):
        result = validator.validate(valid_msg(msg_type="COMMAND"))
        assert not result.ok and result.payload is None

    def test_mutation_before_validate_is_what_gets_authenticated(self, validator):
        """Mutating before validate() is attacker-controlled input, not a TOCTOU bug: the MAC
        covers the ORIGINAL payload, so a pre-validate mutation correctly fails integrity."""
        m = valid_msg(payload={"reading": 42})
        m.payload["reading"] = 9999
        result = validator.validate(m)
        assert not result.ok and result.stage == "integrity"

    def test_mutation_after_validate_does_not_affect_returned_snapshot(self, validator):
        m = valid_msg(payload={"reading": 42, "nested": {"flag": True}})
        result = validator.validate(m)
        assert result.ok
        snapshot = result.payload

        m.payload["reading"] = -1
        m.payload["nested"]["flag"] = False

        assert snapshot == {"reading": 42, "nested": {"flag": True}}

    def test_mutating_the_snapshot_does_not_affect_the_original_message(self, validator):
        m = valid_msg(payload={"reading": 42})
        snapshot = validator.validate(m).payload
        snapshot["reading"] = -1
        assert m.payload["reading"] == 42

    def test_nested_mutable_structures_cannot_bypass_the_snapshot_boundary(self, validator):
        m = valid_msg(payload={"list": [1, 2, {"deep": "value"}]})
        result = validator.validate(m)
        snapshot = result.payload

        m.payload["list"][0] = 999
        m.payload["list"][2]["deep"] = "tampered"

        assert snapshot == {"list": [1, 2, {"deep": "value"}]}

    def test_downstream_consumer_unaffected_by_mutation_between_validate_and_use(self, validator):
        m = valid_msg(payload={"x": 1})
        result = validator.validate(m)
        assert result.ok
        m.payload["x"] = "attacker-controlled"   # simulated concurrent/later mutation
        assert result.payload == {"x": 1}


# =================================================================================== CONCURRENCY (D10, D12)

class TestConcurrency:
    def test_concurrent_identical_duplicate_only_accepted_once(self, validator):
        """Direct test of the TOCTOU race D10 fixes: many threads validate the SAME message
        (same seq) concurrently. Without an atomic check-then-commit, more than one thread
        could read last_seq as 'not yet seen' before either writes, and accept a duplicate."""
        m = valid_msg(seq=1)
        n_threads = 32
        results = []
        lock = threading.Lock()

        def worker():
            r = validator.validate(m)
            with lock:
                results.append(r)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _: worker(), range(n_threads)))

        accepted = [r for r in results if r.ok]
        rejected_replay = [r for r in results if not r.ok and r.stage == "replay"]

        assert len(accepted) == 1, "exactly one concurrent duplicate must be accepted"
        assert len(rejected_replay) == n_threads - 1
        assert validator.last_seq["device-a"] == 1
        assert validator.stats["accepted"] == 1
        assert validator.stats["replay"] == n_threads - 1

    def test_concurrent_distinct_sequences_no_lost_stat_increments(self, validator):
        """D12: stats counters must not lose increments under concurrent access, even though
        audit I/O now happens outside the lock."""
        n = 100
        msgs = [valid_msg(seq=s) for s in range(1, n + 1)]

        def worker(msg):
            return validator.validate(msg)

        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(worker, msgs))

        # Out-of-order concurrent delivery CAN legitimately cause some of these to be
        # rejected as replay (a lower seq arriving after a higher one has already committed) -
        # that is correct behaviour, not a bug. What must hold unconditionally:
        accepted = sum(1 for r in results if r.ok)
        replayed = sum(1 for r in results if not r.ok and r.stage == "replay")

        assert accepted + replayed == n, "every message must be accounted for exactly once"
        assert validator.stats.get("accepted", 0) == accepted
        assert validator.stats.get("replay", 0) == replayed
        assert validator.last_seq["device-a"] == max(
            m.seq for m, r in zip(msgs, results) if r.ok
        )

    def test_audit_io_does_not_block_other_senders(self, registry, roles):
        """D12: a slow audit sink must not serialise validate() calls for unrelated senders.
        We simulate 'slow' with a small sleep inside record() and check overall wall time is
        much less than n_calls * sleep_time, proving the lock isn't held during audit I/O."""

        class SlowAudit:
            def record(self, *args, **kwargs):
                time.sleep(0.02)

        v = MessageValidator(registry=registry, roles=roles, max_age_s=30.0,
                              clock=lambda: FIXED_NOW, audit=SlowAudit())

        # All of these are rejections (unknown sender) -> each one triggers an audit call.
        msgs = [signed(f"unknown-{i}", 1, FIXED_NOW, "TELEMETRY", {}, key=KEY_A) for i in range(20)]

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=20) as ex:
            list(ex.map(v.validate, msgs))
        elapsed = time.monotonic() - start

        # If the lock were held during audit I/O, this would serialise to ~20 * 0.02s = 0.4s+.
        # Run concurrently with the lock released during I/O, it should be well under that.
        assert elapsed < 0.3, (
            f"validate() calls took {elapsed:.3f}s - audit I/O may be holding the state lock "
            "(D12 regression)"
        )


# =================================================================================== STANDALONE SIGN / VERIFY

class TestStandaloneSignAndVerify:
    def test_sign_then_verify_round_trip(self):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"x": 1})
        signed_m = sign_message(m, KEY_A)
        assert verify_tag(signed_m, KEY_A)

    def test_verify_fails_with_wrong_key(self):
        m = sign_message(SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}), KEY_A)
        assert not verify_tag(m, KEY_B)

    def test_verify_fails_after_tampering(self):
        m = sign_message(SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"x": 1}), KEY_A)
        tampered = m.__class__(**{**m.__dict__, "payload": {"x": 999}})
        assert not verify_tag(tampered, KEY_A)

    def test_verify_never_raises_on_garbage_tag(self):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}, tag=object())  # not even a str
        assert verify_tag(m, KEY_A) is False  # must not raise

    def test_sign_raises_SecureMessageError_for_non_serialisable_payload(self):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"bad": {1, 2, 3}})
        with pytest.raises(SecureMessageError):
            sign_message(m, KEY_A)


class TestSignMessageValidation:
    """E4: sign_message() rejects malformed fields/keys consistently with what the receiver's
    schema check would reject, instead of happily signing something no validator could accept."""

    def test_sign_rejects_malformed_seq(self):
        with pytest.raises(SecureMessageError):
            sign_message(SecureMessage("device-a", True, FIXED_NOW, "TELEMETRY", {}), KEY_A)
        with pytest.raises(SecureMessageError):
            sign_message(SecureMessage("device-a", -1, FIXED_NOW, "TELEMETRY", {}), KEY_A)

    def test_sign_rejects_malformed_timestamp(self):
        with pytest.raises(SecureMessageError):
            sign_message(SecureMessage("device-a", 1, float("nan"), "TELEMETRY", {}), KEY_A)

    def test_sign_rejects_empty_sender_or_msg_type(self):
        with pytest.raises(SecureMessageError):
            sign_message(SecureMessage("", 1, FIXED_NOW, "TELEMETRY", {}), KEY_A)
        with pytest.raises(SecureMessageError):
            sign_message(SecureMessage("device-a", 1, FIXED_NOW, "", {}), KEY_A)

    @pytest.mark.parametrize("bad_key", [b"", "", "not-bytes", None, 12345])
    def test_sign_rejects_invalid_keys(self, bad_key):
        m = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {})
        with pytest.raises(SecureMessageError):
            sign_message(m, bad_key)


class TestVerifyTagInputValidation:
    """E3: verify_tag() must return False, never raise, for any malformed/unsupported input."""

    def test_verify_tag_handles_malformed_message_and_key_without_raising(self):
        valid = SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {})
        assert verify_tag(None, KEY_A) is False
        assert verify_tag("not-a-message", KEY_A) is False
        assert verify_tag(valid, b"") is False
        assert verify_tag(valid, "not-bytes") is False
        assert verify_tag(valid, None) is False

    def test_verify_tag_rejects_non_ascii_tag(self):
        m = sign_message(SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {}), KEY_A)
        bad = m.__class__(**{**m.__dict__, "tag": "ünïcödé" * 9})
        assert verify_tag(bad, KEY_A) is False

    def test_canonical_serialisation_is_order_independent(self):
        s1 = sign_message(SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"b": 2, "a": 1}), KEY_A)
        s2 = sign_message(SecureMessage("device-a", 1, FIXED_NOW, "TELEMETRY", {"a": 1, "b": 2}), KEY_A)
        assert s1.tag == s2.tag