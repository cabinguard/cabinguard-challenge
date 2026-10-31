from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import pytest

from src.cybersecurity.command_gateway import (
    AuthorizedCommand,
    AuthorizationIssuer,
    AuthorizationVerifier,
    CommandSigner,
    GatewayConfigError,
    LegacyMockVehicleAdapter,
    SecureCommandGateway,
    SequenceLimitError,
    SimulatedActuator,
    StaticCommandPolicy,
    new_authority,
)
from src.cybersecurity.secure_message import MAX_SEQ, MessageValidator, SecureMessage, sign_message

CommandSequenceError = SequenceLimitError

try:
    from p3_support import ROLES, STATES
except ImportError:
    ROLES = {
        "guardian": {"VEHICLE_COMMAND"},
        "detector": set(),
        "wearable": {"SENSOR_TELEMETRY"},
    }
    STATES = {
        "EMERGENCY": ["SLOW_DOWN", "STOP_SAFELY", "NOTIFY_DRIVER"],
        "NORMAL": ["NOTIFY_DRIVER"],
    }


@dataclass
class _AuditEntry:
    event: str
    stage: str
    sender: str
    reason: str = ""
    extra: str = ""


class RecordingAudit:
    def __init__(self):
        self._entries = []

    def record(self, event, stage, sender, reason="", extra=""):
        self._entries.append(_AuditEntry(event, stage, sender, reason, extra))

    def entries(self):
        return list(self._entries)


def _make_validator(registry, roles, audit=None):
    try:
        return MessageValidator(registry, roles, max_age_s=2.0, clock=lambda: 100.0, audit=audit)
    except TypeError:
        try:
            return MessageValidator(registry, roles, 2.0, clock=lambda: 100.0)
        except TypeError:
            return MessageValidator(registry, roles, 2.0)


@pytest.fixture
def keys():
    return {
        "guardian-01": b"G" * 32,
        "detector-01": b"D" * 32,
        "wearable-01": b"W" * 32,
    }


@pytest.fixture
def registry(keys):
    return {
        "guardian-01": {"key": keys["guardian-01"], "role": "guardian"},
        "detector-01": {"key": keys["detector-01"], "role": "detector"},
        "wearable-01": {"key": keys["wearable-01"], "role": "wearable"},
    }


@pytest.fixture
def audit():
    return RecordingAudit()


@pytest.fixture
def validator(registry, audit):
    return _make_validator(registry, ROLES, audit)


@pytest.fixture
def world(validator, audit):
    issuer, verifier = new_authority()
    act = SimulatedActuator(verifier)
    state = {"s": "EMERGENCY"}
    gw = SecureCommandGateway(
        validator, act, lambda: state["s"], StaticCommandPolicy(STATES), issuer, audit=audit,
    )
    return gw, act, state, verifier


def test_authentic_guardian_command_permitted_by_state_executes(world, keys):
    gw, act, _, _ = world
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0)
    assert r.accepted and r.executed and act.executed == ["SLOW_DOWN"]


def test_policy_blocks_authentic_command_in_wrong_state(world, keys, audit):
    gw, act, state, _ = world
    state["s"] = "NORMAL"
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("STOP_SAFELY", 100.0), 100.0)
    assert not r.accepted and r.stage == "command_policy" and act.executed == []
    assert any(e.event == "COMMAND_REJECTED" and e.stage == "command_policy" for e in audit.entries())


def test_unknown_guardian_state_fails_closed(world, keys):
    gw, act, state, _ = world
    state["s"] = "MADE_UP"
    assert not gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0).accepted


def test_state_provider_failure_fails_closed(validator, keys):
    issuer, verifier = new_authority()
    act = SimulatedActuator(verifier)

    def boom():
        raise RuntimeError("x")

    gw = SecureCommandGateway(validator, act, boom, StaticCommandPolicy(STATES), issuer)
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0)
    assert not r.accepted and r.stage == "state" and act.executed == []


def test_ai_detector_authentic_but_unauthorised(world, keys, audit):
    gw, act, _, _ = world
    m = sign_message(
        SecureMessage("detector-01", 1, 100.0, "VEHICLE_COMMAND", {"cmd": "STOP_SAFELY"}),
        keys["detector-01"],
    )
    r = gw.submit(m, 100.0)
    assert not r.accepted and r.stage == "authorization" and act.executed == []
    assert any(e.event == "MESSAGE_REJECTED" and e.stage == "authorization" for e in audit.entries())


def test_sensor_identity_cannot_command(world, keys):
    gw, act, _, _ = world
    m = sign_message(
        SecureMessage("wearable-01", 1, 100.0, "VEHICLE_COMMAND", {"cmd": "SLOW_DOWN"}),
        keys["wearable-01"],
    )
    assert not gw.submit(m, 100.0).accepted and act.executed == []


def test_replayed_command_rejected(world, keys):
    gw, act, _, _ = world
    m = CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0)
    assert gw.submit(m, 100.0).accepted
    assert gw.submit(m, 100.1).stage == "replay" and act.executed == ["SLOW_DOWN"]


def test_stale_and_tampered_rejected(world, keys):
    gw, act, _, _ = world
    s = CommandSigner("guardian-01", keys["guardian-01"])
    assert gw.submit(s.sign("SLOW_DOWN", 90.0), 100.0).stage == "freshness"
    forged = replace(s.sign("NOTIFY_DRIVER", 100.0), payload={"cmd": "STOP_SAFELY"})
    assert gw.submit(forged, 100.0).stage == "integrity" and act.executed == []


@pytest.mark.parametrize("payload", [{"cmd": "FIRE_ENGINE"}, {"cmd": 7}, {"cmd": "SLOW_DOWN", "extra": 1}, {}])
def test_unknown_or_malformed_commands_rejected(world, keys, payload):
    gw, act, _, _ = world
    m = sign_message(SecureMessage("guardian-01", 1, 100.0, "VEHICLE_COMMAND", payload), keys["guardian-01"])
    r = gw.submit(m, 100.0)
    assert not r.accepted and r.stage == "command_policy" and act.executed == []


def test_actuator_exception_fails_closed(validator, keys):
    issuer, _ = new_authority()

    class Bad:
        def execute(self, a):
            raise RuntimeError("fault")

    gw = SecureCommandGateway(validator, Bad(), lambda: "EMERGENCY", StaticCommandPolicy(STATES), issuer)
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0)
    assert not r.accepted and r.stage == "actuator"


def test_misconfigured_role_table_refused(registry):
    roles = dict(ROLES, detector={"VEHICLE_COMMAND"})
    v = _make_validator(registry, roles)
    issuer, verifier = new_authority()
    with pytest.raises(GatewayConfigError):
        SecureCommandGateway(v, SimulatedActuator(verifier), lambda: "NORMAL", StaticCommandPolicy(STATES), issuer)


def test_actuator_refuses_forged_and_foreign_tokens(world):
    gw, act, _, verifier = world
    forged = AuthorizedCommand("STOP_SAFELY", 1.0, "guardian-01", 1, "0" * 64)
    assert not act.execute(forged) and act.refused == 1
    other_issuer, _ = new_authority()
    assert not act.execute(other_issuer.issue("STOP_SAFELY", 1.0, "guardian-01", 2))
    assert not act.execute("STOP_SAFELY") and act.executed == []


def test_token_is_single_use_and_bound_to_content(world, keys):
    gw, act, _, _ = world
    issuer, verifier = new_authority()
    a = issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 5)
    assert verifier.consume(a) and not verifier.consume(a)
    b = issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 6)
    assert not verifier.consume(replace(b, command="STOP_SAFELY"))
    assert "tag" not in repr(b)


def test_legacy_adapter_asserts_source_only_after_authentication(validator, keys):
    calls = []

    class Veh:
        def execute(self, t, command, source, state):
            calls.append((t, command, source, state))
            return True

    issuer, verifier = new_authority()
    ad = LegacyMockVehicleAdapter(Veh(), verifier, lambda: "EMERGENCY")
    gw = SecureCommandGateway(validator, ad, lambda: "EMERGENCY", StaticCommandPolicy(STATES), issuer)
    assert gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0).executed
    assert calls == [(100, "SLOW_DOWN", "GUARDIAN", "EMERGENCY")]
    assert not ad.execute(AuthorizedCommand("STOP_SAFELY", 1.0, "g", 1, "0" * 64)) and len(calls) == 1


def test_signer_validation_and_repr(keys):
    s = CommandSigner("guardian-01", keys["guardian-01"])
    for bad in ("", None):
        with pytest.raises(ValueError):
            s.sign(bad, 1.0)
    with pytest.raises(ValueError):
        s.sign("X", float("nan"))
    assert s.sign("X", 1.0).seq == 1 and s.sign("X", 1.0).seq == 2
    assert keys["guardian-01"].hex() not in repr(s)


def test_submit_uses_authenticated_payload_snapshot_not_msg_payload(world, keys):
    gw, act, _, _ = world
    m = CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0)
    orig = gw.v.validate

    def mutating_validate(msg, now):
        r = orig(msg, now)
        payload = getattr(msg, "payload", None)
        if isinstance(payload, dict):
            payload.clear()
            payload["cmd"] = "STOP_SAFELY"
        return r

    gw.v.validate = mutating_validate
    r = gw.submit(m, 100.0)
    assert r.accepted and r.executed
    assert act.executed == ["SLOW_DOWN"]


def test_CG1_mutation_after_validate_does_not_change_executed_command(world, keys):
    test_submit_uses_authenticated_payload_snapshot_not_msg_payload(world, keys)


def test_concurrent_consume_accepts_token_once():
    issuer, verifier = new_authority()
    auth = issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 7)
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: verifier.consume(auth), range(64)))
    assert sum(1 for ok in results if ok) == 1
    assert verifier.consume(auth) is False


def test_signer_sequences_unique_under_concurrency(keys, validator):
    s = CommandSigner("guardian-01", keys["guardian-01"])
    n = 50
    with ThreadPoolExecutor(max_workers=8) as ex:
        messages = list(ex.map(lambda i: s.sign("SLOW_DOWN", 100.0), range(n)))
    assert sorted(m.seq for m in messages) == list(range(1, n + 1))
    for m in sorted(messages, key=lambda x: x.seq):
        assert validator.validate(m, 100.0).ok


def test_concurrent_signing_produces_unique_sequence_numbers(keys, validator):
    test_signer_sequences_unique_under_concurrency(keys, validator)


def test_signer_enforces_max_seq(keys):
    s = CommandSigner("guardian-01", keys["guardian-01"], first_seq=MAX_SEQ)
    assert s.sign("SLOW_DOWN", 100.0).seq == MAX_SEQ
    with pytest.raises((SequenceLimitError, CommandSequenceError)):
        s.sign("SLOW_DOWN", 100.1)
    with pytest.raises(ValueError):
        CommandSigner("guardian-01", keys["guardian-01"], first_seq=MAX_SEQ + 1)


def test_sequence_exhaustion_raises_command_sequence_error(keys):
    test_signer_enforces_max_seq(keys)


def test_consumed_token_cannot_replay_after_many_later_tokens():
    issuer, verifier = new_authority()
    first = issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 1)
    assert verifier.consume(first) is True
    for seq in range(2, 64):
        assert verifier.consume(issuer.issue("SLOW_DOWN", 1.0, "guardian-01", seq)) is True
    assert verifier.consume(first) is False
    assert verifier.consume(issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 1)) is False


def test_state_transition_before_actuation_fails_closed(validator, keys):
    issuer, verifier = new_authority()
    act = SimulatedActuator(verifier)
    calls = {"n": 0}

    def state():
        calls["n"] += 1
        return "EMERGENCY" if calls["n"] == 1 else "NORMAL"

    gw = SecureCommandGateway(validator, act, state, StaticCommandPolicy(STATES), issuer)
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("STOP_SAFELY", 100.0), 100.0)
    assert not r.accepted and r.stage == "command_policy"
    assert act.executed == []
    assert calls["n"] >= 2


def test_ARCHITECTURAL_GAP_token_has_no_state_binding_for_a_deferred_actuator():
    issuer, verifier = new_authority()
    act = SimulatedActuator(verifier)
    token = issuer.issue("STOP_SAFELY", 100.0, "guardian-01", 1)
    assert act.execute(token) is True
    assert act.executed == ["STOP_SAFELY"]


def test_issuer_verifier_reject_short_secrets_and_bad_memory():
    with pytest.raises(ValueError):
        AuthorizationIssuer(b"short")
    with pytest.raises(ValueError):
        AuthorizationVerifier(b"short")
    with pytest.raises(ValueError):
        AuthorizationVerifier(b"K" * 32, memory=0)


def test_policy_rejects_malformed_and_ignores_caller_mutation():
    with pytest.raises(GatewayConfigError):
        StaticCommandPolicy({})
    with pytest.raises(GatewayConfigError):
        StaticCommandPolicy({"EMERGENCY": "SLOW_DOWN"})
    with pytest.raises(GatewayConfigError):
        StaticCommandPolicy({"EMERGENCY": ["SLOW_DOWN", ""]})
    table = {"EMERGENCY": ["SLOW_DOWN"]}
    p = StaticCommandPolicy(table)
    table["EMERGENCY"] = ["STOP_SAFELY"]
    table["HACKED"] = ["LAUNCH"]
    assert p.check("STOP_SAFELY", "EMERGENCY")[0] is False
    assert p.check("SLOW_DOWN", "HACKED")[0] is False
    assert p.check("SLOW_DOWN", "EMERGENCY")[0] is True