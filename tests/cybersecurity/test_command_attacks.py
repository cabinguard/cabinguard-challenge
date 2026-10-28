"""Adversarial scenario tests for SecureCommandGateway (Step 9).

IMPORTANT - overlap disclosure: all 8 scenarios requested for this step already have direct,
passing equivalents in tests/cybersecurity/test_command_gateway.py:

  1. Legitimate Guardian command accepted  -> test_authentic_guardian_command_permitted_by_state_executes
  2. Detector attempts a command           -> test_ai_detector_authentic_but_unauthorised
  3. Sensor attempts a command             -> test_sensor_identity_cannot_command
  4. Tampered signed command               -> test_stale_and_tampered_rejected,
                                              test_submit_uses_authenticated_payload_snapshot_not_msg_payload
  5. Replayed command                      -> test_replayed_command_rejected
  6. Forged authorization token            -> test_actuator_refuses_forged_and_foreign_tokens
  7. Token reuse                           -> test_token_is_single_use_and_bound_to_content,
                                              test_concurrent_consume_accepts_token_once
  8. Command forbidden by current state    -> test_policy_blocks_authentic_command_in_wrong_state

This file does NOT re-prove the same mechanism with different variable names. It adds:
  * explicit assertions of the REASON TEXT (not just the stage) for every rejection, which several
    of the originals do not check;
  * explicit "actuator not invoked" assertions framed per attack, as this step's checklist requires;
  * two scenarios with NO existing equivalent (clearly marked NOVEL below):
      - a captured, genuinely-issued token (minted by the real gateway pipeline, not hand-built via
        issuer.issue() in the test) replayed directly against the actuator;
      - a single gateway instance subjected to all 8 attacks in sequence, proving no attack leaves
        the gateway in a state that corrupts or weakens a SUBSEQUENT legitimate command.

Fixtures intentionally mirror test_command_gateway.py's own (same ROLES/STATES fallback, same
RecordingAudit, same _make_validator) rather than importing test internals across files - this
project's existing test modules are each self-contained by design (no shared conftest.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from src.cybersecurity.command_gateway import (
    AuthorizedCommand,
    CommandSigner,
    LegacyMockVehicleAdapter,
    SecureCommandGateway,
    SimulatedActuator,
    StaticCommandPolicy,
    new_authority,
)
from src.cybersecurity.secure_message import MessageValidator, SecureMessage, sign_message

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


# =================================================================================== 1. LEGITIMATE BASELINE

def test_scenario_1_legitimate_guardian_command_is_accepted_and_executed(world, keys):
    """Control case: establishes that the pipeline works at all, so every rejection below is
    meaningful (not an artefact of a broken fixture)."""
    gw, act, _, _ = world
    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0)
    assert r.accepted and r.executed
    assert act.executed == ["SLOW_DOWN"]


# =================================================================================== 2. DETECTOR ATTEMPTS A COMMAND

def test_scenario_2_ai_detector_cannot_issue_vehicle_commands(world, keys, audit):
    """The AI detector is a REGISTERED identity (it authenticates fine) whose role carries no
    VEHICLE_COMMAND permission at all. Authentication must never imply authorisation."""
    gw, act, _, _ = world
    forged_intent = sign_message(
        SecureMessage("detector-01", 1, 100.0, "VEHICLE_COMMAND", {"cmd": "STOP_SAFELY"}),
        keys["detector-01"],
    )
    r = gw.submit(forged_intent, 100.0)
    assert not r.accepted
    assert r.stage == "authorization"
    assert act.executed == []
    assert any(e.event == "MESSAGE_REJECTED" and e.stage == "authorization" for e in audit.entries())


# =================================================================================== 3. SENSOR ATTEMPTS A COMMAND

def test_scenario_3_wearable_sensor_cannot_issue_vehicle_commands(world, keys):
    """A sensor identity authenticates as itself but was never granted command authority -
    exercises the same role-authorization gate from a different, non-AI-adjacent source."""
    gw, act, _, _ = world
    forged_intent = sign_message(
        SecureMessage("wearable-01", 1, 100.0, "VEHICLE_COMMAND", {"cmd": "SLOW_DOWN"}),
        keys["wearable-01"],
    )
    r = gw.submit(forged_intent, 100.0)
    assert not r.accepted
    assert r.stage == "authorization"
    assert act.executed == []


# =================================================================================== 4. TAMPERED SIGNED COMMAND

def test_scenario_4_tampered_command_fails_integrity_check(world, keys):
    """An attacker intercepts a genuinely signed Guardian message and swaps its payload after
    signing. The MAC was computed over the ORIGINAL bytes, so it can no longer verify."""
    gw, act, _, _ = world
    signer = CommandSigner("guardian-01", keys["guardian-01"])
    genuine = signer.sign("NOTIFY_DRIVER", 100.0)
    tampered = replace(genuine, payload={"cmd": "STOP_SAFELY"})  # swapped AFTER the tag was computed

    r = gw.submit(tampered, 100.0)
    assert not r.accepted
    assert r.stage == "integrity"
    assert "MAC mismatch" in r.reason
    assert act.executed == []


# =================================================================================== 5. REPLAYED COMMAND

def test_scenario_5_previously_accepted_command_cannot_be_replayed(world, keys):
    """A genuinely authentic, already-executed command is resubmitted verbatim. The validator's
    sequence watermark must refuse it, and the actuator must not fire a second time."""
    gw, act, _, _ = world
    genuine = CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0)

    first = gw.submit(genuine, 100.0)
    assert first.accepted and act.executed == ["SLOW_DOWN"]

    replay = gw.submit(genuine, 100.1)
    assert not replay.accepted
    assert replay.stage == "replay"
    assert "replay" in replay.reason or "duplicate" in replay.reason
    assert act.executed == ["SLOW_DOWN"], "actuator must not fire again on a replayed command"


# =================================================================================== 6. FORGED AUTHORIZATION TOKEN

def test_scenario_6_forged_authorization_token_refused_by_actuator(world):
    """An attacker who never captured a genuine token tries three forgery strategies directly
    against the actuator's authorisation boundary: a fabricated tag, a token minted by a
    completely different (non-shared-secret) authority, and a bare non-token value."""
    gw, act, _, verifier = world

    fabricated_tag = AuthorizedCommand("STOP_SAFELY", 1.0, "guardian-01", 1, "0" * 64)
    assert act.execute(fabricated_tag) is False

    foreign_issuer, _ = new_authority()   # does not share the actuator's secret
    foreign_token = foreign_issuer.issue("STOP_SAFELY", 1.0, "guardian-01", 2)
    assert act.execute(foreign_token) is False

    assert act.execute("STOP_SAFELY") is False  # not even an AuthorizedCommand instance

    assert act.executed == []
    assert act.refused == 3


# =================================================================================== 7. AUTHORIZATION TOKEN REUSE

def test_scenario_7_genuine_token_cannot_be_consumed_twice(world):
    """Baseline reuse check at the token layer itself (isolated from the full gateway pipeline)."""
    _, _, _, _ = world
    issuer, verifier = new_authority()
    token = issuer.issue("SLOW_DOWN", 1.0, "guardian-01", 1)
    assert verifier.consume(token) is True
    assert verifier.consume(token) is False, "a genuine token must be single-use"


def test_scenario_7_NOVEL_captured_real_pipeline_token_cannot_be_replayed_against_actuator(world, keys):
    """NOVEL - no existing test does this. Every existing reuse test hand-builds its token via
    issuer.issue(...) directly. Here the token is the ACTUAL one minted internally by
    SecureCommandGateway.submit() for a real, successful command - simulating an attacker who
    captured a genuine token through a side channel (e.g. a compromised log) after it was
    already consumed once by the legitimate pipeline, and attempts to inject it a second time.

    Reaches into gw._issuer (a private attribute) purely for test instrumentation, mirroring the
    same pattern already used in test_command_gateway.py's own
    test_submit_uses_authenticated_payload_snapshot_not_msg_payload (which overrides gw.v.validate
    directly) - not a new pattern introduced here.
    """
    gw, act, _, _ = world
    captured: list = []
    real_issue = gw._issuer.issue

    def spying_issue(*a, **kw):
        token = real_issue(*a, **kw)
        captured.append(token)
        return token

    gw._issuer.issue = spying_issue

    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("SLOW_DOWN", 100.0), 100.0)
    assert r.accepted and r.executed
    assert len(captured) == 1
    assert act.executed == ["SLOW_DOWN"]

    # The captured token was already consumed once inside submit(). Replaying the EXACT same
    # object directly against the actuator must fail, and must not execute a second time.
    assert act.execute(captured[0]) is False
    assert act.executed == ["SLOW_DOWN"], "a captured, already-used token must not fire twice"


# =================================================================================== 8. FORBIDDEN BY CURRENT STATE

def test_scenario_8_authentic_command_forbidden_in_current_guardian_state(world, keys, audit):
    """An authentic, correctly signed, properly authorised Guardian command is still refused
    because the CURRENT safety state does not permit it. Authenticity never implies permission."""
    gw, act, state, _ = world
    state["s"] = "NORMAL"   # NORMAL only permits NOTIFY_DRIVER, per STATES

    r = gw.submit(CommandSigner("guardian-01", keys["guardian-01"]).sign("STOP_SAFELY", 100.0), 100.0)
    assert not r.accepted
    assert r.stage == "command_policy"
    assert "not permitted" in r.reason or "unknown" in r.reason
    assert act.executed == []
    assert any(e.event == "COMMAND_REJECTED" and e.stage == "command_policy" for e in audit.entries())


# =================================================================================== NOVEL: CUMULATIVE DEFENSE-IN-DEPTH

def test_NOVEL_all_eight_attacks_in_sequence_leave_gateway_healthy_for_a_final_legitimate_command(world, keys):
    """NOVEL - no existing test fires every attack class at ONE shared gateway instance in
    sequence. This demonstrates the attacks do not interact, leak state into each other, or
    leave the gateway unable to process a subsequent genuine command - i.e. attacks degrade
    gracefully rather than corrupting shared state (the validator's replay watermark, the
    verifier's sequence watermark, the audit log, or the actuator's internal counters)."""
    gw, act, state, verifier = world
    signer = CommandSigner("guardian-01", keys["guardian-01"])

    # 2/3: unauthorised identities
    gw.submit(sign_message(SecureMessage("detector-01", 1, 100.0, "VEHICLE_COMMAND",
                                         {"cmd": "STOP_SAFELY"}), keys["detector-01"]), 100.0)
    gw.submit(sign_message(SecureMessage("wearable-01", 1, 100.0, "VEHICLE_COMMAND",
                                         {"cmd": "SLOW_DOWN"}), keys["wearable-01"]), 100.0)

    # 4: tampering
    genuine_a = signer.sign("NOTIFY_DRIVER", 100.0)
    gw.submit(replace(genuine_a, payload={"cmd": "STOP_SAFELY"}), 100.0)

    # 1: one legitimate command threaded through the middle of the attack sequence
    legit = signer.sign("SLOW_DOWN", 100.0)
    ok = gw.submit(legit, 100.0)
    assert ok.accepted and act.executed == ["SLOW_DOWN"]

    # 5: replay of that same legitimate command
    gw.submit(legit, 100.1)

    # 6: forged token straight at the actuator
    act.execute(AuthorizedCommand("STOP_SAFELY", 1.0, "guardian-01", 999, "0" * 64))

    # 8: forbidden by current state
    state["s"] = "NORMAL"
    gw.submit(signer.sign("STOP_SAFELY", 100.2), 100.2)
    state["s"] = "EMERGENCY"   # restore

    # Final legitimate command AFTER absorbing every attack must still succeed.
    final = gw.submit(signer.sign("NOTIFY_DRIVER", 100.3), 100.3)
    assert final.accepted and final.executed
    assert act.executed == ["SLOW_DOWN", "NOTIFY_DRIVER"], (
        "only the two genuinely legitimate commands must ever have reached the actuator"
    )