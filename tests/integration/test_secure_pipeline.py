from dataclasses import replace

import src
from src.cybersecurity.command_gateway import (
    CommandSigner,
    SecureCommandGateway,
    StaticCommandPolicy,
    new_authority,
)
from src.cybersecurity.secure_message import MessageValidator
from src.integration.secure_vehicle_adapter import SecureVehicleAdapter
from src.cabinguard.vehicle_simulator import SimulatedVehicle
def test_authorized_simulated_notification():
    key = b"G" * 32  # Clé fictive réservée aux tests

    registry = {
        "guardian-01": {
            "key": key,
            "role": "guardian",
        }
    }

    roles = {
        "guardian": {"VEHICLE_COMMAND"},
    }

    policy = StaticCommandPolicy({
        "NORMAL": ["LOG_EVENT"],
        "VERIFY": ["LOG_EVENT"],
        "ASSIST": ["LOG_EVENT"],
        "MINIMUM_RISK_RESPONSE": [
            "LOG_EVENT",
            "SIMULATE_MINIMUM_RISK_RESPONSE",
            "SIMULATE_NOTIFY",
        ],
    })

    validator = MessageValidator(
        registry,
        roles,
        max_age_s=2.0,
        clock=lambda: 100.0,
    )

    issuer, verifier = new_authority()

    vehicle = SimulatedVehicle()
    adapter = SecureVehicleAdapter(verifier, vehicle)

    gateway = SecureCommandGateway(
        validator=validator,
        actuator=adapter,
        state_provider=lambda: "MINIMUM_RISK_RESPONSE",
        policy=policy,
        issuer=issuer,
    )

    signer = CommandSigner("guardian-01", key)

    message = signer.sign("SIMULATE_NOTIFY", 100.0)
    result = gateway.submit(message, 100.0)

    assert result.accepted is True
    assert result.executed is True
    assert adapter.authorized_commands == ["SIMULATE_NOTIFY"]

    # L'adaptateur ne transmet pas encore la décision au simulateur.
    assert vehicle.event_log == []
def test_critical_command_rejected_in_normal_state():
    key = b"G" * 32  # Clé fictive de test

    registry = {
        "guardian-01": {
            "key": key,
            "role": "guardian",
        }
    }

    roles = {
        "guardian": {"VEHICLE_COMMAND"},
    }

    policy = StaticCommandPolicy({
        "NORMAL": ["LOG_EVENT"],
        "VERIFY": ["LOG_EVENT"],
        "ASSIST": ["LOG_EVENT"],
        "MINIMUM_RISK_RESPONSE": [
            "LOG_EVENT",
            "SIMULATE_MINIMUM_RISK_RESPONSE",
            "SIMULATE_NOTIFY",
        ],
    })

    validator = MessageValidator(
        registry,
        roles,
        max_age_s=2.0,
        clock=lambda: 100.0,
    )

    issuer, verifier = new_authority()
    vehicle = SimulatedVehicle()
    adapter = SecureVehicleAdapter(verifier, vehicle)

    gateway = SecureCommandGateway(
        validator=validator,
        actuator=adapter,
        state_provider=lambda: "NORMAL",
        policy=policy,
        issuer=issuer,
    )

    signer = CommandSigner("guardian-01", key)

    message = signer.sign(
        "SIMULATE_MINIMUM_RISK_RESPONSE", 100.0
    )

    result = gateway.submit(message, 100.0)

    assert result.accepted is False
    assert result.executed is False
    assert result.stage == "command_policy"
    assert adapter.authorized_commands == []
    assert vehicle.event_log == []
from src.cabinguard.models import GuardianDecision


def test_critical_decision_without_authorization_is_inactive():
    _, verifier = new_authority()

    vehicle = SimulatedVehicle()
    adapter = SecureVehicleAdapter(verifier, vehicle)

    decision = GuardianDecision(
        state="MINIMUM_RISK_RESPONSE",
        allow_simulated_response=True,
        reason="Critical risk detected",
        event_type="DROWSINESS",
        score=0.95,
        confidence=0.98,
        timestamp=100.0,
        actions=[
            "LOG_EVENT",
            "SIMULATE_MINIMUM_RISK_RESPONSE",
            "SIMULATE_NOTIFY",
        ],
    )

    safe_decision = adapter.prepare_decision(decision)

    assert safe_decision.allow_simulated_response is False
    assert safe_decision.actions == ["LOG_EVENT"]

    record = vehicle.apply_decision(safe_decision)

    assert record["simulated_response"] is False
    assert record["actions"] == ["LOG_EVENT"]
    assert adapter.authorized_commands == []
def was_authorized(self, command: str) -> bool:
        """Indique si une commande a été acceptée par P3."""
        return command in self.authorized_commands
def apply_gateway_result(self, decision, command, result):
        """Prépare une décision selon le résultat P3 courant."""
        safe_decision = self.prepare_decision(decision)

        if (
            decision.state == "MINIMUM_RISK_RESPONSE"
            and decision.allow_simulated_response is True
            and command == "SIMULATE_MINIMUM_RISK_RESPONSE"
            and command in decision.actions
            and result.accepted is True
            and result.executed is True
        ):
            return replace(
                decision,
                allow_simulated_response=True,
                actions=["LOG_EVENT", command],
            )

        return safe_decision
from src.cybersecurity.command_gateway import CommandResult


def test_rejected_gateway_result_blocks_critical_response():
    _, verifier = new_authority()
    vehicle = SimulatedVehicle()
    adapter = SecureVehicleAdapter(verifier, vehicle)

    decision = GuardianDecision(
        state="MINIMUM_RISK_RESPONSE",
        allow_simulated_response=True,
        reason="Critical risk detected",
        event_type="DROWSINESS",
        score=0.95,
        confidence=0.98,
        timestamp=100.0,
        actions=[
            "LOG_EVENT",
            "SIMULATE_MINIMUM_RISK_RESPONSE",
            "SIMULATE_NOTIFY",
        ],
    )

    rejected_result = CommandResult(
        accepted=False,
        stage="command_policy",
        reason="Command not permitted",
        executed=False,
    )

    safe_decision = adapter.apply_gateway_result(
        decision,
        "SIMULATE_MINIMUM_RISK_RESPONSE",
        rejected_result,
    )

    record = vehicle.apply_decision(safe_decision)

    assert record["simulated_response"] is False
    assert record["actions"] == ["LOG_EVENT"]
    assert adapter.authorized_commands == []
def test_forged_gateway_result_cannot_authorize_response():
    _, verifier = new_authority()
    vehicle = SimulatedVehicle()
    adapter = SecureVehicleAdapter(verifier, vehicle)

    decision = GuardianDecision(
        state="MINIMUM_RISK_RESPONSE",
        allow_simulated_response=True,
        reason="Critical risk detected",
        event_type="DROWSINESS",
        score=0.95,
        confidence=0.98,
        timestamp=100.0,
        actions=[
            "LOG_EVENT",
            "SIMULATE_MINIMUM_RISK_RESPONSE",
            "SIMULATE_NOTIFY",
        ],
    )

    # Fausse autorisation, créée sans passer par P3
    forged_result = CommandResult(
        accepted=True,
        stage="accepted",
        reason="Fake authorization",
        executed=True,
    )

    safe_decision = adapter.apply_gateway_result(
        decision,
        "SIMULATE_MINIMUM_RISK_RESPONSE",
        forged_result,
    )

    assert safe_decision.allow_simulated_response is False
    assert safe_decision.actions == ["LOG_EVENT"]
def test_authorization_cannot_be_used_for_another_command():
    issuer, verifier = new_authority()
    adapter = SecureVehicleAdapter(verifier)

    # P3 autorise uniquement une notification simulée.
    token = issuer.issue(
        "SIMULATE_NOTIFY",
        100.0,
        "guardian-01",
        1,
    )

    assert adapter.execute(token) is True

    # Cette autorisation ne doit pas couvrir une autre commande.
    matching = [
        item for item in adapter._pending_authorizations
        if item["command"] == "SIMULATE_MINIMUM_RISK_RESPONSE"
        and item["timestamp"] == 100.0
    ]

    assert matching == []