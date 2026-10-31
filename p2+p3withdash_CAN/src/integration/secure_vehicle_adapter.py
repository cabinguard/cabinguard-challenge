"""Secure adapter between P3 and the P2 vehicle simulator."""

from dataclasses import replace

from src.cybersecurity.command_gateway import (
    AuthorizedCommand,
    AuthorizationVerifier,
)


class SecureVehicleAdapter:
    """Accept authenticated P3 commands; fail closed by default."""

    def __init__(self, verifier: AuthorizationVerifier, vehicle=None):
        self._verifier = verifier
        self._vehicle = vehicle
        self.authorized_commands = []
        self.refused_commands = 0
        self._pending_authorizations = []

    def execute(self, auth: AuthorizedCommand) -> bool:
        if not self._verifier.consume(auth):
            self.refused_commands += 1
            return False

        self.authorized_commands.append(auth.command)
        self._pending_authorizations.append({
            "command": auth.command,
            "timestamp": auth.ts,
            "sender_id": auth.sender_id,
            "sequence": auth.seq,
        })
        return True

    def prepare_decision(self, decision):
        """Disable simulated responses unless explicitly authorized."""
        return replace(
            decision,
            allow_simulated_response=False,
            actions=["LOG_EVENT"],
        )

    def apply_gateway_result(self, decision, command, result):
        """A CommandResult alone is never proof of authorization."""
        return self.prepare_decision(decision)