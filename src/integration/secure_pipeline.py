"""P2-P3 secure integration coordinator (simulation only)."""


from src.cybersecurity.command_gateway import AuthorizedCommand

from dataclasses import replace
class SecurePipelineCoordinator:
    """Coordinates safety decisions and authenticated commands."""

    def __init__(self, gateway, adapter, signer):
        self.gateway = gateway
        self.adapter = adapter
        self.signer = signer
        self._pending_decision = None
        self._authorized_decision = None
    def process_decision(self, decision, now=None):
        """Default to a safe decision until P3 authorization is verified."""
        safe_decision = self.adapter.prepare_decision(decision)

        if (
            decision.state != "MINIMUM_RISK_RESPONSE"
            or not decision.allow_simulated_response
        ):
            return safe_decision

        command = "SIMULATE_MINIMUM_RISK_RESPONSE"

        # Only request commands explicitly permitted by this decision.
        if command not in decision.actions:
            return safe_decision

        timestamp = float(decision.timestamp)

        # Register the current P2 decision before contacting P3.
        if not self.adapter.begin_decision(decision, command):
            return safe_decision

        try:
            # Sign the command as the Guardian.
            message = self.signer.sign(command, timestamp)

            # P3 validates authentication, replay protection,
            # permissions, and the current Guardian state.
            result = self.gateway.submit(
                message,
                timestamp if now is None else now,
            )

            # A gateway result alone cannot authorize execution.
            # The adapter must have received the verified token.
            if (
                result.accepted is True
                and result.executed is True
                and self.adapter._authorized_decision is decision
            ):
                return replace(
                    decision,
                    allow_simulated_response=True,
                    actions=["LOG_EVENT", command],
                )

            return safe_decision

        except Exception:
            return safe_decision

        finally:
            # Never leave authorization state pending.
            self.adapter._pending_decision = None
            self.adapter._authorized_decision = None

        # Create an authenticated Guardian command.
        message = self.signer.sign(command, timestamp)

        # Submit it to P3 for security and policy verification.
        result = self.gateway.submit(
            message,
            timestamp if now is None else now,
        )

        # A successful result is not sufficient proof that the
        # authorization belongs to this exact decision.
        # Remain fail-closed until decision binding is implemented.
        return safe_decision
    def begin_decision(self, decision, command):
        """Register one pending decision before gateway submission."""
        self._pending_decision = None
        self._authorized_decision = None

        if (
            decision.state != "MINIMUM_RISK_RESPONSE"
            or decision.allow_simulated_response is not True
            or command not in decision.actions
            or command != "SIMULATE_MINIMUM_RISK_RESPONSE"
        ):
            return False

        self._pending_decision = (decision, command)
        return True
    def execute(self, auth: AuthorizedCommand) -> bool:
            """Accept a P3 token only for the pending Guardian decision."""

            pending = self._pending_decision

            if pending is None:
                self.refused_commands += 1
                return False

            decision, command = pending

            if (
                auth.command != command
                or auth.ts != float(decision.timestamp)
                or decision.state != "MINIMUM_RISK_RESPONSE"
                or decision.allow_simulated_response is not True
                or command not in decision.actions
            ):
                self.refused_commands += 1
                return False

            if not self._verifier.consume(auth):
                self.refused_commands += 1
                return False

            # The token is authentic, single-use, and matches the
            # pending decision's command and timestamp.
            self._authorized_decision = decision
            self._pending_decision = None

            self.authorized_commands.append(auth.command)

            self._pending_authorizations.append({
                "command": auth.command,
                "timestamp": auth.ts,
                "sender_id": auth.sender_id,
                "sequence": auth.seq,
            })

            return True