"""Interface fictive de véhicule : journalise, ne commande aucun matériel réel."""
from dataclasses import asdict
from .models import GuardianDecision, VehicleContext


class SimulatedVehicle:
    def __init__(self, initial_speed_kmh: float = 0.0):
        self.context = VehicleContext(speed_kmh=initial_speed_kmh, driving_context="SIMULATED")
        self.event_log: list[dict] = []

    def update_context(self, speed_kmh: float, driving_context: str, timestamp: float):
        try:
            speed = float(speed_kmh)
        except (TypeError, ValueError):
            speed = 0.0
        if speed != speed or speed in (float("inf"), float("-inf")):   # NaN / inf
            speed = 0.0
        self.context = VehicleContext(
            speed_kmh=max(0.0, speed),
            driving_context=driving_context,
            vehicle_available=True,
            timestamp=timestamp
        )

    def apply_decision(self, decision: GuardianDecision) -> dict:
        record = {
            "timestamp": decision.timestamp,
            "state": decision.state,
            "event_type": decision.event_type,
            "simulated_response": decision.allow_simulated_response,
            "actions": decision.actions,
            "reason": decision.reason,
            "vehicle_speed_kmh_before": self.context.speed_kmh,
            "note": "SIMULATION ONLY — no real vehicle command sent"
        }
        # Deliberately do not change speed/steering/brakes: this is a safe event simulator.
        self.event_log.append(record)
        return record

    def summary(self) -> dict:
        return {"context": asdict(self.context), "events_logged": len(self.event_log), "events": self.event_log}
