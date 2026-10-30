"""Pipeline Personne 2 : blocs 5/6/7 raccordés au Safety Guardian et simulateur."""
from dataclasses import replace
from .models import DetectionResult, SensorStatus, SecurityStatus
from .safety_guardian import SafetyGuardian
from .vehicle_simulator import SimulatedVehicle
from .temporal_analysis import TemporalAnalyzer
from .digital_twin import DriverDigitalTwin
from .system_health import SystemSensorHealthManager


class CabinGuardPipeline:
    def __init__(self, config_path: str | None = None, secure_adapter=None):
        self.guardian = SafetyGuardian(config_path)
        self.vehicle = SimulatedVehicle()
        self.secure_adapter = secure_adapter
        self.secure_coordinator = None
        self.temporal = TemporalAnalyzer(window_seconds=5.0, min_persistence_seconds=2.0)
        self.digital_twin = DriverDigitalTwin(min_calibration_samples=10)
        self.health_manager = SystemSensorHealthManager()

    def process(self, detection: DetectionResult, sensors: SensorStatus,
                security: SecurityStatus, speed_kmh: float = 50.0,
                driving_context: str = "URBAN_SIMULATION",
                features: dict | None = None) -> dict:
        """`features` contient les features numériques normalisées des autres modèles (optionnel)."""
        self.vehicle.update_context(speed_kmh, driving_context, detection.timestamp)

        availability = {
            "camera": sensors.camera_available is True,
            "fsr": sensors.fsr_available is True,
            "physiology": sensors.physiology_available is True,
            "microphone": sensors.microphone_available is True,
            "context": sensors.context_available is True and self.vehicle.context.vehicle_available is True,
        }
        health = self.health_manager.evaluate(
            availability=availability, quality=sensors.quality,
            communication=sensors.communication_status,
            processing=sensors.processing_status,
            vehicle_interface=sensors.vehicle_interface_available)

        temporal = self.temporal.update(detection)
        # Le score temporel est une métadonnée d'appui, pas une probabilité médicale.
        # La persistance fournie par Personne 1 reste prise en compte si elle est supérieure.
        effective_detection = replace(
            detection,
            persistence_seconds=max(detection.persistence_seconds, temporal["persistence_seconds"]),
            data_quality=min(detection.data_quality, _mean_reliability(health.sensor_reliability))
            if sensors.quality else detection.data_quality,
        )
        twin = self.digital_twin.update(
            features or {}, event_type=detection.event_type,
            data_quality=effective_detection.data_quality,
            allow_adaptation=(detection.event_type == "NORMAL" and health.system_health_state == "FULL"))

        # Une panne de l'interface véhicule est reportée explicitement au Guardian.
        vehicle_context = self.vehicle.context
        if sensors.vehicle_interface_available is not True:
            vehicle_context.vehicle_available = False

        decision = self.guardian.evaluate(
            effective_detection, sensors, security, vehicle_context
        )

        # Default to a safe, logging-only decision.
        safe_decision = replace(
            decision,
            allow_simulated_response=False,
            actions=["LOG_EVENT"],
        )

        # Critical simulated responses require the P3 coordinator.
        if self.secure_coordinator is not None:
            try:
                safe_decision = self.secure_coordinator.process_decision(
                    decision,
                    now=detection.timestamp,
                )
            except Exception:
                pass  # Fail closed: keep the logging-only decision.

        record = self.vehicle.apply_decision(safe_decision)
        record.update({
            "degraded_mode": decision.degraded_mode or health.system_health_state != "FULL",
            "rejected_checks": decision.rejected_checks,
            "temporal_analysis": temporal,
            "digital_twin": twin,
            "system_health": {
                "state": health.system_health_state,
                "sensor_reliability": health.sensor_reliability,
                "fusion_weights": health.fusion_weights,
                "faults": health.faults,
                "critical_path_available": health.critical_path_available,
            },
        })
        return record


def _mean_reliability(values: dict) -> float:
    vals = list(values.values())
    return sum(vals) / len(vals) if vals else 1.0
