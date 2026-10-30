"""Bloc 7 — état de santé des capteurs, communications, traitement et interface véhicule."""
from dataclasses import dataclass, field
from math import isfinite

@dataclass
class HealthReport:
    system_health_state: str
    sensor_reliability: dict
    faults: list[str] = field(default_factory=list)
    fusion_weights: dict = field(default_factory=dict)
    critical_path_available: bool = True

class SystemSensorHealthManager:
    """Les champs absents sont indisponibles, jamais interprétés comme une mesure normale."""
    def evaluate(self, availability: dict | None = None, quality: dict | None = None,
                 communication: dict | None = None, processing: dict | None = None,
                 vehicle_interface: bool = True) -> HealthReport:
        availability = availability or {}; quality = quality or {}
        communication = communication or {}; processing = processing or {}
        modalities = ("camera", "fsr", "physiology", "microphone", "context")
        reliability, faults, weights = {}, [], {}
        available_count = 0
        for name in modalities:
            avail = availability.get(name, False) is True
            q = quality.get(name, 1.0 if avail else 0.0)
            try: q = float(q)
            except (TypeError, ValueError): q = 0.0
            if not isfinite(q) or not 0 <= q <= 1: q = 0.0
            r = q if avail else 0.0
            reliability[name] = round(r, 3); weights[name] = round(r, 3)
            if avail: available_count += 1
            else: faults.append(f"{name}_unavailable")
            if avail and r < 0.5: faults.append(f"{name}_low_quality")
        for group_name, group in (("communication", communication), ("processing", processing)):
            for name, ok in group.items():
                if ok is not True: faults.append(f"{group_name}_{name}_unavailable")
        if vehicle_interface is not True: faults.append("vehicle_interface_unavailable")
        core_failure = (available_count == 0 or vehicle_interface is not True or
                        any(v is not True for v in processing.values()))
        if core_failure: state = "SAFE FALLBACK"
        elif len(faults) == 0: state = "FULL"
        elif available_count >= 3: state = "DEGRADED"
        else: state = "LIMITED"
        total = sum(weights.values())
        if total > 0: weights = {k: round(v/total, 4) for k, v in weights.items()}
        else: weights = {k: 0.0 for k in weights}
        return HealthReport(state, reliability, faults, weights, not core_failure)
