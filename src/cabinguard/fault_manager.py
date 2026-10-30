"""Bibliothèque de scénarios de défaut : décrit des pannes et construit les entrées correspondantes."""
from dataclasses import fields
from .models import SensorStatus, SecurityStatus, VehicleContext

FAULTS = {
    "camera_unavailable": {"camera_available": False},
    "bracelet_absent": {"physiology_available": False},
    "stale_message": {"message_fresh": False},
    "tampered_message": {"integrity_valid": False},
    "unauthenticated_sender": {"sender_authenticated": False},
    "unauthorized_command": {"command_authorized": False},
    "replay_or_sequence_error": {"sequence_valid": False},
    "all_sensors_missing": {"camera_available": False, "fsr_available": False, "physiology_available": False},
    "vehicle_unavailable": {"vehicle_available": False},
}

# Une panne doit-elle empêcher la réponse simulée critique ? (politique de démonstration)
# L'absence du bracelet seule ne bloque pas : ce n'est pas une urgence (voir README).
FAULT_BLOCKS_RESPONSE = {name: name != "bracelet_absent" for name in FAULTS}


def fault_scenario(name: str) -> dict:
    """Retourne les champs modifiés par la panne `name`."""
    if name not in FAULTS:
        raise ValueError(f"Scénario inconnu: {name}. Choix: {', '.join(FAULTS)}")
    return dict(FAULTS[name])


def build_faulted_inputs(name: str, timestamp: float = 0.0):
    """Construit (SensorStatus, SecurityStatus, VehicleContext) sains, puis applique la panne."""
    overrides = fault_scenario(name)
    sensors = SensorStatus(camera_available=True, fsr_available=True, physiology_available=True, timestamp=timestamp)
    security, vehicle = SecurityStatus(), VehicleContext(speed_kmh=50.0, vehicle_available=True, timestamp=timestamp)
    for obj in (sensors, security, vehicle):
        names = {f.name for f in fields(obj)}
        for key, value in overrides.items():
            if key in names:
                setattr(obj, key, value)
    return sensors, security, vehicle
