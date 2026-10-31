"""Safety Guardian : contrôle des conditions avant toute réponse simulée.

Principe : fail-closed. Toute valeur absente, mal typée, non finie ou hors plage
est rejetée et ne peut jamais augmenter l'autorité du système. Les booléens de
disponibilité / sécurité doivent être *exactement* ``True`` (une chaîne « False »
est truthy en Python : elle ne doit pas passer pour un « oui »).
"""
import json
import math
import numbers
from pathlib import Path
from .models import DetectionResult, SensorStatus, SecurityStatus, VehicleContext, GuardianDecision
from .state_machine import choose_state


DEFAULT_CONFIG = {
    "minimum_event_score": 0.80,
    "minimum_confidence": 0.70,
    "minimum_data_quality": 0.70,
    "minimum_persistence_seconds": 3.0,
    "minimum_cross_modal_agreement": 0.60,
    "require_integrity_valid": True,
    "require_message_fresh": True,
    "require_sender_authenticated": True,
    "require_command_authorized": True,
    "allow_simulated_response_only": True,
    # Noms d'événements à confirmer avec la Personne 1 (voir docs/INTERFACES.md §7).
    "known_event_types": [
        "NORMAL",
        "STROKE_LIKE_PATTERN",
        "SEIZURE_LIKE_MOTION_PATTERN",
        "CARDIAC_EVENT_LIKE_PATTERN",
    ],
}

_UNIT_KEYS = ("minimum_event_score", "minimum_confidence",
              "minimum_data_quality", "minimum_cross_modal_agreement")
SECURITY_CHECKS = ("integrity_invalid", "stale_message", "sender_not_authenticated",
                   "command_not_authorized", "sequence_invalid")


def _is_number(value) -> bool:
    """Vrai pour un réel fini (int, float, numpy float...) ; faux pour bool, NaN, inf, None, str."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _in_unit_range(value) -> bool:
    return _is_number(value) and 0.0 <= value <= 1.0


def validate_config(cfg: dict) -> dict:
    """Refuse une configuration incohérente plutôt que de l'appliquer silencieusement."""
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Clés de configuration inconnues (faute de frappe ?) : {sorted(unknown)}")
    for key in _UNIT_KEYS:
        if not _in_unit_range(cfg[key]):
            raise ValueError(f"{key} doit être un nombre dans [0, 1], reçu : {cfg[key]!r}")
    if not _is_number(cfg["minimum_persistence_seconds"]) or cfg["minimum_persistence_seconds"] < 0:
        raise ValueError("minimum_persistence_seconds doit être un nombre >= 0")
    for key, value in cfg.items():
        if key.startswith(("require_", "allow_")) and not isinstance(value, bool):
            raise ValueError(f"{key} doit être un booléen, reçu : {value!r}")
    if cfg["allow_simulated_response_only"] is not True:
        raise ValueError("Phase 1 : seules les réponses simulées existent ; "
                         "allow_simulated_response_only doit rester true.")
    types = cfg["known_event_types"]
    if not (isinstance(types, list) and all(isinstance(t, str) for t in types) and "NORMAL" in types):
        raise ValueError("known_event_types doit être une liste de chaînes contenant 'NORMAL'")
    return cfg


class SafetyGuardian:
    def __init__(self, config_path: str | None = None):
        self.config = DEFAULT_CONFIG.copy()
        if config_path:
            path = Path(config_path)
            self.config.update(json.loads(path.read_text(encoding="utf-8")))
        validate_config(self.config)

    def evaluate(self, detection: DetectionResult, sensors: SensorStatus,
                 security: SecurityStatus, vehicle: VehicleContext) -> GuardianDecision:
        rejected = []
        cfg = self.config

        # Valeurs numériques : une valeur mal formée ne doit jamais augmenter l'autorité.
        for name, value in [
            ("event_score", detection.event_score),
            ("confidence", detection.confidence),
            ("data_quality", detection.data_quality),
            ("cross_modal_agreement", detection.cross_modal_agreement),
        ]:
            if not _in_unit_range(value):
                rejected.append(f"{name}_out_of_range")

        if (not _is_number(detection.timestamp) or not _is_number(detection.persistence_seconds)
                or detection.timestamp < 0 or detection.persistence_seconds < 0):
            rejected.append("invalid_timestamp_or_persistence")

        if detection.event_type not in cfg["known_event_types"]:
            rejected.append("unknown_event_type")

        camera = sensors.camera_available is True
        fsr = sensors.fsr_available is True
        physiology = sensors.physiology_available is True
        sensors_available = camera or fsr or physiology
        vehicle_available = vehicle.vehicle_available is True

        # Politique de démonstration : la caméra est requise pour autoriser une réponse critique simulée.
        if not camera:
            rejected.append("camera_unavailable")
        if not sensors_available:
            rejected.append("no_sensor_available")

        if cfg["require_integrity_valid"] and security.integrity_valid is not True:
            rejected.append("integrity_invalid")
        if cfg["require_message_fresh"] and security.message_fresh is not True:
            rejected.append("stale_message")
        if cfg["require_sender_authenticated"] and security.sender_authenticated is not True:
            rejected.append("sender_not_authenticated")
        if cfg["require_command_authorized"] and security.command_authorized is not True:
            rejected.append("command_not_authorized")
        if security.sequence_valid is not True:
            rejected.append("sequence_invalid")

        event_present = (
            detection.event_type != "NORMAL"
            and _in_unit_range(detection.event_score)
            and detection.event_score >= cfg["minimum_event_score"]
        )
        evidence_sufficient = (
            not rejected
            and detection.confidence >= cfg["minimum_confidence"]
            and detection.data_quality >= cfg["minimum_data_quality"]
            and detection.persistence_seconds >= cfg["minimum_persistence_seconds"]
            and detection.cross_modal_agreement >= cfg["minimum_cross_modal_agreement"]
        )
        security_ok = not any(x in rejected for x in SECURITY_CHECKS)

        # Le contexte véhicule n'est PAS mélangé aux preuves : sinon la branche ASSIST
        # de la machine à états serait inatteignable (VERIFY serait toujours choisi avant).
        state, reason = choose_state(
            event_present=event_present,
            evidence_sufficient=evidence_sufficient,
            security_ok=security_ok,
            vehicle_available=vehicle_available,
            sensors_available=sensors_available,
        )

        rejected_out = list(rejected)
        if not vehicle_available:
            rejected_out.append("vehicle_context_unavailable")

        allow = state == "MINIMUM_RISK_RESPONSE"      # Phase 1 : réponse simulée uniquement
        if rejected_out:
            reason += " Contrôles échoués : " + ", ".join(rejected_out) + "."
        if not physiology:
            reason += " Bracelet physiologique absent : il n'est pas considéré comme une urgence en soi."
        return GuardianDecision(
            state=state,
            allow_simulated_response=allow,
            reason=reason,
            event_type=detection.event_type,
            score=float(detection.event_score) if _is_number(detection.event_score) else 0.0,
            confidence=float(detection.confidence) if _is_number(detection.confidence) else 0.0,
            timestamp=detection.timestamp,
            rejected_checks=rejected_out,
            degraded_mode=not (camera and fsr and physiology),
            actions=(["LOG_EVENT", "SIMULATE_MINIMUM_RISK_RESPONSE", "SIMULATE_NOTIFY"] if allow else ["LOG_EVENT"]),
        )
