"""Contrats partagés entre détection, cybersécurité et Safety Guardian."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    event_type: str = "NORMAL"
    event_score: float = 0.0
    confidence: float = 1.0
    timestamp: float = 0.0
    persistence_seconds: float = 0.0
    cross_modal_agreement: float = 1.0
    data_quality: float = 1.0
    risk_level: str = "LOW"
    evidence: list[str] = field(default_factory=list)


@dataclass
class SensorStatus:
    camera_available: bool = True
    fsr_available: bool = True
    physiology_available: bool = False
    microphone_available: bool = False
    context_available: bool = True
    timestamp: float = 0.0
    notes: list[str] = field(default_factory=list)
    quality: dict[str, float] = field(default_factory=dict)
    communication_status: dict[str, bool] = field(default_factory=dict)
    processing_status: dict[str, bool] = field(default_factory=dict)
    vehicle_interface_available: bool = True


@dataclass
class SecurityStatus:
    integrity_valid: bool = True
    message_fresh: bool = True
    sender_authenticated: bool = True
    command_authorized: bool = True
    sequence_valid: bool = True
    reason: str = "checks_passed"


@dataclass
class VehicleContext:
    speed_kmh: float = 0.0
    driving_context: str = "UNKNOWN"
    vehicle_available: bool = True
    timestamp: float = 0.0


@dataclass
class GuardianDecision:
    state: str
    allow_simulated_response: bool
    reason: str
    event_type: str
    score: float
    confidence: float
    timestamp: float
    rejected_checks: list[str] = field(default_factory=list)
    degraded_mode: bool = False
    actions: list[str] = field(default_factory=list)
