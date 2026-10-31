"""Démonstration bout-en-bout avec scénarios synthétiques (aucun véhicule réel)."""
import json
from pathlib import Path
from src.cabinguard.models import DetectionResult, SensorStatus, SecurityStatus
from src.integration.secure_factory import build_secure_pipeline

CONFIG = str(Path(__file__).resolve().parent / "configs" / "safety.json")

SEIZURE = "SEIZURE_LIKE_MOTION_PATTERN"


def strong(t, event=SEIZURE):
    """Détection solide : tous les seuils sont dépassés."""
    return DetectionResult(event_type=event, event_score=0.95, confidence=0.95, timestamp=t,
                           persistence_seconds=5.0, cross_modal_agreement=0.9, data_quality=0.95)


def sensors(t, camera=True, fsr=True, physio=True):
    return SensorStatus(camera_available=camera, fsr_available=fsr, physiology_available=physio, timestamp=t)


def build_cases():
    """Retourne la liste ordonnée des scénarios : (titre, détection, capteurs, sécurité, vitesse km/h)."""
    return [
        ("CAS 1 — Conduite normale",
         DetectionResult(event_type="NORMAL", event_score=0.12, confidence=0.92, timestamp=1.0,
                         persistence_seconds=0.5, cross_modal_agreement=0.9, data_quality=0.95),
         sensors(1.0), SecurityStatus(), 42),
        ("CAS 2 — Motif suspect, confirmation insuffisante",
         DetectionResult(event_type="STROKE_LIKE_PATTERN", event_score=0.86, confidence=0.62, timestamp=2.0,
                         persistence_seconds=1.0, cross_modal_agreement=0.45, data_quality=0.80,
                         evidence=["synthetic_face_feature", "synthetic_fsr_feature"]),
         sensors(2.0, physio=False), SecurityStatus(), 38),
        ("CAS 3 — Conditions de simulation remplies",
         DetectionResult(event_type=SEIZURE, event_score=0.91, confidence=0.88, timestamp=3.0,
                         persistence_seconds=4.0, cross_modal_agreement=0.82, data_quality=0.91),
         sensors(3.0), SecurityStatus(), 55),
        ("CAS 4 — Message périmé / risque de rejeu",
         DetectionResult(event_type="CARDIAC_EVENT_LIKE_PATTERN", event_score=0.95, confidence=0.94, timestamp=4.0,
                         persistence_seconds=5.0, cross_modal_agreement=0.90, data_quality=0.92),
         sensors(4.0), SecurityStatus(message_fresh=False), 60),
        ("CAS 5 — Bracelet absent mais autres capteurs disponibles",
         DetectionResult(event_type="NORMAL", event_score=0.15, confidence=0.90, timestamp=5.0,
                         persistence_seconds=0.5, cross_modal_agreement=0.85, data_quality=0.88),
         sensors(5.0, physio=False), SecurityStatus(), 30),
        ("CAS 6 — Tous les capteurs indisponibles",
         strong(6.0), sensors(6.0, camera=False, fsr=False, physio=False), SecurityStatus(), 45),
        ("CAS 7 — Émetteur non authentifié",
         strong(7.0), sensors(7.0), SecurityStatus(sender_authenticated=False), 45),
        ("CAS 8 — Commande non autorisée",
         strong(8.0), sensors(8.0), SecurityStatus(command_authorized=False), 45),
        ("CAS 9 — Intégrité du message invalide",
         strong(9.0), sensors(9.0), SecurityStatus(integrity_valid=False), 45),
        ("CAS 10 — Séquence de message invalide",
         strong(10.0), sensors(10.0), SecurityStatus(sequence_valid=False), 45),
        ("CAS 11 — Caméra indisponible, autres capteurs disponibles",
         strong(11.0), sensors(11.0, camera=False), SecurityStatus(), 45),
    ]


def run_all():
    """Exécute tous les scénarios et retourne [(titre, enregistrement)]."""
    results = []
    for title, detection, sens, security, speed in build_cases():
        pipeline = build_secure_pipeline(CONFIG)       # pipeline neuf (chaîne P3 incluse) : cas indépendants
        results.append((title, pipeline.process(detection, sens, security, speed_kmh=speed)))
    return results


def main():
    for title, record in run_all():
        print("\n" + "=" * 72)
        print(title)
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
