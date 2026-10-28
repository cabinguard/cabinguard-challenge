"""Machine à états de sécurité, indépendante du modèle d'IA."""

ALLOWED_STATES = {"NORMAL", "VERIFY", "ASSIST", "MINIMUM_RISK_RESPONSE"}


def choose_state(
    event_present: bool,
    evidence_sufficient: bool,
    security_ok: bool,
    vehicle_available: bool,
    sensors_available: bool = True,
) -> tuple[str, str]:
    """Retourne l'état et sa justification. Politique de démonstration.

    Ordre de priorité (le premier test vrai gagne) :
    capteurs -> sécurité -> événement -> preuves -> véhicule -> réponse.
    """
    if not sensors_available:
        return (
            "ASSIST",
            "Aucun capteur disponible : système en mode dégradé, "
            "aucune réponse simulée critique.",
        )

    if not security_ok:
        return (
            "VERIFY",
            "Vérification de sécurité échouée : "
            "aucune réponse simulée critique.",
        )

    if not event_present:
        return "NORMAL", "Aucun motif suffisamment suspect."

    if not evidence_sufficient:
        return (
            "VERIFY",
            "Signal suspect mais preuves insuffisantes "
            "ou qualité dégradée.",
        )

    if not vehicle_available:
        return (
            "ASSIST",
            "Véhicule ou contexte indisponible : "
            "pas de réponse véhicule.",
        )

    return (
        "MINIMUM_RISK_RESPONSE",
        "Conditions de démonstration remplies ; "
        "réponse simulée uniquement.",
    )
