"""Bloc 5 — Analyse temporelle des hypothèses, sans diagnostic ni action véhicule."""
from collections import deque
from dataclasses import asdict
from statistics import mean
from typing import Optional
from .models import DetectionResult


class TemporalAnalyzer:
    """Analyse une fenêtre glissante de résultats de détection (timestamps en secondes)."""
    def __init__(self, window_seconds: float = 5.0, min_persistence_seconds: float = 2.0,
                 max_history: int = 500):
        if window_seconds <= 0 or min_persistence_seconds < 0:
            raise ValueError("window_seconds doit être > 0 et persistence >= 0")
        self.window_seconds = float(window_seconds)
        self.min_persistence_seconds = float(min_persistence_seconds)
        self.history = deque(maxlen=max_history)

    def update(self, result: DetectionResult) -> dict:
        t = float(result.timestamp)
        if self.history and t < self.history[-1]["timestamp"]:
            # Out-of-order input is recorded as a discontinuity, never used to increase persistence.
            return {"timestamp": t, "window_seconds": self.window_seconds,
                    "event_type": result.event_type, "persistence_seconds": 0.0,
                    "repetitions": 0, "trend": "TIMESTAMP_DISCONTINUITY",
                    "temporal_confidence": 0.0, "confirmed_persistent": False,
                    "samples_in_window": 0}
        self.history.append({"timestamp": t, "event_type": result.event_type,
                             "event_score": _unit(result.event_score),
                             "confidence": _unit(result.confidence)})
        while self.history and t - self.history[0]["timestamp"] > self.window_seconds:
            self.history.popleft()
        same = [x for x in self.history if x["event_type"] == result.event_type and result.event_type != "NORMAL"]
        persistence = max(0.0, same[-1]["timestamp"] - same[0]["timestamp"]) if same else 0.0
        scores = [x["event_score"] for x in same]
        trend = "STABLE"
        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            trend = "INCREASING" if delta > 0.10 else "DECREASING" if delta < -0.10 else "STABLE"
        repetitions = len(same)
        persistent = bool(same) and persistence >= self.min_persistence_seconds
        avg_score = mean(scores) if scores else _unit(result.event_score)
        avg_conf = mean([x["confidence"] for x in same]) if same else _unit(result.confidence)
        # Heuristic evidence score, not a medically validated probability.
        temporal_conf = min(1.0, 0.50 * avg_score + 0.25 * avg_conf + 0.15 * min(1.0, persistence / max(self.min_persistence_seconds, 1.0)) + 0.10 * min(1.0, repetitions / 5.0))
        return {"timestamp": t, "window_seconds": self.window_seconds,
                "event_type": result.event_type, "persistence_seconds": persistence,
                "repetitions": repetitions, "trend": trend,
                "temporal_confidence": round(temporal_conf, 4),
                "confirmed_persistent": persistent,
                "samples_in_window": len(self.history),
                "onset_timestamp": same[0]["timestamp"] if same else None}


def _unit(value) -> float:
    try:
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")): return 0.0
        return min(1.0, max(0.0, v))
    except (TypeError, ValueError):
        return 0.0
