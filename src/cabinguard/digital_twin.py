"""Bloc 6 — baseline personnel explicable. Features numériques uniquement."""
from collections import defaultdict
from math import sqrt, isfinite


class DriverDigitalTwin:
    def __init__(self, min_calibration_samples: int = 10, z_clip: float = 5.0):
        if min_calibration_samples < 2: raise ValueError("min_calibration_samples doit être >= 2")
        self.min_calibration_samples = int(min_calibration_samples)
        self.z_clip = float(z_clip)
        self._calibration = defaultdict(list)
        self.baseline = {}
        self.ready = False
        self.calibration_samples = 0

    def update(self, features: dict, event_type: str = "NORMAL", data_quality: float = 1.0,
               allow_adaptation: bool = True) -> dict:
        clean = {str(k): float(v) for k, v in (features or {}).items()
                 if _valid_number(v)}
        # Baseline learns only from sufficiently good, explicitly normal observations.
        can_learn = (allow_adaptation and event_type == "NORMAL" and _valid_number(data_quality)
                     and 0.0 <= float(data_quality) <= 1.0 and float(data_quality) >= 0.80)
        if can_learn and not self.ready:
            for key, value in clean.items(): self._calibration[key].append(value)
            self.calibration_samples += 1
            self._refresh_baseline()
        elif can_learn and self.ready:
            # Conservative online adaptation, only on normal/high-quality samples.
            for key, value in clean.items():
                if key in self.baseline:
                    b = self.baseline[key]
                    b["mean"] = 0.98 * b["mean"] + 0.02 * value
        deviations = {}
        z_values = []
        if self.ready:
            for key, value in clean.items():
                if key not in self.baseline: continue
                b = self.baseline[key]
                z = abs(value - b["mean"]) / max(b["std"], 1e-6)
                z = min(self.z_clip, z)
                deviations[key] = round(z, 4); z_values.append(z / self.z_clip)
        deviation = (sum(z_values) / len(z_values)) if z_values else None
        return {"ready": self.ready, "calibration_samples": self.calibration_samples,
                "baseline_features": sorted(self.baseline), "deviations": deviations,
                "deviation_score": round(deviation, 4) if deviation is not None else None,
                "adapted": bool(can_learn),
                "note": "Baseline personnalisé; deviation_score n'est pas une probabilité médicale."}

    def _refresh_baseline(self):
        eligible = {k: vals for k, vals in self._calibration.items() if len(vals) >= self.min_calibration_samples}
        if self.calibration_samples >= self.min_calibration_samples and eligible:
            self.baseline = {k: {"mean": sum(v)/len(v),
                                 "std": max(sqrt(sum((x-sum(v)/len(v))**2 for x in v)/max(1, len(v)-1)), 1e-3),
                                 "count": len(v)} for k, v in eligible.items()}
            self.ready = bool(self.baseline)


def _valid_number(v):
    if isinstance(v, bool): return False
    try:
        x = float(v)
        return isfinite(x)
    except (TypeError, ValueError): return False
