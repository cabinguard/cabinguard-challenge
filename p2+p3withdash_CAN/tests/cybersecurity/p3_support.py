"""Constants/helpers for the Person 3 suites (TEST FIXTURE values only: not Person 1's feature spec nor
Person 2's Guardian states)."""
FEATS = {"phys": ("hr", "hrv"), "face": ("blink",)}
BOUNDS = {"hr": (30.0, 220.0), "hrv": (0.0, 300.0), "blink": (0.0, 60.0)}
ROLES = {"wearable": {"SENSOR_TELEMETRY"}, "camera": {"SENSOR_TELEMETRY"},
        "guardian": {"VEHICLE_COMMAND"}, "detector": set()}
SOURCES = {"wearable-01": ("phys",), "camera-01": ("face",)}
STATES = {"NORMAL": ["NOTIFY_DRIVER"], "EMERGENCY": ["NOTIFY_DRIVER", "SLOW_DOWN", "STOP_SAFELY"]}


def good_phys(q=0.9, **over):
    f = {"hr": 70.0, "hrv": 50.0}
    f.update(over)
    return {"phys": {"q": q, "f": f}}
