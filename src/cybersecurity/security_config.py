"""Security configuration, separated from application logic (SIMULATION).

Parses the ``security`` / ``sensor_trust`` sections of the project's cybersecurity YAML (same shape as the
reference ``configs/cybersecurity.yaml``) into an immutable, validated object. Rules:

  * **No secrets in configuration.** Any mapping key whose *tokens* look like secret material
    (``key``, ``secret``, ``password``, ``token``, …) is rejected, so a key committed to YAML fails
    loudly instead of being used. Matching is token-based (snake/kebab/camelCase), not substring:
    ``api_key`` / ``hmacKey`` are rejected; ``keyboard_enabled`` is not. A trailing numeric suffix on
    a token is stripped before matching (``token2`` -> ``token``, ``apikey9`` -> ``apikey``), so a
    sequentially- or version-numbered field name cannot bypass the check. Keys are supplied at run
    time through ``keys.load_key_from_env`` / a secret store.
  * Sender identities and roles are data (``identities``); keys are supplied separately and joined by
    ``build_registry``.
  * ``identities`` may be empty. Sensor sources do **not** have to appear in ``security.identities``:
    authenticated identities may be provisioned by another trusted runtime mechanism and passed to
    ``build_registry``. When both maps are present they are checked independently (valid identifiers,
    known roles); overlap is not required.
  * Plausibility ``bounds`` are *sanity limits derived from each feature's definition*; they belong to the
    feature owner (Person 1) and must be confirmed with them. They are not clinical thresholds.
  * Optional fields: a **missing** key uses the documented default. An **explicitly supplied**
    non-mapping (including ``null``) is a configuration error, not a silent default.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .keys import check_key

# Token match only (see _field_tokens). Substring match would flag keyboard_enabled.
_SECRET_TOKENS = frozenset({
    "key", "keys", "secret", "secrets", "password", "passwd",
    "token", "credential", "credentials", "apikey",
})
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Strips a TRAILING run of digits from an already-extracted token (e.g. "token2" -> "token",
# "apikey9" -> "apikey"), so a numbered/versioned field name cannot evade _SECRET_TOKENS by
# appending a digit. Only trims the end of the token; does not split digits appearing elsewhere
# (e.g. "esp32" -> "esp", which was never a secret token either way).
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
DEFAULT_TELEMETRY_TYPES = ("WEARABLE_TELEMETRY", "SENSOR_TELEMETRY")


class SecurityConfigError(ValueError):
    pass


def _field_tokens(name: str) -> list[str]:
    s = _CAMEL_RE.sub(r"\1_\2", str(name))
    s = _ACRONYM_RE.sub(r"\1_\2", s)
    tokens = _TOKEN_RE.findall(s.lower())
    return [_TRAILING_DIGITS_RE.sub("", t) for t in tokens]


def _scan_for_secrets(node, path="") -> None:
    if isinstance(node, Mapping):
        for k, v in node.items():
            if any(t in _SECRET_TOKENS for t in _field_tokens(k)):
                raise SecurityConfigError(
                    f"secret-like field {path + str(k)!r} is not allowed in configuration"
                )
            _scan_for_secrets(v, f"{path}{k}.")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _scan_for_secrets(v, f"{path}{i}.")


def _num(x, what: str, *, positive=False, nonneg=False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SecurityConfigError(f"{what} must be a finite number")
    if positive and x <= 0:
        raise SecurityConfigError(f"{what} must be > 0")
    if nonneg and x < 0:
        raise SecurityConfigError(f"{what} must be >= 0")
    return float(x)


def _name(x, what: str) -> str:
    if not isinstance(x, str) or not x or x.strip() != x:
        raise SecurityConfigError(f"{what} must be a non-empty string without surrounding whitespace")
    return x


def _require_mapping(value, what: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SecurityConfigError(f"{what} must be a mapping")
    return value


def _optional_mapping(parent: Mapping, key: str, what: str) -> Mapping:
    """Missing key → empty mapping. Present but not a mapping (incl. null) → error."""
    if key not in parent:
        return {}
    return _require_mapping(parent[key], what)


def _str_seq(value, what: str, *, allow_empty=False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SecurityConfigError(f"{what} must be a list of non-empty strings")
    if not value and not allow_empty:
        raise SecurityConfigError(f"{what} must be a list of non-empty strings")
    out = []
    seen = set()
    for item in value:
        name = _name(item, what)
        if name in seen:
            raise SecurityConfigError(f"{what} contains duplicate {name!r}")
        seen.add(name)
        out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class SecurityConfig:
    max_age_s: float
    max_future_skew_s: float | None
    roles: Mapping[str, frozenset]
    sources: Mapping[str, tuple]
    bounds: Mapping[str, tuple]
    integrity_hold_s: float
    telemetry_types: tuple
    guardian_role: str
    identities: Mapping[str, str]          # sender_id -> role (may be empty if supplied elsewhere)

    def __post_init__(self):
        # Frozen dataclass blocks *reassignment*, not mutation of a dict already stored on a field.
        if not isinstance(self.roles, Mapping):
            raise SecurityConfigError("roles must be a mapping")
        if not isinstance(self.sources, Mapping):
            raise SecurityConfigError("sources must be a mapping")
        if not isinstance(self.bounds, Mapping):
            raise SecurityConfigError("bounds must be a mapping")
        if not isinstance(self.identities, Mapping):
            raise SecurityConfigError("identities must be a mapping")
        frozen_roles = {k: frozenset(v) for k, v in self.roles.items()}
        frozen_sources = {k: tuple(v) for k, v in self.sources.items()}
        frozen_bounds = {k: tuple(v) for k, v in self.bounds.items()}
        frozen_ids = dict(self.identities)
        object.__setattr__(self, "roles", MappingProxyType(frozen_roles))
        object.__setattr__(self, "sources", MappingProxyType(frozen_sources))
        object.__setattr__(self, "bounds", MappingProxyType(frozen_bounds))
        object.__setattr__(self, "identities", MappingProxyType(frozen_ids))
        object.__setattr__(self, "telemetry_types", tuple(self.telemetry_types))


def parse_security_config(raw: Mapping) -> SecurityConfig:
    if not isinstance(raw, Mapping):
        raise SecurityConfigError("configuration root must be a mapping")
    _scan_for_secrets(raw)
    if "security" not in raw:
        raise SecurityConfigError("'security' mapping is required")
    sec = _require_mapping(raw["security"], "'security'")
    st = _optional_mapping(raw, "sensor_trust", "'sensor_trust'")

    max_age = _num(sec.get("max_age_s"), "security.max_age_s", positive=True)
    skew = sec.get("max_future_skew_s")
    skew = None if skew is None else _num(skew, "security.max_future_skew_s", nonneg=True)

    roles_raw = sec.get("roles")
    if not isinstance(roles_raw, Mapping) or not roles_raw:
        raise SecurityConfigError("security.roles must be a non-empty mapping")
    roles = {}
    for r, perms in roles_raw.items():
        role = _name(r, "security.roles key")
        roles[role] = frozenset(_str_seq(perms, f"security.roles[{role!r}]", allow_empty=True))

    guardian = _name(sec.get("guardian_role", "guardian"), "security.guardian_role")
    if guardian not in roles:
        raise SecurityConfigError("guardian_role must be one of security.roles")

    sources_raw = _optional_mapping(st, "sources", "sensor_trust.sources")
    sources = {}
    for sid, mods in sources_raw.items():
        sender = _name(sid, "sensor_trust.sources key")
        sources[sender] = _str_seq(mods, f"sensor_trust.sources[{sender!r}]")

    bounds_raw = _optional_mapping(st, "bounds", "sensor_trust.bounds")
    bounds = {}
    for f, b in bounds_raw.items():
        feat = _name(f, "sensor_trust.bounds key")
        if not isinstance(b, (list, tuple)) or isinstance(b, (str, bytes)) or len(b) != 2:
            raise SecurityConfigError(f"sensor_trust.bounds[{feat!r}] must be [lo, hi]")
        lo, hi = _num(b[0], f"bounds[{feat}].lo"), _num(b[1], f"bounds[{feat}].hi")
        if lo > hi:
            raise SecurityConfigError(f"sensor_trust.bounds[{feat!r}]: lo > hi")
        bounds[feat] = (lo, hi)

    hold = _num(st.get("integrity_hold_s", 1.0), "sensor_trust.integrity_hold_s", nonneg=True)
    if "telemetry_types" not in st:
        types = DEFAULT_TELEMETRY_TYPES
    else:
        types = _str_seq(st["telemetry_types"], "sensor_trust.telemetry_types")

    ids_raw = _optional_mapping(sec, "identities", "security.identities")
    ids = {}
    for sid, role in ids_raw.items():
        sender = _name(sid, "security.identities key")
        if not isinstance(role, str) or role not in roles:
            raise SecurityConfigError(f"identity {sender!r} references unknown role")
        ids[sender] = role
    # sources ⊈ identities is intentional: identities may be provisioned at runtime.

    return SecurityConfig(max_age, skew, roles, sources, bounds, hold, types, guardian, ids)


def load_security_config(path: str | Path) -> SecurityConfig:
    try:
        import importlib
        yaml = importlib.import_module("yaml")
    except ImportError as e:             # pragma: no cover
        raise SecurityConfigError("pyyaml is required to load YAML configuration") from e
    with open(path, encoding="utf-8") as fh:
        return parse_security_config(yaml.safe_load(fh))


def build_registry(identities: Mapping[str, str], keys: Mapping[str, bytes]) -> dict:
    """Join identities (config) with keys (runtime secret store) into the MessageValidator registry."""
    reg = {}
    for sid, role in identities.items():
        if not isinstance(sid, str) or not sid:
            raise SecurityConfigError("identity sender_id must be a non-empty string")
        if sid not in keys:
            raise SecurityConfigError(f"no key supplied for identity {sid!r}")
        reg[sid] = {"key": check_key(keys[sid], f"key for {sid!r}"), "role": role}
    extra = set(keys) - set(identities)
    if extra:
        raise SecurityConfigError("keys supplied for identities not declared in configuration")
    return reg