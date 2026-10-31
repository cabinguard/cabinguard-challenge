"""
tests/cybersecurity/test_security_config.py

Unit tests for security_config.py: parsing/validation, deep immutability, the absent-vs-malformed
optional-field distinction, secret-field detection, and role/source/identity handling.

Self-contained by design (no shared conftest.py), same rationale as the other cybersecurity test
modules in this suite.

NOTE on open policy decisions (see review thread): two tests below (marked OPEN DECISION) assert
the CURRENT behavior of this specific implementation, which differs from an earlier draft reviewed
for this module. They are not a statement that this behavior is correct - they exist so that if
the policy is changed, these tests fail loudly and visibly rather than the change going unnoticed.
"""
from __future__ import annotations

import pytest

from src.cybersecurity.security_config import (
    DEFAULT_TELEMETRY_TYPES,
    SecurityConfig,
    SecurityConfigError,
    build_registry,
    parse_security_config,
)


def _base_raw() -> dict:
    """A minimal, valid configuration. Tests mutate their own copy, never share one."""
    return {
        "security": {
            "max_age_s": 30.0,
            "max_future_skew_s": 2.0,
            "roles": {
                "wearable": ["SENSOR_TELEMETRY"],
                "guardian": ["VEHICLE_COMMAND"],
            },
            "guardian_role": "guardian",
            "identities": {
                "wearable-01": "wearable",
            },
        },
        "sensor_trust": {
            "sources": {
                "wearable-01": ["phys", "face"],
            },
            "bounds": {
                "hr": [30.0, 220.0],
            },
            "integrity_hold_s": 1.0,
            "telemetry_types": ["SENSOR_TELEMETRY"],
        },
    }


# =================================================================================== BASELINE

def test_valid_config_parses():
    cfg = parse_security_config(_base_raw())
    assert cfg.max_age_s == 30.0
    assert cfg.roles["wearable"] == frozenset({"SENSOR_TELEMETRY"})
    assert cfg.sources["wearable-01"] == ("phys", "face")
    assert cfg.bounds["hr"] == (30.0, 220.0)
    assert cfg.identities["wearable-01"] == "wearable"
    assert cfg.telemetry_types == ("SENSOR_TELEMETRY",)


def test_telemetry_types_defaults_when_absent():
    raw = _base_raw()
    del raw["sensor_trust"]["telemetry_types"]
    cfg = parse_security_config(raw)
    assert cfg.telemetry_types == DEFAULT_TELEMETRY_TYPES


def test_sensor_trust_section_entirely_absent_is_allowed():
    raw = {"security": _base_raw()["security"]}
    del raw["security"]["identities"]
    cfg = parse_security_config(raw)
    assert dict(cfg.sources) == {} and dict(cfg.bounds) == {}
    assert cfg.telemetry_types == DEFAULT_TELEMETRY_TYPES


# =================================================================================== DEEP IMMUTABILITY

class TestDeepImmutability:

    def test_mutating_source_dict_after_parse_does_not_affect_config(self):
        raw = _base_raw()
        cfg = parse_security_config(raw)
        raw["security"]["roles"]["wearable"].append("VEHICLE_COMMAND")
        raw["sensor_trust"]["sources"]["wearable-01"].append("new_modality")
        assert cfg.roles["wearable"] == frozenset({"SENSOR_TELEMETRY"})
        assert cfg.sources["wearable-01"] == ("phys", "face")

    def test_writing_through_roles_mapping_raises(self):
        cfg = parse_security_config(_base_raw())
        with pytest.raises(TypeError):
            cfg.roles["wearable"] = frozenset({"VEHICLE_COMMAND"})

    def test_writing_through_sources_mapping_raises(self):
        cfg = parse_security_config(_base_raw())
        with pytest.raises(TypeError):
            cfg.sources["wearable-01"] = ("x",)

    def test_writing_through_bounds_mapping_raises(self):
        cfg = parse_security_config(_base_raw())
        with pytest.raises(TypeError):
            cfg.bounds["hr"] = (0.0, 1.0)

    def test_writing_through_identities_mapping_raises(self):
        cfg = parse_security_config(_base_raw())
        with pytest.raises(TypeError):
            cfg.identities["wearable-01"] = "guardian"

    def test_directly_constructed_config_is_also_frozen(self):
        cfg = SecurityConfig(
            max_age_s=30.0, max_future_skew_s=None,
            roles={"wearable": ["SENSOR_TELEMETRY"]},
            sources={"wearable-01": ["phys"]},
            bounds={"hr": [30.0, 220.0]},
            integrity_hold_s=1.0,
            telemetry_types=["SENSOR_TELEMETRY"],
            guardian_role="wearable",
            identities={"wearable-01": "wearable"},
        )
        assert cfg.roles["wearable"] == frozenset({"SENSOR_TELEMETRY"})
        with pytest.raises(TypeError):
            cfg.roles["wearable"] = frozenset()


# =================================================================================== MAPPING TYPE VALIDATION

class TestMappingTypeValidation:

    @pytest.mark.parametrize("bad", [["not", "a", "mapping"], "a-string", 42, True])
    def test_sources_rejects_non_mapping(self, bad):
        raw = _base_raw()
        raw["sensor_trust"]["sources"] = bad
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    @pytest.mark.parametrize("bad", [["not", "a", "mapping"], "a-string", 42])
    def test_bounds_rejects_non_mapping(self, bad):
        raw = _base_raw()
        raw["sensor_trust"]["bounds"] = bad
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    @pytest.mark.parametrize("bad", [["not", "a", "mapping"], "a-string", 42])
    def test_identities_rejects_non_mapping(self, bad):
        raw = _base_raw()
        raw["security"]["identities"] = bad
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_sensor_trust_explicit_null_rejected_not_defaulted(self):
        raw = _base_raw()
        raw["sensor_trust"] = None
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_bounds_value_as_string_rejected_not_treated_as_two_char_list(self):
        """Two-character string ('30') has len() == 2, same as a real [lo, hi] pair - must not be
        silently accepted as one."""
        raw = _base_raw()
        raw["sensor_trust"]["bounds"]["hr"] = "30"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)


# =================================================================================== ABSENT vs MALFORMED

class TestAbsentVsMalformedOptionalFields:

    def test_absent_identities_defaults_to_empty_and_is_allowed(self):
        raw = _base_raw()
        del raw["security"]["identities"]
        cfg = parse_security_config(raw)
        assert dict(cfg.identities) == {}

    def test_explicit_empty_list_identities_is_rejected_not_defaulted(self):
        raw = _base_raw()
        raw["security"]["identities"] = []
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_absent_sources_and_bounds_default_to_empty(self):
        raw = _base_raw()
        del raw["sensor_trust"]["sources"]
        del raw["sensor_trust"]["bounds"]
        del raw["security"]["identities"]
        cfg = parse_security_config(raw)
        assert dict(cfg.sources) == {} and dict(cfg.bounds) == {}


# =================================================================================== IDENTIFIER / NAME VALIDATION

class TestIdentifierValidation:

    @pytest.mark.parametrize("bad_sid", ["", "   ", 123, None])
    def test_source_id_must_be_a_clean_nonempty_string(self, bad_sid):
        raw = _base_raw()
        raw["sensor_trust"]["sources"] = {bad_sid: ["phys"]}
        raw["security"]["identities"] = {}
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    @pytest.mark.parametrize("bad_sid", ["", "   ", 123, None])
    def test_identity_sender_id_must_be_a_clean_nonempty_string(self, bad_sid):
        raw = _base_raw()
        raw["security"]["identities"] = {bad_sid: "wearable"}
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    @pytest.mark.parametrize("bad_role", ["", "   ", 123, None])
    def test_role_name_must_be_a_clean_nonempty_string(self, bad_role):
        raw = _base_raw()
        raw["security"]["roles"] = {bad_role: ["SENSOR_TELEMETRY"]}
        raw["security"]["guardian_role"] = bad_role
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_leading_or_trailing_whitespace_in_a_name_is_rejected(self):
        """Stricter than plain non-empty: this implementation rejects names with surrounding
        whitespace, e.g. a stray trailing space from a copy-pasted YAML value."""
        raw = _base_raw()
        raw["security"]["roles"] = {" wearable ": ["SENSOR_TELEMETRY"]}
        raw["security"]["guardian_role"] = "wearable"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)


# =================================================================================== ROLE PERMISSION VALIDATION

class TestRolePermissionValidation:

    def test_empty_permission_string_rejected(self):
        raw = _base_raw()
        raw["security"]["roles"]["wearable"] = ["SENSOR_TELEMETRY", ""]
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_duplicate_permission_rejected(self):
        raw = _base_raw()
        raw["security"]["roles"]["wearable"] = ["SENSOR_TELEMETRY", "SENSOR_TELEMETRY"]
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_duplicate_modality_in_source_rejected(self):
        raw = _base_raw()
        raw["sensor_trust"]["sources"]["wearable-01"] = ["phys", "phys"]
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_empty_source_modality_list_rejected(self):
        """Unlike role permission lists (see OPEN DECISION test below), a source's modality list
        is NOT allowed to be empty - a source reporting nothing is almost certainly a mistake."""
        raw = _base_raw()
        raw["sensor_trust"]["sources"]["wearable-01"] = []
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_OPEN_DECISION_empty_role_permission_list_is_currently_allowed(self):
        """Documents CURRENT behavior, not an endorsement: this implementation allows a role to be
        declared with zero permissions (inert/reserved role). Flagged for confirmation in review -
        if the policy changes to reject this, this test should be updated to pytest.raises(...).
        """
        raw = _base_raw()
        raw["security"]["roles"]["idle_role"] = []
        cfg = parse_security_config(raw)                 # currently does NOT raise
        assert cfg.roles["idle_role"] == frozenset()


# =================================================================================== TELEMETRY TYPES VALIDATION

class TestTelemetryTypesValidation:

    def test_bare_string_is_rejected_not_silently_exploded_into_characters(self):
        raw = _base_raw()
        raw["sensor_trust"]["telemetry_types"] = "SENSOR_TELEMETRY"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_empty_list_rejected(self):
        raw = _base_raw()
        raw["sensor_trust"]["telemetry_types"] = []
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_duplicate_entries_rejected(self):
        raw = _base_raw()
        raw["sensor_trust"]["telemetry_types"] = ["SENSOR_TELEMETRY", "SENSOR_TELEMETRY"]
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_non_string_entry_rejected(self):
        raw = _base_raw()
        raw["sensor_trust"]["telemetry_types"] = ["SENSOR_TELEMETRY", 123]
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)


# =================================================================================== SECRET-FIELD DETECTION

class TestSecretFieldDetection:

    @pytest.mark.parametrize("field", ["key", "secret", "password", "token",
                                        "api_key", "authToken", "client_secret",
                                        "db_password", "KEY", "credentials"])
    def test_obvious_secret_like_fields_are_rejected(self, field):
        raw = _base_raw()
        raw["security"][field] = "whatever"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    @pytest.mark.parametrize("field", ["keyboard_enabled", "monkey_mode", "tokenizer_version",
                                        "passwordless_login", "secretary_role"])
    def test_compound_words_containing_secret_substrings_are_not_false_positives(self, field):
        raw = _base_raw()
        raw["security"][field] = "whatever"
        cfg = parse_security_config(raw)                 # must not raise
        assert cfg is not None

    @pytest.mark.parametrize("field", ["token2", "secret1", "apikey9"])
    def test_digit_suffixed_secret_word_without_separator_is_rejected(self, field):
        """token2 / secret1 / apikey9 must be rejected. Trailing digits are stripped from each
        token before matching, so these are treated as token / secret / apikey."""
        raw = _base_raw()
        raw["security"][field] = "whatever"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)


# =================================================================================== SOURCES / IDENTITIES

class TestSourceIdentityRelationship:

    def test_sources_allowed_with_no_identities_at_all_external_provisioning(self):
        raw = _base_raw()
        raw["security"]["identities"] = {}
        cfg = parse_security_config(raw)
        assert cfg.sources
        assert dict(cfg.identities) == {}

    def test_matching_identity_and_source_passes(self):
        cfg = parse_security_config(_base_raw())
        assert "wearable-01" in cfg.sources and "wearable-01" in cfg.identities

    def test_identity_referencing_unknown_role_rejected(self):
        raw = _base_raw()
        raw["security"]["identities"]["ghost-01"] = "nonexistent-role"
        with pytest.raises(SecurityConfigError):
            parse_security_config(raw)

    def test_OPEN_DECISION_source_with_no_matching_identity_is_currently_accepted(self):
        """Documents CURRENT behavior, not an endorsement. This implementation performs NO
        consistency check between sensor_trust.sources and security.identities, even when
        identities is non-empty for OTHER sources. This is the exact design question flagged in
        review as needing explicit confirmation - a source declared here with genuinely no
        identity anywhere (declarative or external) would currently load without error. If the
        policy is changed to require every source to have SOME identity when identities is
        non-empty, this test should be updated to pytest.raises(...).
        """
        raw = _base_raw()
        raw["sensor_trust"]["sources"]["rogue-sensor"] = ["phys"]   # no matching identity anywhere
        cfg = parse_security_config(raw)                           # currently does NOT raise
        assert "rogue-sensor" in cfg.sources
        assert "rogue-sensor" not in cfg.identities


# =================================================================================== build_registry

class TestBuildRegistry:

    def test_builds_registry_from_identities_and_keys(self):
        identities = {"wearable-01": "wearable"}
        keys = {"wearable-01": b"K" * 32}
        reg = build_registry(identities, keys)
        assert reg["wearable-01"]["role"] == "wearable"
        assert reg["wearable-01"]["key"] == b"K" * 32

    def test_missing_key_for_declared_identity_rejected(self):
        with pytest.raises(SecurityConfigError):
            build_registry({"wearable-01": "wearable"}, {})

    def test_extra_key_for_undeclared_identity_rejected(self):
        with pytest.raises(SecurityConfigError):
            build_registry({}, {"ghost": b"K" * 32})

    @pytest.mark.parametrize("bad_sid", ["", 123, None])
    def test_build_registry_rejects_invalid_sender_id(self, bad_sid):
        with pytest.raises(SecurityConfigError):
            build_registry({bad_sid: "wearable"}, {bad_sid: b"K" * 32})

    def test_works_with_parsed_config_identities_mappingproxy(self):
        cfg = parse_security_config(_base_raw())
        keys = {"wearable-01": b"K" * 32}
        reg = build_registry(cfg.identities, keys)
        assert reg["wearable-01"]["role"] == "wearable"