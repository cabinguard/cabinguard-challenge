"""
tests/cybersecurity/test_baseline_guard.py

Deterministic tests for authenticated baseline files (src/cybersecurity/baseline_guard.py).

Covers: HMAC authentication, inner SHA-256 checked at sign AND load, driver binding,
unacceptable status, rollback, same-version content conflicts, duplicate JSON keys,
signed-envelope schema, version-store failures, atomic writes, audit redaction.

Honest limitation documented by test: HMAC authenticates the writer. It does NOT
stop baseline poisoning through a legitimately signed update of a VALID baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import pytest

from src.cybersecurity.audit_log import SecurityAuditLog
from src.cybersecurity.baseline_guard import (
    MAC_FIELD,
    MAX_FILE_BYTES,
    BaselineIntegrityError,
    BaselineRollbackError,
    BaselineVersionConflictError,
    InMemoryVersionSource,
    VerifiedBaseline,
    _inner_sha,
    _mac,
    load_authenticated_baseline,
    sign_baseline_blob,
    sign_baseline_file,
    write_signed_baseline,
)
from src.cybersecurity.keys import generate_key



def blob(driver="drv-A", version=3, feats=None, status="VALID"):
    content = {
        "driver_id": driver,
        "status": status,
        "baseline_version": version,
        "features": feats or {"hr": {"mu": 60.0}},
    }
    return {"content": content, "sha256": _inner_sha(content)}


@pytest.fixture
def key():
    return generate_key()


@pytest.fixture
def path(tmp_path, key):
    p = tmp_path / "b.json"
    write_signed_baseline(p, blob(), key)
    return p


def load(path, key, **kw):
    args = dict(
        expected_driver_id="drv-A",
        version_source=InMemoryVersionSource(),
        accepted_statuses=("VALID",),
    ) | kw
    return load_authenticated_baseline(path, key, **args)


# =================================================================================== HAPPY PATH

def test_roundtrip_ok_and_returns_content(path, key):
    v = load(path, key)
    assert isinstance(v, VerifiedBaseline) and v.version == 3 and v.content["driver_id"] == "drv-A"


def test_mac_present_and_file_remains_plaintext_json(path):
    raw = json.loads(path.read_text())
    assert len(raw[MAC_FIELD]) == 64 and raw["content"]["features"] == {"hr": {"mu": 60.0}}


def test_identical_same_version_reload_is_idempotent(path, key):
    vs = InMemoryVersionSource()
    assert load(path, key, version_source=vs).version == 3
    assert load(path, key, version_source=vs).version == 3
    assert vs.floor("drv-A") == 3


# =================================================================================== AUTH / TAMPER

def test_wrong_key_rejected(path, key):
    with pytest.raises(BaselineIntegrityError):
        load(path, generate_key())


def test_content_tamper_with_recomputed_sha_still_rejected(path, key):
    raw = json.loads(path.read_text())
    raw["content"]["features"]["hr"]["mu"] = 999.0
    raw["sha256"] = _inner_sha(raw["content"])
    path.write_text(json.dumps(raw))
    with pytest.raises(BaselineIntegrityError, match="authentication"):
        load(path, key)


def test_mac_stripped_or_unsigned_rejected(path, key):
    raw = json.loads(path.read_text())
    raw.pop(MAC_FIELD)
    path.write_text(json.dumps(raw))
    with pytest.raises(BaselineIntegrityError, match="unsigned"):
        load(path, key)


@pytest.mark.parametrize("data", [b"", b"not json", b"[]", b'{"content": 1}', b"\xff\xfe"])
def test_corrupt_files_fail_safely(tmp_path, key, data):
    p = tmp_path / "x.json"
    p.write_bytes(data)
    with pytest.raises(BaselineIntegrityError):
        load(p, key)


def test_missing_file_and_oversize(tmp_path, key):
    with pytest.raises(BaselineIntegrityError):
        load(tmp_path / "nope.json", key)
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (MAX_FILE_BYTES + 5))
    with pytest.raises(BaselineIntegrityError):
        load(big, key)


def test_driver_substitution_rejected(path, key):
    with pytest.raises(BaselineIntegrityError, match="different driver"):
        load(path, key, expected_driver_id="drv-B")


# =================================================================================== SIGN-TIME CHECKSUM (review item 1)

def test_sign_rejects_inner_checksum_mismatch(key):
    b = blob()
    b["sha256"] = "0" * 64
    with pytest.raises(BaselineIntegrityError, match="checksum"):
        sign_baseline_blob(b, key)


def test_write_signed_baseline_rejects_inner_checksum_mismatch(tmp_path, key):
    p = tmp_path / "m.json"
    b = blob()
    b["sha256"] = "0" * 64
    with pytest.raises(BaselineIntegrityError, match="checksum"):
        write_signed_baseline(p, b, key)
    assert not p.exists()


def test_load_detects_inner_sha_mismatch_even_when_mac_covers_it(tmp_path, key):
    """Defense in depth: a MAC computed over (content, wrong sha256) is still refused at load."""
    p = tmp_path / "m.json"
    b = blob()
    wrong = "0" * 64
    envelope = {"content": b["content"], "sha256": wrong, MAC_FIELD: _mac(key, b["content"], wrong)}
    p.write_text(json.dumps(envelope))
    with pytest.raises(BaselineIntegrityError, match="checksum"):
        load(p, key)
    assert load(p, key, verify_inner_sha256=False).version == 3


# =================================================================================== STATUS / VERSION SCHEMA

@pytest.mark.parametrize("status", ["INVALID_CONTAMINATED", "INVALID_INSUFFICIENT", None, ""])
def test_signing_cannot_launder_an_invalid_baseline(tmp_path, key, status):
    p = tmp_path / "c.json"
    write_signed_baseline(p, blob(status=status), key)
    with pytest.raises(BaselineIntegrityError, match="status"):
        load(p, key)


def test_accepted_statuses_is_mandatory_and_nonempty(path, key):
    with pytest.raises(TypeError):
        load_authenticated_baseline(path, key, expected_driver_id="drv-A",
                                    version_source=InMemoryVersionSource())
    with pytest.raises(BaselineIntegrityError):
        load(path, key, accepted_statuses=())


@pytest.mark.parametrize("ver", [None, 0, -1, "3", True, 2.5])
def test_bad_version_rejected(tmp_path, key, ver):
    p = tmp_path / "v.json"
    b = blob()
    b["content"]["baseline_version"] = ver
    b["sha256"] = _inner_sha(b["content"])
    write_signed_baseline(p, b, key)
    with pytest.raises(BaselineIntegrityError):
        load(p, key)


def test_sign_rejects_malformed_blob(key):
    with pytest.raises(BaselineIntegrityError):
        sign_baseline_blob({"content": "nope", "sha256": "0" * 64}, key)
    with pytest.raises(BaselineIntegrityError):
        sign_baseline_blob(["not", "a", "dict"], key)


def test_sign_rejects_extra_envelope_keys(key):
    b = blob()
    b["evil"] = 1
    with pytest.raises(BaselineIntegrityError):
        sign_baseline_blob(b, key)


def test_load_rejects_extra_envelope_keys_even_with_valid_mac(path, key):
    raw = json.loads(path.read_text())
    raw["extra"] = "nope"
    path.write_text(json.dumps(raw))
    with pytest.raises(BaselineIntegrityError, match="malformed|unsigned"):
        load(path, key)


def test_duplicate_json_keys_rejected(tmp_path, key):
    p = tmp_path / "dup.json"
    body = blob()
    signed = sign_baseline_blob(body, key)
    # Duplicate "content" at the envelope level. stdlib json.loads would keep the last value.
    p.write_text(
        '{"content": %s, "sha256": "%s", "%s": "%s", "content": {}}'
        % (json.dumps(signed["content"]), signed["sha256"], MAC_FIELD, signed[MAC_FIELD])
    )
    with pytest.raises(BaselineIntegrityError, match="duplicate"):
        load(p, key)


def test_non_finite_number_rejected_at_sign_and_load(tmp_path, key):
    b = blob(feats={"hr": {"mu": float("nan")}})
    with pytest.raises(BaselineIntegrityError):
        sign_baseline_blob(b, key)
    b2 = blob(feats={"hr": {"mu": float("inf")}})
    b2["sha256"] = hashlib.sha256(json.dumps(b2["content"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(BaselineIntegrityError):
        write_signed_baseline(tmp_path / "inf.json", b2, key)


# =================================================================================== ROLLBACK / SAME-VERSION CONFLICT

def test_rollback_refused_and_floor_is_monotonic(tmp_path, key):
    vs = InMemoryVersionSource()
    new, old = tmp_path / "new.json", tmp_path / "old.json"
    write_signed_baseline(new, blob(version=5), key)
    write_signed_baseline(old, blob(version=4), key)
    assert load(new, key, version_source=vs).version == 5 and vs.floor("drv-A") == 5
    with pytest.raises(BaselineRollbackError):
        load(old, key, version_source=vs)
    assert load(new, key, version_source=vs).version == 5
    assert vs.floor("drv-A") == 5


def test_initial_floor_enforced(path, key):
    with pytest.raises(BaselineRollbackError):
        load(path, key, version_source=InMemoryVersionSource({"drv-A": 10}))


def test_same_version_different_content_rejected(tmp_path, key):
    vs = InMemoryVersionSource()
    a, bpath = tmp_path / "a.json", tmp_path / "b.json"
    write_signed_baseline(a, blob(version=4, feats={"hr": {"mu": 60.0}}), key)
    write_signed_baseline(bpath, blob(version=4, feats={"hr": {"mu": 72.0}}), key)
    assert load(a, key, version_source=vs).version == 4
    with pytest.raises(BaselineVersionConflictError, match="same version"):
        load(bpath, key, version_source=vs)
    assert vs.floor("drv-A") == 4
    assert load(a, key, version_source=vs).content["features"]["hr"]["mu"] == 60.0


def test_newer_version_with_different_content_is_allowed(tmp_path, key):
    vs = InMemoryVersionSource()
    a, bpath = tmp_path / "a.json", tmp_path / "b.json"
    write_signed_baseline(a, blob(version=4, feats={"hr": {"mu": 60.0}}), key)
    write_signed_baseline(bpath, blob(version=5, feats={"hr": {"mu": 72.0}}), key)
    load(a, key, version_source=vs)
    v = load(bpath, key, version_source=vs)
    assert v.version == 5 and v.content["features"]["hr"]["mu"] == 72.0


def test_failed_load_does_not_advance_floor(tmp_path, key):
    vs = InMemoryVersionSource()
    p = tmp_path / "t.json"
    write_signed_baseline(p, blob(version=9), key)
    with pytest.raises(BaselineIntegrityError):
        load(p, generate_key(), version_source=vs)
    assert vs.floor("drv-A") == 0


def test_version_store_failure_fails_closed(path, key):
    class Boom(InMemoryVersionSource):
        def accept(self, driver_id, version, fingerprint):
            raise OSError("trusted store unavailable")

    with pytest.raises(BaselineIntegrityError, match="version source"):
        load(path, key, version_source=Boom())


def test_in_memory_version_source_is_not_persistent():
    """InMemoryVersionSource is a test double. A new instance does not remember floors."""
    vs = InMemoryVersionSource()
    vs.accept("drv-A", 7, "a" * 64)
    assert vs.floor("drv-A") == 7
    assert InMemoryVersionSource().floor("drv-A") == 0


def test_concurrent_conflicting_same_version_only_one_content_wins(tmp_path, key):
    vs = InMemoryVersionSource()
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_signed_baseline(p1, blob(version=7, feats={"hr": {"mu": 60.0}}), key)
    write_signed_baseline(p2, blob(version=7, feats={"hr": {"mu": 61.0}}), key)
    results = []
    lock = threading.Lock()

    def worker(p):
        try:
            mu = load(p, key, version_source=vs).content["features"]["hr"]["mu"]
            with lock:
                results.append(("ok", mu))
        except BaselineVersionConflictError:
            with lock:
                results.append(("conflict", None))
        except Exception as e:  # pragma: no cover
            with lock:
                results.append(("err", e))

    threads = [threading.Thread(target=worker, args=(p1,)),
               threading.Thread(target=worker, args=(p2,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)
    assert [r[0] for r in results].count("ok") == 1
    assert [r[0] for r in results].count("conflict") == 1
    assert vs.floor("drv-A") == 7


# =================================================================================== FEATURES / SIGN-EXISTING / ATOMIC / AUDIT

def test_required_features_enforced(path, key):
    assert load(path, key, required_features=["hr"])
    with pytest.raises(BaselineIntegrityError):
        load(path, key, required_features=["hr", "blink"])


def test_sign_existing_file_from_unsigned_baseline_save(tmp_path, key):
    p = tmp_path / "plain.json"
    p.write_text(json.dumps(blob()))
    with pytest.raises(BaselineIntegrityError):
        load(p, key)
    sign_baseline_file(p, key)
    assert load(p, key).version == 3


def test_atomic_write_leaves_no_temp_and_survives_failure(tmp_path, key, monkeypatch):
    p = tmp_path / "a.json"
    write_signed_baseline(p, blob(), key)
    good = p.read_bytes()
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        write_signed_baseline(p, blob(version=9), key)
    assert p.read_bytes() == good and [f.name for f in tmp_path.iterdir()] == ["a.json"]


def test_errors_and_audit_do_not_leak_secrets(path, key):
    log = SecurityAuditLog()
    mac = json.loads(path.read_text())[MAC_FIELD]
    with pytest.raises(BaselineIntegrityError) as e:
        load(path, generate_key(), audit=log)
    dump = repr(log.to_dicts()) + str(e.value)
    assert mac not in dump and key.hex() not in dump and "drv-A" not in dump
    assert log.entries()[0].event == "BASELINE_REJECTED"


def test_short_key_refused(path):
    with pytest.raises(Exception):
        load(path, b"short")


def test_HMAC_does_not_prevent_legitimately_signed_poisoned_baseline(tmp_path, key):
    """HMAC authenticates the writer, not the physiology.

    A VALID baseline whose feature values were influenced through the legitimate
    update path (compromised sensor, poisoned observations, etc.) is accepted.
    Baseline poisoning is an update-policy problem, not a MAC problem.
    """
    p = tmp_path / "poison.json"
    write_signed_baseline(p, blob(feats={"hr": {"mu": 999.0}}), key)
    v = load(p, key)
    assert v.content["status"] == "VALID"
    assert v.content["features"]["hr"]["mu"] == 999.0