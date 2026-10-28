"""Authenticated Guardian -> simulated-vehicle command path (TB2, SIMULATION; no real actuation).

Pipeline for every command (a ``SecureMessage`` of type VEHICLE_COMMAND):

  MessageValidator  (schema -> authentication -> integrity -> freshness -> replay -> role authorisation)
  -> sender must hold the guardian role (defence in depth: re-checked here, not only in the role table)
  -> payload must be exactly {"cmd": <str>} with the command inside the policy's allow-list
     (payload is the validator's authenticated snapshot, never a re-read of ``msg.payload`` - CG1)
  -> command policy evaluated against the Guardian's OWN current state (``state_provider``; never
     caller-supplied)  -> re-check at the actuation boundary  -> single-use token  -> actuator port.

Authenticated != safe: an authentic Guardian command that the policy forbids in the current state is refused.
The AI detector is a registered identity whose role has NO permissions: authentication does not authorise.

Interfaces (nothing from Person 2 is imported or re-implemented)
  * ``CommandPolicy``   : known_commands() and check(command, state) -> (allowed, reason). Person 2's
                          ``ALLOWED_COMMANDS`` table plugs in through ``StaticCommandPolicy``. Unknown state
                          or unknown command => refuse (fail closed).
  * ``ActuatorPort``    : execute(AuthorizedCommand) -> bool.

Bypass resistance - what this does and does not give you
  The gateway issues an ``AuthorizedCommand`` bound to (command, ts, sender, seq) by an HMAC under a secret
  shared only between the *issuer* (held by the gateway) and the *verifier* (held by the actuator). An
  actuator built on the verifier refuses anything not issued by the gateway and refuses reuse. This removes
  every *accidental or convenience* path that passes a source string. It is NOT isolation: Python offers no
  in-process privilege boundary, and code that can import the issuer can mint tokens. The architectural
  requirement is therefore that the vehicle object exposes ONLY an authorisation-checked entry point. Today's
  ``MockVehicle.execute(t, command, source, state)`` trusts a caller-supplied ``source`` string; wrapping it
  with ``LegacyMockVehicleAdapter`` does not remove that method. Closing it needs a change owned by Person 2
  (see docs/P3_INTEGRATION_CHECKLIST.md). Real systems need ECU-level per-command MACs (out of scope).

Token threat model (expiry / actuator binding intentionally NOT added)
  Tokens are in-process HMAC capabilities, not network credentials:
    * ``new_authority()`` hands the issuer to the gateway and the verifier to the actuator in the same
      address space. The secret never leaves the process.
    * Freshness of the *command message* is already enforced by ``MessageValidator`` (max_age_s, replay,
      seq). The token additionally binds ``(command, ts, sender_id, seq)`` so it cannot be applied to a
      different command or a different Guardian message.
    * Single-use is a per-sender high-water mark on ``seq``, under a lock (CG2). A bounded deque of
      remembered identities is NOT used: wrapping that deque would forget ``(sender, seq)`` and let the
      same token authenticate again.
    * An ``exp`` timestamp or ``actuator_id`` in the MAC would be a breaking change to ``_auth_bytes`` /
      ``AuthorizedCommand``. In-process it buys nothing: a stolen token implies the process is already
      compromised, and a leaked secret lets an attacker mint arbitrary tokens regardless of ``exp``.
      If tokens later cross a process or network boundary, bind ``actuator_id`` and ``exp`` into
      ``_auth_bytes`` and check them in ``consume(auth, now, actuator_id)`` as a deliberate version bump
      (``cabinguard/cmd-auth/v2``).

Hardening (CG1/CG2/CG3)
  * CG1 - ``submit()`` authorises ``ValidationResult.payload`` (atomically produced with MAC
    verification), never a second read of ``msg.payload``.
  * CG2 - ``AuthorizationVerifier.consume()`` and ``CommandSigner.sign()`` serialise mutation of
    the watermark and sequence allocation. Failed ``sign_message`` does not consume a sequence.
  * CG3 - ``check_key`` on issuer/verifier secrets; ``MAX_SEQ`` on the signer; ``StaticCommandPolicy``
    rejects malformed tables and freezes its mapping.
  * Actuation-boundary re-check - ``state_provider`` + ``policy.check`` run again immediately before
    ``issue``/``execute``, so a Guardian state transition after the first check cannot admit a
    command the policy would now refuse.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

from .audit_safe import as_safe
from .keys import check_key
from .secure_message import MAX_SEQ, MessageValidator, SecureMessage, sign_message

COMMAND_MSG_TYPE = "VEHICLE_COMMAND"


class GatewayConfigError(ValueError):
    pass


class SequenceLimitError(OverflowError):
    """Raised when ``CommandSigner`` has issued ``MAX_SEQ`` and cannot allocate another sequence."""


CommandSequenceError = SequenceLimitError


# ----------------------------------------------------------------------------------------- policy interface
class CommandPolicy(Protocol):
    def known_commands(self) -> frozenset: ...
    def check(self, command: str, state) -> tuple: ...


class StaticCommandPolicy:
    """Allow-list per Guardian state. The table is supplied by the safety owner; none is defined here.

    Fail closed: unknown state, unknown command, or a malformed table => refuse / refuse to construct.
    The caller's mapping is copied and frozen; mutating it afterwards has no effect.
    """

    def __init__(self, allowed: Mapping):
        if not isinstance(allowed, Mapping) or not allowed:
            raise GatewayConfigError("command policy table must be a non-empty mapping of state -> commands")
        frozen = {}
        known: set = set()
        for state, cmds in allowed.items():
            if not isinstance(state, str) or not state or state.strip() != state:
                raise GatewayConfigError("policy state names must be non-empty strings")
            if isinstance(cmds, (str, bytes)) or not isinstance(cmds, Iterable):
                raise GatewayConfigError(
                    f"policy[{state!r}] must be an iterable of command-name strings, not a bare string"
                )
            seen: set = set()
            cleaned = []
            for c in cmds:
                if not isinstance(c, str) or not c or c.strip() != c:
                    raise GatewayConfigError("command names must be non-empty strings")
                if c in seen:
                    raise GatewayConfigError(f"duplicate command {c!r} for state {state!r}")
                seen.add(c)
                cleaned.append(c)
            frozen[state] = frozenset(cleaned)
            known.update(cleaned)
        self._allowed = MappingProxyType(frozen)
        self._known = frozenset(known)

    def known_commands(self) -> frozenset:
        return self._known

    def check(self, command: str, state) -> tuple:
        if state not in self._allowed:
            return False, "unknown Guardian state"
        if command not in self._allowed[state]:
            return False, "command not permitted in current Guardian state"
        return True, "permitted by response policy"


# ----------------------------------------------------------------------------------------- authorisation tokens
@dataclass(frozen=True)
class AuthorizedCommand:
    command: str
    ts: float
    sender_id: str
    seq: int
    tag: str

    def __repr__(self) -> str:
        return f"AuthorizedCommand(command={self.command!r}, sender_id={self.sender_id!r}, seq={self.seq})"


def _auth_bytes(command: str, ts: float, sender_id: str, seq: int) -> bytes:
    return json.dumps(["cabinguard/cmd-auth/v1", command, ts, sender_id, seq],
                      separators=(",", ":"), allow_nan=False).encode()


class AuthorizationIssuer:
    def __init__(self, secret: bytes):
        self._secret = check_key(secret, "authorization issuer secret")

    def issue(self, command: str, ts: float, sender_id: str, seq: int) -> AuthorizedCommand:
        tag = hmac.new(self._secret, _auth_bytes(command, ts, sender_id, seq), hashlib.sha256).hexdigest()
        return AuthorizedCommand(command, ts, sender_id, seq, tag)


class AuthorizationVerifier:
    """Held by the actuator. ``consume`` is True once per genuine token, False for forgeries and reuse.

    Replay barrier is a per-sender high-water mark on ``seq``, not a bounded deque. A deque of
    remembered identities can forget ``(sender, seq)`` when it wraps, which would let the same
    HMAC token authenticate again. Commands are issued with a strictly increasing ``seq`` per
    Guardian, so ``seq <= last_consumed[sender]`` is always replay (or a stale out-of-order
    token) and is refused. ``memory`` is accepted and validated for API compatibility; it is
    not used as a forgetting window.

    CG2: MAC verification may run without the lock; admission into the watermark is serialised
    so two concurrent ``consume()`` calls cannot both accept the same token.
    """

    def __init__(self, secret: bytes, memory: int = 4096):
        if isinstance(memory, bool) or not isinstance(memory, int) or memory < 1:
            raise ValueError("verifier memory must be a positive integer")
        self._secret = check_key(secret, "authorization verifier secret")
        self._last_seq: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume(self, auth: object) -> bool:
        if not isinstance(auth, AuthorizedCommand):
            return False
        try:
            expected = hmac.new(self._secret, _auth_bytes(auth.command, auth.ts, auth.sender_id, auth.seq),
                                hashlib.sha256).hexdigest()
            ok = hmac.compare_digest(auth.tag.encode("utf-8"), expected.encode())
        except (TypeError, ValueError, UnicodeError):
            return False
        if not ok:
            return False
        with self._lock:
            last = self._last_seq.get(auth.sender_id, -1)
            if auth.seq <= last:
                return False
            self._last_seq[auth.sender_id] = auth.seq
            return True


def new_authority() -> tuple:
    """(issuer, verifier) sharing a fresh CSPRNG secret. Give the issuer to the gateway, the verifier to the actuator."""
    s = secrets.token_bytes(32)
    return AuthorizationIssuer(s), AuthorizationVerifier(s)


# ----------------------------------------------------------------------------------------- actuator side
class ActuatorPort(Protocol):
    def execute(self, auth: AuthorizedCommand) -> bool: ...


class SimulatedActuator:
    """Reference actuator: records commands, performs nothing, refuses anything without a genuine token."""

    def __init__(self, verifier: AuthorizationVerifier):
        self._verifier = verifier
        self.executed: list = []
        self.refused = 0
        self._lock = threading.Lock()

    def execute(self, auth: AuthorizedCommand) -> bool:
        if not self._verifier.consume(auth):
            with self._lock:
                self.refused += 1
            return False
        with self._lock:
            self.executed.append(auth.command)
        return True


class LegacyMockVehicleAdapter:
    """Adapts an object exposing ``execute(t, command, source, state) -> bool`` (the current MockVehicle
    signature) to ``ActuatorPort``. Authentication happens BEFORE the label "GUARDIAN" is asserted. Does NOT
    close the legacy direct path on the wrapped object - see module docstring."""

    def __init__(self, vehicle, verifier: AuthorizationVerifier, state_provider: Callable):
        self._vehicle, self._verifier, self._state = vehicle, verifier, state_provider

    def execute(self, auth: AuthorizedCommand) -> bool:
        if not self._verifier.consume(auth):
            return False
        return bool(self._vehicle.execute(int(auth.ts), auth.command, "GUARDIAN", self._state()))


# ----------------------------------------------------------------------------------------- signer
class CommandSigner:
    """Guardian-side signer: own key, strictly increasing sequence.

    CG2: sequence allocation is atomic. ``sign_message`` runs while holding the lock so a failed
    sign does not consume a sequence and two threads cannot be issued the same seq. CG3: ``MAX_SEQ``
    is enforced; wraparound is refused rather than silently reused.
    """

    def __init__(self, sender_id: str, key: bytes, first_seq: int = 1):
        if not isinstance(sender_id, str) or not sender_id:
            raise ValueError("sender_id must be a non-empty string")
        if isinstance(first_seq, bool) or not isinstance(first_seq, int) or first_seq < 0:
            raise ValueError("first_seq must be a non-negative integer")
        if first_seq > MAX_SEQ:
            raise ValueError("first_seq exceeds MAX_SEQ")
        self.sender_id, self._key, self._seq = sender_id, check_key(key, f"key for {sender_id!r}"), first_seq
        self._lock = threading.Lock()

    def sign(self, command: str, ts: float) -> SecureMessage:
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError("ts must be a finite number")
        with self._lock:
            if self._seq > MAX_SEQ:
                raise SequenceLimitError(f"sequence space exhausted for {self.sender_id!r}")
            seq = self._seq
            m = sign_message(
                SecureMessage(self.sender_id, seq, float(ts), COMMAND_MSG_TYPE, {"cmd": command}),
                self._key,
            )
            self._seq = seq + 1
            return m

    def __repr__(self) -> str:
        with self._lock:
            nxt = self._seq
        return f"CommandSigner(sender_id={self.sender_id!r}, next_seq={nxt})"


# ----------------------------------------------------------------------------------------- gateway
@dataclass
class CommandResult:
    accepted: bool
    stage: str            # validator stage | 'authorization' | 'command_policy' | 'state' | 'actuator' | 'accepted'
    reason: str
    executed: bool = False


class SecureCommandGateway:
    def __init__(self, validator: MessageValidator, actuator: ActuatorPort, state_provider: Callable,
                 policy: CommandPolicy, issuer: AuthorizationIssuer, audit=None,
                 guardian_role: str = "guardian", history: int = 1000):
        if not isinstance(guardian_role, str) or not guardian_role:
            raise GatewayConfigError("guardian_role must be a non-empty string")
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise GatewayConfigError("history must be a positive integer")
        if not callable(state_provider):
            raise GatewayConfigError("state_provider must be callable")
        # Fail closed on misconfiguration: only the guardian role may hold the command permission.
        for role, perms in validator.roles.items():
            if COMMAND_MSG_TYPE in perms and role != guardian_role:
                raise GatewayConfigError(f"role {role!r} must not hold {COMMAND_MSG_TYPE}; only {guardian_role!r} may")
        self.v, self.actuator, self.state_provider = validator, actuator, state_provider
        self.policy, self._issuer, self.guardian_role = policy, issuer, guardian_role
        self.audit = as_safe(audit if audit is not None else validator.audit, validator.registry.keys())
        self.results: deque = deque(maxlen=history)
        self._lock = threading.Lock()

    def _fin(self, res: CommandResult, sender: str = "<unregistered>") -> CommandResult:
        if not res.accepted and res.stage != "accepted" and self.audit is not None and res.stage in (
                "authorization", "command_policy", "state", "actuator"):
            self.audit.record("COMMAND_REJECTED", res.stage, sender, res.reason)   # validator logs its own stages
        with self._lock:
            self.results.append(res)
        return res

    def submit(self, msg: SecureMessage, now: float) -> CommandResult:
        r = self.v.validate(msg, now)
        if not r.ok:
            return self._fin(CommandResult(False, r.stage, r.reason))
        # Envelope copies taken immediately after a successful validate. Payload MUST be the
        # validator's authenticated snapshot (CG1 / ST1): never msg.payload, which another thread
        # could mutate between MAC verification and this read.
        sender = msg.sender_id
        seq = msg.seq
        ts = msg.ts
        msg_type = msg.msg_type
        payload = r.payload
        if self.v.registry[sender]["role"] != self.guardian_role or msg_type != COMMAND_MSG_TYPE:
            return self._fin(CommandResult(False, "authorization", "only the Guardian may issue vehicle commands"), sender)
        cmd = payload.get("cmd") if isinstance(payload, Mapping) and set(payload) == {"cmd"} else None
        if not isinstance(cmd, str) or cmd not in self.policy.known_commands():
            return self._fin(CommandResult(False, "command_policy", "unknown or malformed command"), sender)
        try:
            state = self.state_provider()
        except Exception:
            return self._fin(CommandResult(False, "state", "Guardian state unavailable (fail closed)"), sender)
        try:
            allowed, why = self.policy.check(cmd, state)
        except Exception:
            allowed, why = False, "policy evaluation failed (fail closed)"
        if not allowed:
            return self._fin(CommandResult(False, "command_policy", why), sender)
        try:
            # Actuation-boundary re-check: a Guardian state transition after the first
            # check must not admit a command the policy would now refuse.
            state_now = self.state_provider()
            allowed_now, why_now = self.policy.check(cmd, state_now)
        except Exception:
            allowed_now, why_now = False, "policy evaluation failed (fail closed)"
        if not allowed_now:
            return self._fin(CommandResult(False, "command_policy", why_now), sender)
        try:
            executed = bool(self.actuator.execute(self._issuer.issue(cmd, ts, sender, seq)))
        except Exception:
            return self._fin(CommandResult(False, "actuator", "actuator error (fail closed)"), sender)
        if not executed:
            return self._fin(CommandResult(False, "actuator", "actuator refused the command"), sender)
        return self._fin(CommandResult(True, "accepted", why_now, True), sender)