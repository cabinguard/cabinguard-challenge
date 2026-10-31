"""Experiments A-I. Every number reported anywhere is computed here from executed code.
All conclusions are valid ONLY inside this synthetic evaluation environment."""
from __future__ import annotations
import copy
import json
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..communication.can_frame import CanFrameError, CanReceiver, encode_frame
from ..communication.emergency_message import ALLOWED_KEYS
from ..cybersecurity.secure_message import MessageValidator, SecureMessage, sign_message
from ..detectors.models import RAW, EventModel, apply_abstention, generic_features, personalised_features
from ..digital_twin.baseline import Baseline
from ..digital_twin.calibration import CalibrationError, build_baseline
from ..digital_twin.deviation import compute_deviations
from ..fusion.quality_aware_fusion import evidence_scores, transparent_predict
from ..fusion.temporal_confirmation import confirm_series
from ..pipeline import prepare_inputs, run_guardian, score_frame
from ..schemas import EVENT_CLASSES, EVENT_ONLY, NORMAL, GuardianState as S
from ..simulation.generator import generate_dataset, make_profile, simulate_session
from . import faults as F
from .metrics import cm_metrics, confusion, evaluate_predictions


@dataclass
class Ctx:
    cfg: dict
    seed: int
    df: pd.DataFrame
    bl: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    med: pd.Series
    fscale: dict
    models: dict = field(default_factory=dict)
    theta: float = 0.5
    n_train_rows_excluded: int = 0
    n_train_rows: int = 0


def feats(ctx: Ctx, name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "A":
        return generic_features(df, ctx.med)
    if name == "B":
        return personalised_features(df, ctx.bl, ctx.cfg)[0]
    if name == "B_noctx":
        return personalised_features(df, ctx.bl, ctx.cfg, use_context=False)[0]
    if name == "C":
        return personalised_features(df, ctx.bl, ctx.cfg, quality_aware=True)[0]
    raise KeyError(name)


def predict(ctx: Ctx, name: str, df: pd.DataFrame) -> np.ndarray:
    if name in ("W", "W_noctx"):
        cfg = ctx.cfg if name == "W" else {**ctx.cfg, "fusion": {**ctx.cfg["fusion"], "turn_gate": 2.0}}
        _, z, g = personalised_features(df, ctx.bl, ctx.cfg)
        ev = evidence_scores(z, g, df["turn_frac"], cfg)
        return transparent_predict(ev, ctx.theta, ctx.cfg["fusion"]["sufficiency_min"])
    if name == "C":
        X, _, g = personalised_features(df, ctx.bl, ctx.cfg, quality_aware=True)
        return np.array(EVENT_CLASSES)[apply_abstention(ctx.models["C"].predict_proba(X), g).argmax(1)]
    if name == "C_raw":
        return ctx.models["C"].predict(feats(ctx, "C", df))
    return ctx.models[name].predict(feats(ctx, name, df))


def build_context(cfg: dict, seed: int, df: pd.DataFrame | None = None, singles: bool = False) -> Ctx:
    df = generate_dataset(cfg, seed) if df is None else df
    bl = {d: build_baseline(g, d, cfg, assume_normal=True) for d, g in df[df.scenario == "CALIBRATION"].groupby("driver_id")}
    dd = df[df.scenario != "CALIBRATION"].reset_index(drop=True)
    tr, va, te = (dd[dd.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    ctx = Ctx(cfg, seed, df, bl, tr, va, te, tr[RAW].median(), F.feature_scale(tr))
    ctx.models["A"] = EventModel(cfg, seed).fit(generic_features(tr, ctx.med), tr.label)
    ctx.models["B"] = EventModel(cfg, seed).fit(feats(ctx, "B", tr), tr.label)
    ctx.models["B_noctx"] = EventModel(cfg, seed).fit(feats(ctx, "B_noctx", tr), tr.label)
    aug = F.augment_for_training(tr, np.random.default_rng([seed, 99]), cfg, ctx.fscale)
    trc = pd.concat([tr, aug]).reset_index(drop=True)
    Xc, zc, gc = personalised_features(trc, bl, cfg, quality_aware=True)
    evc = evidence_scores(zc, gc, trc["turn_frac"], cfg)
    keep = (trc.label == NORMAL).values.copy()
    for c in EVENT_ONLY:                      # never teach the model "no reliable evidence => event"
        m = (trc.label == c).values
        keep[m] = evc[c]["sufficiency"].values[m] >= cfg["fusion"]["sufficiency_min"]
    ctx.n_train_rows_excluded = int((~keep).sum()); ctx.n_train_rows = int(len(keep))
    ctx.models["C"] = EventModel(cfg, seed).fit(Xc[keep], trc.label[keep])
    # transparent weighted-evidence fusion: threshold tuned on VALIDATION drivers only
    best = (-1, 0.5)
    for th in np.arange(0.15, 0.81, 0.05):
        ctx.theta = float(th)
        f1 = cm_metrics(confusion(va.label.values, predict(ctx, "W", va)))["macro_f1"]
        best = max(best, (f1, float(th)))
    ctx.theta = best[1]
    if singles:
        cols_b = list(feats(ctx, "B", tr).columns)
        from ..schemas import CONTEXT_COLS, FEATURE_SPECS
        for m in ("steer", "motion", "phys", "visual", "speech"):
            cols = [c for c in cols_b if (c in FEATURE_SPECS and FEATURE_SPECS[c][0] == m)] + CONTEXT_COLS
            ctx.models[f"single_{m}"] = (cols, EventModel(cfg, seed).fit(feats(ctx, "B", tr)[cols], tr.label))
    return ctx


def _ev(ctx, name, df, B=200):
    r = evaluate_predictions(df, predict(ctx, name, df), B=B, seed=ctx.seed)
    return r


# ----------------------------------------------------------------------------------- A
def exp_A(ctx: Ctx) -> dict:
    out = {n: _ev(ctx, n, ctx.test) for n in ("A", "B", "C_raw", "C", "W")}
    out["_note"] = ("A generic RF on raw features; B personalised RF (robust deviations from calibration baseline); "
                    "C personalised + quality-aware (gating + augmentation); W transparent weighted-evidence rules.")
    return out


# ----------------------------------------------------------------------------------- B
def exp_B(ctx: Ctx) -> dict:
    out = {}
    for m in ("steer", "motion", "phys", "visual", "speech"):
        cols, mdl = ctx.models[f"single_{m}"]
        pred = mdl.predict(feats(ctx, "B", ctx.test)[cols])
        r = cm_metrics(confusion(ctx.test.label.values, pred)); out[f"single_{m}"] = r
    out["multimodal_B"] = cm_metrics(confusion(ctx.test.label.values, predict(ctx, "B", ctx.test)))
    out["multimodal_C"] = cm_metrics(confusion(ctx.test.label.values, predict(ctx, "C", ctx.test)))
    return out


# ----------------------------------------------------------------------------------- C
FAULTS = {
    "clean": lambda d, r, fs: d,
    "lowq_phys": lambda d, r, fs: F.degrade_quality(d, ["phys"], r, fs, 3.5),
    "lowq_motion": lambda d, r, fs: F.degrade_quality(d, ["motion"], r, fs, 3.5),
    "lowq_visual+speech": lambda d, r, fs: F.degrade_quality(d, ["visual", "speech"], r, fs, 3.5),
    "lowq_all": lambda d, r, fs: F.degrade_quality(d, ["steer", "motion", "phys", "visual", "speech"], r, fs, 3.5),
    "missing_phys(rPPG)": lambda d, r, fs: F.drop_modality(d, ["phys"]),
    "missing_BLE(motion+phys)": lambda d, r, fs: F.ble_disconnect(d),
    "missing_camera(visual+speech)": lambda d, r, fs: F.camera_unavailable(d),
    "missing_FSR": lambda d, r, fs: F.fsr_missing(d),
    "missing_BLE+camera": lambda d, r, fs: F.camera_unavailable(F.ble_disconnect(d)),
}


def exp_C(ctx: Ctx) -> dict:
    out = {}
    for k, fn in FAULTS.items():
        d = fn(ctx.test, np.random.default_rng([ctx.seed, 7, len(out)]), ctx.fscale)
        out[k] = {n: {kk: v for kk, v in evaluate_predictions(d, predict(ctx, n, d), B=100, seed=ctx.seed).items()
                      if kk in ("macro_f1", "fpr", "fnr", "macro_f1_ci95", "recall")} for n in ("A", "B", "C_raw", "C", "W")}
    return out


# ----------------------------------------------------------------------------------- D
def exp_D(ctx: Ctx) -> dict:
    te = ctx.test[ctx.test.label == NORMAL]
    flags = {"turn": "gt_conf_turn", "vibration": "gt_conf_vib", "lighting_glitch": "gt_conf_glitch",
             "stress_HR_rise": "gt_conf_stress", "rPPG_motion_artifact": "gt_conf_motion_art",
             "FSR_contact_dropout": "gt_conf_fsr_drop"}
    preds = {n: predict(ctx, n, te) for n in ("A", "B", "B_noctx", "C", "W", "W_noctx")}
    out = {}
    for name, col in list(flags.items()) + [("none", None)]:
        m = (te[col] == 1).values if col else (te[list(flags.values())].sum(axis=1) == 0).values
        out[name] = {"n_windows": int(m.sum()), **{n: float((p[m] != NORMAL).mean()) for n, p in preds.items()}}
    return out


# ----------------------------------------------------------------------------------- E
def _alarm_stats(ctx, fz_all, d, n, m, tau):
    fa_sessions = fa_windows = tot_norm_windows = 0; lat = []; det = det_cls = n_ev = n_norm = 0
    for sid, g in d.groupby("session_id", sort=False):
        f = fz_all.loc[g.index]
        pc, sc = f.event_class.values, f.score.values
        if n == 1:
            alarm = (sc >= tau)
            cls = pc
        else:
            cls, _ = confirm_series(pc, sc, n, m, tau)
            alarm = cls != ""
        lab_norm = (g.label.values == NORMAL)
        fa = alarm & lab_norm
        fa_windows += int(fa.sum()); tot_norm_windows += int(lab_norm.sum())
        if g.scenario.iloc[0] == "NORMAL_DRIVING":
            n_norm += 1; fa_sessions += int(fa.any())
        else:
            n_ev += 1; on = int(g.gt_onset.iloc[0]); post = alarm & ~lab_norm
            if post.any():
                det += 1; i = int(np.argmax(post)); lat.append(int(g.t.values[i]) - on)
                det_cls += int(cls[i] == g.scenario.iloc[0])
    return {"false_alarm_sessions_normal": fa_sessions / max(n_norm, 1), "n_normal_sessions": n_norm,
            "false_alarms_per_hour": fa_windows / max(tot_norm_windows, 1) * 3600,
            "false_alarm_window_rate": fa_windows / max(tot_norm_windows, 1),
            "detection_rate": det / max(n_ev, 1), "class_correct_at_first_alarm": det_cls / max(det, 1),
            "median_latency_s": float(np.median(lat)) if lat else None, "n_event_sessions": n_ev}


def exp_E(ctx: Ctx) -> dict:
    d = prepare_inputs(ctx.test, ctx.cfg); fz = score_frame(d, ctx.models["C"], ctx.bl, ctx.cfg)
    out = {}
    for n, m in ((1, 1), (5, 3), (10, 7), (15, 10)):
        out[f"n{n}_m{m}"] = _alarm_stats(ctx, fz, d, n, m, ctx.cfg["confirmation"]["tau"])
    return out


# ----------------------------------------------------------------------------------- F / H
def _summarise_runs(runs: dict, d: pd.DataFrame, fz: pd.DataFrame, cfg: dict) -> dict:
    sv = cfg["safety"]["score_verify"]
    norm = {k: 0 for k in ("n", "reach_VERIFY", "reach_ASSIST", "reach_MRR", "reach_DEGRADED", "direct_inst_action", "direct_confirmed_action")}
    ev = {k: 0 for k in ("n", "detected_VERIFY", "reach_ASSIST", "reach_MRR", "reach_NOTIFY", "reach_DEGRADED_any_time", "false_verify_pre_onset")}
    lv, lm = [], []
    for sid, g in d.groupby("session_id", sort=False):
        r = runs[sid]; first = r["first"]; f = fz.loc[g.index]
        if g.scenario.iloc[0] == "NORMAL_DRIVING":
            norm["n"] += 1
            for s, k in (("VERIFY", "reach_VERIFY"), ("ASSIST", "reach_ASSIST"), ("MINIMUM_RISK_RESPONSE", "reach_MRR"), ("DEGRADED", "reach_DEGRADED")):
                norm[k] += int(s in first)
            norm["direct_inst_action"] += int((f.score >= sv).any())
            norm["direct_confirmed_action"] += int((f.confirmed & (f.score >= sv)).any())
        else:
            ev["n"] += 1; on = int(g.gt_onset.iloc[0])
            vt = [t.t for t in r["transitions"] if t.dst == "VERIFY" and t.t >= on]
            ev["false_verify_pre_onset"] += int(any(t.dst == "VERIFY" and t.t < on for t in r["transitions"]))
            if vt:
                ev["detected_VERIFY"] += 1; lv.append(vt[0] - on)
            for s, k in (("ASSIST", "reach_ASSIST"), ("MINIMUM_RISK_RESPONSE", "reach_MRR"), ("NOTIFY", "reach_NOTIFY")):
                ev[k] += int(s in first and first[s] >= on)
            ev["reach_DEGRADED_any_time"] += int("DEGRADED" in first)
            if "MINIMUM_RISK_RESPONSE" in first:
                lm.append(first["MINIMUM_RISK_RESPONSE"] - on)
    ev["median_latency_to_VERIFY_s"] = float(np.median(lv)) if lv else None
    ev["median_latency_to_MRR_s"] = float(np.median(lm)) if lm else None
    return {"normal_sessions": norm, "event_sessions": ev}


def run_sessions(ctx, df_in, security_on=True, keep=None):
    d = prepare_inputs(df_in, ctx.cfg, security_on); fz = score_frame(d, ctx.models["C"], ctx.bl, ctx.cfg)
    runs = {sid: run_guardian(g, fz.loc[g.index], ctx.cfg, keep_trace=(keep is not None and sid in keep))
            for sid, g in d.groupby("session_id", sort=False)}
    return d, fz, runs


def exp_F(ctx: Ctx) -> dict:
    te, out = ctx.test, {}
    scen = {
        "clean": lambda d, r: d,
        "camera_unavailable(60-end)": lambda d, r: F.camera_unavailable(d, 60, 999),
        "FSR_missing(60-end)": lambda d, r: F.fsr_missing(d, 60, 999),
        "BLE_disconnect(60-end)": lambda d, r: F.ble_disconnect(d, 60, 999),
        "camera+BLE(60-end)": lambda d, r: F.camera_unavailable(F.ble_disconnect(d, 60, 999), 60, 999),
        "poor_rPPG_quality(60-end)": lambda d, r: F.degrade_quality(d, ["phys"], r, ctx.fscale, 3.5, 60, 999),
        "all_modalities_low_quality(60-end)": lambda d, r: F.degrade_quality(d, ["steer", "motion", "phys", "visual", "speech"], r, ctx.fscale, 3.5, 60, 999),
        "stale_data(60-end)": lambda d, r: F.stale_data(d, 60, 999),
        "transient_stale(100-110)": lambda d, r: F.stale_data(d, 100, 110),
        "processing_failure(60-end)": lambda d, r: F.processing_failure(d, 60, 999),
    }
    for i, (k, fn) in enumerate(scen.items()):
        d = fn(te, np.random.default_rng([ctx.seed, 11, i]))
        dd, fz, runs = run_sessions(ctx, d)
        out[k] = _summarise_runs(runs, dd, fz, ctx.cfg)
    return out


def exp_H(ctx: Ctx) -> dict:
    ex_ev = ctx.test[(ctx.test.scenario == "STROKE_LIKE_PATTERN")].session_id.iloc[0]
    ex_no = ctx.test[(ctx.test.scenario == "NORMAL_DRIVING") & (ctx.test.gt_conf_turn == 1)].session_id.iloc[0]
    d, fz, runs = run_sessions(ctx, ctx.test, keep={ex_ev, ex_no})
    summ = _summarise_runs(runs, d, fz, ctx.cfg)
    per_class = {}
    for c in EVENT_ONLY:
        ss = [sid for sid, g in d.groupby("session_id", sort=False) if g.scenario.iloc[0] == c]
        on = {sid: int(d[d.session_id == sid].gt_onset.iloc[0]) for sid in ss}
        lv = [runs[s]["first"].get("VERIFY", 10**9) - on[s] for s in ss]
        lm = [runs[s]["first"].get("MINIMUM_RISK_RESPONSE", 10**9) - on[s] for s in ss]
        per_class[c] = {"n": len(ss), "reach_MRR": sum(1 for s in ss if "MINIMUM_RISK_RESPONSE" in runs[s]["first"]),
                        "median_latency_VERIFY_s": float(np.median([x for x in lv if x < 10**8])) if any(x < 10**8 for x in lv) else None,
                        "median_latency_MRR_s": float(np.median([x for x in lm if x < 10**8])) if any(x < 10**8 for x in lm) else None}
    notes = [r["notification"] for r in runs.values() if r["notification"]]
    # latency measurements
    X = feats(ctx, "C", ctx.test.iloc[:200]); t0 = []
    for i in range(100):
        s = time.perf_counter(); ctx.models["C"].predict_proba(X.iloc[[i]]); t0.append((time.perf_counter() - s) * 1e3)
    tr = lambda sid: [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in runs[sid]["trace"]]
    return {"summary": summ, "per_class": per_class,
            "notifications_sent": len(notes), "notifications_validated": sum(n["validated"] for n in notes),
            "notification_keys_all_in_allowlist": all(set(n["msg"]) <= ALLOWED_KEYS for n in notes),
            "example_notification": notes[0]["msg"] if notes else None,
            "example_event_transitions": [t.__dict__ for t in runs[ex_ev]["transitions"]],
            "example_event_session": ex_ev, "example_normal_session": ex_no,
            "trace_event": tr(ex_ev), "trace_normal": tr(ex_no),
            "vehicle_log_event": [e.__dict__ for e in runs[ex_ev]["vehicle_log"]],
            "latency": {"model_inference_ms_mean_single_window": float(np.mean(t0)), "model_inference_ms_p95": float(np.percentile(t0, 95)),
                        "guardian_step_us_mean": float(np.mean([r["step_us"] for r in runs.values()]))}}


# ----------------------------------------------------------------------------------- G
def exp_G(ctx: Ctx, n_legit=200, n_att=100) -> dict:
    rng = np.random.default_rng(ctx.seed); kw, ke, kx = rng.bytes(32), rng.bytes(32), rng.bytes(32)
    reg = {"wearable-01": {"key": kw, "role": "wearable"}, "edge-01": {"key": ke, "role": "edge"}}
    roles = {r: set(v) for r, v in ctx.cfg["security"]["roles"].items()}
    v = MessageValidator(reg, roles, ctx.cfg["security"]["max_age_s"])
    mk = lambda sid, seq, ts, ty, key, p=None: sign_message(SecureMessage(sid, seq, ts, ty, p or {"hr": 70.0}), key)
    res = {"legit": {"sent": 0, "accepted": 0}}; lat = []; sent_ok = []
    for i in range(n_legit):
        m = mk("wearable-01", i + 1, 1000.0 + i, "WEARABLE_TELEMETRY", kw); t0 = time.perf_counter()
        r = v.validate(m, 1000.0 + i + rng.uniform(0, 0.5)); lat.append((time.perf_counter() - t0) * 1e6)
        res["legit"]["sent"] += 1; res["legit"]["accepted"] += int(r.ok); sent_ok.append(m)
    base = 1000.0 + n_legit
    attacks = {}
    for i in range(n_att):
        now = base + i
        att = {"tampered": SecureMessage("wearable-01", 10_000 + i, now, "WEARABLE_TELEMETRY", {"hr": 190.0},
                                          mk("wearable-01", 10_000 + i, now, "WEARABLE_TELEMETRY", kw).tag),
               "stale": mk("wearable-01", 20_000 + i, now - 60, "WEARABLE_TELEMETRY", kw),
               "replayed_after_delay": sent_ok[int(rng.integers(0, len(sent_ok)))],
               "unknown_sender": mk("wearable-99", 30_000 + i, now, "WEARABLE_TELEMETRY", kx),
               "forged_tag(wrong key)": mk("wearable-01", 40_000 + i, now, "WEARABLE_TELEMETRY", kx),
               "unauthorized_command": mk("wearable-01", 50_000 + i, now, "VEHICLE_COMMAND", kw, {"cmd": "DISABLE_BRAKES"})}
        j = int(rng.integers(0, len(sent_ok)))
        att["replayed_within_freshness_window"] = sent_ok[j]
        for k, m in att.items():
            r = v.validate(m, sent_ok[j].ts + 0.5 if k == "replayed_within_freshness_window" else now); a = attacks.setdefault(k, {"generated": 0, "rejected": 0, "stages": {}})
            a["generated"] += 1; a["rejected"] += int(not r.ok); a["stages"][r.stage] = a["stages"].get(r.stage, 0) + 1
    can = {"key": rng.bytes(32)}; rx = CanReceiver(can["key"]); frames = []; ok_n = 0
    for i in range(60):
        fr = encode_frame(S.NORMAL, NORMAL, 0.1, 0.2, 0b1111, i, can["key"]); frames.append(fr)
        try: rx.decode(fr); ok_n += 1
        except CanFrameError: pass
    can_att = {"modified": 0, "replayed": 0, "forged_key": 0}
    for fr in frames[-20:]:
        for k, bad in (("modified", bytes([fr[0], 255]) + fr[2:]), ("replayed", fr),
                       ("forged_key", encode_frame(S.NORMAL, NORMAL, 0.1, 0.2, 15, 61, rng.bytes(32)))):
            try: rx.decode(bad)
            except CanFrameError: can_att[k] += 1
    # integration: forged wearable telemetry during a normal drive, validation ON vs OFF
    normal = ctx.test[ctx.test.scenario == "NORMAL_DRIVING"]
    d = F.tamper_wearable(normal, 100, 140, np.random.default_rng(ctx.seed))
    integ = {}
    for on in (True, False):
        dd, fz, runs = run_sessions(ctx, d, security_on=on)
        integ["validation_ON" if on else "validation_OFF"] = _summarise_runs(runs, dd, fz, ctx.cfg)["normal_sessions"]
    return {"message_layer": {"legit_acceptance_rate": res["legit"]["accepted"] / res["legit"]["sent"],
                              "validation_latency_us_mean": float(np.mean(lat)), "validation_latency_us_p95": float(np.percentile(lat, 95)),
                              "attacks": {k: {**a, "rejection_rate": a["rejected"] / a["generated"]} for k, a in attacks.items()},
                              "validator_stats": v.stats},
            "can_frames": {"legit_accepted": ok_n, "attacks_each": 20, "rejected": can_att},
            "integration_forged_wearable_in_normal_sessions": integ}


# ----------------------------------------------------------------------------------- I
def exp_I(ctx: Ctx) -> dict:
    cfg, seed = ctx.cfg, ctx.seed; T = cfg["dataset"]["calibration_len_s"]
    clean_flag = sum(1 for b in ctx.bl.values() if not b.is_valid)
    cont = {c: [0, 0] for c in EVENT_ONLY}
    for i in range(cfg["dataset"]["n_drivers"]):
        prof = make_profile(np.random.default_rng([seed, 1, i]), f"D{i:02d}")
        for c in EVENT_ONLY:
            s = simulate_session(prof, "CALIBRATION", np.random.default_rng([seed, 5, i, EVENT_ONLY.index(c)]), cfg, "x",
                                 calm=True, inject_event=c, onset=int(0.4 * T))
            b = build_baseline(s, prof.driver_id, cfg, assume_normal=True)
            cont[c][0] += int(b.status == "INVALID_CONTAMINATED"); cont[c][1] += 1
    short = [build_baseline(g.head(60), d, cfg, assume_normal=True).status for d, g in ctx.df[ctx.df.scenario == "CALIBRATION"].groupby("driver_id")]
    nocam = [build_baseline(F.drop_modality(g, ["visual"]), d, cfg, assume_normal=True).status for d, g in ctx.df[ctx.df.scenario == "CALIBRATION"].groupby("driver_id")]
    try:
        build_baseline(ctx.df[ctx.df.scenario == "CALIBRATION"].head(10), "x", cfg); refused = False
    except CalibrationError:
        refused = True
    return {"clean_calibrations_flagged_invalid": f"{clean_flag}/{len(ctx.bl)}",
            "contaminated_calibration_detected": {c: f"{a}/{n}" for c, (a, n) in cont.items()},
            "truncated_60_windows_flagged_insufficient": f"{sum(s == 'INVALID_INSUFFICIENT' for s in short)}/{len(short)}",
            "no_camera_flagged_insufficient": f"{sum(s == 'INVALID_INSUFFICIENT' for s in nocam)}/{len(nocam)}",
            "calibration_without_explicit_normal_assumption_refused": refused}


# ----------------------------------------------------------------------------------- demo
def demo_two_drivers(ctx: Ctx) -> dict:
    """Same raw observation, two drivers: personalised deviation vs generic output."""
    te_d = sorted(ctx.test.driver_id.unique())
    med = {d: {f: ctx.bl[d].features[f]["median"] for f in ("fsr_asym", "face_asym", "hr_mean")} for d in te_d}
    lo = min(te_d, key=lambda d: med[d]["fsr_asym"] + med[d]["face_asym"] * 2)
    hi = max(te_d, key=lambda d: med[d]["fsr_asym"] + med[d]["face_asym"] * 2)
    base = ctx.test[(ctx.test.driver_id == hi) & (ctx.test.label == NORMAL) & (ctx.test.gt_conf_turn == 0)
                    & (ctx.test.gt_conf_glitch == 0) & (ctx.test.gt_conf_vib == 0) & (ctx.test.gt_conf_stress == 0)]
    row = base.iloc[[int((base.fsr_asym - base.fsr_asym.median()).abs().argmin())]]   # typical normal window of the high-baseline driver
    obs = row.copy(); obs.index = [0]
    obs_lo = obs.copy(); obs_lo["driver_id"] = lo                                 # SAME raw numbers, other driver
    obs_hi = obs.copy()
    res = {"high_baseline_driver": hi, "low_baseline_driver": lo, "raw_observation": {f: float(obs[f].iloc[0]) for f in ("fsr_asym", "face_asym", "hr_mean", "imu_rms")}}
    for tag, o, d in (("driver_high", obs_hi, hi), ("driver_low", obs_lo, lo)):
        z = compute_deviations(o, ctx.bl[d]).iloc[0]
        pA = ctx.models["A"].predict_proba(generic_features(o, ctx.med))[0]
        pB = ctx.models["B"].predict_proba(feats(ctx, "B", o))[0]
        res[tag] = {"baseline_median": {f: ctx.bl[d].features[f]["median"] for f in ("fsr_asym", "face_asym", "hr_mean")},
                    "z": {f: float(z[f]) for f in ("fsr_asym", "face_asym", "hr_mean")},
                    "P_event_generic_A": float(1 - pA[0]), "P_event_personalised_B": float(1 - pB[0]),
                    "pred_A": EVENT_CLASSES[int(pA.argmax())], "pred_B": EVENT_CLASSES[int(pB.argmax())]}
    cal = ctx.df[ctx.df.scenario == "CALIBRATION"]
    res["calibration_distributions"] = {d: {f: cal[cal.driver_id == d][f].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist() for f in ("fsr_asym", "face_asym", "hr_mean")} for d in (hi, lo)}
    return res


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, cls=_Enc)
