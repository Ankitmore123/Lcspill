"""prereg_lib.py -- shared code for predict_only.py and score.py (stdlib + advisor.py)."""
import argparse, hashlib, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advisor import CostModel, load_profile, resolve_geometry  # noqa: E402

MIN_GAIN_PCT = 1.0      # advisor threshold: recommend SWAP only if predicted gain >= 1 %
TOL_PP = 1.5            # pass band, pre-registered (PREREG_W3W4.md)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cost_model(profile_path):
    prof = load_profile(profile_path)
    ns = argparse.Namespace(model=None, num_layers=None, num_kv_heads=None,
                            head_dim=None, dtype=None, block_size=None)
    return CostModel(resolve_geometry(ns, prof), prof)


def load_log(path):
    if not os.path.isfile(path):
        sys.exit(f"[prereg] ERROR: missing log {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def done_rec(recs, path):
    d = [r for r in recs if r["t"] == "done"]
    if len(d) != 1:
        sys.exit(f"[prereg] ERROR: {path}: expected one 'done' record, found {len(d)} (run crashed?)")
    d = d[0]
    if not d["finished"] == d["exact_len"] == d["submitted"]:
        sys.exit(f"[prereg] ERROR: {path}: incomplete run {d}")
    return d


def predict_gpu(cm, path):
    """Prediction for one GPU from the RECOMPUTE log only."""
    recs = load_log(path)
    d = done_rec(recs, path)
    modes = {r["mode"] for r in recs if r["t"] == "preempt"}
    if modes - {"RECOMPUTE"}:
        sys.exit(f"[prereg] ERROR: {path} is not a recompute run (preempt modes {modes})")
    lens = [r["seq_len"] for r in recs if r["t"] == "preempt"]
    Tr = d["gen_wall_s"]
    sum_r = sum(cm.recompute_ms(L) for L in lens) / 1e3
    sum_s = sum(cm.swap_ms(L) for L in lens) / 1e3
    return {"Tr": Tr, "Tp": Tr - sum_r + sum_s, "n": len(lens),
            "meanL": st.mean(lens) if lens else 0.0, "maxL": max(lens) if lens else 0,
            "sumR": sum_r, "sumS": sum_s, "f": sum_r / Tr, "gen_tokens": d["gen_tokens"],
            "sha256": sha256(path)}


def predict_run(cm, runs, wl, k):
    g = [predict_gpu(cm, os.path.join(runs, f"{wl}_rec_r{k}", f"gpu{i}.jsonl")) for i in (0, 1)]
    Tr, Tp = max(x["Tr"] for x in g), max(x["Tp"] for x in g)
    return {"k": k, "Tr": Tr, "Tp": Tp, "pred_pct": 100 * (Tr / Tp - 1), "gpus": g}


def measure_run(runs, wl, k, rec_gpus):
    Ts, toks = [], []
    for i in (0, 1):
        path = os.path.join(runs, f"{wl}_swap_r{k}", f"gpu{i}.jsonl")
        recs = load_log(path)
        d = done_rec(recs, path)
        if not any(r["t"] == "meta" and r.get("force_swap") for r in recs):
            sys.exit(f"[prereg] ERROR: {path} is not a forced-swap run")
        if d["gen_tokens"] != rec_gpus[i]["gen_tokens"]:
            sys.exit(f"[prereg] ERROR: {path}: arms generated different token counts")
        Ts.append(d["gen_wall_s"])
    Tr = max(x["Tr"] for x in rec_gpus)
    return {"Ts": max(Ts), "meas_pct": 100 * (Tr / max(Ts) - 1)}


def msd(xs):
    xs = list(xs)
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def recommend(pred_pct):
    return "SWAP" if pred_pct >= MIN_GAIN_PCT else "RECOMPUTE"


def repeats_for(spec, wl):
    """'3' -> 3 for every workload; 'W3=3,W4=5' -> per workload (default 3)."""
    if "=" not in spec:
        return int(spec)
    d = dict(x.split("=") for x in spec.split(","))
    return int(d.get(wl, 3))
