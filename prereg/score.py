#!/usr/bin/env python3
"""
score.py -- score predictions against measured swap runs, mean +/- SD over repeats.

For W3/W4 it checks that the frozen predictions (predictions_W3W4.json) match a fresh
recomputation from the SAME recompute logs (sha256 checked) -> proves nothing was retuned.
W1/W2 are scored too but labelled IN-SAMPLE (the profile was validated on them).

Usage:
  python score.py --profile t4_profile_B.json --profile t4_profile_C.json --runs runs \
      --prereg predictions_W3W4.json --workloads W1,W2,W3,W4 --json-out score_W1-W4.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_lib as L  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="append", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--workloads", default="W1,W2,W3,W4")
    p.add_argument("--repeats", default="W1=3,W2=3,W3=3,W4=5", help="'3' or 'W3=3,W4=5'")
    p.add_argument("--json-out", default="score_W1-W4.json")
    a = p.parse_args()
    pre = json.load(open(a.prereg))

    for pp in a.profile:
        name = os.path.basename(pp)
        if name in pre["profiles"] and L.sha256(pp) != pre["profiles"][name]:
            sys.exit(f"[score] ERROR: {name} changed since pre-registration")

    results, W = {}, 118
    for pp in a.profile:
        name = os.path.basename(pp)
        primary = name == pre["primary_profile"]
        cm = L.cost_model(pp)
        print("=" * W)
        print(f" SCORE  profile {name} ({'PRIMARY' if primary else 'secondary'})   "
              f"pass band |pred - meas| <= {L.TOL_PP} pp on the mean over repeats")
        print("=" * W)
        print(f" {'WL':<4}{'status':<14}{'preempt/GPU':>12}{'mean L':>8}{'measured gain % (mean, sd, range)':>36}"
              f"{'predicted %':>17}{'error pp':>10}{'advisor':>11}{'decision':>10}{'verdict':>9}")
        for wl in a.workloads.split(","):
            runs = [L.predict_run(cm, a.runs, wl, k) for k in range(L.repeats_for(a.repeats, wl))]
            meas = [L.measure_run(a.runs, wl, r["k"], r["gpus"]) for r in runs]
            pm, ps = L.msd(r["pred_pct"] for r in runs)
            mm, ms = L.msd(x["meas_pct"] for x in meas)
            lo, hi = min(x["meas_pct"] for x in meas), max(x["meas_pct"] for x in meas)
            err = pm - mm
            frozen = pre["workloads"].get(wl, {}).get(name)
            if frozen:
                for r in runs:
                    for i in (0, 1):
                        key = f"{wl}_rec_r{r['k']}/gpu{i}.jsonl"
                        if r["gpus"][i]["sha256"] != frozen["rec_log_sha256"][key]:
                            sys.exit(f"[score] ERROR: {key} changed since pre-registration")
                if abs(frozen["pred_pct_mean"] - pm) > 1e-9:
                    sys.exit(f"[score] ERROR: {wl} prediction differs from the frozen one")
                status = "PRE-REG"
            else:
                status = "in-sample"
            rec = L.recommend(pm)
            decision_ok = (rec == "SWAP") == (mm >= L.MIN_GAIN_PCT)
            verdict = "PASS" if abs(err) <= L.TOL_PP else "FAIL"
            n = L.st.median(x["n"] for r in runs for x in r["gpus"])
            mL = L.st.median(x["meanL"] for r in runs for x in r["gpus"])
            print(f" {wl:<4}{status:<14}{n:>12.0f}{mL:>8.0f}"
                  f"{f'{mm:+.2f} +/- {ms:.2f} [{lo:+.1f},{hi:+.1f}]':>36}"
                  f"{f'{pm:+.2f} +/- {ps:.2f}':>17}{err:>+10.2f}{rec:>11}"
                  f"{('agrees' if decision_ok else 'DISAGREES'):>10}{verdict:>9}")
            results.setdefault(name, {})[wl] = {
                "status": status, "measured_pct": [x["meas_pct"] for x in meas],
                "predicted_pct": [r["pred_pct"] for r in runs], "meas_mean": mm, "meas_sd": ms,
                "pred_mean": pm, "pred_sd": ps, "error_pp": err, "recommendation": rec,
                "decision_ok": decision_ok, "verdict": verdict,
                "T_rec_s": [r["Tr"] for r in runs], "T_swap_s": [x["Ts"] for x in meas],
                "T_swap_pred_s": [r["Tp"] for r in runs]}
        print()
    json.dump(results, open(a.json_out, "w"), indent=1)
    print(f" verdict  = PASS iff |mean predicted - mean measured| <= {L.TOL_PP} pp (pre-registered)")
    print(f" decision = advisor says SWAP iff predicted >= {L.MIN_GAIN_PCT}%; 'agrees' if the measured mean is on "
          f"the same side of {L.MIN_GAIN_PCT}% (reported, not part of the verdict)")
    print(f" wrote {os.path.abspath(a.json_out)}")


if __name__ == "__main__":
    main()
