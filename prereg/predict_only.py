#!/usr/bin/env python3
"""
predict_only.py -- PRE-REGISTERED prediction. Reads ONLY the recompute runs
(<WL>_rec_r<k>/gpu<g>.jsonl) and the calibration profile(s); writes the advisor's
predicted swap gain to a JSON file that must be committed BEFORE any swap arm runs.

Refuses to run if a swap-arm folder for the workload already exists.

Usage:
  python predict_only.py --profile t4_profile_B.json --profile t4_profile_C.json \
      --runs runs --workloads W3,W4 --out predictions_W3W4.json
The FIRST --profile is the primary one (pass/fail is judged on it).
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_lib as L  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="append", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--workloads", default="W3,W4")
    p.add_argument("--repeats", default="W1=3,W2=3,W3=3,W4=5", help="'3' or 'W3=3,W4=5'")
    p.add_argument("--out", default="predictions_W3W4.json")
    p.add_argument("--force", action="store_true", help="testing only: ignore existing swap folders")
    a = p.parse_args()
    wls = a.workloads.split(",")

    for wl in wls:
        if glob.glob(os.path.join(a.runs, f"{wl}_swap_r*")) and not a.force:
            sys.exit(f"[predict_only] REFUSED: {wl}_swap_* already exists -> this would not be a "
                     "pre-registered prediction.")

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "primary_profile": os.path.basename(a.profile[0]),
           "min_gain_pct": L.MIN_GAIN_PCT, "tol_pp": L.TOL_PP,
           "profiles": {os.path.basename(pp): L.sha256(pp) for pp in a.profile}, "workloads": {}}

    print("=" * 92)
    print(" PRE-REGISTERED PREDICTION (recompute runs only; swap arms not yet run)")
    print("=" * 92)
    for wl in wls:
        out["workloads"][wl] = {}
        for pp in a.profile:
            cm = L.cost_model(pp)
            runs = [L.predict_run(cm, a.runs, wl, k) for k in range(L.repeats_for(a.repeats, wl))]
            m, s = L.msd(r["pred_pct"] for r in runs)
            g = [x for r in runs for x in r["gpus"]]
            name = os.path.basename(pp)
            out["workloads"][wl][name] = {
                "pred_pct_per_repeat": [r["pred_pct"] for r in runs],
                "pred_pct_mean": m, "pred_pct_sd": s, "recommendation": L.recommend(m),
                "T_rec_s": [r["Tr"] for r in runs], "T_swap_pred_s": [r["Tp"] for r in runs],
                "preemptions_per_gpu": [x["n"] for x in g], "mean_preempt_len": [x["meanL"] for x in g],
                "f_recompute_frac": [x["f"] for x in g],
                "rec_log_sha256": {f"{wl}_rec_r{r['k']}/gpu{i}.jsonl": r["gpus"][i]["sha256"]
                                   for r in runs for i in (0, 1)}}
            tag = "PRIMARY" if pp == a.profile[0] else "secondary"
            print(f"\n {wl}  [{name}, {tag}]")
            print(f" {'run':<5}{'T_rec s':>9}{'T_pred s':>10}{'preempt g0/g1':>15}{'mean L':>8}"
                  f"{'f (g0)':>8}{'sumR-sumS s':>13}{'pred gain':>11}")
            for r in runs:
                g0, g1 = r["gpus"]
                print(f" r{r['k']:<4}{r['Tr']:>9.1f}{r['Tp']:>10.1f}{g0['n']:>8}/{g1['n']:<6}"
                      f"{g0['meanL']:>8.0f}{g0['f']:>8.3f}{g0['sumR'] - g0['sumS']:>13.2f}"
                      f"{r['pred_pct']:>10.2f}%")
            print(f" -> predicted gain {m:+.2f} % (sd {s:.2f} over repeats)   "
                  f"RECOMMENDATION: {L.recommend(m)}")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 92)
    print(f" wrote {os.path.abspath(a.out)}")
    print(f" sha256 {L.sha256(a.out)}")
    print(" NEXT: commit + push this file (and this printout) BEFORE running any swap arm.")
    print("=" * 92)


if __name__ == "__main__":
    main()
