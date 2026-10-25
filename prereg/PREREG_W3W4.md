# Pre-registration: out-of-sample test of the swap-vs-recompute advisor (W3, W4)

**Project:** LCSpill Phase 2. **Written:** 25 Oct 2026, before any W3/W4 run exists.
**Rule:** this file, `run_wl.py`, `prereg_lib.py`, `predict_only.py` and `score.py` are committed
*before* the W3/W4 recompute arms run. `predictions_W3W4.json` is committed *before* any swap arm runs.
After that nothing here changes. Any later edit is a new commit and is reported as a deviation.

## 1. Claim under test

Given only (a) the calibration profile `t4_profile_B.json` (kv_profile.py, 24 Oct) and
(b) the log of the **default vLLM run** (always-recompute), the advisor predicts the end-to-end
throughput gain of switching preemption to host swap, on workloads it has never seen.

W1/W2 are in-sample: the model was validated on them. W3/W4 are the test.

## 2. Fixed setup (identical for every run)

| | |
|---|---|
| GPU | Kaggle 2× Tesla T4, one independent vLLM instance per GPU, both started together (barrier) |
| Software | stock vLLM 0.2.7, fresh env, torch 2.1.2+cu121 |
| Model | facebook/opt-1.3b, fp16, block_size 16, swap_space 6 GiB per GPU (W3: 8 GiB, see §3) |
| Requests | 12-token prompt, greedy, `ignore_eos=True` (every request generates exactly `max_tokens`), FCFS |
| Harness | `kvlog.py` (23 Oct, unchanged) + `run_wl.py`; swap-in = Python per-block loop (0.2.7 kernel bug) |
| Arms | `rec` = vLLM default (RECOMPUTE); `swap` = `_preempt` forced to SWAP |
| Wall time T | first `add_request` → last request finished, per GPU; run time = slower GPU |

## 3. Workloads

| WL | Requests / GPU | `max_tokens` | `gpu_mem_util` | Repeats | Status | Design intent |
|---|---|---|---|---|---|---|
| W1 | 256 | 512 fixed | 0.4 | 3 | in-sample (23 Oct) | heavy pressure, short preemptions (mean L ≈ 150) |
| W2 | 256 | U[64, 1024] | 0.4 | 3 | in-sample (23 Oct) | heavy pressure, mixed lengths |
| **W3** | 128 | U[768, 1536] | **0.5** | 3 | **test** | long outputs; fewer but ~2× longer preemptions; more GPU blocks |
| **W4** | 256 | U[128, 448] | **0.8** | 5 | **test** | low pressure: preemptions rare or absent |

Output lengths for W3/W4 come from `random.Random(0)`: 2N lengths are drawn, the ids shuffled, and GPU g
takes `ids[g::2]`. Every repeat and both arms are therefore the same workload.

W3 uses `swap_space` 8 GiB instead of 6. W1's swap arm peaked at 1,838 swapped blocks out of
2,048 (90 %), and a scheduler simulation (which reproduces W1's peak exactly) puts W3's peak at
2,060–2,274 blocks. vLLM 0.2.7 aborts when CPU swap space runs out, so at 6 GiB the W3 swap arm would fail.
At 8 GiB (2,730 blocks) the predicted peak is ~83 %. Swap space does not affect the recompute arm's scheduling.

W4 gets 5 repeats because each run is only ~30–40 s, so relative timing noise is larger.

## 4. Quantities

Per GPU, from the **recompute** log only (`preempt.seq_len` = tokens re-prefilled):

```
T_swap_pred = T_rec − Σ R(L_i) + Σ S(L_i)        R, S from t4_profile_B.json (advisor.CostModel)
predicted gain (run k) = max_g T_rec / max_g T_swap_pred − 1
measured  gain (run k) = max_g T_rec / max_g T_swap      − 1     (same total tokens in both arms)
```

Each workload reports **mean ± sample SD over repeats**, plus the range.

## 5. Criteria (decided now)

1. **Primary (pass/fail):** for each of W3 and W4, |mean predicted − mean measured| ≤ **1.5 pp**,
   using the profile B prediction.
   *Why 1.5 pp:* the measured per-repeat SD of W2 is 1.52 pp, and the in-sample errors of the means are
   +0.36 pp (W1) and −0.37 pp (W2).
2. **Decision (reported):** the advisor recommends SWAP iff the predicted gain is ≥ 1 %.
   It "agrees" if the measured mean is on the same side of 1 %.
   For W4 the design intent is RECOMPUTE ("not worth switching").
3. **Secondary (reported, not pass/fail):** the profile C prediction, and per-repeat errors.

## 6. Procedure and exclusions

1. Commit this file and the scripts. Record the commit hash below.
2. Harness check: replay the W1 r0 assignment with `run_wl.py` (recompute arm).
   Expected: 591 preemptions on both GPUs, 862 GPU blocks, wall time within ±5 % of 104–106 s.
   If the preemption count differs, stop and use the original 23 Oct runner with these W3/W4 parameters.
3. Run the recompute arms: W3 r0–r2, W4 r0–r4.
4. Run `predict_only.py`. Commit `predictions_W3W4.json` and its printout. Record the hash below.
5. Run the swap arms: W3 r0–r2, W4 r0–r4.
6. Run `score.py`. It checks the sha256 of every recompute log and that the frozen prediction is
   reproduced exactly.

- **Invalid run:** a run is invalid only if it crashed or is incomplete (no `done` record, or
  finished ≠ submitted). An invalid run is rerun with the same repeat index, and the rerun is noted
  here. No other run is dropped.
- **On failure:** report it as is, with the decomposition (Σ R and Σ S, model vs measured).
  Do not refit the profile or change any parameter for W3/W4. Model fixes (for example, batched
  re-prefill) are future work and get tested on new workloads.

## 7. Known limitations (stated in advance)

- The runs use 2 GPUs at once; profile B was measured on 1 GPU (`gpu_mem_util` 0.85).
- The profile's prefill fit range is L = 32–2000. W3 preemptions stay inside it (prompt + max_tokens ≤ 1548).
- If W4 has zero preemptions, the prediction is exactly 0 % by construction. The test then shows that
  the advisor says "don't switch", and the measured value is the swap-vs-recompute noise floor.
  It does not test the cost model.

## 8. Commit log

| Step | Commit | Time (IST) |
|---|---|---|
| Pre-registration + scripts | `f5783be` (f5783be4d4b92abb59690b766d22feba45e2bdd8) | 2026-10-25 17:47 IST |
| Harness check result | OK (gpu0 -1.2%, gpu1 +1.6%, eager=True) | 2026-10-25 18:17 IST |
| predictions_W3W4.json | `b8ff65b` (b8ff65b6f3c14d9b1dbce1da0bf002f23cf5faea) | 2026-10-25 19:14 IST |
| Swap arms + score | | |
