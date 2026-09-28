# LCSpill — Novelty, Feasibility & Plan (Conclusion)

*As of 2026-09-29 · Ankit More*

---

## TL;DR

LCSpill is feasible as a **simulation-first project validated by a minimal real prototype on 2×T4**. It is not feasible as a full reproduction of FuseSpill.

- **Contribution II as currently worded ("Adaptive Cost Model") is not novel.** FuseSpill's paper already claims an adaptive α. Keep the work, reframe the claim as **closed-loop runtime α control** vs. FuseSpill's open-loop analytical α.
- **"Filling the stub" is reproduction work.** It's fine for the BE report, but it isn't a paper contribution.
- **Pure Python simulation alone will not pass a reviewer.** Calibrated and validated simulation will, for a workshop or IEEE conference paper. A TPDS-level journal is out of reach.
- **Do not tell the guide real implementation is "impossible."** A minimal prototype is possible. Pitch a scoped plan instead.

---

## 1. What FuseSpill actually does with α

FuseSpill's offload split policy is **adaptive in claim, analytical and open-loop in method, and manually tuned in evaluation.**
Source: Jiang et al., *Efficient KV Cache Spillover Management on Memory-Constrained GPU for LLM Inference*, IEEE TPDS vol. 37 no. 1, Jan 2026.

| Aspect | What the paper says | Where |
|---|---|---|
| Definition of α | Proportion of spilled sequences decoded on the auxiliary device | Sec. IV-B3 |
| How α is set | Cost model balances primary vs. auxiliary time: `T_pr = T_pf + (1−α)·T_dc`, `T_ax = T_sp + α·T_dc`, target `T_ax ≈ T_pr` | Sec. IV-B3, eq. 5–7 |
| Inputs | Roofline estimates: peak FLOPS, memory bandwidth, PCIe bandwidth, model hyperparameters | Sec. IV-B1 |
| "Adaptive" claim | α reduced when auxiliary is weaker, raised when closer to primary; offload triggered by device load + cost model | Sec. IV-A, IV-C3 |
| How α was evaluated | "Experimental and manual adjustments" of α | Sec. V-D3 |
| Accuracy | Predicted optimal α ≈ 0.72, measured ≈ 0.6 (≈12% deviation, vicuna-7b on A10) | Sec. V-D3, Fig. 15 |
| Their proposed fix | Run experiments, add a scaling factor to the cost model | Sec. V-D3 |

The paper also argues against fixed, one-size-fits-all spillover policies (Sec. I, III-B). So "static vs. adaptive" is already FuseSpill's framing. It isn't a gap you can claim.

---

## 2. Novelty verdict per contribution

Contribution II survives only if reframed. The other two need their differences stated explicitly.

| Contribution | As currently written | Verdict | What to claim instead |
|---|---|---|---|
| I. Recency Stratified Sequence Selection (RSSS) | Selects sequences by KV cache age | Likely OK, but check | State clearly how recency differs from FuseSpill's length-aware selection (Sec. IV-E) |
| II. Adaptive Cost Model / filling the stub | "Adaptive α" + implementing `compute_offload_portion()` | **Not novel as worded** | **Closed-loop runtime α control:** a PID controller driven by measured signals, replacing FuseSpill's open-loop analytical estimate, with no offline calibration |
| III. Simulation-first methodology | Simulator for parameter sweeps | Methodology, not a result | Credible only if calibrated and validated against real runs (see Section 5) |

**The winnable claim for Contribution II:** it matches FuseSpill's analytical α in steady state, and beats it under non-stationary workloads (shifting arrival rate or length mix) and under cost-model miscalibration, with no manual tuning.

**Required baseline:** FuseSpill's own analytical α (eq. 5–7, about 20 lines of Python), not just the hardcoded stub. Beating a stub proves nothing.

**Fallback if PID underperforms:** online auto-calibration of FuseSpill's cost model. This automates the scaling-factor fix the paper leaves manual.

**Discarded idea:** pivoting to 32K+ long-context optimization. It wasn't grounded in the project scope, so ignore it.

---

## 3. Fix before writing: α mismatch and unverified claims

Your Context.md and FuseSpill define α differently, and a reviewer will catch it.

- **FuseSpill α:** proportion of *sequences* decoded on the auxiliary device.
- **LCSpill Context.md α:** fraction of *a sequence's KV cache* on CPU vs. GPU.

Either adopt FuseSpill's definition, or state explicitly that you redefine α and why. You cannot claim to improve their α while measuring a different quantity.

- [ ] Open the FuseSpill repo (github.com/JIANGJZ/TDoppelad) and confirm `compute_offload_portion()` returns a hardcoded value; note the file and line numbers
- [ ] Settle one α definition and update Context.md
- [ ] Write the RSSS vs. length-aware-selection distinction in one paragraph

---

## 4. Feasibility and hardware

The simulator is the product; real hardware calibrates and validates it. The trap to avoid is porting FuseSpill's actual codebase to Kaggle first.

### Testbed comparison

| | FuseSpill testbed | LCSpill (Kaggle) |
|---|---|---|
| GPUs | 2×A10, 4090+3090, 4090+A10 (24 GB each) | 2×T4 (16 GB each) |
| Model | 7B fp16 (~14 GB weights) | 7B fp16 leaves ~1 GB for KV, which is unusable. Use 1B–3B, or 4-bit 7B |
| Host RAM | 500 GB | ~30 GB |
| Stack | vLLM 0.2.7, Ray 2.5.1, Python 3.8 | Much newer Python/CUDA; old vLLM likely conflicts |
| GPU time | Unlimited | ~30 GPU-hrs/week, sessions can die |

### Component feasibility

| Component | Feasibility | Notes |
|---|---|---|
| Simulator (III) | Fully feasible, CPU only | Build from FuseSpill's roofline + PCIe equations |
| Analytical-α baseline + PID (II) | Fully feasible | Runs inside the simulator |
| RSSS (I) | Fully feasible | Runs inside the simulator |
| T4 calibration runs | Feasible with small models | Force spillover by capping KV budget (e.g. vLLM `gpu_memory_utilization`); measure PCIe bandwidth, prefill/decode throughput vs. batch size, swap latency |
| Minimal real prototype | Feasible, moderate risk | HF transformers, 2×T4, 1–3B model, move KV GPU0 → CPU → GPU1, run fixed / analytical / PID α |
| Full FuseSpill reproduction | **Not feasible** | Old multi-process vLLM + Ray stack; months of work that isn't your contribution |

**Risk to check this week:** which vLLM version runs on T4 (compute capability 7.5) in Kaggle. The fallback is plain HF transformers.

**Extra hardware options:**
- Ask the college/department about a GPU server, which would be free.
- Rent 2×3090 or 4090 on Vast.ai / RunPod: roughly ₹2–4k for 30–50 hours (approximate, verify current prices). Use this only for a final validation run.

---

## 5. Will simulation hold up with reviewers?

Pure simulation will not. Calibrated simulation validated against a small real prototype will, for a workshop or IEEE conference paper. **Target row 3.**

| Evidence | BE report | Workshop / IEEE conf. | Journal (TPDS, MLSys) |
|---|---|---|---|
| Pure Python sim, assumed parameters | OK | Rejected | Rejected |
| Sim calibrated from real T4 measurements | OK | Borderline | Rejected |
| **Calibrated sim + validated against a minimal real prototype** | OK | **Acceptable** | Weak |
| Full real implementation | OK | OK | OK |

A reviewer won't ask "why simulate?" They'll ask "how do I know your simulator is right?" Answer it with four things:

1. **Calibration:** use measured T4 numbers, not spec sheets.
2. **Validation:** run 2–3 small real configurations and show sim vs. real throughput within ~10–15%.
3. **Rank consistency:** show the policy ordering (fixed < analytical < PID) is the same in sim and real. This is the strongest defense, since the claim is comparative.
4. **Sensitivity analysis:** show the conclusion holds across a parameter range.

**Precedent to cite (verify details first):** Vidur (MLSys 2024), an LLM inference simulator validated against real runs; Splitwise's simulator.

**If calibration "fails":** calibration itself is basic PyTorch microbenchmarking and won't fail. Validation can. If it does, report the error honestly, lean on rank consistency, and add missing overheads as measured constants (e.g. synchronization, which cost FuseSpill ~50% GPU utilization). The worst case is BE report only, no paper.

---

## 6. Pitch to the guide

Go in with a scoping proposal, not a list of reasons it can't be done.

- **You can defend:** full FuseSpill reproduction isn't feasible, and a journal paper is unrealistic.
- **You can't defend:** "real implementation is impossible." A minimal prototype is possible.

### Points to make

1. **Hardware gap:** FuseSpill used 24 GB GPUs and 500 GB host RAM. We have 2×16 GB T4, ~30 GB RAM, and ~30 GPU-hrs/week. A 7B fp16 model leaves almost no KV room on a T4.
2. **Software gap:** FuseSpill runs on vLLM 0.2.7 / Ray 2.5.1 / Python 3.8, which conflict with Kaggle. Its `compute_offload_portion()` is a stub (show the exact lines).
3. **Team and time:** two people, about six months, alongside coursework.
4. **Accepted methodology:** calibrated, validated simulation is standard in LLM serving research.

### Proposal

- Simulator calibrated with real T4 measurements
- Minimal real prototype on 2×T4 with a 1–3B model and forced spillover, to validate the simulator
- Comparison: fixed α vs. FuseSpill analytical α vs. PID α, plus RSSS
- Target: BE report for sure; IEEE conference or workshop paper if results hold

### Likely questions

| Guide asks | Answer |
|---|---|
| Can't you use college lab GPUs? | Check before the meeting and give the actual answer |
| Why not rent GPUs? | ~₹2–4k for 30–50 hrs on 2×3090; offered as optional final validation |
| Isn't simulation weak? | Alone, yes. That's why we validate against a real prototype and show the ranking holds in both |

---

## 7. Timeline and next actions

Front-load the risky real-hardware work by building a tiny prototype first, so problems surface in week 2, not month 4. The job starts in January, so most engineering must land by December.

1. **Oct 2026:** tiny prototype (KV move GPU0 → CPU → GPU1 on Kaggle T4s); T4 calibration microbenchmarks; confirm the α definition and the stub
2. **Nov 2026:** simulator + fixed-α and FuseSpill analytical-α baselines
3. **Dec 2026:** PID controller + RSSS in the simulator; steady-state, non-stationary and miscalibration experiments
4. **Jan–Mar 2027:** finish prototype validation (sim vs. real, rank consistency); write-up
5. **Apr 2027:** BE report + paper submission

### This week

- [ ] Verify the stub in the FuseSpill repo and screenshot the lines
- [ ] Check which vLLM / HF setup runs on Kaggle T4
- [ ] Ask the department about GPU server access
- [ ] Draft the guide pitch from Section 6
- [ ] Rename Contribution II in Context.md to "closed-loop runtime α control"
