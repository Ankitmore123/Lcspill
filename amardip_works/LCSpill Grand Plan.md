# LCSpill Grand Plan

Oct 20, 2026 · @Sahil More

## Goal and end state

We will show that FuseSpill's GPU-to-GPU migration design admits requests to GPU 1 that GPU 1 later throws back, fix that with growth-aware admission, and measure the gain against released FuseSpill.

**Done means:**

1. Released FuseSpill, our fixes, and single-GPU vLLM measured with at least 3 runs each, on two workloads (identical prompts, and mixed-length ShareGPT).
2. One new mechanism, growth-aware admission for GPU 1, built behind a flag and shown to reduce evictions and change throughput by a measured amount.
3. Supporting results kept as they stand: the paper-vs-code audit, copy costs, and the race (about 2,000 per run, fixed to 0).
4. A report where every claim points to a log line or a file:line.

The companion doc *LCSpill: What We Found So Far* holds the measured results; this doc holds the plan.

## Where we stand

Five results are established; four questions are open, and Phases 1–2 close them.

| Result | Evidence | Status |
| --- | --- | --- |
| Paper vs code audit: cost model is a stub, α hardcoded, copy blocks compute | file:line references | Established |
| Copy costs: 82 ms as released, 19–23 ms pinned, 20.5 ms P2P (1,024-token request) | Microbenchmark, 2 sessions | Established |
| Race: host blocks reused before GPU 1 reads them, 1,890–2,475 per run; fix gives 0 | Counter, 6 runs | Established |
| Released FuseSpill 1,572 tokens/s (4 runs) vs single-GPU vLLM 1,216 (2 runs) | `results.jsonl` | Established, synthetic workload only |
| Released FuseSpill: 495 requests thrown back from GPU 1 for recompute | v4 counter, queue log | Established, 1 run |

**Open:**

- **Pinned runs crash.** `AssertionError: Waiting sequence group should have only one prompt sequence`, then a hang until `timeout` kills the run. Cause not yet known (Phase 1).
- **Does the race fix cost throughput?** Two runs fall inside released FuseSpill's own spread (Phase 2).
- **Are evictions stable across runs?** One run so far (Phase 2).
- **Every run so far uses 512 identical prompts.** A mixed-length workload is still needed (Phase 4).

**Withdrawn:** "pinned copies made the system much slower." That run crashed; it was not slow.

## Rules for every experiment

Every run follows the same protocol, so any two results can be compared.

1. **One setup notebook.** After `pip install -e` succeeds, use Save Version, so a session reset costs minutes, not an hour.
2. **Never edit the repo by hand.** Every change is a patch cell gated by a flag file. With every flag off, the code is FuseSpill as released.
3. **One run script** (`cell3.py`, current version v5). One command per run; one line appended to `results.jsonl`; one log per run.
4. **Fixed configuration:** OPT-1.3B, GPU memory 0.4, 512 requests × 512 tokens, 8 GiB host buffer. Change one thing at a time and write down what changed.
5. **At least 3 runs per variant.** Report mean and range. Released FuseSpill alone spans 1,355–1,761 tokens/s, so differences under about 15% need more runs.
6. **After every run**, save the RESULT line and the queue sample: `grep "waiting =" <log> | awk 'NR%100==1'`. Also save any `Traceback` or `FS_DIAG` lines.
7. **Before every run**, clean up (`pkill -9 -f "ray::"; pkill -9 -f cell3.py`), and wrap the run in `timeout 900`.
8. **No RESULT line = no number.** A crash or timeout is logged as such, with its log tail. It is never described as "slow".
9. **Every claim cites a log line or a file:line.** That includes anything an LLM says.
10. **Patches are applied in order:** cell 5, cell 6, cell 7, and later ones in sequence. Each prints `patched` or `already patched`; anything else means stop.

## Roadmap

&#91;embedded content: LCSpill roadmap · 7 phases, 7 discussion slots\]

Phases 0–2 settle the baseline and the crash; Phase 3 is the contribution; Phases 4–5 test it under realistic conditions. Each diamond is a discussion slot with its pass criterion.

## Phase by phase

Each phase ends in a discussion slot. The next phase starts only when its pass criteria are met. Dates are a proposal; shift them to fit the review calendar.

### Phase 0: Clarity (21–23 Oct) · owner: Amardip

- **Day 1, paper:** read Sections III-B, IV-A, IV-C and V-A, alongside `fusespill_paper_vs_code.md`.
- **Day 2, code:** follow one request through `scheduler.py` (`MultiScheduler`: `_schedule_main`, `_schedule_aux`, `_preempt`, `_aux_preempt_by_recompute`), then `multi_block_manager.py` (`swap_out`, `swap_in`, `can_swap_in`, the allocator), `multi_worker.py` (`execute_model` on both workers), `multi_gpu_cache_engine.py`, `cpu_cache_engine.py`, `csrc/cache_kernels.cu` (`swap_blocks`), `llm_engine.py` (`step`, `handle_*_result`) and `asy_submmitter.py`.
- **Day 3, logs:** explain every field of a `waiting = …` line, using `v4_base_a.log`.
- **Deliverable:** `phase0_answers.md`, with five questions answered using file:line evidence:
  1. Where does GPU 0 wait for the copy?
  2. When is a host block freed, and when is it actually safe to free it?
  3. What check decides whether GPU 1 accepts a request?
  4. What happens when GPU 1 runs out of room mid-decode?
  5. Why do the waiting-queue numbers grow while GPU 1 is full?

  Plus a corrected `stage4.md`, and the stale claims removed from the handwritten notes.
- **Pass:** Ankit picks 5 random file:line references and all 5 check out.

### Phase 1: Fix the crash (24–25 Oct) · owner: Amardip

- Apply cell 7 (daemon threads, the `FS_DIAG` line, the `FS_FIX2` flag). Run `pinbpfix 8` once.
- If it crashes: from the `FS_DIAG` lines, write down which cause it is, a duplicate entry or a wrong state. Send that to Claude for a patch.
- **Deliverable:** `phase1_crash.md`, 1 paragraph plus the log lines.
- **Pass:** `pinbpfix` completes with a RESULT line, or the cause is identified from `FS_DIAG` output.

### Phase 2: Lock the baseline (26–28 Oct) · owner: Amardip

- Runs: `base` ×2 more (5 total), `basefix` ×3, `defer` ×1 more (3 total), single-GPU `base` ×1 more (3 total), `pinbpfix` ×3.
- **Deliverable:** `phase2_baseline.md`, one table: variant, runs, mean and range of tokens/s, races, evictions, duplicates dropped.
- **Pass:** every variant has 3 or more runs; evictions are measured on at least 3 `base` runs.

### Phase 3: Build growth-aware admission (29 Oct–6 Nov) · owners: Claude writes the patch, Amardip runs it, Ankit reviews

- **Design note first** (1 page, Ankit approves): admit a request to GPU 1 only if its blocks *at full length* fit beside what GPU 1 already holds at their full lengths. Full length = prompt + `max_tokens` (the length predictor comes in Phase 4).
- Patch behind flag `FS_ADMIT`, plus variant `admit` in `cell3.py`.
- Runs: `admit` ×3 vs `base` (from Phase 2).
- **Deliverable:** design note, patch, results table.
- **Pass:** evictions drop clearly below `base`'s; throughput change reported with its range, whichever direction it goes.

### Phase 4: Realistic workload (7–13 Nov) · owner: Amardip

- Switch the prompts to ShareGPT, which the paper uses. Output lengths vary per request.
- Use the length predictor's estimate for full length, and compare against `max_tokens`.
- Runs: `base`, `admit` (`max_tokens`), `admit` (predictor), ×3 each.
- **Pass:** same table as Phase 3, on mixed lengths.

### Phase 5: Retest the faster copy path (14–17 Nov) · owner: Amardip

- Runs: `admit` + pinned buffer, ×3.
- **Question:** does a faster copy help once GPU 1 stops throwing requests back?
- **Pass:** a yes or no with numbers. Either answer goes in the report.

### Phase 6: Write-up (18 Nov–8 Dec) · owner: Ankit, with Amardip

- Chapters: problem; FuseSpill and the paper-vs-code audit; measurements; race; admission design; evaluation; future work (P2P, overlapped transfer).
- **Pass:** every claim in the report traced to a log line or file:line.

## Discussion slots

Each slot is 30 minutes and has one job: check the deliverable against its pass criteria, then decide the next step.

**Format:**

- Amardip shows the deliverable (10 min).
- Ankit checks evidence at random (10 min).
- The decision is written in the last column (10 min).

| Slot | Date | Reviews | Decision to take |
| --- | --- | --- | --- |
| 1 | 23 Oct | Phase 0 answers + corrected `stage4.md` | Does Amardip own the code path? If not, one more day of Phase 0 |
| 2 | 25 Oct | Phase 1 crash note | Crash fixed, or which patch to write next |
| 3 | 28 Oct | Phase 2 baseline table | Freeze the baseline numbers; approve the Phase 3 design note |
| 4 | 6 Nov | Phase 3 admission results | Keep admission as the main contribution, or adjust the design |
| 5 | 13 Nov | Phase 4 ShareGPT results | Final evaluation set agreed |
| 6 | 17 Nov | Phase 5 pinned-buffer result | Report outline locked |
| 7 | Weekly from 24 Nov | Draft chapters | Chapter sign-off |

**Between slots:** a 5-line daily update in the group chat: what ran, RESULT lines, blockers. No update = the day counts as not done.

## Work schema

All work lives in the GitHub repo `Ankitmore123/Lcspill`, laid out as below, so anyone can find any result in under a minute.

```
Lcspill/
  notebook/        lcspill_handover.ipynb (setup + every cell, in order)
  patches/         cell5_fix.py, cell6_backpressure.py, cell7_crashfix.py, ... (never edited after use; new version = new file)
  runs/            results.jsonl (every RESULT line), one log per run, queue samples
  phases/          phase0_answers.md, phase1_crash.md, phase2_baseline.md, ...
  reference/       fusespill_paper_vs_code.md, the paper PDF
  CLAIMS.md        every claim the report will make, each with its evidence
```

**Run log names:** `<phase>_<variant>_<swapGiB>_<run letter>.log`, for example `p2_base_8_c.log` or `p3_admit_8_a.log`.

**Fields every RESULT line carries** (written by `cell3.py` v5):

| Field | Meaning |
| --- | --- |
| `variant`, `swap_gib`, `multi` | What was run |
| `elapsed_s`, `gen_tok_per_s` | Speed |
| `migrated` | Requests moved to GPU 1 |
| `races` | Host blocks reused before GPU 1 read them |
| `gpu1_evictions` | Requests GPU 1 threw back for recompute |
| `bp_offloads_skipped`, `bp_spills_recomputed` | Backpressure activity |
| `duplicates_dropped` | Duplicate waiting entries removed (`FS_FIX2`) |
| `handoff_ms_p50_p90` | Time from leaving GPU 0 to being scheduled on GPU 1 |

**`CLAIMS.md` format, one row per claim:**

| Claim | Evidence (log line or file:line) | Runs | Status |
| --- | --- | --- | --- |
| Released FuseSpill reuses host blocks before GPU 1 reads them | `results.jsonl` races field; `multi_block_manager.py` `swap_in` | 6 | Established |

## Roles and risks

| Who | Owns |
| --- | --- |
| Ankit | Decisions at every slot, evidence checks, design-note approval, the report |
| Amardip | Runs, logs, code reading, phase deliverables, daily update |
| Claude | Patch code, design review, checking claims against code and paper |

| Risk | Sign | Fallback |
| --- | --- | --- |
| `pinbpfix` still crashes after cell 7 | No RESULT line; `FS_DIAG` printed | Drop the pinned path from the main story; it becomes a measured negative result. Phases 3–4 do not depend on it. |
| Admission cuts evictions but not throughput | `admit` within `base`'s range | Report it as is: fewer wasted copies and recomputes, with the reason GPU 1 still limits throughput |
| The length predictor is not usable in the code | Phase 4 setup fails | Use the dataset's true lengths as an oracle upper bound, and say so in the report |
| Kaggle session resets mid-phase | Setup missing | Saved notebook version; patches are re-applied from `patches/` in order |
| Run-to-run noise hides the effect | Ranges overlap | More runs (5 per variant) before any claim |
| Time runs out | Phase 6 starts late | Stop after Phase 3. Audit + race + evictions + admission is a complete BE project |
