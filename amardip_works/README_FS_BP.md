# FS_BP Host-Buffer Backpressure — Experiment README

## 1. What was already implemented

The FuseSpill migration experiment already had these variants:

- `base` — original FuseSpill migration behavior.
- `defer` — delays CPU-block reuse/freeing to remove the observed swap-in race.
- `pin` — `defer` + pinned `/dev/shm` host KV buffer for faster transfers.

Earlier observed results:
- Base multi-GPU: 1559.9 tok/s, 2151 races.
- Base multi-GPU: 1354.6 tok/s, 2055 races.
- Defer: 1427.3 tok/s, 0 races.
- Defer: 1169.9 tok/s, 0 races.
- PIN did not complete because CPU swap space was exhausted.

## 2. New 10 GiB PIN investigation

The host buffer was increased from 8 GiB to 10 GiB.

The 10 GiB PIN run still aborted with:

`RuntimeError: Aborted due to the lack of CPU swap space.`

The scheduler queue showed host-side accumulation. `swapped` reached values including:

- 188
- 267
- 295
- 317
- 333

This showed that increasing the buffer alone did not solve the underlying host-buffer pressure.

## 3. New change: FS_BP host-buffer backpressure

A new optional feature flag was added:

`/kaggle/working/FS_BP`

The scheduler changes are marked with `# FS_BP`.

The feature is OFF unless the flag file exists.

When FS_BP is ON:

1. Proactive GPU-1 offload is skipped if it would leave less than the configured reserve of host-buffer blocks.
2. If a forced spill cannot fit in the host buffer, the request falls back to recompute instead of aborting the whole run.
3. The request goes back to the waiting queue and is re-prefilled on GPU 0.

The reserve is read from the FS_BP file. `cell3.py` uses:

`0.25`

meaning approximately 25% of CPU host-buffer blocks are kept as reserve.

## 4. New counters

The scheduler now records:

- `_fs_bp_skipped` — proactive offloads skipped because host-buffer room was insufficient.
- `_fs_bp_recompute` — forced spills handled by recomputation.
- `gpu1_evictions` — GPU-1 eviction/recompute-related counter exposed by the benchmark.

## 5. New cell3.py variant

`cell3.py` now supports:

- `base` = original FuseSpill
- `defer` = race fix
- `pin` = race fix + pinned `/dev/shm` buffer
- `pinbp` = pin + host-buffer backpressure

For `pinbp`, the script enables `FS_DEFER`, `FS_PIN`, and `FS_BP`, with a 0.25 reserve.

The feature flags are created before importing vLLM so the worker/scheduler code sees them at startup.

## 6. Benchmark result available

A BASE multi-GPU run completed successfully:

- variant: `base`
- swap space: 8 GiB
- model: `facebook/opt-1.3b`
- utilization: 0.4
- prompts: 512
- output tokens: 512
- elapsed: 148.83 s
- generation throughput: **1761.4 tok/s**
- migrated: 510
- races: **2475**
- handoff p50/p90: **3436.7 / 17794.8 ms**

This gives the new BASE comparison point.

## 7. Why FS_BP was added

The observed problem is:

```text
More requests
    |
    v
More KV blocks swapped to host
    |
    v
Host buffer fills
    |
    v
CPU swap space exhausted
    |
    v
PIN aborts
```

FS_BP changes the policy to:

```text
Host buffer getting full
        |
        +--> skip proactive offload
        |
        +--> forced spill does not fit
                 |
                 v
              recompute
                 |
                 v
          request waits/re-prefills
```

The goal is to prevent the complete run from aborting because the host buffer is exhausted, even if this costs some throughput due to recomputation.

## 8. Next experiment

The next direct comparison is:

`pin` vs `pinbp`

using the same 2x T4 setup, `multi=1`, utilization 0.4, 512 prompts and 512 output tokens.

For `pinbp`, record:

- `gen_tok_per_s`
- `races`
- `bp_offloads_skipped`
- `bp_spills_recomputed`
- `gpu1_evictions`
- `handoff_ms_p50_p90`

The first success criterion is that `pinbp` avoids the CPU swap-space abort. After that, compare throughput and recomputation overhead with BASE and PIN.

## 9. Evidence

The ZIP accompanying this README contains four screenshots supplied from the Kaggle notebook showing the implementation/results evidence.
