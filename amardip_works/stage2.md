# Stage 2 — Microbenchmarks, Hooks & Baseline

## Objective

Run the Section 3 microbenchmarks, install the FuseSpill migration hooks/fix, create the benchmark runner, and establish the required baseline measurements.

## Section 3 — Microbenchmark

The register/P2P and copy microbenchmarks were completed on the 2× T4 setup.

### Key observations

- GPU 0 → GPU 1 P2P was reported as `True`.
- Pinned-memory transfers were substantially faster than mmap/pageable transfers in the per-block measurements.
- Fused transfers reduced the per-block overhead for mmap/pageable memory.
- The register/P2P test completed and reported GPU 0 → GPU 1 P2P transfer timing.

### Measured copy results

| Blocks | mmap/pageable per-block | mmap/pageable fused | pinned per-block | pinned fused |
|---:|---:|---:|---:|---:|
| 128 | 10.37 ms | 6.63 ms | 2.52 ms | 3.84 ms |
| 512 | 41.05 ms | 27.56 ms | 11.40 ms | 15.45 ms |
| 1024 | 80.21 ms | 55.05 ms | 21.60 ms | 29.80 ms |
| 2048 | 160.09 ms | 109.75 ms | 43.98 ms | 59.61 ms |

## Section 4 — Hooks / Fix

The migration-related changes were applied to:

- `vllm/worker/cpu_cache_engine.py`
- `vllm/core/multi_block_manager.py`
- `vllm/core/scheduler.py`
- `vllm/engine/llm_engine.py`

The profiling patch in:

- `vllm/worker/multi_worker.py`

was already present and therefore was not reverted.

### Important note

`cell5` reported:

```text
already patched
already patched
already patched
already patched
```

for the four target files rather than literally printing `patched` four times. The files were nevertheless verified as present with the expected modifications.

## Benchmark Runner

`cell3.py` was created for the required throughput runs.

Benchmark format:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/kaggle/working/TDoppeladler \
$PY /kaggle/working/cell3.py throughput <multi> 0.4 <variant>
```

## Required Baseline Runs

### Single GPU — base

```text
elapsed_s: 212.44
gen_tok_per_s: 1234.0
migrated: 0
races: null
```

### Two GPU — base, Run A

```text
elapsed_s: 168.05
gen_tok_per_s: 1559.9
migrated: 499
races: 2151
handoff_ms_p50_p90: [3297.5, 19763.1]
```

### Two GPU — base, Run B

```text
elapsed_s: 193.52
gen_tok_per_s: 1354.6
migrated: 512
races: 2055
handoff_ms_p50_p90: [5248.2, 20985.5]
```

## Stage 2 Artifacts

The required baseline artifacts were:

- `results.jsonl`
- `r_t0_base.log`
- `r_t1_base_a.log`
- `r_t1_base_b.log`

A Stage 2 artifact ZIP was also created containing these three logs/results files.

## Outcome

Stage 2 established the baseline needed for the later `defer` and `pin` experiments and captured the baseline migration/race behavior.
