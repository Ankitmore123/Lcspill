# Stage 3 — Migration Variants: Defer & Pin

## Objective

Evaluate the two migration variants required after the Stage 2 baseline:

- `defer`
- `pin`

Each variant is intended to be run twice using the 2-GPU benchmark.

## Required Commands

### Defer

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/kaggle/working/TDoppeladler \
$PY /kaggle/working/cell3.py throughput 1 0.4 defer
```

Run twice.

### Pin

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=/kaggle/working/TDoppeladler \
$PY /kaggle/working/cell3.py throughput 1 0.4 pin
```

Run twice.

## Required Measurements

For every run record:

- `gen_tok_per_s`
- `races`
- `handoff_ms_p50_p90`
- elapsed time
- whether the run completed normally

The final comparison should calculate:

```text
mean throughput ± run-to-run spread
mean races
handoff p50 / p90
```

for each variant.

## Comparison Baseline

The Stage 2 base runs provide the reference:

| Variant | Run | Throughput | Races | Handoff p50 / p90 |
|---|---|---:|---:|---:|
| base | A | 1559.9 tok/s | 2151 | 3297.5 / 19763.1 ms |
| base | B | 1354.6 tok/s | 2055 | 5248.2 / 20985.5 ms |

Average base throughput:

```text
1457.25 tok/s
```

Average base race count:

```text
2103
```

## Deliverable

Create:

```text
night/results_table.md
```

The table should contain one row per run and a summary for each variant.

It should also include a sentence answering whether `pin` beats `base` by more than the run-to-run spread.

## Hang Rule

If any run hangs for more than 8 minutes:

1. Kill the Ray/cell3 processes.
2. Save the last 30 lines of the log.
3. Record the run as a hang.
4. Continue with the remaining runs.

## Outcome

Stage 3 is the experimental comparison stage for the deferred/freeing and pinned-memory migration variants. Its results are used as the quantitative basis for the lifecycle and overlap analysis in later stages.
