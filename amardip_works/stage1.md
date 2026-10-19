# Stage 1 — Environment Setup & Verification

## Objective

Prepare the Kaggle environment required for the FuseSpill / TDoppeladler migration work and verify that the expected Python environment and GPUs are available.

## Environment

- Platform: Kaggle
- Repository: `TDoppeladler`
- Conda environment: `fusespill`
- Python: `3.8.20`
- GPU count: `2`
- GPU setup: 2× NVIDIA T4

## Verification

The Section 2 / Cell 0 verification confirmed:

```text
Python 3.8.20
2 GPUs detected
```

The required environment was therefore available for the subsequent microbenchmark, hook installation, and multi-GPU benchmark stages.

## Stage 1 Outcome

The environment was rebuilt/verified successfully and the two-GPU setup was ready for the FuseSpill migration experiments.

## Evidence

The Section 2 Cell 0 output and setup notes were captured as the Stage 1 verification evidence.

