# Task 3 Deliverable: Baselines and Metrics

## Baselines

- **FuseSpill Default (Hardcoded Stub):** Proves the fundamental limitation of the base paper's unimplemented offload function.
- **Fixed Alpha (α = 0.5):** Proves that a static 50/50 memory split degrades performance compared to our dynamic PID approach under varying workloads.
- **Greedy Eviction (FCFS):** Proves that our Recency Stratified Sequence Selection (RSSS) is superior to a naive, oldest-first eviction policy.

## Metrics

- **Throughput (tokens/sec):** Demonstrates the macro-level system efficiency and overall improvement in user-facing generation speed.
- **Per-token Decode Latency (ms):** Provides micro-level proof of reduced PCIe thrashing during the memory-bound decode phase.
- **PCIe Transfer Volume (GB):** Acts as the direct, undeniable proof that the PID controller and RSSS are successfully reducing traffic over the CPU-GPU bus.
- **Peak GPU Memory Utilization (%):** Proves the PID controller safely maximizes the 16 GB Kaggle T4 VRAM limit without causing OOM crashes.
- **Controller Overhead (% of decode step time):** Proves that the computational cost of the PID math and RSSS sorting does not cancel out the optimization gains.

---

## Baseline / Metric: Why + How

### 1. Baselines (The Systems We Compare Against)

| Baseline | Why (Justification for Paper) | How (Measurement Methodology) |
| --- | --- | --- |
| **FuseSpill Default (Hardcoded Stub)** | Highlights the fundamental limitation of the base paper by exposing the unimplemented α stub. | Run the reference FuseSpill code out-of-the-box without modifying the `compute_offload_portion()` function. |
| **Fixed Alpha (α = 0.5)** | Proves that a static 50/50 memory split degrades performance under varying workloads compared to our dynamic PID approach. | Override the offload function to return a constant 0.5, forcing exactly half the cache to CPU for every sequence. |
| **Greedy Eviction (FCFS)** | Proves that our Recency Stratified Sequence Selection (RSSS) is vastly superior to a naive, oldest-first eviction policy. | Implement a standard First-In-First-Out (FIFO) queue for eviction and scheduling, pushing the oldest KV blocks to CPU first. |

### 2. Metrics (The Proof of Improvement)

| Metric | Why (Justification for Paper) | How (Measurement Methodology) |
| --- | --- | --- |
| **Throughput (tokens/sec)** | Demonstrates the macro-level system efficiency and overall improvement in user-facing generation speed. | Divide the total number of tokens generated across the batch by the total end-to-end execution time using `time.perf_counter()`. |
| **Per-token Decode Latency (ms)** | Provides micro-level proof of reduced PCIe thrashing during the memory-bound decode phase. | Calculate the average time interval between consecutive generated tokens specifically during the autoregressive decoding loop. |
| **PCIe Transfer Volume (GB)** | Serves as the direct, undeniable proof that the PID controller and RSSS are successfully reducing traffic over the CPU-GPU bus. | Track and accumulate the exact byte sizes of all tensors passed through `tensor.to('cpu')` and `tensor.to('cuda')` during simulation/profiling. |
| **Peak GPU Utilization (%)** | Proves the PID controller safely rides the 16 GB hardware limit to maximize T4 VRAM usage without causing OOM crashes. | Record `torch.cuda.max_memory_allocated(device)` divided by total device memory (16 GB) at the end of an inference run. |
| **Controller Overhead (%)** | Proves that the computational cost of the PID math and RSSS sorting does not cancel out the optimization gains. | Wrap the RSSS and PID functions in timer blocks; divide their execution time by the total time taken for a single forward pass. |

> **Note:** We will add one sentence in the text stating that all LCSpill outputs were strictly validated against a GPU-only baseline to ensure bit-for-bit exact matching and lossless offloading, saving us from needing "Correctness" as a table metric.
