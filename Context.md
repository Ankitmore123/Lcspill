# LCSpill — Project Context File

> **Purpose:** Paste this into any LLM (Gemini, DeepSeek, ChatGPT, Claude, Mistral) before asking it to help with this project. Keep it updated as work progresses. Both team members should use this as the shared starting prompt.

---

## 1. Project Identity

- **Project Name:** LCSpill
- **Type:** BE Capstone Project, Final Year Computer Engineering, Army Institute of Technology (Pune, SPPU)
- **Base Paper:** FuseSpill framework — published in IEEE Transactions on Parallel and Distributed Systems (TPDS), January 2026
- **Core Problem:** Optimizing KV cache spillover (CPU↔GPU offloading) for LLM inference on memory-constrained hardware
- **Hardware:** Kaggle dual-NVIDIA T4 GPUs (16 GB VRAM each)
- **Team:** 2 members (Ankit More — team lead; [TEAMMATE NAME])
- **Target Output:** BE project report + potential workshop paper submission

---

## 2. Our Contributions (What's New)

We propose **three contributions** on top of the FuseSpill framework:

### Contribution I — Recency Stratified Sequence Selection (RSSS)
- A batch scheduling algorithm that stratifies incoming sequences by their KV cache "age" (how recently tokens were generated)
- Prioritizes sequences whose KV cache entries are most likely to be reused, reducing unnecessary CPU→GPU transfers
- Replaces FuseSpill's default FCFS/uniform batch selection

### Contribution II — Adaptive Cost Model (filling the stub)
- **Key finding:** The reference FuseSpill repository's core function `compute_offload_portion()` is an **unimplemented stub** — it returns a hardcoded value instead of dynamically computing how much KV cache to offload
- We implement a **Dynamic Alpha PID-style controller** that adjusts the offload ratio (α) in real time based on:
  - Current GPU memory utilization
  - Transfer bandwidth (PCIe throughput between CPU↔GPU)
  - Sequence length and batch size
- The PID controller treats the target GPU memory utilization as the setpoint and adjusts α to minimize thrashing

### Contribution III — Simulation-First Methodology
- Since real dual-GPU KV cache offloading is hard to benchmark reproducibly on Kaggle (preemption, variable load), we built a **simulation layer** that models:
  - GPU memory allocation/deallocation
  - PCIe transfer latency
  - Batch scheduling decisions
  - Cache hit/miss rates
- Validated against real T4 runs for calibration, then used for systematic parameter sweeps

---

## 3. System Architecture

```
Input: Batch of sequences with varying lengths
         │
         ▼
┌─────────────────────────┐
│  Recency Stratified     │  ← Contribution I
│  Sequence Selection     │
│  (batch scheduler)      │
└──────────┬──────────────┘
           │ ordered batch
           ▼
┌─────────────────────────┐
│  FuseSpill Core Engine  │  ← Base framework
│  (KV cache management)  │
│                         │
│  ┌───────────────────┐  │
│  │ compute_offload_  │  │
│  │ portion() — NOW   │  │  ← Contribution II
│  │ PID controller    │  │
│  └───────────────────┘  │
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ GPU     │ │ CPU     │
│ Memory  │ │ Memory  │
│ (T4×2)  │ │ (RAM)   │
│ KV Hot  │ │ KV Cold │
└─────────┘ └─────────┘
           │
           ▼
┌─────────────────────────┐
│  Simulation Layer       │  ← Contribution III
│  (for parameter sweeps  │
│   and reproducibility)  │
└─────────────────────────┘
```

---

## 4. Current State

> **Update this section as you work. Delete old items.**

- [ ] Architecture diagrams — [TODO / IN PROGRESS / DONE]
- [ ] Methodology section write-up — [TODO / IN PROGRESS / DONE]
- [ ] Simulation code — [TODO / IN PROGRESS / DONE]
- [ ] Simulation results & plots — [TODO / IN PROGRESS / DONE]
- [ ] Related work survey — [TODO / IN PROGRESS / DONE]
- [ ] Experimental setup section — [TODO / IN PROGRESS / DONE]
- [ ] Results & analysis section — [TODO / IN PROGRESS / DONE]
- [ ] Abstract — [TODO / IN PROGRESS / DONE]
- [ ] Introduction — [TODO / IN PROGRESS / DONE]
- [ ] Conclusion — [TODO / IN PROGRESS / DONE]
- [ ] Full paper draft assembled — [TODO / IN PROGRESS / DONE]
- [ ] BE report formatting (college template) — [TODO / IN PROGRESS / DONE]

---

## 5. Key Terminology

| Term | Definition |
|------|-----------|
| **KV Cache** | Key-Value cache storing attention keys and values from prior tokens during autoregressive LLM inference; grows linearly with sequence length |
| **Spillover / Offloading** | Moving KV cache entries from GPU VRAM to CPU RAM when GPU memory is full, and fetching them back when needed |
| **Offload Portion (α)** | The fraction of a sequence's KV cache that lives on CPU vs GPU; α=0 means fully on GPU, α=1 means fully offloaded |
| **FuseSpill** | The base framework (IEEE TPDS Jan 2026) that fuses KV cache management with spillover decisions during LLM serving |
| **compute_offload_portion()** | The function in FuseSpill's codebase that should dynamically compute α — found to be an unimplemented stub returning a hardcoded value |
| **Recency Stratification** | Our method of bucketing sequences by how recently their KV cache was accessed, to prioritize hot cache entries on GPU |
| **PID Controller** | Proportional-Integral-Derivative controller; we use a PID-style feedback loop where the setpoint is target GPU memory utilization and the output is α |
| **Dynamic Alpha** | Our adaptive α that changes per scheduling cycle based on real-time memory/bandwidth signals, replacing the static hardcoded value |
| **PCIe Transfer** | The CPU↔GPU data bus; transfer latency is a major cost in spillover and must be modeled accurately |
| **Thrashing** | Repeatedly offloading and re-fetching the same KV cache entries, wasting PCIe bandwidth |
| **T4 GPU** | NVIDIA Tesla T4 — 16 GB VRAM, available free on Kaggle; our hardware constraint |

---

## 6. Paper Skeleton

Target format: IEEE conference style (two-column)

| Section | What Goes Here |
|---------|---------------|
| **Abstract** | Problem (KV cache memory bottleneck), gap (FuseSpill's unimplemented stub + static scheduling), our approach (RSSS + PID α + simulation), key result (X% improvement in [metric]) — write LAST |
| **I. Introduction** | Motivation (LLM inference on constrained hardware), why KV cache spillover matters, what FuseSpill does, what's missing, our contributions (3 bullets), paper organization |
| **II. Related Work** | KV cache optimization (PagedAttention/vLLM, FlexGen, etc.), offloading strategies, PID/feedback control in systems, simulation methodologies for GPU systems |
| **III. Background** | FuseSpill architecture summary, the KV cache lifecycle, the stub discovery, problem formulation |
| **IV. Methodology** | RSSS algorithm (pseudocode + explanation), PID controller design (control loop, tuning, setpoint), simulation framework design |
| **V. Experimental Setup** | Hardware (Kaggle T4×2), models tested, sequence length distributions, baselines (original FuseSpill with hardcoded α, FCFS scheduling), metrics (throughput, latency, memory utilization, cache hit rate) |
| **VI. Results & Analysis** | Performance comparison tables/charts, ablation study (RSSS alone vs PID alone vs both), sensitivity analysis (varying batch sizes, sequence lengths), simulation vs real-run calibration |
| **VII. Conclusion** | Summary of contributions, limitations (Kaggle constraints, simulation-first caveats), future work (real multi-GPU deployment, integration with vLLM) |
| **References** | BibTeX via Zotero/Google Scholar |

---

## 7. Constraints & Practical Notes

- **Hardware:** Kaggle free tier — dual T4 GPUs, 30 hrs/week GPU quota, sessions can be preempted
- **Reproducibility:** Kaggle environments are ephemeral; all code must be self-contained in notebooks or a GitHub repo
- **Team size:** 2 people doing a project scoped for 4 — be ruthless about scope. Simulation-first is a feature, not a compromise
- **Base code:** FuseSpill reference repo exists but has the stub issue; we fork and extend, not rewrite
- **College requirements:** BE report has a specific format (title page, certificate, acknowledgment, abstract, chapters, references, appendix) — different from the IEEE paper format. Both deliverables needed.
- **LaTeX:** Paper in Overleaf (IEEE template); BE report can be in Overleaf or Word depending on college template

---

## 8. How to Use This File with an LLM

**Starting a new chat:**
```
[Paste this entire file]

I need help with: [your specific task]
```

**Asking for a paper section draft:**
```
[Paste this entire file]

Draft Section IV (Methodology) for our IEEE-format paper.
Focus on the RSSS algorithm first. Include pseudocode.
Keep it technical but concise — target 1.5 pages two-column.
```

**Asking for diagram descriptions:**
```
[Paste this entire file]

Describe a system architecture diagram I should create in draw.io
that shows the full LCSpill pipeline. List every box, arrow,
and label. I will draw it manually.
```

**Asking for code help:**
```
[Paste this entire file]

Here is our current compute_offload_portion() implementation:
[paste code]

The PID controller is oscillating when batch size changes sharply.
Help me tune the gains or suggest a modification.
```
