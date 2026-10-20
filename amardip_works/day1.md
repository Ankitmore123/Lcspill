# FuseSpill Paper Reading Notes — Day 1

**Date:** Oct 21, 2026  
**Author:** Amardip  
**Context:** Phase 0 (Clarity) — Study of Sections III-B, IV-A, IV-C, and V-A  

---

## 📌 Executive Summary

Yeh document FuseSpill paper ke core sections (III-B, IV-A, IV-C, V-A) ka structured Hindi-English (Hinglish) breakdown hai. Isme existing KV cache spillover ke challenges, Primary & Auxiliary architecture, adaptive swap mechanisms, aur experimental testbed details ko systematically compile kiya gaya hai.

---

## 1. Section III-B: Challenges in Existing KV Cache Spillover

Current inference frameworks (jaise vLLM) mein memory saturation ke waqt KV cache manage karne ke 3 major fundamental challenges identify kiye gaye hain:

### ⚠️ Challenge 1: Static / One-Size-Fits-All Approach
* **Problem:** KV cache spillover handle karne ke liye koi comprehensive runtime analysis available nahi hai.
* **Impact:** Deployment ke waqt developers ko ek fixed strategy (**Swap** ya **Recompute**) pehle se chunni padti hai.
* **Consequence:** Runtime dynamics (arrival rate, prompt burst, generation length) change hone par yeh static strategy fit nahi baithti, jisse hardware resources ka **suboptimal utilization** hota hai.

---

### 📉 Challenge 2: Chhota Prefill Batch Size (Underutilized Compute)
* **Problem:** Memory-constrained GPUs par memory ka bada hissa pehle se generate ho rahe sequences ke accumulated KV cache se occupy rehta hai.
* **Impact:** Naye requests ke liye prefill stage ka batch size kafi chhota rakhna padta hai.
* **Consequence:** GPU cores compute-starved rehte hain aur hardware utilization poora nahi ho pata.

---

### 🎯 Challenge 3: Inefficient Sequence Selection for Spillover
* **Problem:** Spillover trigger hone par sequences ko random selection ya arrival time (FIFO) ke basis par select kiya jata hai.
* **Root Cause:** Har sequence ki future generation length ($G$) alag hoti hai:
  $$\text{Memory Benefit} = S \times G$$
  *(jahan $S$ = current sequence length, $G$ = remaining generation steps)*
* **Consequence:** Bina future length estimate kiye kisi bhi sequence ko swap karne se transmission aur eviction overhead unnecessarily badh jata hai.

| Challenge | Root Cause | Direct Impact |
| :--- | :--- | :--- |
| **1. Static Strategy** | Fixed choice at deployment (Swap vs Recompute) | Runtime workload shifts par poor performance |
| **2. Small Prefill Batch** | GPU memory decoding KV caches se clogged hona | Compute saturation aur utilization down |
| **3. Inefficient Eviction** | Random/FIFO selection without predicting $S \times G$ | Higher transfer overhead & suboptimal benefit |

---

## 2. Section IV-A: System Architecture & Device Specialization

FuseSpill heterogenous multi-GPU inference ke liye **Primary-Auxiliary Device Separation** model introduce karta hai.

```
                      ┌────────────────────────────────────────┐
                      │             Incoming Requests          │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │           PRIMARY DEVICE            │
                       │   (Prefill + Non-offloaded Decode)  │
                       └──────────────────┬──────────────────┘
                                          │ (Offload KV Cache via PCIe)
                                          ▼
                       ┌─────────────────────────────────────┐
                       │          AUXILIARY DEVICE           │
                       │   (Specialized Offloaded Decode)    │
                       └─────────────────────────────────────┘
```

### 🔹 Roles of Devices

* **Primary Device:**
  * Sabhi sequences ke liye **Prefill iterations** aur **Non-offloaded decoding** execute karta hai.
  * Central inference hub ki tarah act karta hai.
* **Auxiliary Device:**
  * Primary device se swap-out kiye gaye KV cache ki **Decoding iterations** ko specialize aur execute karta hai.

### 💡 Auxiliary Device Selection & Cost-Efficiency
* **Low-Cost Hardware Friendly:** Auxiliary device ke liye top-tier ya expensive GPUs (e.g., A100/H100) ki mandatory requirement nahi hoti. Saste aur consumer-grade devices jaise **GTX 3090, A10, ya CPU** use ho sakte hain, jisse overall TCO (Total Cost of Ownership) drastically drop hoti hai.
* **Do Mandatory Requirements:**
  1. **Memory Capacity:** Model weights load karne ke liye sufficient VRAM.
  2. **Compute Capability:** Primary device ke computational throughput ka kam se kam **20%** capable hona chahiye.
* **Dynamic Offload Ratio ($\alpha$):**
  * Queue bottlenecks aur task accumulation ko prevent karne ke liye offloading ratio $\alpha$ ko hardware speed aur bandwidth ke mutabiq dynamically adjust kiya jata hai.

---

## 3. Section IV-C: Spillover Mechanism & Adaptive Operations

### ⚙️ 1. KV Cache Swap Orchestrator & Memory Cache Engine
* **KV Cache Swap Orchestrator:** PCIe-connected devices ke beech fast KV cache transmission aur separation-based spillover coordination handle karta hai.
* **Memory Cache Engine:** CPU Host Memory (RAM) ko Primary aur Auxiliary devices ke beech central **"Distribution Hub"** ki tarah manage karta hai.
* **Low Overhead via Shared Memory:** OS ke zero-copy / shared memory mechanisms (`mmap` / `unmap`) ka use karta hai, jisse KV cache serialization/deserialization latency minimize ho jati hai.

---

### 🚀 2. Separation-Based Spillover Handling (Key Benefits)
* **Prefill Headroom Creation:** Spillover sequences ko auxiliary device par offload karke Primary device par naye prefill tokens ke liye large memory space unlock hoti hai $\rightarrow$ Prefill batch size aur GPU utilization boost hoti hai.
* **Dramatic Queue & Latency Reduction:**
  * **Traditional Swap:** 10 sequences ka average waiting queue time $\approx \mathbf{5.5\,T_d}$ hota hai.
  * **Separation-Based (Auxiliary GTX 3090):** Queuing time khatam ho kar **0** ho jata hai, aur decoding time sirf $\mathbf{1.6\,T_d}$ reh jata hai.

---

### 🔄 3. Adaptive Offload Operations (3 Memory Operations)

Primary device par memory free aur balance rakhne ke liye 3 explicit operations define kiye gaye hain:

```
[Memory Management Operations]
       ├── 1. Release Operation  ──► Sequence complete hone par memory free (Highest Priority)
       ├── 2. Spillover Operation ──► Out-of-memory hone par mandatory host swap (Urgent Priority)
       └── 3. Offload Operation   ──► Device saturation optimize karne proactive transfer (Dynamic Priority)
```

1. **Release Operation:** Sequence generation finish hote hi uska KV cache immediately deallocate kar diya jata hai.
2. **Spillover Operation:** Memory saturation ke waqt mandatory taur par KV blocks ko host memory mein swap karta hai.
3. **Offload Operation:** Proactive step jo primary aur auxiliary devices ke saturation ko balance karne ke liye execute kiya jata hai.
   * *Priority Hierarchy:* Offload ki priority **Release** aur **Spillover** se kam hoti hai.
   * *Dynamic Balancing:* Workload arrival rate (traffic intensity) ke according spillover aur offload ka balance automatically adjust hota hai taaki throughput maximize rahe.

---

## 4. Section V-A: Evaluation & Implementation Details

FuseSpill prototype ki performance ko alag-alag hardware nodes, LLM models, aur realistic workloads par benchmark kiya gaya hai.

### 🛠️ Framework & Tech Stack
* **Core Implementation:** Python
* **Serving & Inference Engines:** PyTorch v2.1.2 aur vLLM v0.2.7
* **Distributed Runtime:** Ray framework (v2.5.1) for inter-device management & communication
* **CUDA Version:** CUDA 12.1 (Python v3.8)

---

### 🖥️ Testbed (Hardware Platforms)

Experiments 3 distinct multi-GPU setups par conduct kiye gaye:

| Platform | Primary GPU | Auxiliary GPU | Host CPU & RAM | Interconnect |
| :--- | :--- | :--- | :--- | :--- |
| **Platform 1** | Nvidia A10 (24 GB) | Nvidia A10 (24 GB) | Intel Xeon Platinum 8358P (500 GB) | PCIe 4.0 (16 GB/s, No NVLink) |
| **Platform 2** | Nvidia RTX 4090 (24 GB) | Nvidia RTX 3090 (24 GB) | Intel Xeon Platinum 8358P (500 GB) | PCIe 4.0 (16 GB/s, No NVLink) |
| **Platform 3** | Nvidia RTX 4090 (24 GB) | Nvidia A10 (24 GB) | Intel Xeon Platinum 8358P (500 GB) | PCIe 4.0 (16 GB/s, No NVLink) |

> ℹ️ **Notice:** GPUs ke beech koi NVLink nahi hai; system commodity PCIe 4.0 bus par rely karta hai, jo realistic heterogeneous edge/cloud clusters ko mimic karta hai.

---

### 🧠 Models and Workload
* **Target Models (7B Class):**
  1. `LLaMA-7b`
  2. `Baichuan-7b`
  3. `Aquila-7b`
* **Dataset & Request Distribution:**
  * **ShareGPT_V3** real-world conversation dataset.
  * Prefill prompt length variable range: **4 tokens se 2048 tokens**.

---

### 📊 Key Evaluation Metrics
* **End-to-End Generation Speed (Primary Metric):**
  $$\text{Throughput} = \frac{\text{Total Generated Tokens}}{\text{Total Response Time}} \quad (\text{tokens/s})$$
* **Inference Latency Impact:** Separation-based offloading se single-sequence response latency par aane wala variation track kiya jata hai.

---

### ⚖️ Baselines for Comparison
* **Standard Spillover:** State-of-the-art inference engines ke default **Swap-to-Host** aur **Recomputation** mechanisms.
* **Fairness Protocol:** Baseline methods ko bhi identical hardware devices assign kiye gaye aur requests uniformly balance ki gayi.
* **Alternative Parallel Paradigms:**
  * **Tensor Parallelism (TP)**
  * **Splitwise** distributed pipeline architecture (Ray-based execution)
