# FlexGen: Graph Traversal, Cost Model, Granularity, and Static Policy

## 1. Graph Traversal, Cost Model, and Linear Programming

FlexGen converts the entire **LLM inference process** into a **computational graph**, where each computation step is represented as a node.

Moving data between the **GPU, CPU, and disk** involves I/O operations, which introduce latency. To account for this overhead, FlexGen uses an **Analytical Cost Model** to estimate the execution time of different computation and memory-transfer operations.

FlexGen then uses **Linear Programming (LP)** to determine an efficient execution and memory-placement strategy.

The optimizer considers:

* Computation time
* GPU ↔ CPU data-transfer time
* CPU ↔ disk data-transfer time
* Available GPU memory
* Available CPU memory
* Available disk storage
* Other hardware constraints

The goal is to find a strategy that minimizes **total execution time** and maximizes **inference throughput**.

### Example

Suppose:

* GPU memory = **16 GB**
* Model memory requirement = **40 GB**

The complete model cannot fit into GPU memory.

FlexGen can therefore determine which data should remain on the GPU and which data should be placed on the CPU or disk, while attempting to minimize unnecessary data transfers.

```text
             ┌─────────────────┐
             │ Computational   │
             │     Graph       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Analytical Cost │
             │      Model      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Linear          │
             │ Programming     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Optimized Data  │
             │ Placement &     │
             │ Execution Plan  │
             └─────────────────┘
```

---

## 2. Granularity

**Granularity** defines the level at which data is divided, managed, and moved during memory allocation and execution.

FlexGen uses different levels of granularity for different types of data.

| Data          | Granularity      | Purpose                                    |
| ------------- | ---------------- | ------------------------------------------ |
| Model Weights | **Layer-level**  | Manage and move weights as complete layers |
| KV Cache      | **Tensor-level** | Provide finer memory allocation            |
| Activations   | **Tensor-level** | Provide finer memory allocation            |

### 2.1 Layer Granularity — Model Weights

For **model weights**, FlexGen uses **layer-level granularity**.

The weights belonging to an entire layer are treated as a single unit for storage and movement.

Instead of tracking thousands of small weight components individually, the system can manage an entire layer as one unit:

```text
Layer 1
Layer 2
Layer 3
...
Layer N
```

This reduces runtime management and scheduling overhead.

### Why Layer Granularity?

Managing weights at the layer level provides:

* Lower runtime overhead
* Simpler scheduling
* Fewer individual data-management operations
* More predictable execution

---

### 2.2 Tensor Granularity — KV Cache and Activations

For the **KV cache and activations**, FlexGen uses a finer **tensor-level granularity**.

These components can be divided and managed at the tensor level, providing greater flexibility when allocating memory.

This is particularly useful when there are small amounts of available memory that can still be utilized.

### Simple Analogy

Think of a large box containing many objects.

**Layer granularity:**

> Move the entire box.

**Tensor granularity:**

> Move individual objects inside the box.

Therefore:

* **Layer granularity** → lower management overhead
* **Tensor granularity** → finer memory control

---

## 3. Static Policy

A **static policy** is a pre-computed memory-placement and execution strategy that remains fixed during inference.

Before inference begins, FlexGen's optimizer determines how different components should be distributed across:

* GPU
* CPU
* Disk

For example, the optimizer could determine a particular placement ratio:

$$
w_g = 20\%, \qquad
w_c = 50\%, \qquad
w_d = 30\%
$$

where:

* $w_g$ = portion assigned to GPU
* $w_c$ = portion assigned to CPU
* $w_d$ = portion assigned to disk

The optimizer similarly determines the placement strategy for:

* Model weights
* KV cache
* Activations

### Execution

Once inference begins, the policy is generally **not re-optimized at every step**.

Even though the internal memory state changes during execution, the predefined execution and offloading strategy remains fixed.

This avoids the runtime overhead that would be introduced by continuously solving the optimization problem.

---

## 4. Why Use a Static Policy?

A static policy provides several advantages:

### Lower Runtime Overhead

The optimization problem is solved before execution rather than repeatedly during inference.

### Predictable Execution

The system knows in advance where data should be located and when it should be moved.

### Reduced Optimization Cost

Continuously recomputing the optimal placement would itself consume computational resources.

### Hardware-Aware Planning

The optimizer can incorporate memory capacities and data-transfer costs before execution starts.

---

## 5. Overall FlexGen Workflow

The complete process can be summarized as:

```text
┌───────────────────────────┐
│   LLM Inference Graph     │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Analytical Cost Model   │
│                           │
│ • Computation cost        │
│ • GPU ↔ CPU transfer      │
│ • CPU ↔ Disk transfer     │
│ • Memory constraints      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Linear Programming      │
│        Optimizer          │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Data Placement Plan     │
│                           │
│ GPU / CPU / Disk          │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      Granularity          │
│                           │
│ Weights → Layer           │
│ KV Cache → Tensor         │
│ Activations → Tensor      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      Static Policy        │
│                           │
│ Pre-computed execution    │
│ and offloading strategy   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      LLM Inference        │
└───────────────────────────┘
```

---

## 6. Key Takeaways

### Computational Graph

Represents the LLM inference process as a sequence of computational operations and data dependencies.

### Analytical Cost Model

Estimates the cost of computation and data movement between GPU, CPU, and disk.

### Linear Programming

Uses these estimated costs and hardware constraints to determine an efficient execution and memory-placement strategy.

### Granularity

Determines the size of the units managed by the system:

* **Weights → Layer-level**
* **KV Cache → Tensor-level**
* **Activations → Tensor-level**

### Static Policy

The optimized execution and memory-placement strategy is determined before inference and remains fixed during execution.

---

## 7. One-Line Summary

> **FlexGen uses an analytical cost model and Linear Programming to pre-compute an efficient GPU/CPU/disk execution strategy, manages different data types at appropriate granularities, and executes inference using a static policy to reduce runtime overhead.**
