# FuseSpill BE Project — Complete Notes (22 Points)

## 1. Project Idea
Humara BE project GPU-to-GPU KV-cache migration aur FuseSpill ke migration path ko study/optimize karne par based hai.

## 2. Normal GPU-to-GPU Migration Path
Current migration path roughly:
`GPU A -> PCIe -> Host Memory -> PCIe -> GPU B`

Isliye host memory migration path ka important part hai.

## 3. Shared Host Memory
Host side par shared memory / mmap buffer use hota hai:
`GPU A -> Host shared memory (mmap) -> GPU B`

## 4. FuseSpill ka Main Idea
FuseSpill migration karta hai, aur hum dekh rahe hain ki transfer ko better pack/optimize kaise kiya ja sakta hai aur migration ko compute ke saath overlap kiya ja sakta hai.

## 5. Block Fusion
Bahut saare small blocks ko ek contiguous buffer me combine kiya ja sakta hai.
Example:
`3000 small 64 KB blocks -> 1 contiguous buffer`

## 6. Fused Copy ka Benefit
Small individual copies ko larger contiguous copy me combine karne se copy overhead reduce ho sakta hai.

## 7. KV Cache Migration
LLM inference me KV cache GPU memory use karta hai. GPU memory pressure hone par KV blocks ko host memory me migrate kiya ja sakta hai aur zarurat par auxiliary GPU par wapas laya ja sakta hai.

## 8. Why KV Migration Matters
KV cache continuously use/grow hota hai. GPU memory limited hone par migration necessary ho sakti hai, isliye migration performance LLM throughput ko affect karti hai.

## 9. Two-GPU Setup
Experiments Kaggle ke 2 x NVIDIA T4 GPU environment par hue.

Model: `facebook/opt-1.3b`  
Benchmark: `512 prompts`, `512 max tokens`, GPU utilization `0.4`.

## 10. PCIe / Host Memory Bottleneck
Theoretical PCIe speed high ho sakti hai, lekin actual transfer host-memory behavior, copy granularity aur synchronization ki wajah se lower ho sakta hai.

## 11. Pageable Host Memory
Pageable normal CPU host memory hai. GPU transfer ke liye iska performance pinned memory se generally slower ho sakta hai.

## 12. Pinned Host Memory
Pinned memory ko CUDA DMA transfer ke liye efficiently use kar sakta hai. Microbenchmark me pinned memory transfers pageable transfers se significantly faster aaye.

## 13. Microbenchmark Observation
128-block example:
- Pageable per-block: `10.37 ms`
- Pageable fused: `6.63 ms`
- Pinned per-block: `2.52 ms`
- Pinned fused: `3.84 ms`

Isse pata chalta hai ki host-memory type aur transfer granularity dono important hain.

## 14. Main Experimental Variants
- **Base** = original FuseSpill + race counting
- **Defer** = race fix; CPU block ko transfer complete hone tak defer karna
- **Pin** = defer fix + pinned `/dev/shm` host buffer
- **PinBP** = Pin + host-buffer backpressure

## 15. Base Result
Initial multi-GPU base runs:
- Run 1: `1559.9 tok/s`, `2151 races`
- Run 2: `1354.6 tok/s`, `2055 races`

Later base run:
- `1761.4 tok/s`
- `2475 races`

Main observation: Base fast hai, lekin CPU swap-block reuse race observe hoti hai.

## 16. Defer Result
Defer me CPU block ko immediately free karne ke bajay migration complete hone tak hold kiya gaya.

Results:
- `1427.3 tok/s`, `0 races`
- `1169.9 tok/s`, `0 races`

Matlab race fix effective tha, lekin throughput overhead aa sakta hai.

## 17. Pin Result
Pinned host buffer ka purpose transfer ko faster banana tha.

Lekin full benchmark complete nahi hua. Run abort/timeout hua aur log me:
`RuntimeError: Aborted due to the lack of CPU swap space.`

Isliye Pin ke liye valid final throughput comparison claim nahi karna chahiye.

## 18. Pin Failure ka Meaning
Pin benchmark me host/CPU swap-buffer capacity exhaust hui aur run abort hua.

Log me:
`AssertionError: Waiting sequence group should have only one prompt sequence.`
bhi dikha.

Isliye exact evidence ke basis par ise sirf "system slow tha" kehna sahi nahi hai; run resource-exhaustion/abort state me gaya.

## 19. FS_BP — Host-Buffer Backpressure
Pin ki host-buffer exhaustion problem ko handle karne ke liye **FS_BP** feature add kiya gaya.

Flag:
`/kaggle/working/FS_BP`

Feature default me **OFF** hai.

## 20. FS_BP ka 25% Reserve
`cell3.py` me FS_BP reserve:
`0.25`

Matlab approximately 25% host-buffer blocks reserve rakhe jaate hain.

Agar proactive offload karne se free host-buffer space reserve se neeche chali jaati hai, scheduler us offload ko skip karta hai.

## 21. Forced Spill + Recompute Fallback
Agar forced spill ke time host buffer me enough space nahi hai, system poori run ko abort karne ke bajay recompute path use karne ki koshish karta hai.

Request waiting queue me wapas ja sakti hai aur GPU 0 par dobara prefill/recompute hoti hai.

Flow:
`GPU pressure -> KV migration needed -> host space check`

- Enough space -> **offload**
- Not enough space -> **skip proactive offload**
- Forced spill still needed -> **recompute instead of abort**

## 22. Final Research Direction
Overall direction:
`GPU memory pressure -> KV-cache migration -> host-buffer check -> safe/faster transfer -> race prevention -> backpressure -> recompute fallback`

Main goal hai ki LLM throughput improve ho, KV migration safe rahe, host-buffer exhaustion avoid ho, aur migration ko compute ke saath better overlap kiya ja sake.

### Key Takeaway
- **Base:** fast, but race-prone.
- **Defer:** races `0`, but slower.
- **Pin:** faster-transfer idea, but host-buffer exhaustion caused benchmark failure.
- **FS_BP / PinBP:** host-buffer ko protect karne ke liye reserve + backpressure + recompute fallback add kiya gaya.
