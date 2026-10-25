#!/usr/bin/env python3
"""
run_wl.py -- run ONE arm of ONE workload on ONE GPU with the kvlog hooks.
Launched twice in parallel (gpu 0 and gpu 1) by run_pair() in the notebook.

Writes <out>/gpu<g>.jsonl in the same schema as the 23 Oct W1/W2 runs:
  {"t":"meta", ...assign...}  step/preempt/swap_* records (kvlog)  {"t":"done", ...}

Workloads (fixed here, pre-registered in PREREG_W3W4.md -- do not edit after committing):
  W3  long outputs, lower pressure per step, longer preemptions
  W4  low pressure: preemptions rare or absent -> advisor should say "don't switch"
Replay mode (--assign-from) re-runs an existing W1/W2 assignment for the harness check.
"""
import argparse, json, os, random, sys, time

COMMON = dict(block_size=16, seed=0, prompt_len=12, model="facebook/opt-1.3b")
WORKLOADS = {
    #       requests per GPU, max_tokens ~ U[lo, hi], gpu_memory_utilization
    "W3": dict(n=128, lo=768, hi=1536, gpu_mem_util=0.5, swap_space_gb=8),   # sim peak 2274 swapped blocks > 2048 at 6 GiB
    "W4": dict(n=256, lo=128, hi=448, gpu_mem_util=0.8, swap_space_gb=6),
}
PROMPT = ("The history of the city begins with a small settlement on the bank of the river, "
          "where traders from the north and the south met every season to exchange goods.")


def make_assign(spec, gpu, seed):
    """2*n requests with seeded random output lengths, shuffled, split across the 2 GPUs.
    Same seed -> identical assignment in every repeat and in both arms."""
    rng = random.Random(seed)
    outs = [rng.randint(spec["lo"], spec["hi"]) for _ in range(2 * spec["n"])]
    ids = list(range(2 * spec["n"]))
    rng.shuffle(ids)
    return [[i, outs[i]] for i in ids[gpu::2]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wl", required=True)
    p.add_argument("--mode", choices=["rec", "swap"], required=True)
    p.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kvexp", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--assign-from", help="dir of an existing run: replay its meta.assign + gpu_mem_util")
    p.add_argument("--enforce-eager", action="store_true",
                   help="only if the 23 Oct runs used enforce_eager=True")
    p.add_argument("--barrier-timeout", type=float, default=1200)
    a = p.parse_args()

    if a.assign_from:
        meta0 = json.loads(open(os.path.join(a.assign_from, f"gpu{a.gpu}.jsonl")).readline())
        assign, util, swap_gb = meta0["assign"], meta0["gpu_mem_util"], meta0["swap_space_gb"]
    else:
        spec = WORKLOADS[a.wl]
        assign, util = make_assign(spec, a.gpu, COMMON["seed"]), spec["gpu_mem_util"]
        swap_gb = spec["swap_space_gb"]

    sys.path.insert(0, a.kvexp)
    import kvlog                                   # patches Scheduler / LLMEngine / CacheEngine
    os.makedirs(a.out, exist_ok=True)
    kvlog.install(os.path.join(a.out, f"gpu{a.gpu}.jsonl"), force_swap=(a.mode == "swap"))

    from vllm import LLM, SamplingParams
    import vllm
    kw = {"enforce_eager": True} if a.enforce_eager else {}
    llm = LLM(model=COMMON["model"], dtype="float16", seed=COMMON["seed"],
              gpu_memory_utilization=util, swap_space=swap_gb,
              block_size=COMMON["block_size"], **kw)
    eng = llm.llm_engine

    ids_full = eng.tokenizer(PROMPT).input_ids
    ids = ids_full[:COMMON["prompt_len"]]
    assert len(ids) == COMMON["prompt_len"], "prompt shorter than 12 tokens"

    kvlog.log(t="meta", gpu=a.gpu, vllm=vllm.__version__, wl=a.wl, mode=a.mode,
              force_swap=(a.mode == "swap"), swap_space_gb=swap_gb,
              gpu_mem_util=util, block_size=COMMON["block_size"], seed=COMMON["seed"],
              prompt_len_orig=len(ids_full), prompt_len_used=len(ids), assign=assign,
              num_gpu_blocks=eng.cache_config.num_gpu_blocks,
              num_cpu_blocks=eng.cache_config.num_cpu_blocks,
              enforce_eager=a.enforce_eager, harness="run_wl.py")

    # barrier: both GPUs start generating together (the run ends when the slower one ends)
    open(os.path.join(a.out, f"ready{a.gpu}"), "w").close()
    other = os.path.join(a.out, f"ready{1 - a.gpu}")
    t_wait = time.time()
    while not os.path.exists(other):
        if time.time() - t_wait > a.barrier_timeout:
            sys.exit(f"[run_wl] gpu{a.gpu}: other GPU never became ready")
        time.sleep(0.05)

    want = {str(rid): m for rid, m in assign}
    finished = exact = gen_tokens = 0
    t0 = time.time()
    for rid, m in assign:                          # FCFS order = list order
        eng.add_request(str(rid), None,
                        SamplingParams(temperature=0.0, max_tokens=m, ignore_eos=True),
                        prompt_token_ids=ids)
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                n = len(o.outputs[0].token_ids)
                finished += 1
                gen_tokens += n
                exact += int(n == want[o.request_id])
    t1 = time.time()
    kvlog.log(t="done", submitted=len(assign), finished=finished, exact_len=exact,
              gen_tokens=gen_tokens, gen_start=t0, gen_end=t1, gen_wall_s=t1 - t0,
              tput=gen_tokens / (t1 - t0))
    kvlog._f.close()
    print(f"[run_wl] {a.wl} {a.mode} gpu{a.gpu}: {t1 - t0:.1f} s, {finished}/{len(assign)} finished, "
          f"{exact} exact, blocks gpu={eng.cache_config.num_gpu_blocks} cpu={eng.cache_config.num_cpu_blocks}")
    sys.exit(0 if finished == exact == len(assign) else 3)


if __name__ == "__main__":
    main()
