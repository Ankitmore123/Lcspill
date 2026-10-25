import json, time
import torch
from vllm.core.scheduler import PreemptionMode, Scheduler
from vllm.engine.llm_engine import LLMEngine
from vllm.worker.cache_engine import CacheEngine

_f = None


def log(**r):
    _f.write(json.dumps(r) + "\n")


def install(path, force_swap=False):
    global _f
    _f = open(path, "w", buffering=1)
    preempt0, sched0, step0 = Scheduler._preempt, Scheduler.schedule, LLMEngine.step

    # 1. Force SWAP (unchanged from 23 Oct) + log what recompute would have to re-prefill
    def _preempt(self, seq_group, blocks_to_swap_out, preemption_mode=None):
        if force_swap:
            preemption_mode = PreemptionMode.SWAP
        mode = preemption_mode or (PreemptionMode.RECOMPUTE
                                   if seq_group.get_max_num_running_seqs() == 1 else PreemptionMode.SWAP)
        seqs = seq_group.get_seqs()
        bt = self.block_manager.block_tables
        log(t="preempt", time=time.time(), rid=seq_group.request_id, mode=mode.name,
            seq_len=sum(s.get_len() for s in seqs),
            num_blocks=sum(len(bt[s.seq_id]) for s in seqs if s.seq_id in bt))
        return preempt0(self, seq_group, blocks_to_swap_out, preemption_mode)

    # 2. Per-step composition, taken from the scheduler output
    def schedule(self):
        out = sched0(self)
        so = out[1]
        seqs = [s for sg in so.scheduled_seq_groups for s in sg.get_seqs() if not s.is_finished()]
        if so.prompt_run and seqs:
            lens = [s.get_len() for s in seqs]
            info = dict(kind="prefill", num_seqs=len(seqs), real_tokens=sum(lens),
                        padded_tokens=len(lens) * max(lens),
                        n_reprefill=sum(s.get_len() > len(s.data.prompt_token_ids) for s in seqs))
        elif seqs:
            info = dict(kind="decode", num_seqs=len(seqs), real_tokens=len(seqs),
                        padded_tokens=len(seqs), n_reprefill=0)
        else:
            info = dict(kind="idle", num_seqs=0, real_tokens=0, padded_tokens=0, n_reprefill=0)
        info["vllm_batched_tokens"] = so.num_batched_tokens
        self._kv_info = info
        return out

    def step(self):
        torch.cuda.synchronize()
        t0 = time.time()
        res = step0(self)
        torch.cuda.synchronize()
        t1 = time.time()
        bm = self.scheduler.block_manager
        log(t="step", start=t0, end=t1, ms=(t1 - t0) * 1e3,
            free=bm.get_num_free_gpu_blocks(), total=bm.num_total_gpu_blocks, **self.scheduler._kv_info)
        return res

    # 3. Swap-in: Python per-block copy loop (stock 0.2.7 swap_blocks asserts on CPU->GPU)
    def swap_in_py(self, src_to_dst):
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                (sk, sv), (dk, dv) = self.cpu_cache[i], self.gpu_cache[i]
                for s, d in src_to_dst.items():
                    dk[d].copy_(sk[s], non_blocking=True)
                    dv[d].copy_(sv[s], non_blocking=True)
                self.events[i].record(stream=self.cache_stream)

    def timed(fn, kind):
        def w(self, src_to_dst):
            torch.cuda.synchronize()
            t0 = time.time()
            r = fn(self, src_to_dst)
            torch.cuda.synchronize()
            log(t=kind, start=t0, ms=(time.time() - t0) * 1e3, blocks=len(src_to_dst))
            return r
        return w

    Scheduler._preempt, Scheduler.schedule, LLMEngine.step = _preempt, schedule, step
    CacheEngine.swap_in = timed(swap_in_py, "swap_in")                 # Python loop
    CacheEngine.swap_out = timed(CacheEngine.swap_out, "swap_out")     # vLLM cache_ops.swap_blocks
