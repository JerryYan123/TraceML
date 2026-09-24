#!/usr/bin/env python3
"""Standalone vLLM generation worker for the TraceML labelers.

Runs in its own process (optionally another Python via TRACEML_VLLM_PYTHON) and
imports nothing from traceml_toolkit, so any environment with vLLM can run it.

Input JSONL rows:  {"idx": int, "prompt": str}   (chat-rendered prompts)
Output JSONL rows: {"idx": int, "text": str, "n_tokens": int, "finish_reason": str}

Two deliberate settings:

* Prefix caching is disabled. With it on, a batch of near-identical prompts
  (the normal case within one run, whose versions share most of their code)
  drives greedy decoding into a repetition loop and every row comes back
  unparsed (measured: 0/16 parsed with caching, 16/16 without, same prompts).
* The output file is written to a temporary name and renamed when complete,
  then the process exits immediately: vLLM 0.8.5 can hang in engine teardown
  after generation has finished, and the parent only waits for the file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    with open(args.input) as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop=["<|im_end|>"])
    outputs = llm.generate([r["prompt"] for r in rows], sampling)

    tmp = args.output + ".partial"
    with open(tmp, "w") as fh:
        for r, o in zip(rows, outputs):
            c = o.outputs[0]
            fh.write(json.dumps({"idx": r["idx"], "text": c.text,
                                 "n_tokens": len(c.token_ids),
                                 "finish_reason": c.finish_reason}) + "\n")
    os.replace(tmp, args.output)
    print(f"[vllm_worker] wrote {len(rows)} generations to {args.output}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
