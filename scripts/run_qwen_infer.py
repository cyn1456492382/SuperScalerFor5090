#!/usr/bin/env python3
"""Simple distributed inference wrapper for causal LM (Qwen-3-0.6B).

This script is intended to be launched with `accelerate launch --num_processes 8`.
It uses `device_map='auto'` so `accelerate` will shard the model across the 8 GPUs.

Notes:
- Ensure `accelerate` is installed and you use the same python env for launch.
- For HF hub models that require `trust_remote_code`, we enable it by default.
"""
import argparse
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--ctx_len", type=int, default=8192)
    p.add_argument("--prompt", type=str, default="Hello")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        print("Missing dependencies: install transformers and torch. Error:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer and model: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # Some models don't have pad_token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Let accelerate/transformers shard the model across devices
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.ctx_len)

    # Move inputs to the first device (accelerate maps tensors automatically in many cases)
    # If tensors need to be moved, device_map='auto' + accelerate handles it.

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0.0),
        temperature=args.temperature if args.temperature > 0.0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("Generating...")
    # generate will be handled by the sharded model
    outputs = model.generate(**generation_kwargs)

    # Only decode the sequences (they are on CPU or moved automatically)
    if isinstance(outputs, dict):
        sequences = outputs["sequences"]
    else:
        sequences = outputs

    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    print("\n--- OUTPUT ---\n")
    for text in decoded:
        print(text)


if __name__ == "__main__":
    main()
