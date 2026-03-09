#!/usr/bin/env python3
"""
Fetch Qwen-3-0.6B config from Hugging Face and save a local JSON for profiler/runtime.

Usage: python3 scripts/fetch_qwen_config.py --model Qwen/Qwen-3-0.6B

This script requires network access and `transformers` installed in the environment.
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen-3-0.6B")
    p.add_argument("--out", type=str, default="profiler/qwen_config.json")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoConfig
    except Exception as e:
        print("Please install transformers in this environment:")
        print("  pip install transformers")
        print("Error:", e)
        sys.exit(1)

    print(f"Fetching config for {args.model} from HF Hub...")
    try:
        cfg = AutoConfig.from_pretrained(args.model)
    except Exception as e:
        print("Failed to download config:", e)
        sys.exit(2)

    # map common fields to profiler/runtime expectations
    out = {
        "num_layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None)),
        "seq_len": getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", None)),
        "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None)),
        "ffn_hidden_size": getattr(cfg, "intermediate_size", None) or (getattr(cfg, "hidden_size", None) * 4 if getattr(cfg, "hidden_size", None) else None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None)),
        "kv_channels": getattr(cfg, "kv_channels", None) or (getattr(cfg, "hidden_size", None) // getattr(cfg, "num_attention_heads", 1) if getattr(cfg, "hidden_size", None) else None),
        "vocab_size": getattr(cfg, "vocab_size", getattr(cfg, "vocab_size", None)),
        "params_dtype": "fp16"
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote profiler config to {out_path}")


if __name__ == "__main__":
    main()

