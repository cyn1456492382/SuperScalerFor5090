#!/usr/bin/env python3
"""
Download and save Qwen tokenizer to runtime/vocabs/qwen.

Usage:
  python3 scripts/fetch_qwen_tokenizer.py --model Qwen/Qwen-3-0.6B --out runtime/vocabs/qwen

This requires `transformers` installed and network access.
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen-3-0.6B")
    p.add_argument("--out", type=str, default="runtime/vocabs/qwen3-vocab.json")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("Please install transformers: pip install transformers")
        print("Error:", e)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer for {args.model} ...")
    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tok.save_pretrained(str(out_dir))
    except Exception as e:
        print("Failed to download or save tokenizer:", e)
        sys.exit(2)

    print(f"Saved tokenizer to {out_dir}")


if __name__ == '__main__':
    main()

