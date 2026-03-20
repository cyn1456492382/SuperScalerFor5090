#!/usr/bin/env python3
"""
Extract arguments from qwen_execute.log, expand special lists into columns named by operations.
Output files are saved under: /{model_name}/{model_size}/{encoder_seq_length}/
"""

import os
import re
import ast
import csv
from collections import OrderedDict

LOG_FILE = "./qwen_execute.log"

# Output filenames (no path, just base names)
OUTPUT_CSV_BASE = "arguments.csv"
LISTS_CSV_BASE = "arguments_lists.csv"
MODEL_CONFIG_CSV_BASE = "model_config.csv"
EXEC_PARAMS_CSV_BASE = "execution_params.csv"

# Keys to be extracted into a separate model configuration file
MODEL_CONFIG_KEYS = [
    "ffn_hidden_size",
    "hidden_size",
    "kv_channels",
    "max_position_embeddings",
    "model_name",
    "model_size",
    "num_attention_heads",
    "num_layers",
    "num_query_groups",
    "params_dtype",
    "vocab_size"
]

# Keys to be extracted into a separate execution parameters file
EXEC_PARAMS_KEYS = [
    "checkpoint_activations",
    "global_batch_size",
    "micro_batch_size",
    "num_ops_in_each_stage",
    "num_stages",
    "resharding_stages"
]

def parse_arguments_block(lines):
    """Parse arguments block lines into a dict."""
    args_dict = OrderedDict()
    pattern = re.compile(r'^\s*(\S+)\s+\.+\s+(.*)$')
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            key = m.group(1)
            val_str = m.group(2).strip()
            try:
                val = ast.literal_eval(val_str)
            except:
                val = val_str
            args_dict[key] = val
        # ignore lines that don't match (e.g., continuation lines)
    return args_dict

def extract_arguments_section(log_text):
    """Extract the last arguments block from log text using new markers."""
    lines = log_text.splitlines()
    start_marker = "------------------------ arguments ------------------------"
    end_marker = "-------------------- end of arguments ---------------------"
    start_indices = [i for i, line in enumerate(lines) if line.strip() == start_marker]
    if not start_indices:
        raise ValueError("No arguments block start marker found")
    start_idx = start_indices[-1]
    end_idx = None
    for i in range(start_idx+1, len(lines)):
        if lines[i].strip() == end_marker:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("No arguments block end marker after start")
    args_lines = lines[start_idx+1:end_idx]
    return args_lines

def extract_op_names(log_text, rank=3):
    """Extract operation names from [rank X all ops] line."""
    pattern = re.compile(rf'^\s*\[rank {rank} all ops\]\s+(.*)$', re.MULTILINE)
    match = pattern.search(log_text)
    if not match:
        raise ValueError(f"No [rank {rank} all ops] line found")
    quoted_names = re.findall(r'"([^"]*)"', match.group(1))
    return quoted_names

def extract_recompute_op_names(log_text, rank=3):
    """Extract recompute operation names from [rank X recompute ops] line.
       Handles both quoted and unquoted lists."""
    pattern = re.compile(rf'^\s*\[rank {rank} recompute ops\]\s+(.*)$', re.MULTILINE)
    match = pattern.search(log_text)
    if not match:
        return []  # not present in all logs
    # Try to extract quoted names first
    quoted_names = re.findall(r'"([^"]*)"', match.group(1))
    if quoted_names:
        return quoted_names
    # Fallback: split by whitespace (assumes no quotes)
    ops_str = match.group(1).strip()
    return ops_str.split()

def extract_complete_parallel_lists(log_text):
    """Extract complete data-parallel-size and model-parallel-size lists."""
    dp_pattern = re.compile(r'data-parallel-size:\s*(\[\[.*?\]\])', re.DOTALL)
    mp_pattern = re.compile(r'tensor-model-parallel size:\s*(\[\[.*?\]\])', re.DOTALL)

    dp_match = dp_pattern.search(log_text)
    mp_match = mp_pattern.search(log_text)

    if not dp_match or not mp_match:
        raise ValueError("Could not find complete parallel lists")

    dp_list_str = dp_match.group(1)
    mp_list_str = mp_match.group(1)

    try:
        dp_list = ast.literal_eval(dp_list_str)
        mp_list = ast.literal_eval(mp_list_str)
    except Exception as e:
        raise ValueError(f"Failed to parse parallel lists: {e}")

    # They are nested lists [[...]]; extract inner list
    if isinstance(dp_list, list) and len(dp_list) == 1 and isinstance(dp_list[0], list):
        dp_list = dp_list[0]
    if isinstance(mp_list, list) and len(mp_list) == 1 and isinstance(mp_list[0], list):
        mp_list = mp_list[0]
    return dp_list, mp_list

def build_recompute_list(op_names, recompute_op_names):
    """Build 0/1 list indicating recompute for each op."""
    recompute_set = set(recompute_op_names)
    return [1 if name in recompute_set else 0 for name in op_names]

def build_algo_list(op_names, algo_val_from_args):
    """Build algo list. If truncated, assume all zeros."""
    if isinstance(algo_val_from_args, list):
        # Expecting [[...]]; extract inner list if possible
        if len(algo_val_from_args) == 1 and isinstance(algo_val_from_args[0], list):
            inner = algo_val_from_args[0]
            if len(inner) == len(op_names):
                return inner
            else:
                print("Warning: algo list length mismatch, using all zeros")
                return [0] * len(op_names)
        elif len(algo_val_from_args) == len(op_names):
            return algo_val_from_args
        else:
            return [0] * len(op_names)
    else:
        return [0] * len(op_names)

def main():
    # --- 1. Read log file ---
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # --- 2. Extract arguments section ---
    args_lines = extract_arguments_section(log_text)
    args_dict = parse_arguments_block(args_lines)

    # --- 3. Extract operation names and recompute ops ---
    op_names = extract_op_names(log_text, rank=3)
    print(f"Found {len(op_names)} operations")

    recompute_op_names = extract_recompute_op_names(log_text, rank=3)
    print(f"Found {len(recompute_op_names)} recompute operations")

    dp_list, mp_list = extract_complete_parallel_lists(log_text)
    print(f"DP list length: {len(dp_list)}, MP list length: {len(mp_list)}")

    # --- 4. Build special lists ---
    recompute_list = build_recompute_list(op_names, recompute_op_names)
    algo_val = args_dict.get('algo_of_each_op', [])
    algo_list = build_algo_list(op_names, algo_val)

    # --- 5. Get dynamic path components ---
    # model_name from args_dict
    model_name = args_dict.get('model_name')
    if not model_name:
        raise ValueError("Required parameter 'model_name' not found in arguments.")
    model_name = str(model_name)

    # encoder_seq_length from args_dict
    encoder_seq_length = args_dict.get('encoder_seq_length')
    if not encoder_seq_length:
        raise ValueError("Required parameter 'encoder_seq_length' not found in arguments.")
    encoder_seq_length = str(encoder_seq_length)

    # model_size from log_path parameter in arguments
    log_path = args_dict.get('log_path')
    if not log_path:
        raise ValueError("Required parameter 'log_path' not found in arguments.")
    # Get the directory of the log_path, then its basename as model_size
    log_dir = os.path.dirname(str(log_path))
    model_size = os.path.basename(log_dir)
    if not model_size:
        raise ValueError("Could not determine model_size from log_path directory.")
    print(f"Extracted model_size: {model_size} (from log_path: {log_path})")

    # --- 6. Build output directory ---
    output_dir = os.path.join(model_name, model_size, encoder_seq_length)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- 7. Build full output paths ---
    output_csv = os.path.join(output_dir, OUTPUT_CSV_BASE)
    lists_csv = os.path.join(output_dir, LISTS_CSV_BASE)
    model_config_csv = os.path.join(output_dir, MODEL_CONFIG_CSV_BASE)
    exec_params_csv = os.path.join(output_dir, EXEC_PARAMS_CSV_BASE)

    # --- 8. Filter args for main arguments CSV ---
    special_keys = ['algo_of_each_op', 'data_parallel_size_of_each_op',
                    'model_parallel_size_of_each_op', 'recompute_ops']

    filtered_args = {}
    for k, v in args_dict.items():
        if k in special_keys:
            continue
        if 'log' in k:
            continue
        if v is None:
            continue
        filtered_args[k] = v

    # --- 9. Write main CSV (arguments.csv) ---
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['parameter', 'value'])
        writer.writeheader()
        for key, val in filtered_args.items():
            writer.writerow({'parameter': key, 'value': str(val)})
    print(f"Main CSV written to {output_csv}")

    # --- 10. Write lists CSV (arguments_lists.csv) ---
    list_columns = ['op'] + special_keys
    with open(lists_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list_columns)
        writer.writeheader()
        for i, op in enumerate(op_names):
            row = {'op': op}
            row['algo_of_each_op'] = algo_list[i] if i < len(algo_list) else ''
            row['data_parallel_size_of_each_op'] = dp_list[i] if i < len(dp_list) else ''
            row['model_parallel_size_of_each_op'] = mp_list[i] if i < len(mp_list) else ''
            row['recompute_ops'] = recompute_list[i] if i < len(recompute_list) else ''
            writer.writerow(row)
    print(f"Lists CSV written to {lists_csv}")

    # --- 11. Write model configuration CSV (model_config.csv) ---
    model_rows = []
    for key in MODEL_CONFIG_KEYS:
        if key in filtered_args:
            model_rows.append({'parameter': key, 'value': str(filtered_args[key])})
    if model_rows:
        with open(model_config_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['parameter', 'value'])
            writer.writeheader()
            writer.writerows(model_rows)
        print(f"Model configuration CSV written to {model_config_csv}")
    else:
        print("No model configuration keys found; skipping model_config.csv")

    # --- 12. Write execution parameters CSV (execution_params.csv) ---
    exec_rows = []
    for key in EXEC_PARAMS_KEYS:
        if key in args_dict:
            exec_rows.append({'parameter': key, 'value': str(args_dict[key])})
    if exec_rows:
        with open(exec_params_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['parameter', 'value'])
            writer.writeheader()
            writer.writerows(exec_rows)
        print(f"Execution parameters CSV written to {exec_params_csv}")
    else:
        print("No execution parameters keys found; skipping execution_params.csv")

if __name__ == '__main__':
    main()