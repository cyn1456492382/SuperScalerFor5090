# coding=utf-8
"""Pretrain entry for Qwen models (wrapper around Megatron pretrain pipeline).

This script mirrors `pretrain_gpt.py` but sets default args for Qwen-like models.
It will read `profiler/qwen_config.json` if present to populate model sizes.
"""
import json
import os
from functools import partial
import torch
from megatron import get_args, print_rank_0, get_timers, mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import FlexGPTModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


# def _load_qwen_defaults():
#     # default placeholders; prefer reading profiler/qwen_config.json
#     defaults = dict(
#         num_layers=24,
#         seq_length=8192,
#         hidden_size=2048,
#         ffn_hidden_size=2048 * 4,
#         num_attention_heads=32,
#         kv_channels=2048 // 32,
#         padded_vocab_size=51200,
#         num_query_groups=8
#     )
#     try:
#         cfg_path = os.path.join(os.path.dirname(__file__), "..", "profiler", "qwen_config.json")
#         cfg_path = os.path.abspath(cfg_path)
#         if os.path.exists(cfg_path):
#             with open(cfg_path, "r") as f:
#                 data = json.load(f)
#             defaults.update({
#                 "num_layers": data.get("num_layers", defaults["num_layers"]),
#                 "seq_length": data.get("seq_len", defaults["seq_length"]),
#                 "hidden_size": data.get("hidden_size", defaults["hidden_size"]),
#                 "ffn_hidden_size": data.get("ffn_hidden_size", defaults["ffn_hidden_size"]),
#                 "num_attention_heads": data.get("num_attention_heads", defaults["num_attention_heads"]),
#                 "kv_channels": data.get("kv_channels", defaults["kv_channels"]),
#                 "padded_vocab_size": data.get("vocab_size", defaults["padded_vocab_size"]),
#                 "num_query_groups": data.get("num_query_groups", None)
#             })
#     except Exception:
#         pass
#     return defaults


def model_provider(pre_process=True, post_process=True):
    print_rank_0('building Qwen-like GPT model ...')
    model = FlexGPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    args = get_args()
    vocab_size = getattr(args, "vocab_size", 50257)
    tokens = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
    loss_mask =  (torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
    attention_mask = (torch.rand((args.micro_batch_size, 1, args.seq_length, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
    position_ids = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length
    return tokens, loss_mask, position_ids, attention_mask


def loss_func(loss_mask, output_tensor):
    losses = output_tensor["output"].float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model, extra_tensors_):
    args = get_args()
    timers = get_timers()
    timers('batch-generator').start()
    tokens, loss_mask, position_ids, attention_mask = get_batch(data_iterator)
    input_tensors = {}
    input_tensors["enc_input_ids"] = tokens
    input_tensors["enc_position_ids"] = position_ids
    extra_tensors = {}
    extra_tensors["enc_attention_mask"] = attention_mask
    if extra_tensors_ is not None:
        for key in extra_tensors_:
            extra_tensors[key] = extra_tensors_[key]
    timers('batch-generator').stop()
    if mpu.is_pipeline_last_stage():
        output_tensor = model(input_tensors, extra_tensors)
        ouput_extra_tensors = None
    else:
        output_tensor, ouput_extra_tensors = model(input_tensors, extra_tensors)
    return output_tensor, ouput_extra_tensors, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    # Use synthetic data unless user configures data pipeline
    return None, None, None


if __name__ == "__main__":
    # qwen_defaults = _load_qwen_defaults()
    # # Map to Megatron arg names expected by pretrain
    # args_defaults = {
    #     'seq_length': qwen_defaults['seq_length'],
    #     'hidden_size': qwen_defaults['hidden_size'],
    #     'ffn_hidden_size': qwen_defaults['ffn_hidden_size'],
    #     'num_attention_heads': qwen_defaults['num_attention_heads'],
    #     'kv_channels': qwen_defaults['kv_channels'],
    #     'num_layers': qwen_defaults['num_layers'],
    #     'padded_vocab_size': qwen_defaults['padded_vocab_size'],
    #     # Tokenizer defaults: prefer local runtime/vocabs/qwen if present
    #     'tokenizer_type': 'GPT2BPETokenizer',
    #     'vocab_file': 'vocabs/qwen/vocab.json',
    #     'merge_file': 'vocabs/qwen/merges.txt',
    #     'tokenizer_name_or_path': None,
    # }

    forward_step_func = forward_step

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
    #          args_defaults={'tokenizer_type': 'Qwen3Tokenizer'})
