"""Fused MoE kernel."""

import functools
import json
import os
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe import gather_scatter_kernel

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import is_hip

import pycublas.trtllm_moe_grouped_gemm as moe_kernel

logger = init_logger(__name__)

# Ampere FP8 kernel
from torch.utils.cpp_extension import load

ampere_fp8 = load(
    name="ampere_fp8",
    sources=[
        os.path.join(
            os.path.dirname(__file__), "csrc", "moe_align_block_size_kernels.cu"
        )
    ],
)

moe_kernel_cfg = {
    4096: {
        1: 4,
        32: 5,
        64: 4,
        96: 5,
        128: 4,
        160: 17,
        192: 17,
        224: 16,
        256: 16,
        512: 23,
        768: 26,
        1024: 26,
        1280: 23,
        1536: 23,
        1792: 26,
        2048: 26,
        2304: 27,
        2560: 27,
        2816: 27,
        3072: 27,
        3328: 27,
        3584: 26,
        3840: 26,
        4096: 26,
    },
    6400: {
        1: 4,
        32: 4,
        64: 4,
        96: 4,
        128: 4,
        160: 16,
        192: 16,
        224: 16,
        256: 16,
        512: 20,
        768: 26,
        1024: 26,
        1280: 22,
        1536: 20,
        1792: 26,
        2048: 26,
        2304: 20,
        2560: 21,
        2816: 26,
        3072: 27,
        3328: 26,
        3584: 26,
        3840: 26,
        4096: 26,
    },
}


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    expert_off = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=topk_ids.device
    )
    expert_length = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=topk_ids.device
    )

    ampere_fp8.ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_off,
        expert_length,
    )
    return (
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_off.to(torch.int64),
        expert_length,
    )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    routing_func: Callable = torch.topk,
    cfg_id=20,
) -> torch.Tensor:
    # Check constraints.
    M, K = hidden_states.shape
    E, _, N = w1.shape
    block_m = 16
    block_k = 128
    splitk = 4
    ME = M + E // topk

    hidden_states_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float16)

    if routing_func != torch.topk:
        topk_weights, topk_ids = routing_func(gating_output, topk)
    
    hidden_states = torch.cat((hidden_states, torch.empty((E//topk, K), dtype=hidden_states.dtype, device=hidden_states.device)), dim=0)
    topk_weights = torch.cat((topk_weights, torch.empty((E//topk, topk), dtype=topk_weights.dtype, device=topk_weights.device)), dim=0)
    topk_ids = torch.cat((topk_ids, torch.arange(0,E,1, dtype=topk_ids.dtype, device=topk_ids.device).view((E//topk, topk))), dim=0)

    sorted_token_ids, expert_ids, num_tokens_post_padded, expert_off, expert_length = (
        moe_align_block_size(topk_ids, block_m, E)
    )

    intermediate_cache3 = torch.empty(
        (ME, topk, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache = torch.empty(
        (sorted_token_ids.size(0), K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # hidden states -> sorted hidden states
    gather_scatter_kernel.invoke_moe_gather(
        hidden_states,
        gathered_cache,
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        block_m,
        block_k,
        topk,
        4,
    )

    gathered_cache_1 = torch.empty(
        (sorted_token_ids.size(0), N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache_2 = torch.empty(
        (sorted_token_ids.size(0), N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache_3 = torch.empty(
        (sorted_token_ids.size(0), K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    total_rows_before_expert = expert_off[:E]

    fc1_cfg_id = moe_kernel_cfg[K][
        min(
            moe_kernel_cfg[K].keys(),
            key=lambda x: abs(x - sorted_token_ids.size(0) // 16),
        )
    ]
    fc2_cfg_id = moe_kernel_cfg[N // 2][
        min(
            moe_kernel_cfg[N // 2].keys(),
            key=lambda x: abs(x - sorted_token_ids.size(0) // 16),
        )
    ]

    moe_kernel.grouped_gemm(
        gathered_cache.view(torch.float16),
        w1.view(torch.int8),
        w1_scale.to(hidden_states.dtype),
        total_rows_before_expert,
        gathered_cache_1,
        5,
        cfg_id,
    )

    # ret_gathered_cache_1 = torch.empty(
    #     (ME, 2, N),
    #     device=hidden_states.device,
    #     dtype=hidden_states.dtype,
    # )

    # gather_scatter_kernel.invoke_moe_scatter(
    #     gathered_cache_1,
    #     ret_gathered_cache_1.view(-1, N),
    #     sorted_token_ids,
    #     num_tokens_post_padded,
    #     topk_ids,
    #     block_m,
    #     block_k,
    #     topk,
    #     splitk,
    # )

    # return ret_gathered_cache_1

    ops.silu_and_mul(gathered_cache_2, gathered_cache_1.view(-1, N))
    moe_kernel.grouped_gemm(
       gathered_cache_2.view(torch.float16),
       w2.view(torch.int8),
       w2_scale.view(hidden_states.dtype),
       total_rows_before_expert,
       gathered_cache_3,
       5,
       cfg_id,
    )

    gather_scatter_kernel.invoke_moe_scatter(
        gathered_cache_3,
        intermediate_cache3.view(-1, K),
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        block_m,
        block_k,
        topk,
        splitk,
        topk_weights=topk_weights,
    )

    intermediate_cache3 = intermediate_cache3[:M,:,:].to(hidden_states_dtype)

    if inplace:
        hidden_states = hidden_states[:M, :].view(dtype=hidden_states_dtype)
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=hidden_states,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
