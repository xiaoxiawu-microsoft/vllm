"""Fused MoE kernel."""

import functools
import json
import os
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe import gather_scatter_kernel
from vllm.model_executor.layers.fused_moe import grouped_gemm_kernel

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import is_hip

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
    expert_off = torch.empty((num_experts + 1), dtype=torch.int32, device=topk_ids.device)
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
    return sorted_ids, expert_ids, num_tokens_post_pad, expert_off, expert_length


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
) -> torch.Tensor:
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    block_m = 128
    block_k = 128
    splitk = 4

    hidden_states_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float16)

    g_w1_scale = w1_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, N).contiguous()
    g_w2_scale = w2_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, K).contiguous()

    if routing_func != torch.topk:
        topk_weights, topk_ids = routing_func(gating_output, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded, expert_off, expert_length = (
        moe_align_block_size(topk_ids, block_m, E)
    )

    intermediate_cache3 = torch.empty(
        (M, topk, K),
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

    #ggemm_kernel_1 = grouped_gemm_kernel.grouped_gemm(M, N, K, with_scaling=True)
    #ggemm_kernel_2 = grouped_gemm_kernel.grouped_gemm(M, K, N // 2, with_scaling=True)

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

    for i in range(0, E):
        start = expert_off[i]
        end = expert_off[i] + expert_length[i]
        a = gathered_cache[start:end, :]
        w = w1[i, :, :]
        c = gathered_cache_1[start:end, :]
        scale = g_w1_scale[i, :]

        #ggemm_kernel_1.forward(
        #    a, w, scale=scale, output=c
        #)

        w_t = w.t().to(torch.float16)
        torch.mm(a, w_t*scale, out=c)

    ops.silu_and_mul(gathered_cache_2, gathered_cache_1.view(-1, N))
    
    debug_scatter_cache = torch.empty(
        (M, topk, N//2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gather_scatter_kernel.invoke_moe_scatter(
        gathered_cache_2,
        debug_scatter_cache.view(-1, N//2),
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        block_m,
        block_k,
        topk,
        splitk,
    )

    return debug_scatter_cache.to(hidden_states_dtype).view(M*topk, N//2)

    for i in range(0, E):
        start = expert_off[i]
        end = expert_off[i] + expert_length[i]

        a = gathered_cache_2[start:end, :]
        w = w2[i, :, :]
        c = gathered_cache_3[start:end, :]
        scale = g_w2_scale[i, :]

        #ggemm_kernel_2.forward(
        #    a, w, scale=scale,output=c
        #)

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

    intermediate_cache3 = intermediate_cache3.to(hidden_states_dtype)

    if inplace:
        hidden_states = hidden_states.view(dtype=hidden_states_dtype)
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=hidden_states,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
