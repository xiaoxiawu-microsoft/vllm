"""Fused MoE kernel."""

import functools
import json
import os
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)


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
    expert_off = torch.empty((num_experts), dtype=torch.int32, device=topk_ids.device)
    expert_length = torch.empty(
        (num_experts), dtype=torch.int32, device=topk_ids.device
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
    M, _ = hidden_states.shape
    E, N, _ = w1.shape

    if routing_func != torch.topk:
        topk_weights, topk_ids = routing_func(gating_output, topk)

    sorted_ids, expert_ids, num_tokens_post_pad, expert_off, expert_length = (
        moe_align_block_size(topk_ids, 128, E)
    )

    # assert torch.sum(expert_length) == M * topk, f"Number of tokens mismatch, {torch.sum(expert_length)} != {M * topk}"

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    if inplace:
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=hidden_states,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
