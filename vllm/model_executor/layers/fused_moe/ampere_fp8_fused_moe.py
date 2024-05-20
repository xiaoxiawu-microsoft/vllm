"""Fused MoE kernel with FP8 weight using Ampere."""

import vllm
import torch
from vllm import _custom_ops as ops
import cupy
from typing import Dict, Any, Optional, Callable

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_configs,
    moe_align_block_size,
    invoke_fused_moe_kernel,
)

# <todo:wenxh> Kernels performance needs to be optimized
#   such as one thread deals with multiple elements to reduce memory transaction.

convert_fp8e4m3_to_half = cupy.RawKernel(
    r"""
#include "cuda_fp8.h"
#include "cuda_fp16.h"
extern "C" __global__
void convert_fp8e4m3_to_half(const __nv_fp8_storage_t* x, float *scale_p, half* y, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float scale = *scale_p;
    if (tid < size)
        y[tid] = __nv_cvt_fp8_to_halfraw(x[tid], __NV_E4M3) * scale;
}
""",
    "convert_fp8e4m3_to_half",
)

convert_fp8e4m3_to_bfloat16 = cupy.RawKernel(
    r"""
#include "cuda_fp8.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
extern "C" __global__
void convert_fp8e4m3_to_bfloat16(const __nv_fp8_storage_t* x, float* scale_p, __nv_bfloat16* y, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float scale = *scale_p;
    if (tid < size)
        y[tid] = __float2bfloat16(__nv_cvt_fp8_to_halfraw(x[tid], __NV_E4M3) * scale);
}
""",
    "convert_fp8e4m3_to_bfloat16",
)


def dequantize_fp8(t_fp8, scales, dtype=torch.float16):
    s = torch.empty_like(t_fp8, dtype=dtype)
    convert = (
        convert_fp8e4m3_to_half
        if dtype == torch.float16
        else convert_fp8e4m3_to_bfloat16
    )

    expert_num = t_fp8.shape[0]

    expert_in = torch.chunk(t_fp8, expert_num, dim=0)
    expert_out = torch.chunk(s, expert_num, dim=0)

    for i in range(expert_num):
        scale = scales[i]
        convert(
            ((expert_in[i].numel() + 1024 - 1) // 1024,),
            (1024,),
            (expert_in[i].data_ptr(), scale.data_ptr(), expert_out[i].data_ptr(), t_fp8.numel()),
        )
    return s


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    training: bool = False,
    sparse_mixer: bool = False,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    routing_func: Callable = torch.topk,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    This layer works the same as fused_moe, but it is used for the Ampere arch, which does not support fp8.
    By default, to be more comparable to Hopper, we reuse E4M3 configuration.
    <todo:wenxh> Use FP8E4b16 to reduce overhead:
        https://github.com/triton-lang/triton/blob/d7c8b3d7890125f5fc1b9f046e3189baa2665be4/python/triton/language/extra/cuda/utils.py#L34

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
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
    elif is_hip():
        # The MoE kernels are not yet supported on ROCm.
        routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = routing_func(routing_weights, topk)
    else:
        import vllm._moe_C as moe_kernels

        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        token_expert_indicies = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )
        moe_kernels.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),  # TODO(woosuk): Optimize this.
        )
        del token_expert_indicies  # Not used. Will be used in the future.
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        configs = get_moe_configs(E, w2.shape[2], None)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }

            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                }

    if M == 1:
        # expert, hs1, hs2
        topk_w1 = w1.view(torch.uint8)[topk_ids.flatten()]
        topk_w2 = w2.view(torch.uint8)[topk_ids.flatten()]
        topk_ids = torch.arange(
            topk, device=topk_ids.device, dtype=topk_ids.dtype
        ).unsqueeze(0)

        E = topk

        w1_scale = w1_scale[topk_ids.flatten()]
        w1 = dequantize_fp8(topk_w1, w1_scale, dtype=hidden_states.dtype)

    else:
        w1 = dequantize_fp8(w1, w1_scale, dtype=hidden_states.dtype)

    use_fp8 = False
    w1_scale = None
    a1_scale = None
    a2_scale = None

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

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    del w1

    if M == 1:
        w2_scale = w2_scale[topk_ids.flatten()]
        w2 = dequantize_fp8(topk_w2, w2_scale, dtype=hidden_states.dtype)
    else:
        w2 = dequantize_fp8(w2, w2_scale, dtype=hidden_states.dtype)
        
    w2_scale = None

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    del w2

    if inplace:
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=hidden_states,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
