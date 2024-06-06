import vllm
import torch
from vllm import _custom_ops as ops
import argparse
import json
import sys
import os
import shutil

import fused_moe
import ampere_fp8_fused_moe_large_tokens
import ampere_fp8_fused_moe


import functools

def timeit_decorator(times=100):
    def decorator(function_call):
        @functools.wraps(function_call)
        def wrapper(*args, **kwargs):

            # cuda graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3):
                    function_call(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                function_call(*args, **kwargs)

            all_time = 0.0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for j in range(times):
                #function_call(*args, **kwargs)
                g.replay()

            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            all_time = elapsed_time_ms
            
            avg_time = all_time / times
            print(f"{function_call.__name__} average time: {avg_time} ms")
            return function_call(*args, **kwargs)
        
        return wrapper
    return decorator


def sparsemixer(scores, top_k, jitter_eps=0.01):
    assert top_k == 2

    ################ first expert ################

    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts = max_ind

    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    multiplier = multiplier_o

    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2 = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


def moe_perf(
    tokens=1024,
    experts=8,
    topk=2,
    intermediate_size=14336,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)

    hidden_state = torch.randn(tokens, hidden_size).uniform_(-1, 1).cuda().half()

    if use_fp8:
        w1_f32 = torch.randn(experts, intermediate_size * 2, hidden_size).uniform_(-1, 1).cuda()
        w1, ws_scale = ops.scaled_fp8_quant(
            w1_f32.bfloat16(), torch.ones(experts, dtype=torch.float32, device=w1_f32.device)* 10
        )
        w2_f32 = torch.ones(experts, hidden_size, intermediate_size).uniform_(-1, 1).cuda()
        w2, w2s_scale = ops.scaled_fp8_quant(
            w2_f32.half()
        )
        _, h_scale = ops.scaled_fp8_quant(hidden_state)
        print(f"ws_scale: {ws_scale}")
        print(f"w2s_scale: {w2s_scale}")
        ws_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device) * ws_scale
        w2s_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device) * w2s_scale
        fused_moe_f = ampere_fp8_fused_moe_large_tokens.fused_moe
    else:
        w1 = torch.randn(experts, intermediate_size * 2, hidden_size).cuda().half()
        w2 = torch.randn(experts, hidden_size, intermediate_size).cuda().half()
        h_scale = None
        ws_scale = None
        w2s_scale = None
        fused_moe_f = fused_moe.fused_moe

    gatew = torch.randn(hidden_size, experts).cuda().half()
    gating_output = torch.matmul(hidden_state.half(), gatew).float()

    #@timeit_decorator()
    def run_fused_moe(*args, **kwargs):
        return fused_moe_f(*args, **kwargs)

    r1 = run_fused_moe(
        hidden_states=hidden_state,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        topk=topk,
        override_config=config,
        renormalize=False,
        inplace=True,
        use_fp8=use_fp8,
        w1_scale=ws_scale,
        w2_scale=w2s_scale,
        a1_scale=h_scale,
        a2_scale=h_scale,
        routing_func=sparsemixer
    )

    r2 = fused_moe.fused_moe(
        hidden_states=hidden_state,
        w1=w1_f32.half(),
        w2=w2_f32.half(),
        gating_output=gating_output,
        topk=topk,
        override_config=config,
        renormalize=False,
        inplace=True,
        use_fp8=False,
        routing_func=sparsemixer
    )

    print(r1)
    print(r2)
    torch.testing.assert_close(r1, r2, rtol=1e-0, atol=1e-1)

searchspace = [1] + list(range(0, 256, 32))[1:] + list(range(256, 4097, 256))
intermediate_size = 6400
expert_num = 16

for tk in searchspace:
    print(
        tk,
        ",",
        moe_perf(tokens=tk, experts=expert_num, intermediate_size=intermediate_size, use_fp8=True),
    )