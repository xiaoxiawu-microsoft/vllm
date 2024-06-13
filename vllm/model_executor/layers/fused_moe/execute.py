import vllm
import torch
from vllm import _custom_ops as ops
import argparse
import json
import sys
import os
import shutil

import fused_moe
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
    experts=2,
    topk=2,
    intermediate_size=6400,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)

    w1_f32 = torch.ones(experts, hidden_size, intermediate_size * 2).cuda()
    w1 = torch.empty(experts, hidden_size, intermediate_size * 2, dtype=torch.float8_e4m3fn , device=w1_f32.device)
    ws_scale = torch.empty(experts, dtype=torch.float32, device=w1_f32.device)
    for i in range(experts):
        w1[i, :, :], ws_scale[i] = ops.scaled_fp8_quant(w1_f32[i, :, :].half())
    
    w2 = torch.ones(experts, intermediate_size, hidden_size, dtype=torch.float8_e4m3fn, device=w1_f32.device)
    w2s_scale = torch.empty(experts, dtype=torch.float32, device=w1_f32.device)
    w2_f32 = torch.ones(experts, intermediate_size, hidden_size).cuda()
    for i in range(experts):
        w2[i, :, :], w2s_scale[i] = ops.scaled_fp8_quant(w2_f32[i, :, :].half())

    ws_scale = ws_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, w1_f32.size(-1)).contiguous()
    w2s_scale = w2s_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, w2_f32.size(-1)).contiguous()

    searchspace = list(range(32, 256, 32)) + list(range(256, 4097, 256))
    searchspace = [1]
    for tokens in searchspace:
        hidden_state = torch.ones(tokens, hidden_size).cuda().uniform_(-1,1).half()
        gatew = torch.randn(hidden_size, experts).cuda().half()
        gating_output = torch.matmul(hidden_state.half(), gatew).float()
        topk_weights, topk_ids = sparsemixer(gating_output, topk)
        def sparse_mixer_cache(gating_output, topk):
            return topk_weights, topk_ids

        o0 = ampere_fp8_fused_moe.fused_moe(
            hidden_state,
            w1=w1.view(torch.int8),
            w2=w2.view(torch.int8),
            gating_output=gating_output,
            topk=topk,
            override_config=config,
            renormalize=True,
            inplace=False,
            use_fp8=use_fp8,
            w1_scale=ws_scale,
            w2_scale=w2s_scale,
            routing_func=sparse_mixer_cache,
        )
        print(o0)

        o1 = fused_moe.fused_moe(
            hidden_state,
            w1=w1_f32.half().transpose(1,2).contiguous(),
            w2=w2_f32.half().transpose(1,2).contiguous(),
            gating_output=gating_output,
            topk=topk,
            override_config=config,
            renormalize=True,
            inplace=False,
            use_fp8=False,
            routing_func=sparse_mixer_cache,
        )

        print(o1)

        torch.testing.assert_close(o0, o1, rtol=1e-0, atol=1e-1)

moe_perf()