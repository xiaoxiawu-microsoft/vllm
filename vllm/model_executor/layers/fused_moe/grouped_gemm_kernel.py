import bitblas
from bitblas.ops.general_matmul_splitk import MatmulWithSplitK, MatmulConfigWithSplitK
from bitblas.cache import global_operator_cache, get_database_path
from bitblas import auto_detect_nvidia_target
import pytest

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = get_database_path()

# Grouped gemm
def grouped_gemm(
    M,
    N,
    K,
    A_dtype="float16",
    W_dtype="e4m3_float8",
    accum_dtype="float32",
    out_dtype="float16",
    layout="nt",
    with_bias=False,
    group_size=-1,
    with_scaling=False,
    with_zeros=False,
    zeros_mode=None,
    SplitK=4,
):
    if global_operator_cache.size() == 0:
        global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

    matmul_config = MatmulConfigWithSplitK(
        k_split=SplitK,
        M=[16],
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )
    bitblas_matmul = global_operator_cache.get(matmul_config)

    if bitblas_matmul is None:
        bitblas_matmul = MatmulWithSplitK(config=matmul_config, enable_tuning=False)
        global_operator_cache.add(matmul_config, bitblas_matmul)
        global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        print("New operator added to the database")
    return bitblas_matmul


@pytest.mark.parametrize(
    "SplitK,M,N,K,A_dtype,W_dtype,accum_dtype,out_dtype,layout,with_bias,group_size,with_scaling,with_zeros,zeros_mode",
    [
        (4, 1, 12800, 4096, "float16", "e4m3_float8", "float32", "float16", "nt", False, -1, True,
         False, None),
    ],
)
def test_matmul_torch_forward_fp8e4m3(SplitK, M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype,
                                         layout, with_bias, group_size, with_scaling, with_zeros,
                                         zeros_mode):
    import torch
    torch.random.manual_seed(0)
    matmul_config = MatmulConfigWithSplitK(
        k_split=SplitK,
        M=[1, 16],
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )

    matmul = global_operator_cache.get(matmul_config)

    if matmul is None:
        matmul = MatmulWithSplitK(config=matmul_config, enable_tuning=False)
        global_operator_cache.add(matmul_config, matmul)
        global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        print("New operator added to the database")

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    def map_torch_type(intype):

        typemap = {
            'e4m3_float8': torch.float8_e4m3fn,
            'e5m2_float8': torch.float8_e5m2,
        }
        if intype in typemap:
            return typemap[intype]
        else:
            return getattr(torch, intype)

    numpytype_a = map_torch_type(A_dtype)
    numpytype_b = map_torch_type(W_dtype)

    torch_a = torch.rand(M * K).uniform_(-1, 1).reshape(input_shape).type(numpytype_a).cuda()
    torch_b = torch.rand(N * K).uniform_(-1, 1).reshape(weight_shape).type(numpytype_b).cuda()
    ref_out = torch.matmul(torch_a.to(torch.float32),
                           torch_b.t().to(torch.float32)) if layout == "nt" else torch.matmul(
                               torch_a.to(torch.float32), torch_b.to(torch.float32))
    ref_out = ref_out.to(torch.float16)

    # !pip install vllm
    from vllm import _custom_ops as ops

    b, b_scale = ops.scaled_fp8_quant(
        torch_b.to(torch.float32).bfloat16(), torch.ones(1).cuda() * 0.0022
    )

    b_scale = b_scale.expand(N)
    one_scale = torch.ones(N).cuda()

    bitblas_out = torch.empty_like(ref_out)
    matmul.forward(torch_a, torch_b, scale=one_scale.to(torch_a.dtype), output=bitblas_out)
    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)

    torch.testing.assert_close(bitblas_out, ref_out, rtol=1e0, atol=1e-1)

    matmul.forward(torch_a, b, scale=b_scale.to(torch_a.dtype), output=bitblas_out)

    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)
    torch.testing.assert_close(bitblas_out, ref_out, rtol=1e0, atol=1e-1)

