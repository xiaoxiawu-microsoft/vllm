import bitblas
from bitblas.ops.general_matmul_splitk import MatmulWithSplitK, MatmulConfigWithSplitK
from bitblas.cache import global_operator_cache, get_database_path
from bitblas import auto_detect_nvidia_target

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