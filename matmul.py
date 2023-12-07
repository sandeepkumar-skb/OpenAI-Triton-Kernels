import triton
import triton.language as tl
import torch
import logging
logging.basicConfig(level=logging.INFO)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  ):
    pid = tl.program_id(0)

    ngrids_m = tl.cdiv(M, BLOCK_SIZE_M)
    ngrids_n = tl.cdiv(N, BLOCK_SIZE_N)

    m_id = pid // ngrids_n
    n_id = pid % ngrids_n

    offs_m = (m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k*BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k*BLOCK_SIZE_K), other=0.0)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



def matmul(a: torch.Tensor, b: torch.Tensor, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
    )
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    for _ in range(1):
        a = torch.rand((892, 512), device='cuda', dtype=torch.float16)
        b = torch.rand((512, 512), device='cuda', dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        # print(f"triton_output={triton_output}")
        # print(f"torch_output={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            logging.info("✅ Triton and Torch match")
        else:
            logging.info("❌ Triton and Torch differ")

