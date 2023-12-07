import triton
import triton.language as tl
import torch
import logging
logging.basicConfig(level=logging.INFO)


@triton.jit
def dropout_kernel(x_ptr,
                  x_keep_ptr,
                  out_ptr,
                  p,
                  M, N,
                  stride_xm, stride_xn,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    off_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_n = tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    x_keep_ptrs = x_keep_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    out_ptrs = out_ptr + off_m[:, None]*stride_xm + off_n[None, :] * stride_xn

    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        mask = (off_m[:, None] < M) & (off_n[None, :] < N - n*BLOCK_SIZE)
        x = tl.load(x_ptrs, mask=mask)
        x_keep = tl.load(x_keep_ptrs, mask=mask)
        out = tl.where(x_keep, x / (1-p), 0.0)
        tl.store(out_ptrs, out, mask=mask)
        x_ptrs += BLOCK_SIZE * stride_xn
        x_keep_ptrs += BLOCK_SIZE * stride_xn
        out_ptrs += BLOCK_SIZE * stride_xn


def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float):
    """Dropout.
    Driver Function for dropout kernel.
    """
    output = torch.empty_like(x)
    assert x.is_contiguous()
    M, N = x.shape
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), )

    dropout_kernel[grid](x,
                         x_keep,
                         output,
                         p,
                         M, N,
                         x.stride(0), x.stride(1),
                         BLOCK_SIZE=16)
    return output

if __name__ == "__main__":
    # Input tensor
    M, N = 520, 721
    # Probability
    p = 0.5

    # Input Tensor
    x = torch.randn((M, N)).cuda()
    # Dropout Mask
    x_keep = (torch.rand((M, N)) > p).cuda()

    # Torch Output
    out_torch = torch.where(x_keep, x / (1-p), 0.0)

    # Triton Output
    out_tl = dropout(x, x_keep, p)

    # Correctness Check
    assert torch.equal(out_tl, out_torch)
    logging.info("Success")
