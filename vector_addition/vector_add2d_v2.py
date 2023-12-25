import torch
import triton
import triton.language as tl
import logging
logging.basicConfig(level=logging.INFO)

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               M, N,
               stride_m, stride_n,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)

    offs_m = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    offs_n = tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    y_ptrs = y_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    output_ptrs = output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < (N - n*BLOCK_SIZE))
        x = tl.load(x_ptrs, mask=mask)
        y = tl.load(y_ptrs, mask=mask)

        out = x + y
        tl.store(output_ptrs, out, mask=mask)

        x_ptrs += BLOCK_SIZE * stride_n
        y_ptrs += BLOCK_SIZE * stride_n
        output_ptrs += BLOCK_SIZE * stride_n


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda and x.shape == y.shape
    M, N = x.shape

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), )

    add_kernel[grid](x,
                     y,
                     output,
                     M, N,
                     x.stride(0), x.stride(1),
                     BLOCK_SIZE=16)

    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 131, 131
    x = torch.rand((M, N), device='cuda')
    y = torch.rand((M, N), device='cuda')

    output_torch = x + y
    output_triton = add(x, y)
    logging.info(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')


