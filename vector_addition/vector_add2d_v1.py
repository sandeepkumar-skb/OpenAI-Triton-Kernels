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
    grid_id = tl.program_id(axis=0)
    offset_n = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + grid_id * stride_m  + offset_n * stride_n
    y_ptrs = y_ptr + grid_id * stride_m  + offset_n * stride_n

    x= tl.load(x_ptrs, mask=offset_n < N, other=0.0)
    y= tl.load(y_ptrs, mask=offset_n < N, other=0.0)
    out = x + y
    out_ptrs = output_ptr + grid_id * stride_m + offset_n * stride_n
    tl.store(out_ptrs, out, mask=offset_n < N)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda and x.shape == y.shape
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    # grid = (tl.cdiv(M, BLOCK_SIZE), ) # parallelizing across rows.
    grid = lambda meta: (M, )

    add_kernel[grid](x,
                     y,
                     output,
                     M, N,
                     x.stride(0), x.stride(1),
                     BLOCK_SIZE=BLOCK_SIZE)

    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 101, 100001
    x = torch.rand((M, N), device='cuda')
    y = torch.rand((M, N), device='cuda')

    output_torch = x + y
    output_triton = add(x, y)
    logging.info(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')


