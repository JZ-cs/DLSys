
# %%
# Compute Kernel
# --------------

import torch
import triton
import triton.language as tl


@triton.jit
def mul_kernel(
    x_ptr,  
    y_ptr,  
    output_ptr, 
    n_elements,  
    BLOCK_SIZE: tl.constexpr, 
):

    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)




def mul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output



torch.manual_seed(0)
size = 1024
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")

output_triton = mul(x, y)
import os
grand_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f'{grand_path}/tcache_mul.ptx', 'w') as f:
    print(list(mul_kernel.cache[0].values())[0].asm['ptx'], file=f) 