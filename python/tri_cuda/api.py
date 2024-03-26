import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
import operator
from functools import reduce
from typing import Dict
from utils import simple_ptx_info_extract
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
def prod(x):
    return reduce(operator.mul, x, 1)

class TritonOp:
    def __init__(self, op_name) -> None:
        self.op_name = op_name
        self.ptx_code = None
        self.ptx_info = None

    def __repr__(self):
        return self.op_name

    def __hash__(self):
        return self.__repr__().__hash__()
    
    def init_code_from_file(self, ptx_file_path):
        with open(ptx_file_path, 'r') as f:
            self.ptx_code = f.read()
        self.ptx_info = simple_ptx_info_extract(ptx_file_path)
        cuda_module = pycuda.driver.module_from_buffer(self.ptx_code.encode())
        self.kernel_function = cuda_module.get_function(self.ptx_info['kernel_name'])

op_libs: Dict[str, TritonOp]= {}
def get_triton_op(op_name:str):
    if op_name not in op_libs:
        op_obj = TritonOp(op_name)
        op_libs[op_name] = op_obj
        op_obj.init_code_from_file(f'{cur_path}/triton_ops/tcache_{op_name}.ptx')
    
    return op_libs[op_name]


def ewise_op(a, b, out, shp, op_name=None):
    assert op_name is not None
    op_obj = get_triton_op(op_name)
    block = op_obj.ptx_info['block']
    M = prod(shp)
    op_obj.kernel_function(a, b, out, np.int32(M), grid=((M+block[0]-1)//block[0], 1, 1), block=block)

def ewise_add(a, b, out, shp):
    return ewise_op(a, b, out, shp, op_name='add')

def ewise_sub(a, b, out, shp):
    return ewise_op(a, b, out, shp, op_name='sub')

def ewise_mul(a, b, out, shp):
    return ewise_op(a, b, out, shp, op_name='mul')

def ewise_div(a, b, out, shp):
    return ewise_op(a, b, out, shp, op_name='div')

if __name__ == '__main__':
    import sys
    tricuda_dir = os.path.dirname(os.path.abspath(__file__))
    if tricuda_dir not in sys.path:
        sys.path.append(tricuda_dir)
    bsz, l, d = (4, 64, 64)
    a = np.random.randn(bsz, l, d).astype(np.float32)
    b = np.random.randn(bsz, l, d).astype(np.float32)
    c = np.zeros((bsz, l, d), dtype=np.float32)

    ag = cuda.mem_alloc(a.nbytes)
    bg = cuda.mem_alloc(b.nbytes)
    cg = cuda.mem_alloc(c.nbytes)

    cuda.memcpy_htod(ag, a)
    cuda.memcpy_htod(bg, b)
    cuda.memcpy_htod(cg, c)

    ewise_div(ag, bg, cg, tuple(a.shape))
    c_np = a / b

    cuda.memcpy_dtoh(c, cg)
    max_diff = np.max(np.abs(c_np-c))
    print(max_diff)