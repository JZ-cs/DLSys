#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if(gid < size){
    size_t idxs[MAX_VEC_SIZE];
    size_t x = gid;
    size_t pos = offset;
    #pragma unroll
    for(int i = shape.size-1; i >= 0; i--){
      idxs[i] = x % shape.data[i];
      x /= shape.data[i];
      pos += idxs[i]*strides.data[i];
    }
    out[gid] = a[pos];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t idxs[MAX_VEC_SIZE];
    size_t x = gid;
    size_t pos = offset;
    #pragma unroll
    for(int i = shape.size-1; i >= 0; i--){
      idxs[i] = x % shape.data[i];
      x /= shape.data[i];
      pos += idxs[i]*strides.data[i];
    }
    out[pos] = a[gid];
  }                              
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t idxs[MAX_VEC_SIZE];
    size_t x = gid;
    size_t pos = offset;
    #pragma unroll
    for(int i = shape.size-1; i >= 0; i--){
      idxs[i] = x % shape.data[i];
      x /= shape.data[i];
      pos += idxs[i]*strides.data[i];
    }
    out[pos] = val;
  }                              
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
enum class EwiseOp{
  MUL, DIV, MAXIMUM, EQ, GE
};

enum class EwiseFnOp{
  LOG, EXP, TANH
};

enum class ScalarOp{
  MUL, DIV, MAXIMUM, EQ, GE, POWER
};


__global__ void Ewise_apply_op_Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, EwiseOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == EwiseOp::MUL){
      out[gid] = a[gid] * b[gid];
    } else if(op == EwiseOp::DIV){
      out[gid] = a[gid] / b[gid];
    } else if(op == EwiseOp::MAXIMUM){
      out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
    } else if(op == EwiseOp::EQ){
      out[gid] = a[gid] == b[gid];
    } else if(op == EwiseOp::GE){
      out[gid] = a[gid] >= b[gid];
    }
  }
}

__global__ void Fn_apply_op_Kernel(const scalar_t* a, scalar_t* out, size_t size, EwiseFnOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == EwiseFnOp::LOG){
      out[gid] = log(a[gid]);
    } else if(op == EwiseFnOp::EXP){
      out[gid] = exp(a[gid]);
    } else if(op == EwiseFnOp::TANH){
      out[gid] = tanh(a[gid]);
    }
  }
}

__global__ void Scalar_apply_op_Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, ScalarOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    if(op == ScalarOp::MUL){
      out[gid] = a[gid] * val;
    } else if(op == ScalarOp::DIV){
      out[gid] = a[gid] / val;
    } else if(op == ScalarOp::MAXIMUM){
      out[gid] = a[gid] > val ? a[gid] : val;
    } else if(op == ScalarOp::EQ){
      out[gid] = a[gid] == val;
    } else if(op == ScalarOp::GE){
      out[gid] = a[gid] >= val;
    } else if(op == ScalarOp::POWER){
      out[gid] = powf(a[gid], val);
    }
  }
}

/// BEGIN YOUR SOLUTION
// Ewise operation
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Ewise_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EwiseOp::MUL);
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::MUL);
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Ewise_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EwiseOp::DIV);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::DIV);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Ewise_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EwiseOp::MAXIMUM);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::MAXIMUM);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Ewise_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EwiseOp::EQ);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::EQ);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Ewise_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EwiseOp::GE);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::GE);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Scalar_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ScalarOp::POWER);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Fn_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, EwiseFnOp::LOG);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Fn_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, EwiseFnOp::EXP);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  Fn_apply_op_Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, EwiseFnOp::TANH);
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
enum class MATMUL_ALGO{
  naive
};

__global__ void MatmulKernel_Naive(
  const scalar_t* a, const scalar_t* b, scalar_t* out, 
  size_t M, size_t K, size_t N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t rowid = gid / N;
  size_t colid = gid % N;
  if (gid < M*N) {
    scalar_t acc = 0;
    for(size_t i = 0; i < K; i++){
      acc += a[rowid * K + i] * b[i * N + colid];
    }
    out[gid] = acc;
  }
}

void Matmul_algorithm(
  const CudaArray& a, const CudaArray& b, CudaArray* out, 
  uint32_t M, uint32_t K, uint32_t N, 
  MATMUL_ALGO algo){

    if(algo == MATMUL_ALGO::naive){
      dim3 block = dim3(BASE_THREAD_NUM, 1, 1);
      dim3 grid = dim3((M*N+BASE_THREAD_NUM-1)/BASE_THREAD_NUM, 1, 1);
      MatmulKernel_Naive<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, K, N);
    }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t K,
            uint32_t N) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x k
   *   b: comapct 2D array of size k x n
   *   out: compact 2D array of size m x n to write the output to
   *   M: rows of a / out
   *   K: columns of a / rows of b
   *   N: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  Matmul_algorithm(a, b, out, M, K, N, MATMUL_ALGO::naive);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t pos_start = gid * reduce_size;
    scalar_t maxv = a[pos_start];
    for(int i = 0; i < reduce_size; i++){
      maxv = max(maxv, a[pos_start + i]);
    }
    out[gid] = maxv;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t pos_start = gid * reduce_size;
    scalar_t sumv = 0;
    for(int i = 0; i < reduce_size; i++){
      sumv += a[pos_start + i];
    }
    out[gid] = sumv;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
