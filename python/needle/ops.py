"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        try:
            assert lhs.device==rhs.device and out_grad.device==lhs.device, f'lhs:{lhs.device}, rhs:{rhs.device}, out_grad:{out_grad.device}'
        except Exception as e:
            import pdb; pdb.set_trace()
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        res = out_grad * mul_scalar(power_scalar(x, self.scalar-1), self.scalar)
        return res
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * lhs / (-rhs**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.transpose(a, self.axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Permute(TensorOp):
    def __init__(self, axes: tuple =None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert(len(self.axes) == len(a.shape))
        self.backward_map = [None for _ in range(len(self.axes))]
        for i, ax in enumerate(self.axes):
            self.backward_map[ax] = i
        return array_api.permute(a, self.axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return permute(out_grad, tuple(self.backward_map))
        ### END YOUR SOLUTION


def permute(a, axes=None):
    return Permute(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        old_shape = node.inputs[0].shape
        return reshape(out_grad, old_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a_shape_rev = list(node.inputs[0].shape)
        a_shape_rev.reverse()
        o_shape_rev = list(self.shape)
        o_shape_rev.reverse()
        sum_axes = []
        for i, shp in enumerate(a_shape_rev):
            if(shp != o_shape_rev[i]):
                sum_axes.append(i)
        a_shape_len = len(a_shape_rev)
        o_shape_len = len(o_shape_rev)
        for i in range(a_shape_len, o_shape_len):
            sum_axes.append(i)
        for i in range(len(sum_axes)):
            sum_axes[i] = o_shape_len - 1 - sum_axes[i]
        # import pdb; pdb.set_trace()
        res = summation(out_grad, axes=tuple(sum_axes))
        return reshape(res, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        old_shape= list(node.inputs[0].shape)
        if(self.axes is None):
            # sum all
            for i in range(len(old_shape)):
                old_shape[i] = 1
        elif(isinstance(self.axes, int)):
            old_shape[self.axes] = 1
        else:
            for ax in self.axes:
                old_shape[ax] = 1
        lhs = node.inputs[0]
        return broadcast_to(reshape(out_grad, tuple(old_shape)), lhs.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        l_shape = lhs.shape
        r_shape = rhs.shape
        ml = matmul(out_grad, transpose(rhs))
        if(l_shape != ml.shape):
            sum_axes = tuple([i for i in range(len(ml.shape) - len(l_shape))])
            ml =  summation(ml, axes=sum_axes)
        mr = matmul(transpose(lhs), out_grad)
        if(r_shape != ml.shape):
            sum_axes = tuple([i for i in range(len(mr.shape) - len(r_shape))])
            mr =  summation(mr, axes=sum_axes)
        return ml, mr
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a*(-1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * -1
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * power_scalar(node.inputs[0], -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node.inputs[0]) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        mask = a > 0
        # self.save_for_backward = Tensor(mask,device=a.device)
        return a * mask
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # import pdb; pdb.set_trace()
        return out_grad * Tensor(node.inputs[0].cached_data >= 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Z_numpy = Z.numpy()
        # max_Z_kd = array_api.broadcast_to(array_api.max(Z, self.axes, keepdims=True), Z.shape).numpy()
        # # max_Z_kd = numpy.broadcast_to(numpy.max(Z_numpy, self.axes, keepdims=True), Z.shape)
        # e_Z = numpy.exp(Z_numpy-max_Z_kd)
        # max_Z =  numpy.amax(Z_numpy, self.axes)
        # sum_Z_submax = numpy.sum(e_Z, axis=self.axes)
        # self.save_for_backward = (Tensor(NDArray(sum_Z_submax, device=Z.device), device=Z.device), Tensor(NDArray(max_Z, device=Z.device), device=Z.device))
        # res_numpy = numpy.log(sum_Z_submax) + max_Z
        # return NDArray(res_numpy, device=Z.device)

        max_Z_kd = array_api.broadcast_to(array_api.max(Z, self.axes, keepdims=True), Z.shape)
        e_Z = array_api.exp(Z - max_Z_kd)
        max_Z =  array_api.max(Z, self.axes)
        sum_Z_submax = array_api.sum(e_Z, axis=self.axes)
        self.save_for_backward = (Tensor(sum_Z_submax, device=Z.device), Tensor(max_Z, device=Z.device))
        res = array_api.log(sum_Z_submax) + max_Z
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        sum_Z_submax, max_Z = self.save_for_backward
        Z = node.inputs[0]
        # dZ = SUM_RESHAPE(out_grad / (sum_Z_submax * e^(max_Z))) * e^Z
        old_shape= list(Z.shape)
        if(self.axes is None):
            # sum all
            for i in range(len(old_shape)):
                old_shape[i] = 1
        elif(isinstance(self.axes, tuple)):
            for ax in self.axes:
                old_shape[ax] = 1
        else:
            assert isinstance(self.axes, int)
            old_shape[self.axes] = 1
        # import pdb; pdb.set_trace()
        out_grad_broadcasted = broadcast_to(reshape(out_grad, tuple(old_shape)), Z.shape)
        sum_Z_submax_reshaped = reshape(sum_Z_submax, tuple(old_shape))
        max_Z_reshaped = reshape(max_Z, tuple(old_shape))
        return out_grad_broadcasted / broadcast_to(sum_Z_submax_reshaped, Z.shape) * exp(Z - broadcast_to(max_Z_reshaped, Z.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        return out_grad * (1-tanh(inp)*tanh(inp))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

class Sigmoid(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.power((1 + array_api.exp(a*(-1.0))), -1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        return out_grad * (sigmoid(inp) * (1 - sigmoid(inp)))
        ### END YOUR SOLUTION


def sigmoid(a):
    return Sigmoid()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        return array_api.stack(args, axis=self.axis)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grads = split(out_grad, axis=self.axis)
        # import pdb; pdb.set_trace()
        return grads
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        return array_api.split(A, axis=self.axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.dilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # import pdb; pdb.set_trace()
        return array_api.undilate(a, self.axes, self.dilation)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if(self.padding is not None and self.padding > 0):
            _A = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        else:
            _A = A
        unfold_A, h_out, w_out = array_api.im2col(_A, (B.shape[0], B.shape[1]), conv_stride=self.stride) # n*h_out*w_out, kh*kw*c_in
        reshaped_kernel = array_api.kernel_trans(B) # kh*kw*c_in, c_out

        self.h_in = self.padding*2 + A.shape[1]
        self.w_in = self.padding*2 + A.shape[2]
        self.unfold_x = unfold_A
        self.reshaped_kernel = reshaped_kernel
        out = array_api.matmul(unfold_A, reshaped_kernel) # n*h_out*w_out, c_out
        # import pdb; pdb.set_trace()
        res = array_api.reshape(out, (A.shape[0], h_out, w_out, B.shape[-1]))
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]; w = node.inputs[1]
        # roughly, grad_x = out_grad @ w.T, grad_w = x.T @ out_grad
        # x = (n, h, w, c_in)    w = (kh, kw, c_in, c_out)  out_grad = (n, h_out, w_out, c_out)

        n, h_out, w_out, c_out = out_grad.shape
        kh, kw, c_in, c_out = w.shape
        _out_grad = array_api.reshape(out_grad.cached_data.compact(), (n*h_out*w_out, c_out))
        assert len(self.unfold_x.shape) == 2
        assert len(self.reshaped_kernel.shape) == 2
        xT = array_api.transpose(self.unfold_x, (0, 1)) # n*h_out*w_out, kh*kw*c_in -> kh*kw*c_in, n*h_out*w_out
        wT = array_api.transpose(self.reshaped_kernel, (0, 1))# kh*kw*c_in,c_out -> c_out, kh*kw*c_in
        grad_reshaped_w = array_api.matmul(xT, _out_grad) # ->  kh*kw*c_in, c_out
        grad_unfold_x = array_api.matmul(_out_grad, wT)
        grad_w = array_api.reshape(grad_reshaped_w, w.shape).compact()
        grad_x = array_api.col2img(grad_unfold_x, 
                                            (kh, kw), 
                                            self.h_in, self.w_in,
                                            out_grad.shape[1], out_grad.shape[2],
                                            conv_stride=self.stride)
        if(self.padding is not None and self.padding > 0):
            grad_x = Tensor(array_api.unpad(grad_x, ((0,0),(self.padding, self.padding), (self.padding, self.padding), (0,0))))

        return Tensor(grad_x), Tensor(grad_w)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



