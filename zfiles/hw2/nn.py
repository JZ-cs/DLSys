"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features))
        if(bias == True):
            self.bias = Parameter(init.kaiming_uniform(self.out_features, 1).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if(self.bias is None):
            return ops.matmul(X, self.weight)
        else:
            out_shape = X.shape[:-1] + (self.out_features,)
            b_b = ops.broadcast_to(self.bias, out_shape)
            return ops.matmul(X, self.weight) + b_b
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if(len(X.shape) == 1):
            return X
        fla_dim = 1
        for dim in X.shape[1:]:
            fla_dim *= dim
        return ops.reshape(X, (X.shape[0], fla_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        bsz = logits.shape[0]
        y_onehot = init.one_hot(logits.shape[1], y)
        y_onehot_neg = ops.mul_scalar(y_onehot, -1)
        log_softs = logits - ops.broadcast_to(ops.reshape(ops.logsumexp(logits, axes=(1)), (bsz, 1)), logits.shape)
        log_mul_y = ops.multiply(log_softs, y_onehot_neg)
        res = ops.divide_scalar(ops.summation(log_mul_y), (logits.shape[0]))
        return res
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))

        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def compute_estimate(self, x_old=None, x_new=None):
        return ops.mul_scalar(x_old.detach(), 1-self.momentum) + ops.mul_scalar(x_new.detach(), self.momentum)

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if(self.training == False):
            # eval()
            bsz = x.shape[0]
            w_b = ops.broadcast_to(self.weight.detach(), x.shape)
            b_b = ops.broadcast_to(self.bias.detach(), x.shape)
            Ex_r = ops.reshape(self.running_mean.detach(), (1, self.dim))
            Ex_b = ops.broadcast_to(Ex_r, x.shape)
            Var_r = ops.reshape(self.running_var.detach(), (1, self.dim))
            Var_b = ops.broadcast_to(Var_r, x.shape)
            z = (x - Ex_b) / ops.power_scalar(ops.add_scalar(Var_b, self.eps), 0.5)
            return w_b * z + b_b

        bsz = x.shape[0]
        w_b = ops.broadcast_to(self.weight, x.shape)
        b_b = ops.broadcast_to(self.bias, x.shape)
        Ex = ops.divide_scalar(ops.summation(x, axes=(0)), bsz)
        Ex_r = ops.reshape(Ex, (1, self.dim))
        Ex_b = ops.broadcast_to(Ex_r, x.shape)
        Var = ops.divide_scalar(ops.summation(ops.power_scalar(x - Ex_b, 2), axes=(0)), bsz)
        Var_r = ops.reshape(Var, (1, self.dim))
        Var_b = ops.broadcast_to(Var_r, x.shape)

        # running update
        self.running_mean = self.compute_estimate(x_old=self.running_mean, x_new=Ex)
        self.running_var = self.compute_estimate(x_old=self.running_var, x_new=Var)

        z = (x - Ex_b) / ops.power_scalar(ops.add_scalar(Var_b, self.eps), 0.5)
        return w_b * z + b_b
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        bsz, dim = x.shape
        Ex = ops.divide_scalar(ops.summation(x, axes=(1)), dim)
        Ex_b = ops.broadcast_to(ops.reshape(Ex, (bsz,1)), x.shape)
        Var = ops.divide_scalar(ops.summation(ops.power_scalar(x - Ex_b, 2), axes=(1)), dim)
        Var_b = ops.broadcast_to(ops.reshape(Var, (bsz,1)), x.shape)
        z = (x-Ex_b) / ops.power_scalar(ops.add_scalar(Var_b, self.eps), 0.5)
        w_b = ops.broadcast_to(self.weight, x.shape)
        b_b = ops.broadcast_to(self.bias, x.shape)
        return w_b * z + b_b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if(self.training == False or self.p == 0):
            return Identity()(x)
        else:
            mask = init.randb(*x.shape, p=self.p).numpy().astype('float32')
            return ops.divide_scalar(ops.multiply(x, Tensor(mask)), 1-self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



