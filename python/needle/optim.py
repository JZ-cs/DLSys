"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, p in enumerate(self.params):
            if(p.grad is not None):
                # first
                g = p.grad.detach() + p.detach() * self.weight_decay
                if(p not in self.u.keys()):
                    self.u[p] = 0
                gu = g * (1-self.momentum) + self.u[p] * self.momentum
                self.u[p] = gu
                
                delta = p - gu*self.lr
                self.params[i].cached_data = delta.realize_cached_data()
            else:
                raise RuntimeError('grad of param is None !')
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        # import pdb; pdb.set_trace()
        for i, p in enumerate(self.params):
            if(p.grad is not None):
                # first
                g = p.grad.detach() + self.weight_decay * p.detach()
                if(p not in self.m.keys()):
                    self.m[p] = 0
                gm = (1-self.beta1) * g + self.m[p] * self.beta1
                self.m[p] = gm

                if(p not in self.v.keys()):
                    self.v[p] = 0
                gv = (1-self.beta2) * (g * g) + self.v[p] * self.beta2
                self.v[p] = gv

                # import pdb;pdb.set_trace()
                gm_bc = gm / (1 - self.beta1**self.t)
                gv_bc = gv / (1 - self.beta2**self.t)

                delta = p - self.lr * gm_bc / (gv_bc**0.5 + self.eps)
                self.params[i].cached_data = delta.realize_cached_data()
            else:
                print(p)
                raise NotImplementedError()
        ### END YOUR SOLUTION
