"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math

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
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype))
        if(bias == True):
            self.bias = Parameter(init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype).transpose())
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

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.sigmoid(x)
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
        # import pdb; pdb.set_trace()
        bsz = logits.shape[0]
        y_onehot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        y_onehot_neg = ops.mul_scalar(y_onehot, -1)
        log_softs = logits - ops.broadcast_to(ops.reshape(ops.logsumexp(logits, axes=1), (bsz, 1)), logits.shape)
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


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


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
            mask = init.randb(*x.shape, p=self.p, device=x.device, dtype=x.dtype).astype('float32')
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

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
                                    self.in_channels*kernel_size*kernel_size, 
                                    self.out_channels*kernel_size*kernel_size, 
                                    shape=(kernel_size, kernel_size, self.in_channels, self.out_channels), 
                                    device=device,
                                    dtype=dtype))
        
        if(bias == True):
            self.bias = Parameter(init.rand(self.out_channels, low=0.0, high=1.0/(self.in_channels*kernel_size**2)**0.5, device=device,
                                    dtype=dtype))
        else:
            self.bias = None
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.permute(x, (0, 2, 3, 1))
        n, h_in, w_in, c_in = x.shape
        h_out = (h_in-self.kernel_size)//self.stride + 1; w_out = (w_in-self.kernel_size)//self.stride + 1
        padding = 0
        if(h_out != h_in or w_out != w_in):
            # (x_in + k - kernel_size) // stride + 1 = x_out -> k = (x_out-1)*stride + kernel_size - x_in
            # hp_size = (h_out-1)*self.stride + self.kernel_size - h_in
            # wp_size = (w_out-1)*self.stride + self.kernel_size - w_in
            # padding_h = hp_size//2
            # padding_w = wp_size//2
            # assert(padding_h==padding_w)
            padding = self.kernel_size//2
            # re-calculate h_out and w_out
            h_out = (h_in+padding*2-self.kernel_size)//self.stride + 1; w_out = (w_in+padding*2-self.kernel_size)//self.stride + 1
        out = ops.conv(x, self.weight, stride=self.stride, padding=padding)
        if(self.bias is not None):
            bias_reshaped = ops.reshape(self.bias, (1,1,1,self.out_channels))
            bias = ops.broadcast_to(bias_reshaped, (n, h_out, w_out, self.out_channels))
            # import pdb; pdb.set_trace()
            out = out + bias
        return ops.permute(out, (0, 3, 1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        if(bias == True):
            self.bias_ih = Parameter(init.rand(hidden_size, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        if(nonlinearity=='relu'):
            self.non_linear = ReLU()
        else:
            self.non_linear = Tanh()
        
        ### END YOUR SOLUTION
    
    def reshape_broadcast(self, bias:Parameter, shape:tuple):
        if(bias.shape == shape):
            return bias
        assert bias.shape[-1] == shape[-1], f'Can not reshape, bias={bias.shape}, to {shape}'
        _shape = [1 for _ in range(len(shape))]
        _shape[-1] = bias.shape[-1]
        z1 = ops.reshape(bias, tuple(_shape))
        z2 = ops.broadcast_to(z1, shape)
        return z2

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # import pdb; pdb.set_trace()
        xw = ops.matmul(X, self.W_ih)
        # import pdb; pdb.set_trace()
        if(self.bias_ih is not None):
            xw = xw + self.reshape_broadcast(self.bias_ih, xw.shape)
        
        if(h is None):
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        hw = ops.matmul(h, self.W_hh)
        if(self.bias_hh is not None):
            hw = hw + self.reshape_broadcast(self.bias_hh, hw.shape)
        return self.non_linear(xw + hw)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=self.nonlinearity,
                                    device=device, dtype=dtype) for _ in range(self.num_layers)]
        
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if(h0 is None):
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)
        # import pdb; pdb.set_trace()
        h0_list = ops.split(h0, axis=0)
        x_list = ops.split(X, axis=0)

        hn = []
        hs = []
        for j in range(len(x_list)):
            if(j == 0):
                hs.append(self.rnn_cells[0](x_list[j], h=h0_list[0]))
            else:
                hs.append(self.rnn_cells[0](x_list[j], h=hs[j-1]))
        hn.append(hs[-1])
        for i in range(1, self.num_layers):
            for j in range(len(x_list)):
                if(j == 0):
                    hs[j] = self.rnn_cells[i](hs[j], h=h0_list[i])
                else:
                    hs[j] = self.rnn_cells[i](hs[j], h=hs[j-1])
            hn.append(hs[-1])
        output = ops.stack(hs, axis=0)
        h_n = ops.stack(hn, axis=0)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size*4, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(input_size, hidden_size*4, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        if(bias == True):
            self.bias_ih = Parameter(init.rand(hidden_size*4, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size*4, low=-math.sqrt(k), high=math.sqrt(k), device=device, dtype=dtype))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def reshape_broadcast(self, bias:Parameter, shape:tuple):
        if(bias.shape == shape):
            return bias
        assert bias.shape[-1] == shape[-1], f'Can not reshape, bias={bias.shape}, to {shape}'
        _shape = [1 for _ in range(len(shape))]
        _shape[-1] = bias.shape[-1]
        z1 = ops.reshape(bias, tuple(_shape))
        z2 = ops.broadcast_to(z1, shape)
        return z2

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if(h is None):
            h0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        bs, _ = X.shape
        xw = ops.matmul(X, self.W_ih)
        # import pdb; pdb.set_trace()
        if(self.bias_ih is not None):
            xw = xw + self.reshape_broadcast(self.bias_ih, xw.shape)
        
        hw = ops.matmul(h0, self.W_hh)
        if(self.bias_hh is not None):
            hw = hw + self.reshape_broadcast(self.bias_hh, hw.shape)
        
        stacked_res = xw + hw
        stacked_res = ops.reshape(stacked_res, (bs, 4, self.hidden_size))
        hs = ops.split(stacked_res, axis=1)
        i_ = Sigmoid()(hs[0])
        f_ = Sigmoid()(hs[1])
        g_ = Tanh()(hs[2])
        o_ = Sigmoid()(hs[3])
        c_out = f_ * c0 + i_ * g_
        h_out = o_ * Tanh()(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias,
                                    device=device, dtype=dtype) for _ in range(self.num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if(h is None):
            h = (init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype),
                 init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype))
        
        h0_list = ops.split(h[0], axis=0)
        c0_list = ops.split(h[1], axis=0)
        x_list = ops.split(X, axis=0)

        hs = []
        cs = []
        h_finals = []
        c_finals = []
        layer_id = 0
        for j in range(len(x_list)):
            if(j == 0):
                res = self.lstm_cells[layer_id](x_list[j], h=(h0_list[layer_id], c0_list[layer_id]))
                hs.append(res[0])
                cs.append(res[1])
            else:
                res = self.lstm_cells[layer_id](x_list[j], h=(hs[j-1], cs[j-1]))
                hs.append(res[0])
                cs.append(res[1])
        h_finals.append(hs[-1])
        c_finals.append(cs[-1])

        for layer_id in range(1, self.num_layers):
            for j in range(len(x_list)):
                if(j == 0):
                    hs[j], cs[j] = self.lstm_cells[layer_id](hs[j], h=(h0_list[layer_id], c0_list[layer_id]))
                else:
                    hs[j], cs[j] = self.lstm_cells[layer_id](hs[j], h=(hs[j-1], cs[j-1]))
            h_finals.append(hs[-1])
            c_finals.append(cs[-1])
        output = ops.stack(hs, axis=0)
        h_n = ops.stack(h_finals, axis=0)
        c_n = ops.stack(c_finals, axis=0)
        return output,(h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
