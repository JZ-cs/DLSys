import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
# import python.needle as ndl
# import python.needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def make_convbn(a,b,k,s, device=None, dtype="float32"):
    return nn.Sequential(
        nn.Conv(a, b, kernel_size=k, stride=s, device=device, dtype=dtype),
        nn.BatchNorm2d(dim=b, device=device, dtype=dtype),
        nn.ReLU()
    )

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convbn_0 = make_convbn(3,16,7,4,device=device, dtype=dtype)
        self.convbn_1 = make_convbn(16,32,3,2,device=device, dtype=dtype)
        self.residual_0 = nn.Residual(
                        nn.Sequential(
                            make_convbn(32,32,3,1,device=device, dtype=dtype),
                            make_convbn(32,32,3,1,device=device, dtype=dtype)
                        )
        )
        self.convbn_2 = make_convbn(32,64,3,2,device=device, dtype=dtype)
        self.convbn_3 = make_convbn(64,128,3,2,device=device, dtype=dtype)
        self.residual_1 = nn.Residual(
                        nn.Sequential(
                            make_convbn(128,128,3,1,device=device, dtype=dtype),
                            make_convbn(128,128,3,1,device=device, dtype=dtype)
                        )
        )
        self.flat = nn.Flatten()
        self.linear_0 = nn.Linear(128,128,device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(128,10,device=device, dtype=dtype)
        # self.mods = [self.convbn_0, self.convbn_1, self.residual_0, self.convbn_2, self.convbn_3, self.residual_1,
        #                 self.flat, self.linear_0, self.relu, self.linear_1]
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = self.convbn_0(x)
        x = self.convbn_1(x)
        x = self.residual_0(x)
        x = self.convbn_2(x)
        x = self.convbn_3(x)
        x = self.residual_1(x)
        x = self.flat(x)
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        return x
        # for mod in self.mods:
        #     import pdb;pdb.set_trace()
        #     x = mod(x)
        # return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)