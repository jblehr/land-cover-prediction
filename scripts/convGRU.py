# Code based on https://github.com/happyjin/ConvGRU-pytorch/blob/9e932822e11db19b78e3761cb71018850a1247ff/convGRU.py

import os
import torch
from torch import nn
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, input_channels, hidden_channels, kernel_size, bias, cuda_=False):
        """
        Initialize the ConvLSTM cell
        
        Inputs:
            input_dim: (int, int) Height/width of spatial layer (x_dim, x_dim).
            input_channels: (int) Number of channels of spatial layer
                (should be {R, G, B, NIR} = 4).
            param hidden_dim: (int) Number of channels of hidden state.
            kernel_size: (int, int) Size of the convolutional kernel.
            bias: (bool) binary to add a trainable bias term to output.
            cuda: (bool) binary on whether to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.x_dim, self.y_dim = input_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_channels = hidden_channels
        self.bias = bias

        if cuda_:
            self.dtype = torch.cuda.FloatTensor # computation in GPU
        else:
            self.dtype = torch.FloatTensor

        self.conv_gates = nn.Conv2d(in_channels=input_channels + hidden_channels,
                                    out_channels=2*self.hidden_channels,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_channels + hidden_channels,
                                  out_channels=self.hidden_channels, # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(
                    batch_size,
                    self.hidden_channels,
                    self.x_dim,
                    self.y_dim)).type(self.dtype))

    def forward(self, input_current, hidden_current):
        """
            One cell to process one timestep of the input sequence.
            Inputs:
                input_current: 4-d tensor of dims
                    (batch, input_channels, height, width)
                hidden current: 4-d tensor of dims
                    (batch, hidden_channels, height, width)
                current hidden and cell states respectively
            Outputs:
                hidden_next: 4-d tensor as above of next hidden state
        """
        combined = torch.cat([input_current, hidden_current], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        input_state_combined = \
            torch.cat([input_current, reset_gate*hidden_current], dim=1)
        conv_candidate = self.conv_can(input_state_combined)
        candidate = torch.tanh(conv_candidate)

        hidden_next = (1 - update_gate) * hidden_current + update_gate * candidate
        return hidden_next


class ConvGRU(nn.Module):
    def __init__(self, input_dim, input_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, cuda_ = False):
        """
        :param input_dim: (int, int)
            Height and width of input tensor as (height, width).
        :param input_channels: int e.g. 256
            Number of channels of input tensor.
        :param hidden_channels: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels  = self._extend_for_multilayer(hidden_channels, num_layers)

        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.x_dim, self.y_dim = input_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            current_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(ConvGRUCell(input_dim=(self.x_dim, self.y_dim),
                                         input_channels=current_input_channels,
                                         hidden_channels=self.hidden_channels[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         cuda_=cuda_))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_current, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_current = input_current.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_current.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_current.size(1)
        cur_layer_input = input_current

        for layer_idx in range(self.num_layers):
            hidden_current = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next
                # hidden and cell state through ConvGRUCell forward function
                # index dims should be (batch, time, input_channels, x_dim, y_dim)
                hidden_next = self.cell_list[layer_idx](
                    input_current=cur_layer_input[:, t, :, :, :], 
                    hidden_current=hidden_current
                )
                output_inner.append(hidden_next)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([hidden_next])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    cuda_ = torch.cuda.is_available()

    x_dim = y_dim = 128
    input_channels = 4
    hidden_channels = [2,4]
    kernel_size = (3,3) # kernel size for two stacked hidden layer
    num_layers = 2 # number of stacked hidden layer
    model = ConvGRU(input_dim=(x_dim, y_dim),
                    input_channels=input_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    cuda_=cuda_)

    batch_size = 1
    time_steps = 3
    input_current = torch.rand(batch_size, time_steps, input_channels, x_dim, y_dim)  # (b,t,c,h,w)
    layer_output_list, last_state_list = model(input_current)
    pass