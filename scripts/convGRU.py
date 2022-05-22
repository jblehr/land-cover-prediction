# Code based on https://github.com/happyjin/ConvGRU-pytorch/blob/9e932822e11db19b78e3761cb71018850a1247ff/convGRU.py

from pydoc import cli
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import evaluation
from tqdm import tqdm
import dataloaders
import numpy as np
from aim import Run
import os

class ConvGRUCell(nn.Module):
    def __init__(
        self, input_dim, input_channels, hidden_channels, kernel_size, bias, conv_padding_mode, cuda_=False
    ):
        """
        Initialize a single ConvGRU cell

        Inputs:
            input_dim: (int, int) Height/width of spatial layer (x_dim, x_dim).
            input_channels: (int) Number of channels of spatial layer
                (should be {R, G, B, NIR} = 4).
            hidden_channels: (int) Number of channels of hidden state.
            kernel_size: (int, int) Size of the convolutional kernel.
            bias: (bool) binary to add a trainable bias term to output.
            cuda: (bool) binary on whether to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.x_dim, self.y_dim = input_dim
        # self.padding = [kernel_size[0] // 2 + 1, kernel_size[1] // 2]
        self.padding='same'
        self.padding_mode = conv_padding_mode
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.cuda_ = cuda_

        if cuda_:
            self.dtype = torch.cuda.FloatTensor  # computation in GPU
        else:
            self.dtype = torch.FloatTensor

        self.conv_gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2*self.hidden_channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            padding_mode=conv_padding_mode,
            bias=self.bias,
        )   

        if cuda_:
            self.conv_gates.to('cuda')

        self.conv_can = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=self.hidden_channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            padding_mode=conv_padding_mode,
            bias=self.bias,
        )
        if cuda_:
            self.conv_gates.to('cuda')
            self.conv_can.to('cuda')


    def init_hidden(self, batch_size):
        return Variable(
            torch.zeros(batch_size, self.hidden_channels, self.x_dim, self.y_dim)
        ).type(self.dtype)

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
        if self.cuda_:
            input_current = input_current.to('cuda')

        combined = torch.cat([input_current, hidden_current], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        input_state_combined = torch.cat(
            [input_current, reset_gate * hidden_current], dim=1
        )
        conv_candidate = self.conv_can(input_state_combined)
        candidate = torch.tanh(conv_candidate)

        hidden_next = (1 - update_gate) * hidden_current + update_gate * candidate
        return hidden_next


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        input_channels,
        hidden_channels,
        output_dim,
        kernel_size,
        num_layers,
        conv_padding_mode,
        batch_first=False,
        bias=True,
        cuda_=False,
    ):
        """
        A GRU with convolutions between current input channels and hidden
        state channels which persist through time, allowing for spatiotemporal
        nonlinearities to help us predict land use classifications (or whatever else).

        Inputs:
            input_dim: (int, int) Height/width of spatial layer (x_dim, x_dim).
            input_channels: (int) Number of channels of spatial layer
                (should be {R, G, B, NIR} = 4).
            hidden_channels: (int) Number of channels of hidden state.
            kernel_size: (int, int) Size of the convolutional kernel.
            num_layers: (int) Number of ConvLSTM layers (for spatial dimension)
            cuda_: (bool) Whether to use gpu acceleration
            batch_first: (bool) if the first position of array is batch or not
            bias: (bool) binary to add a trainable bias term to output.
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)

        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.x_dim, self.y_dim = input_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.cuda_ = cuda_
        cell_list = []

        for i in range(0, self.num_layers):
            current_input_channels = (
                input_channels if i == 0 else hidden_channels[i - 1]
            )
            cell_list.append(
                ConvGRUCell(
                    input_dim=(self.x_dim, self.y_dim),
                    input_channels=current_input_channels,
                    hidden_channels=self.hidden_channels[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    conv_padding_mode=conv_padding_mode,
                    cuda_=cuda_
                )
            )

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        self.lin_out = nn.Linear(
            in_features=self.hidden_channels[-1], out_features=output_dim
        )

    def forward(self, input_current):
        """
        Fit the ConvGRU on the current input data and past states

        Inputs:
            input_tensor: tensor with dims (batch, time, input_channels, x_dim, y_dim)
            hidden_state: tensor with dims (batch, time, hidden_channels, x_dim, y_dim)

        Outputs:
            layer_output_list, last_state_list
        """
        assert self.batch_first

        layer_output_list = []
        last_state_list = []
        hidden_state = self._init_hidden(batch_size=input_current.size(0))
        seq_len = input_current.size(1)
        cur_layer_input = input_current

        for layer_idx in range(self.num_layers):
            hidden_current = hidden_state[layer_idx]
            output_inner = []
            for time_idx in range(seq_len):
                # input current hidden and cell state then compute the next
                # hidden and cell state through ConvGRUCell forward function
                # index dims should be (batch, time, input_channels, x_dim, y_dim)
                hidden_next = self.cell_list[layer_idx](
                    input_current=cur_layer_input[:, time_idx, :, :, :],
                    hidden_current=hidden_current,
                )
                output_inner.append(hidden_next)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([hidden_next])

        # take last timestep for classification and make the hidden layer the last
        # dim for compatability with nn.Linear to convert our 7 out classes
        class_out = layer_output[:, -1, :, :, :].permute(0, 2, 3, 1)
        class_val = self.lin_out(class_out)

        return class_val

    def fit(self, train_loader, test_loader, optim, lr, momentum, bptt_len=3, epochs = 50, max_norm=False, track_run=False):

        if optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        self.train()
        criterion=torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            losses=[]
            for idx, (batch_x, batch_y) in enumerate(train_loader):

                for timestep in range(batch_x.shape[1] - 1):

                    min_step = max(0, timestep-bptt_len)

                    # For each BPTT step, get all timesteps (t-bptt_len):t, model them
                    inputs = batch_x[:,min_step:timestep+1,:,:]
                    outputs = self(inputs)

                    # Then, choose next timestep target to predict
                    targets = batch_y[:,timestep+1,:,:]

                    # For compatability with CrossEntropyLoss, reshape to ignore
                    # spatial dims and batches for loss - doesn't matter in this
                    # case anyways as we just want pixels to line up properly
                    flat_dim = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]

                    outputs_flat = outputs.reshape(flat_dim, outputs.shape[3])
                    targets_flat = targets.reshape(flat_dim)
                    if self.cuda_:
                        targets_flat = targets_flat.to('cuda')


                    loss = criterion(outputs_flat, targets_flat)
                    if max_norm:
                        nn.utils.clip_grad_norm_(
                            self.parameters(),
                            max_norm=max_norm,
                            norm_type=2
                        )

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(float(loss))

            mean_loss = np.mean(losses)
            train_acc = evaluation.get_accuracy(self, train_loader)
            test_acc = evaluation.get_accuracy(self, test_loader)

            if track_run:
                track_run.track(mean_loss, name='mean_loss', epoch=epoch)
                track_run.track(train_acc, name='train_accuracy', epoch=epoch)
                track_run.track(test_acc, name='test_accuracy', epoch=epoch)

            #TODO: The following works, but many test areas that don't have
            # any changes at all... that might be right, need to do some EDA

            # train_changedAcc = evaluation.get_accuracy(self, train_loader, changed_only=True)
            # track_run.track(train_changedAcc, name='train_changedAccuracy', epoch=epoch)

            # test_changedAcc = evaluation.get_accuracy(self, test_loader, changed_only=True)
            # track_run.track(test_changedAcc, name='test_changedAccuracy', epoch=epoch)

            print(f"After epoch {epoch}:\n  train acc: {train_acc:.3f}, test acc: {test_acc:.3f}")

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":

    # detect if CUDA is available or not
    cuda_ = torch.cuda.is_available()
    if cuda_:
        device = 'cuda'
    else:
        device = 'cpu'

    input_channels = 4
    # input_channels = 7 # for labs_as_features
    hidden_channels = [16, 32, 64]
    n_output_classes = 7
    kernel_size = (3,3)  # kernel size for stacked hidden layer
    num_layers = 3  # number of stacked hidden layers
    n_steps = 2 # only take 2 steps
    batch_size = 3
    train_pct = .8
    cell_width = 128
    lr = .1
    momentum = .001
    bias=True
    optim='adam'
    epochs=10
    conv_padding_mode = 'replicate'
    experiment_desc = 'more AOIS (7) - normalized'
    blur=False
    clip_max_norm = .10

    if blur:
        guassian_sigma = 3.0
        guassian_kernel = int(guassian_sigma * 6) + 1 #from https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
        transform = torchvision.transforms.GaussianBlur(guassian_kernel, guassian_sigma)
    else:
        guassian_sigma = guassian_kernel = transform = None

    poi_list = os.listdir('data/processed/npz/planet')

    STData = dataloaders.SpatiotemporalDataset(
        "data/processed/npz",
        dims = (1024, 1024),
        poi_list=poi_list,
        n_steps=n_steps,
        cell_width=cell_width,
        labs_as_features=False,
        transform=transform
    )

    x_dim = y_dim = STData.cell_width
    n_train = int(len(STData) * train_pct)
    n_test = len(STData) - n_train

    train_dataset, test_dataset = \
        torch.utils.data.random_split(STData, [n_train, n_test])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=batch_size,
        batch_size=None, #batches determined by cell width
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=batch_size,
        batch_size=None, #batches determined by cell width
        shuffle=True
    )
    
    convGRU = ConvGRU(
        input_dim=(x_dim,y_dim),
        num_layers=num_layers,
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_dim=n_output_classes,
        kernel_size=kernel_size,
        batch_first=True,
        conv_padding_mode=conv_padding_mode,
        bias=bias,
        cuda_=cuda_
    )
    if cuda_:
        convGRU = convGRU.to('cuda')

    track_run = Run(experiment=experiment_desc)

    track_run['hparams'] = {
            'lr': lr,
            'batch_size': batch_size,
            'momentum' : momentum,
            'hidden_channels':hidden_channels,
            'conv_kernel_size':kernel_size,
            'hidden_layers':num_layers,
            'bias':bias,
            'epochs':epochs,
            'optim':optim,
            'cell_width':cell_width,
            'conv_padding_mode':conv_padding_mode,
            'guassian_kernel':guassian_kernel,
            'guassian_sigma':guassian_sigma,
            'clip_max_norm':clip_max_norm
    }

    convGRU.fit(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optim=optim,
        lr=lr,
        momentum=momentum,
        epochs=epochs,
        max_norm=clip_max_norm,
        track_run=track_run
    )