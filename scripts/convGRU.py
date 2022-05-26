# Code based on https://github.com/happyjin/ConvGRU-pytorch/blob/9e932822e11db19b78e3761cb71018850a1247ff/convGRU.py

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import evaluation
from tqdm import tqdm
import dataloaders
import numpy as np
import os
import optuna
from optuna.trial import TrialState
import argparse
import json
import logging


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_dim,
        input_channels,
        hidden_channels,
        kernel_size,
        bias,
        conv_padding_mode,
        cuda_=False
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
        self.padding = "same"
        self.padding_mode = conv_padding_mode
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.cuda_ = cuda_
        self.train_report = {}

        if cuda_:
            self.dtype = torch.cuda.FloatTensor  # computation in GPU
        else:
            self.dtype = torch.FloatTensor

        self.conv_gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2
            * self.hidden_channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            padding_mode=conv_padding_mode,
            bias=self.bias,
        )

        if cuda_:
            self.conv_gates.to("cuda")

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
            self.conv_gates.to("cuda")
            self.conv_can.to("cuda")

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
            input_current = input_current.to("cuda")

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
        output_dim,
        input_channels,
        hidden_channels,
        n_output_classes,
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

        self.in_xDim, self.in_yDim = input_dim
        self.out_xDim, self.out_yDim = output_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.cuda_ = cuda_
        self.upsampler = None

        cell_list = []

        for i in range(0, self.num_layers):
            current_input_channels = (
                input_channels if i == 0 else hidden_channels[i - 1]
            )
            cell_list.append(
                ConvGRUCell(
                    input_dim=(self.in_xDim, self.in_yDim),
                    input_channels=current_input_channels,
                    hidden_channels=self.hidden_channels[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    conv_padding_mode=conv_padding_mode,
                    cuda_=cuda_,
                )
            )

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        self.lin_out = nn.Linear(
            in_features=self.hidden_channels[-1], out_features=n_output_classes
        )
        if not (self.in_xDim == self.out_xDim and self.in_yDim == self.out_yDim):
            self.upsampler = torchvision.transforms.Resize(
                (self.out_xDim, self.out_yDim)
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
        last_output = layer_output[:, -1, :, :, :]
        if self.upsampler:
            last_output = self.upsampler(last_output)
        class_val = self.lin_out(last_output.permute(0, 2, 3, 1))

        return class_val

    def fit(
        self,
        train_loader,
        test_loader,
        optim,
        lr,
        momentum,
        trial=None,
        bptt_len=3,
        epochs=50,
        max_norm=False,
        final_train=False,
        model_out=None
    ):

        if optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        self.train()
        criterion = torch.nn.CrossEntropyLoss()
        min_test_loss = None
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            losses = []
            for idx, (batch_x, batch_y) in enumerate(train_loader):

                for timestep in range(bptt_len, batch_x.shape[1] - 1):

                    min_step = max(0, timestep - bptt_len)

                    # For each BPTT step, get all timesteps (t-bptt_len):t, model them
                    inputs = batch_x[:, min_step : timestep + 1, :, :]
                    outputs = self(inputs)

                    # Then, choose next timestep target to predict
                    targets = batch_y[:, timestep + 1, :, :]

                    # For compatibility with CrossEntropyLoss, reshape to ignore
                    # spatial dims and batches for loss - doesn't matter in this
                    # case anyways as we just want pixels to line up properly
                    flat_dim = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]

                    outputs_flat = outputs.reshape(flat_dim, outputs.shape[3])
                    targets_flat = targets.reshape(flat_dim)
                    if self.cuda_:
                        targets_flat = targets_flat.to("cuda")

                    loss = criterion(outputs_flat, targets_flat)
                    if max_norm:
                        nn.utils.clip_grad_norm_(
                            self.parameters(), max_norm=max_norm, norm_type=2
                        )

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(float(loss))

            train_loss = np.mean(losses)
            train_losses.append(train_loss)

            train_acc = evaluation.get_accuracy(self, train_loader)
            train_accs.append(train_acc)

            test_loss = evaluation.get_loss(self, test_loader, criterion, self.cuda_)
            test_losses.append(test_loss)

            test_acc = evaluation.get_accuracy(self, test_loader)
            test_accs.append(test_acc)

            if not min_test_loss or test_loss < min_test_loss:
                min_test_loss = test_loss
                if final_train:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": test_loss,
                        },
                        model_out,
                    )

            logging.info(f"  -- epoch: {epoch}")
            logging.info(f"  -- test acc: {test_acc:.3f}")
            logging.info(f"  -- test loss: {test_loss:.3f}")
            logging.info(f"  -- train acc: {train_acc:.3f}")
            logging.info(f"  -- train loss: {train_loss:.3f}")

            logging.info(f"============== End of epoch {epoch} ============")

            if trial:
                trial.report(test_loss, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        self.train_report = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
        }
        return test_loss

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


def objective(trial, train_dataloader=False, test_dataloader=False, fixed=False):

    cuda_ = torch.cuda.is_available()
    if cuda_:
        print("using GPU backend!")
    else:
        print("using CPU backend.")

    if not trial:
        epochs = 10
        cell_width_pct = 1
        clip_max_norm = 1.184316464877487
        conv_kernel_size = 7
        downsample = True
        downsample_dim = 128
        guassian_blur = False
        hidden_channels = [512]
        bptt_len = 4
        lr = 0.0005326639774392545
        momentum = 0.7679114313544549
        num_layers = 1
        optim = "adam"
        bias = True
        final_train = True
        model_out = 'output/models/ConvGRU_7bptt_12step.pt'

    elif not fixed:
        final_train = False
        epochs = 2
        guassian_blur = trial.suggest_categorical("guassian_blur", [True, False])
        if guassian_blur:
            guassian_sigma = trial.suggest_float("guassian_sigma", 1.0, 5.0)

        downsample = trial.suggest_categorical("downsample", [True, False])
        if downsample:
            downsample_dim = trial.suggest_categorical(
                "downsample_dim", [64, 128, 256, 512]
            )

        optim = trial.suggest_categorical("optim", ["sgd", "adam"])
        clip_max_norm = trial.suggest_float("clip_max_norm", 0.5, 1.5)

        cell_width_pct = trial.suggest_categorical(
            "cell_width_pct", [1, 1 / 2, 1 / 4, 1 / 8]
        )

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.0, 1.0)
        conv_kernel_size = trial.suggest_int("conv_kernel_size", 2, 7)
        bias = True

        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_channels = []
        for layer_idx in range(num_layers):
            hidden_channels_idx = trial.suggest_categorical(
                f"layer_{layer_idx}", [4, 8, 16, 32, 64, 128, 256]
            )
            hidden_channels.append(hidden_channels_idx)
    else:
        # if fixed, fix the 'unimportant' hyperpars according to first pass
        epochs = 15
        final_train = False
        downsample_dim = 128
        downsample = True
        # conv_kernel_size = 6
        clip_max_norm = 1.18
        optim = "adam"
        lr = 0.0005
        momentum = 0.76 # don't think this does anything with adam
        guassian_blur = False
        cell_width_pct = 1
        bias = True

        bptt_len = trial.suggest_int("bptt_len", 2, 11)
        conv_kernel_size = trial.suggest_int("conv_kernel_size", 2, 7)

        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_channels = []
        for layer_idx in range(num_layers):
            hidden_channels_idx = trial.suggest_int(
                f"layer_{layer_idx}",
                low = 50,
                high = 500,
                step = 50
            )
            hidden_channels.append(hidden_channels_idx)

    # Set transform value according to hyperpar choices
    transform = None

    if guassian_blur:
        guassian_kernel = int(guassian_sigma * 3 + 1)
        guassian_kernel = guassian_kernel - int(guassian_kernel % 2 != 1)
        blur_transform = torchvision.transforms.GaussianBlur(
            guassian_kernel, guassian_sigma
        )
        transform = blur_transform

    if downsample:
        downsample_transform = torchvision.transforms.Resize(
            size=(downsample_dim, downsample_dim)
        )
        transform = downsample_transform

    if guassian_blur and downsample:
        transform = torchvision.transforms.Compose(
            [downsample_transform, blur_transform]
        )

    if isinstance(train_dataloader, list):

        train_dataloader = dataloaders.SpatiotemporalDataset(
            "/home/npg/land-cover-prediction/data/processed/npz",
            # "data/processed/npz",
            dims=(1024, 1024),  # Original dims, not post-transformation
            poi_list=train_poi_list,
            n_steps=2,  # start with one prediction (effectively flat CNN)
            cell_width_pct=cell_width_pct,
            labs_as_features=False,
            transform=transform,
            download=False,
            in_memory=True,
        )

    if isinstance(test_dataloader, list):

        test_dataloader = dataloaders.SpatiotemporalDataset(
            "/home/npg/land-cover-prediction/data/processed/npz",
            # "data/processed/npz",
            dims=(1024, 1024),  # Original dims, not post-transformation
            poi_list=test_poi_list,
            n_steps=2,  # start with one prediction (effectively flat CNN)
            cell_width_pct=cell_width_pct,
            labs_as_features=False,
            transform=transform,
            download=False,
            in_memory=True,
        )

    # set up dims properly for convGRU, according to our downsample/cell-width choices
    if downsample:
        in_xDim = in_yDim = int(downsample_dim * cell_width_pct)
        out_xDim = out_yDim = int(1024 * cell_width_pct)
    else:
        in_xDim = in_yDim = out_xDim = out_yDim = int(1024 * cell_width_pct)

    input_channels = 4
    n_output_classes = 7

    convGRU_mod = ConvGRU(
        input_dim=(in_xDim, in_yDim),
        output_dim=(out_xDim, out_yDim),
        num_layers=num_layers,
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        n_output_classes=n_output_classes,
        kernel_size=conv_kernel_size,
        batch_first=True,
        conv_padding_mode="replicate",
        bias=bias,
        cuda_=cuda_,
    )

    if cuda_:
        convGRU_mod = convGRU_mod.to("cuda")

    test_loss = convGRU_mod.fit(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        bptt_len=bptt_len,
        optim=optim,
        lr=lr,
        momentum=momentum,
        epochs=epochs,
        max_norm=clip_max_norm,
        trial=trial,
        final_train=final_train,
        model_out=model_out
    )

    if not final_train:
        logging.info("Training with layers {hidden_channels}.")
    if final_train:
        train_report_path = model_out.replace('.pt', '.json')
        with open(train_report_path, "w") as fp:
            json.dump(convGRU_mod.train_report, fp)

    return test_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", default=False)
    parser.add_argument(
        "--optuna_path",
        default="sqlite:////home/npg/land-cover-prediction/output/midway_optuna.db",
    )
    parsed = parser.parse_args()

    ## FOR FULL TEMPORAL

    # test_poi_list = [
    #     "1700_3100_13_13N",
    #     "2029_3764_13_15N",
    #     "4426_3835_13_33N",
    #     "4397_4302_13_33S",
    #     "5125_4049_13_38N",
    #     "4768_4131_13_35S"
    # ]

    # train_poi_list = [
    #     "1311_3077_13_10N",
    #     "2624_4314_13_20S",
    #     "4622_3159_13_34N",
    #     "4806_3588_13_36N",
    #     "2697_3715_13_20N",
    #     "4791_3920_13_36N",
    #     "1417_3281_13_11N"
    # ]

    # FOR FLAT CNN

    test_poi_list = [
        "1700_3100_13_13N",
        "2029_3764_13_15N",
        "4426_3835_13_33N",
        "4397_4302_13_33S",
        "5125_4049_13_38N",
        "4768_4131_13_35S"
    ]

    train_poi_list = [
        "1311_3077_13_10N",
        "2065_3647_13_16N",
        "2624_4314_13_20S",
        "4622_3159_13_34N",
        "4806_3588_13_36N",
        "3002_4273_13_22S",
        "4881_3344_13_36N",
        "5863_3800_13_43N",
        "1417_3281_13_11N",
        "2006_3280_13_15N",
        "2235_3403_13_17N",
        "2697_3715_13_20N",
        "4421_3800_13_33N",
        "4838_3506_13_36N",
        "5111_4560_13_38S",
        "5926_3715_13_44N",
        "1487_3335_13_11N",
        "2415_3082_13_18N",
        "4791_3920_13_36N",
        "4856_4087_13_36N",
        "5989_3554_13_44N",
    ]

    ## FOR LOCAL TEST

    # test_poi_list = [
    #     "1417_3281_13_11N",
    #     "5125_4049_13_38N"
    # ]

    # train_poi_list = [
    #     "1311_3077_13_10N"
    # ]

    transform = torchvision.transforms.Resize(size=(128, 128))

    train_dataloader = dataloaders.SpatiotemporalDataset(
        "/scratch/npg/data/processed/npz",
        # "/home/npg/land-cover-prediction/data/processed/npz",
        dims=(1024, 1024),  # Original dims, not post-transformation
        poi_list=train_poi_list,
        n_steps=12, 
        cell_width_pct=1,
        labs_as_features=False,
        transform=transform,
        download=False,
        in_memory=True
    )

    test_dataloader = dataloaders.SpatiotemporalDataset(
        "/scratch/npg/data/processed/npz",
        # "/home/npg/land-cover-prediction/data/processed/npz",
        dims=(1024, 1024),  # Original dims, not post-transformation
        poi_list=test_poi_list,
        n_steps=12,  
        cell_width_pct=1,
        labs_as_features=False,
        transform=transform,
        download=False,
        in_memory=True
    )

    if parsed.optuna:

        # fixed_params = {
        #     "downsample_dim": 128,
        #     "downsample": True,
        #     "conv_kernel_size": 6,
        #     "clip_max_norm" : 1.18,
        #     "optim" : 'adam',
        #     "lr": .0005,
        #     "momentum": .76,
        #     "gauassian_blur" : False,
        #     "cell_pct_width" : 1
        #     }

        study = optuna.create_study(
            direction="minimize",
            study_name="midway_loss_withTime",
            storage=parsed.optuna_path,
            load_if_exists=True,
        )

        # Wrap the objective inside a lambda and call objective inside it
        # courtesy of https://www.kaggle.com/general/261870
        objective_preloaded = lambda trial: objective(
            trial, train_dataloader, test_dataloader, fixed=True
        )

        # partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
        # study.sampler = partial_sampler

        study.optimize(objective_preloaded)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logging.info("Study statistics: ")
        logging.info("  Number of finished trials: ", len(study.trials))
        logging.info("  Number of pruned trials: ", len(pruned_trials))
        logging.info("  Number of complete trials: ", len(complete_trials))

        logging.info("Best trial:")
        trial = study.best_trial

        logging.info("  Value: ", trial.value)

        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info("    {}: {}".format(key, value))
    else:
        objective(False, train_dataloader, test_dataloader)
        logging.info("fitting complete.")
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()
