import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import dataloaders
import evaluation
import torchvision
import logging
import json

class LogisticRegression(torch.nn.Module):
    """
    Basic logistic regression that treats each pixel independently. Due to my
    implementation of evaluation for convGRU model, needed to do some trickery
    in the forward pass (basically, for eval, class probs need to be last dim,
    but for upsampler, but for upsampler, x,y needs to be last. Also needed to
    squeeze out the first batch dim so this will NOT work with cell_width_pct
    < 1. Not sure why, but torchvision's Resize does not work with 5 dims, 
    despite the docs saying it only cares that the last two are x,y).
    """
    def __init__(self, input_dim, output_dim, upsample_dim, cuda_):
        super(LogisticRegression, self).__init__()
        self.upsample_dim = upsample_dim
        self.cuda_ = cuda_
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.upsampler = torchvision.transforms.Resize(
            (upsample_dim, upsample_dim)
        )

    def forward(self, x):
        x = x.permute(0,1,3,4,2) #shift channels to last for nn.Linear
        outputs = self.linear(x).permute(0,1,4,2,3) # return x, y to last dim for upsampler

        # Bug in torchvision? Docs say to pass with arbitrary leading dims
        # with x, y as final two, but with batch, it is taking in the last three
        # and considering 3d rather than 2d

        assert x.shape[0] == 1
        x = x.squeeze(0)

        outputs = self.linear(x).permute(0,3,1,2) # return x, y to last dim for upsampler
        upsampled_out = self.upsampler(outputs).permute(0,2,3,1)
        return upsampled_out

def train_logstic(
    train_loader,
    test_loader,
    model_out,
    input_dim=4,
    output_dim=7,
    upsample_dim=1024,
    epochs=50,
    lr=.01,
    momentum=.7,
    criterion = torch.nn.CrossEntropyLoss(),
    cuda_=False,
    final_train = True
    ):
    """Train the logistic regression to the train dataset.

    Args:
        train_loader (DataLoader): Train dataloader
        test_loader (DataLoader): Test dataloader
        input_dim (int, optional): number of input channels
        output_dim (int, optional): number of output classes
        upsample_dim (int, optional): size to eventually upsample to. No effect
            if there's no downsampling in Dataloader transform.
        epochs (int, optional): number of epochs. Defaults to 50.
        lr (float): learning rate
        momentum (float): momentum, only relevant for 'sgd' optimizaer
        criterion (torch criterion): criterion to optimize. Defaults to torch.nn.CrossEntropyLoss().
        cuda_ (bool, optional): whether to use cuda. Defaults to False.
        final_train (bool, optional): If true, save best model and results.json
        model_out (str, optional): Path to save best model if final_train. 
    """
    model = LogisticRegression(
        input_dim,
        output_dim,
        upsample_dim,
        cuda_=cuda_
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()

    min_test_loss = None
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        losses = []
        for idx, (batch_x, batch_y) in enumerate(train_loader):

            for timestep in range(batch_x.shape[1] - 1):

                inputs = batch_x[:, timestep:timestep+1, :, :]
                outputs = model(inputs)

                # Then, choose next timestep target to predict
                targets = batch_y[:, timestep + 1, :, :]

                # For compatibility with CrossEntropyLoss, reshape to ignore
                # spatial dims and batches for loss - doesn't matter in this
                # case anyways as we just want pixels to line up properly
                outputs = outputs.squeeze(1)
                flat_dim = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]

                outputs_flat = outputs.reshape(flat_dim, outputs.shape[3])
                targets_flat = targets.reshape(flat_dim)
                if cuda_:
                    targets_flat = targets_flat.to("cuda")

                loss = criterion(outputs_flat, targets_flat)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                losses.append(float(loss))

        train_loss = np.mean(losses)
        train_losses.append(train_loss)

        train_acc = evaluation.get_accuracy(model, train_loader, bptt_len=0)
        train_accs.append(train_acc)

        test_loss = evaluation.get_loss(model, test_loader, criterion, bptt_len=0, cuda_=cuda_)
        test_losses.append(test_loss)

        test_acc = evaluation.get_accuracy(model, test_loader, bptt_len=0)
        test_accs.append(test_acc)

        if not min_test_loss or test_loss < min_test_loss:
            min_test_loss = test_loss
            if final_train:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
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

    if final_train:
        train_report = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
        }
        train_report_path = model_out.replace('.pt', '.json')
        with open(train_report_path, "w") as fp:
            json.dump(train_report, fp)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('rgb_path', type = str)
    # parser.add_argument('lab_path', type = str)
    # parser.add_argument('--batch_size', type = int, required=False, default=64)
    # parsed = parser.parse_args()
    
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

    transform = torchvision.transforms.Resize(size=(128, 128))

    train_dataloader = dataloaders.SpatiotemporalDataset(
        # "/scratch/npg/data/processed/npz",
        # "/home/npg/land-cover-prediction/data/processed/npz",
        "data/processed/npz",
        dims=(1024, 1024),  # Original dims, not post-transformation
        poi_list=train_poi_list,
        n_steps=3, 
        cell_width_pct=1,
        labs_as_features=False,
        transform=transform,
        download=False,
        in_memory=True
    )

    test_dataloader = dataloaders.SpatiotemporalDataset(
        # "/scratch/npg/data/processed/npz",
        # "/home/npg/land-cover-prediction/data/processed/npz",
        "data/processed/npz",
        dims=(1024, 1024),  # Original dims, not post-transformation
        poi_list=test_poi_list,
        n_steps=3,  
        cell_width_pct=1,
        labs_as_features=False,
        transform=transform,
        download=False,
        in_memory=True
    )

    cuda_ = torch.cuda.is_available()

    train_logstic(
        train_dataloader,
        test_dataloader,
        lr = 0.0005326639774392545,
        momentum = 0.76,
        cuda_=cuda_,
        model_out='output/models/LReg_8step.pt'
    )
