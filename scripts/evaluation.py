import torch
import numpy as np
import dataloaders
import os
import convGRU
import logging
from sklearn.metrics import confusion_matrix

def get_accuracy(model, dataloader, bptt_len, changed_only=False):
    """Calculates accuracy of model on certain dataset.

    Args:
        model (torch model): model to evaluate.
        dataloader (torch dataloader): set to evaluate
        bptt_len (int): amount of temporal data to pass the model. 
        changed_only (bool, optional): if positive, only consider pixels that have
            changed. Hasn't been tested thoroughly as many steps have no changes.

    Returns:
        float of accuracy on dataloader set
    """
    model.eval()
    n_correct = 0
    n_eval = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            print(batch_x.shape)
            for timestep in range(bptt_len, batch_x.shape[1] - 1):

                min_step = max(0, timestep - bptt_len)
                # For each BPTT step, get all timesteps up until now, model them
                inputs = batch_x[:,min_step:timestep+1,:,:]
                outputs = model(inputs)

                # Then, choose next timestep target to predict
                targets = batch_y[:,timestep+1,:,:]

                if changed_only:
                    last_targets = batch_y[:,timestep,:,:]
                    changed = ~torch.eq(targets, last_targets)
                    targets = targets[changed]
                    outputs = outputs[changed]
                if not changed_only or (changed_only and len(outputs) > 0):
                    predicted = outputs.softmax(3).argmax(3)

                    assert targets.size() == predicted.size()
                    if model.cuda_:
                        targets = targets.to('cuda')
                        predicted = predicted.to('cuda')
                    n_correct += int(torch.eq(targets, predicted).sum())
                    n_eval += targets.numel()

    return n_correct / n_eval

def get_loss(model, dataloader, criterion, bptt_len):
    """Calculates loss of model on certain dataset.

    Args:
        model (torch model): model to evaluate.
        dataloader (torch dataloader): set to evaluate
        criterion (torch loss calculator): criterion to evaluate loss
        bptt_len (int): amount of temporal data to pass the model. 

    Returns:
        float of mean loss on dataloader set
    """
    model.eval()
    losses=[]
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            for timestep in range(bptt_len, batch_x.shape[1] - 1):

                min_step = max(0, timestep - bptt_len)
                # For each BPTT step, get all timesteps up until now, model them
                inputs = batch_x[:,min_step:timestep+1,:,:]
                outputs = model(inputs)

                # Then, choose next timestep target to predict
                targets = batch_y[:,timestep+1,:,:]

                # For compatibility with CrossEntropyLoss, reshape to ignore
                # spatial dims and batches for loss - doesn't matter in this
                # case anyways as we just want pixels to line up properly
                flat_dim = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]

                outputs_flat = outputs.reshape(flat_dim, outputs.shape[3])
                targets_flat = targets.reshape(flat_dim)

                if model.cuda_:
                    targets_flat = targets_flat.to('cuda')
                    outputs_flat = outputs_flat.to('cuda')

                loss = criterion(outputs_flat, targets_flat)
                losses.append(float(loss))

    return np.mean(losses)

def get_confusion(model, dataloader, bptt_len):
    """Calculates confusion on a given dataset.

    Args:
        model (torch model): model to evaluate.
        dataloader (torch dataloader): set to evaluate
        bptt_len (int): amount of temporal data to pass the model. 
        changed_only (bool, optional): if positive, only consider pixels that have
            changed. Hasn't been tested thoroughly as many steps have no changes.

    Returns:
        float of accuracy on dataloader set
    """
    model.eval()
    confusion = torch.zeros((7,7)) # 7 land use classes
    denom = torch.zeros(7)
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            for timestep in range(bptt_len, batch_x.shape[1] - 1):
                min_step = max(0, timestep - bptt_len)
                # For each BPTT step, get all timesteps up until now, model them
                inputs = batch_x[:,min_step:timestep+1,:,:]
                outputs = model(inputs)

                # Then, choose next timestep target to predict
                targets = batch_y[:,timestep+1,:,:]

                flat_dim = outputs.shape[0] * outputs.shape[1] * outputs.shape[2]
                
                outputs_flat = outputs.reshape(flat_dim, outputs.shape[3])
                outputs_class = outputs_flat.softmax(1).argmax(1)
                targets_flat = targets.reshape(flat_dim)

                if model.cuda_:
                    targets_flat = targets_flat.to('cuda')
                    outputs_flat = outputs_flat.to('cuda')

                confusion_batch = torch.tensor(confusion_matrix(
                    outputs_class,
                    targets_flat,
                    labels=np.arange(0,7)
                ))

                confusion = confusion.add(confusion_batch)

    return (confusion / (confusion.sum(dim = 1).unsqueeze(1))) * 100

if __name__ == '__main__':
    poi_list = os.listdir('data/processed/npz/planet')
    STData = dataloaders.SpatiotemporalDataset(
        "data/processed/npz",
        dims = (1024, 1024), #Original dims, not post-transformation
        poi_list=poi_list,
        n_steps=12, # start with one year
        cell_width_pct=.5,
        labs_as_features=False,
        transform=None
    )

    test_dataloader = torch.utils.data.DataLoader(
        STData,
        # batch_size=batch_size,
        batch_size=None, #batches determined by cell width
        shuffle=True
    )
    
    criterion=torch.nn.CrossEntropyLoss()

    convGRU_mod = convGRU.ConvGRU(
        input_dim=(1024,1024),
        output_dim=(1024, 1024),
        num_layers=2,
        input_channels=4,
        hidden_channels=[4, 6],
        n_output_classes=7,
        kernel_size=3,
        batch_first=True,
        conv_padding_mode='replicate',
        bias=True,
        cuda_=False
    )

    loss=get_loss(convGRU_mod, test_dataloader, criterion, False)
    loss