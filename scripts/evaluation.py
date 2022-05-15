import torch
import numpy as np

def get_accuracy(model, dataloader, changed_only=False):
    model.eval()
    n_correct = 0
    n_eval = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            for timestep in range(batch_x.shape[1] - 1):

                # For each BPTT step, get all timesteps up until now, model them
                inputs = batch_x[:,0:timestep+1,:,:]
                outputs = model(inputs)

                # Then, choose next timestep target to predict
                targets = batch_y[:,timestep+1,:,:]

                if changed_only:
                    last_targets = batch_y[:,timestep,:,:]
                    changed = ~torch.eq(targets, last_targets)
                    targets = targets[changed]
                    outputs = outputs[changed]
                if not changed_only or (changed_only and len(outputs) > 0):
                    if changed_only:
                        pass
                        changed_only
                    predicted = outputs.softmax(3).argmax(3)

                    assert targets.size() == predicted.size()
                    n_correct += int(np.equal(targets, predicted).sum())
                    n_eval += targets.numel()

    return n_correct / n_eval