import torch
import numpy as np

def get_accuracy(model, dataloader):
    model.eval()
    n_correct = 0
    n_eval = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            log_probs = model(batch_x)
            # sm = torch.nn.Softmax(dim=1)
            # predicted = sm(log_probs).max(axis=1).indices
            predicted = log_probs.max(axis=1).indices
            # dense_y = np.where(batch_y == 1)[1]
            dense_y = batch_y
            n_correct += int(np.equal(dense_y, predicted).sum())
            n_eval += len(batch_y)
    return n_correct / n_eval