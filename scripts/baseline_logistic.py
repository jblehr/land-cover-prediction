import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import dataloaders
import evaluation

class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sm = torch.nn.LogSoftmax()

    def forward(self, x):
        # outputs = torch.sigmoid(self.linear(x))
        # outputs = self.linear(x)
        outputs = self.sm(self.linear(x))
        return outputs

def train_logstic(train_loader, test_loader, input_dim=4, output_dim=7, epochs=50, learning_rate=.01, criterion = torch.nn.NLLLoss()):

    model = LogisticRegression(input_dim,output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    accuracies = []
    
    for epoch in tqdm(range(epochs),desc='Training Epochs'):
        for iter, (batch_x, batch_y) in enumerate(train_loader):
            outputs = model(batch_x)
            # batch_y_ind = torch.tensor(np.where(batch_y == 1)[1]).long()
            batch_y_ind = batch_y
            loss = criterion(outputs, batch_y_ind)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter % 1000 == 0 and iter > 0:
                print(f'At iteration {iter} the loss is {loss:.3f}.')
        acc = evaluation.get_accuracy(model, test_loader)
        accuracies.append(acc)
        print(f'After epoch: accuracy is {acc:.3f}.')
    return model, accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb_path', type = str)
    parser.add_argument('lab_path', type = str)
    parser.add_argument('--batch_size', type = int, required=False, default=64)
    parsed = parser.parse_args()

    # Load dataset
    dataset = dataloaders.FullyIndependentDataset(parsed.rgb_path, parsed.lab_path)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=parsed.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model, accuracies = train_logstic(train_loader, test_loader)

    #final confusion
    test_pred = model(test_set)

    accuracies