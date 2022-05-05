import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # outputs = torch.sigmoid(self.linear(x))
        outputs = self.linear(x)
        return outputs


class FullyIndependentDataset(Dataset):
    def __init__(self, img_path, label_path):

        # Grouping variable names
        self.categorical = ["0", "1", "2", "3", "4", "5", "6"]
        self.target = "land_class"
        self.n_ambigious = 0

        img_cube = np.load(img_path)['arr_0']
        label_cube = np.load(label_path)['arr_0']
        
        fi_rgbn = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2], 4))
        fi_class = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2], 7))

        fi_idx = 0
        for x_pos in range(img_cube.shape[1]):
            for y_pos in range(img_cube.shape[2]):
                fi_rgbn[fi_idx,:] = img_cube[:,x_pos,y_pos]
                one_hot = (label_cube[:,x_pos,y_pos] == 255).astype(int)
                if sum(one_hot) > 1:
                    one_ind = np.where(one_hot == 1)[0]
                    one_hot[one_ind[1:]] = 0
                    self.n_ambigious += 1
                fi_class[fi_idx, :] = one_hot
                fi_idx += 1


        # Save target and predictors
        self.X = torch.tensor(fi_rgbn.astype(np.float32))
        self.y = torch.tensor(fi_class.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx,:], self.y[idx])

def get_accuracy(model, dataloader):
    model.eval()
    n_correct = 0
    n_eval = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            log_probs = model(batch_x)
            sm = torch.nn.Softmax(dim=1)
            predicted = sm(log_probs).max(axis=1).indices
            dense_y = np.where(batch_y == 1)[1]
            n_correct += int(sum(np.equal(dense_y, predicted)))
            n_eval += len(batch_y)
    return n_correct / n_eval

def train_logstic(train_loader, test_loader, input_dim=4, output_dim=7, epochs=50, learning_rate=.01, criterion = torch.nn.CrossEntropyLoss()):

    model = LogisticRegression(input_dim,output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    accuracies = []
    
    for epoch in tqdm(range(epochs),desc='Training Epochs'):
        for iter, (batch_x, batch_y) in enumerate(train_loader):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter % 1000 == 0 and iter > 0:
                print(f'At iteration {iter} the loss is {loss:.3f}.')
        acc = get_accuracy(model, test_loader)
        accuracies.append(acc)
    return model, accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb_path', type = str)
    parser.add_argument('lab_path', type = str)
    parser.add_argument('--batch_size', type = int, required=False, default=64)
    parsed = parser.parse_args()

    # Load dataset
    dataset = FullyIndependentDataset(parsed.rgb_path, parsed.lab_path)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=parsed.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = train_logstic(train_loader, test_loader)