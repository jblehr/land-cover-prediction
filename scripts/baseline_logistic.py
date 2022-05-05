#SOURCE: Adapted from https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be

import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from tqdm import tqdm
import torch
import argparse

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_style("darkgrid")

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))

        return outputs


class FullyIndependentDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        img_cube = np.load(img_path)['arr_0']
        label_cube = np.load(label_path)['arr_0']
        
        fi_rgbn = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2], 4))
        fi_class = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2],))

        fi_idx = 0
        for x_pos in range(img_cube.shape[1]):
            for y_pos in range(img_cube.shape[2]):
                fi_rgbn[fi_idx,:] = img_cube[:,x_pos,y_pos]
                fi_class[fi_idx] = np.where(label_cube[:,x_pos,y_pos] == 255)[0][0]
                fi_idx += 1

        # Grouping variable names
        self.categorical = ["0", "1", "2", "3", "4", "5", "6"]
        self.target = "land_class"

        # Save target and predictors
        self.X = torch.tensor(fi_rgbn)
        self.y = torch.tensor(fi_class)
        pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx,:], self.y[idx])


def model_plot(model,X,y,title,out_path):
    parm = {}
    b = []
    for name, param in model.named_parameters():
        parm[name]=param.detach().numpy()  
    
    w = parm['linear.weight'][0]
    b = parm['linear.bias'][0]
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap='jet')
    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
    plt.plot(u, (0.5-b-w[0]*u)/w[1])
    plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
    plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
    plt.xlabel(r'$\boldsymbol{x_1}$',fontsize=16) # Normally you can just add the argument fontweight='bold' but it does not work with latex
    plt.ylabel(r'$\boldsymbol{x_2}$',fontsize=16)
    plt.title(title)
    plt.savefig(out_path)

def train_logstic(input_dim, output_dim, epochs=50000, learning_rate=.01, chunk_size=64):

    model = LogisticRegression(input_dim,output_dim)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)

    losses = []
    losses_test = []
    Iterations = []
    iter = 0

    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
        x = X_train
        labels = y_train
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), labels) # [200,1] -squeeze-> [200]
        
        loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
        
        optimizer.step() # Updates weights and biases with the optimizer (SGD)
        
        iter+=1

        if iter%10000==0:
            # calculate Accuracy
            with torch.no_grad():

                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)
                
                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)
                
                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    # Train Data
    model_plot(model,X_train,y_train,'Train Data', 'model0_Train.png')

    # Test Dataset Results
    model_plot(model,X_test,y_test,'Test Data', 'model0_Test.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb_path', type = str)
    parser.add_argument('lab_path', type = str)
    parsed = parser.parse_args()

    dat = FullyIndependentDataset(parsed.rgb_path, parsed.lab_path)