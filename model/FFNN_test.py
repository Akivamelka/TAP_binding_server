import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# hyperparameters
hidden = 200
dropout_prob = 0.0

alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc0 = nn.Linear(180, hidden)
        self.fc1 = nn.Linear(hidden, 1)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_prob)
        y1 = torch.sigmoid(self.fc1(x))
        y2 = self.fc2(x)
        return y1, y2

def pep_to_onehot(peptide_list):

    one_hot_data = []

    for line in peptide_list:
        vec = [0] * 180
        for i in range(9):
            vec[i * 20 + alphabet.index(line[i])] = 1
        one_hot_data.append(vec)

    return np.array(one_hot_data)

def compute_output(file_input, file_output, model_path):

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    pre_data = pd.read_csv(file_input, header=None)[0].values
    data_x = pep_to_onehot(pre_data)
    data_x = torch.tensor(data_x)
    data_x = data_x.to(dtype=torch.float)

    # Compute output
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pred1, pred2 = model(data_x.to(device))
    pred1 = pred1.view(-1).tolist()
    pred2 = np.exp(pred2.view(-1).tolist())

    # Print output in file
    data_df = pd.DataFrame(pre_data, columns=['Peptides'])
    pred1_df = pd.DataFrame(pred1, columns=['Proba'])
    pred2_df = pd.DataFrame(pred2, columns=['Concentration'])
    output = pd.concat([data_df, pred1_df, pred2_df], axis=1)
    output.to_csv(file_output, index=False)

def compute_single_output(peptide_seq, model_path):

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    pre_data = np.array([peptide_seq])
    data_x = pep_to_onehot(pre_data)
    data_x = torch.tensor(data_x)
    data_x = data_x.to(dtype=torch.float)

    # Compute output
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pred1, pred2 = model(data_x.to(device))
    pred1 = pred1.view(-1).tolist()
    pred2 = np.exp(pred2.view(-1).tolist())

    return pred1[0], pred2[0]