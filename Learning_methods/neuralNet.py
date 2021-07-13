#import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torch import optim

import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from Preprocess.Preprocces_whole_data import *
from Preprocess.split_data import split_data_and_tags

class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


class NetworkClassifier(nn.Module):
    def __init__(self, _rec_size, first, second):
        super(NetworkClassifier, self).__init__()

        self.size = _rec_size
        self.fc0 = nn.Linear(_rec_size, first)
        self.fc1 = nn.Linear(first, second)
        self.fc2 = nn.Linear(second, 9)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        m = Softmax(1)
        return m(x)


def train_nn(n_epochs, model, train_dataloader, test_dataloader, optimizer):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (receptors_batch, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(receptors_batch)
            #vertoutput = torch.reshape(output, (-1,))
            loss = F.cross_entropy(output.type(torch.float), labels.type(torch.int64))  # cross entropy
            loss.backward()
            optimizer.step()
        eval(model, train_dataloader, "Train")
        eval(model, test_dataloader, "Val")


def neural_network(train_data, train_tags, test_data, test_tags, train_ratio,  params):
    realtrain_data, val_data, realtrain_tags, val_tags = train_test_split(train_data, train_tags, train_size=train_ratio)

    feature_size = len(realtrain_data.columns)
    model = NetworkClassifier(feature_size, params["hidden_size_01"], params["hidden_size_02"])

    train_dataset = Dataset(realtrain_data.to_numpy(dtype=np.float32), realtrain_tags.to_numpy(dtype=np.float32))
    val_dataset = Dataset(val_data.to_numpy(dtype=np.float32), val_tags.to_numpy(dtype=np.float32))
    test_dataset = Dataset(test_data.to_numpy(dtype=np.float32), test_tags.to_numpy(dtype=np.float32))

    batch_size = params["batch_size"]

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    lr = params["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_nn(params["n_epoch"], model, train_dataloader, val_dataloader, optimizer)

    f1 = eval(model, test_dataloader, "Test")
    print(f1)
    return f1

def NN_crossval(data, tags, params: dict, fold=5):
    con_mat_nlst = []
    con_mat_lst = []
    f1_lst = []
    for i in range(fold):
        train_and_val = train_test_split(train, tags, test_size=1/fold)
        conmatn, conmat, f1 = neural_network(*train_and_val, params=params)
        con_mat_nlst.append(conmatn)
        con_mat_lst.append(conmat)
        f1_lst.append(f1)
    con_mat_nlst = np.array(con_mat_nlst)
    mean_nmat = con_mat_nlst.mean(axis=0)
    std_nmat = con_mat_nlst.std(axis=0)

    con_mat_lst = np.array(con_mat_lst)
    mean_mat = con_mat_lst.mean(axis=0)
    std_mat = con_mat_lst.std(axis=0)

    f1_lst = np.array(f1_lst)
    mean_f1 = f1_lst.mean(axis=0)
    std_f1 = f1_lst.std(axis=0)
    return mean_nmat, std_nmat, mean_mat, std_mat, mean_f1, std_f1

def NN_choose_params(data: pd.DataFrame, tags: pd.Series):
    maxconmatn = np.zeros((2, 2))
    maxconmat = np.zeros((2, 2))
    maxf1 = np.array(0)

    for n_est, max_feat, max_depth, min_splt, min_leaf in tqdm(product()):

        paramsgrid = {}

        mean_nmat, std_nmat, mean_mat, std_mat, mean_f1, std_f1 = NN_crossval(data, tags, paramsgrid)

        if mean_nmat.trace() > maxconmatn.trace():
            maxconmatn = mean_nmat
            maxconmatnstd = std_nmat
            conmatnparams = paramsgrid

        if mean_mat.trace() > maxconmat.trace():
            maxconmat = mean_mat
            maxconmatstd = std_mat
            conmatparams = paramsgrid

        if mean_mat.sum() > maxf1.sum():
            maxf1 = mean_f1
            maxf1std = std_f1
            f1params = paramsgrid

    return maxconmatn, maxconmatnstd, conmatnparams, maxconmat, maxconmatstd, conmatparams, maxf1, maxf1std, f1params


def eval(model, test_loader, name):
    model.eval()
    test_loss = 0
    outputfull = None
    targetfull = None
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data) #torch.reshape(, (-1,))
            if outputfull is not None:
                outputfull = torch.cat((outputfull, output), 0)
                targetfull = torch.cat((targetfull, target), 0)
            else:
                outputfull = output
                targetfull = target
            #output_class = torch.Tensor([a.index(max(a)) for a in output.tolist()])
            test_loss += F.cross_entropy(output.type(torch.float), target.type(torch.int64)).item()
    test_loss /= len(test_loader.dataset)
    print(f"{name} loss: {test_loss}")
    output_class = [a.index(max(a)) for a in outputfull.tolist()]
    con_mat_n = confusion_matrix(targetfull.type(torch.int).tolist(), output_class, labels=range(9), normalize="true")
    con_mat = confusion_matrix(targetfull.type(torch.int).tolist(), output_class, labels=range(9), normalize=None)
    f1 = f1_score(targetfull.type(torch.int).tolist(), output_class, labels=range(9), average="micro")
    return con_mat_n, con_mat, f1

if __name__ == "__main__":
    df = data_to_df("mushrooms_data.txt")
    df = odor_to_tag(df)
    train, test, train_tag, test_tag = split_data_and_tags(df)
    paramdict = {"hidden_size_01": 256, "hidden_size_02": 64, "lr": 0.0001, "batch_size": 32, "n_epoch": 30}
    neural_network(train, train_tag, test, train_tag, 0.8, paramdict)
