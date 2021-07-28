import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

"""
A basic dataset for pytorch.
"""
class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


"""
The model that we use for our network.
"""
class Model_multiClass(nn.Module):
    """
    The building of the NN
    """
    def __init__(self, char_num,  hid1_size, hid2_size, num_options, activation_fun, dropout):
        super(Model_multiClass, self).__init__()

        # two layer nn with dropout
        self.layer_1 = nn.Linear(char_num, hid1_size)
        self.layer_2 = nn.Linear(hid1_size, hid2_size)
        self.layer_out = nn.Linear(hid2_size, num_options)

        self.activation = activation_fun
        self.dropout = nn.Dropout(p=dropout)

    """
    The actual NN guessing procces
    """
    def forward(self, inputs):

        x = self.activation(self.layer_1(inputs))
        x = self.activation(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


"""
Load data to pytorch tensors
"""
def loading_data(train_data, train_tag, val_data, val_tag, batch_size):
    train_data = Dataset(torch.FloatTensor(train_data.values), torch.FloatTensor(train_tag.values))
    validation_data = Dataset(torch.FloatTensor(val_data.values), torch.FloatTensor(val_tag.values))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader


"""
For weighted in loss function (in multi classification)
"""
def calculate_weighted_in_train(train_tag, options):
    weighted = np.zeros(options)
    # adding up the counts of each option.
    for i in range(options):
        weighted[i] = list(train_tag).count(i)

    return torch.FloatTensor(weighted)
