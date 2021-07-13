import csv
import numpy as np
import copy

import nni
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from Preprocess.Preprocces_whole_data import *
from Preprocess.split_data import split_data_and_tags

from Learning_methods.NN_funcs_and_classes import loading_data, calculate_weighted_in_train, Model_multiClass


def train_nn(model, train_loader, y_train, optimizer, device, weighted_lst):
    loss_train_total = 0
    y_pred_train_lst = []  # for calculate f1
    y_batch_total = []

    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(x_batch)

        loss = F.cross_entropy(y_pred, y_batch.long(), weight=weighted_lst.to(device))

        loss.backward()
        optimizer.step()

        loss_train_total += loss.item()

        y_pred_train_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_train_softmax, dim=1)

        y_pred_train_lst.extend(y_pred_tags.detach().cpu().numpy())
        y_batch_total.extend(y_batch.cpu().numpy())  # save for comparing the prediction (because shuffle, so y_train is not good to compare)

    sample_weight = [weighted_lst[int(i.item())] for i in y_batch_total]
    f1_micro = f1_score(y_batch_total, y_pred_train_lst, average='micro', sample_weight=sample_weight)
    f1_macro = f1_score(y_batch_total, y_pred_train_lst, average='macro', sample_weight=sample_weight)

    return loss_train_total, f1_micro, f1_macro


def validation(model, validation_loader, y_validation, device, weighted_lst):
    loss_val_total = 0
    y_pred_val_lst = []  # for calculate f1

    model.eval()
    with torch.no_grad():
        for x_batch_val, y_batch_val in validation_loader:
            x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
            y_pred_val = model(x_batch_val)

            loss_val = F.cross_entropy(y_pred_val, y_batch_val.long(), weight=weighted_lst.to(device))

            loss_val_total += loss_val.item()

            y_pred_val_softmax = torch.log_softmax(y_pred_val, dim=1)
            _, y_pred_val_tags = torch.max(y_pred_val_softmax, dim=1)

            y_pred_val_lst.extend(y_pred_val_tags.detach().cpu().numpy())

    sample_weight = [weighted_lst[int(i.item())] for i in y_validation.ravel()]
    f1_micro = f1_score(y_validation.ravel(), y_pred_val_lst, average='micro', sample_weight=sample_weight)
    f1_macro = f1_score(y_validation.ravel(), y_pred_val_lst, average='macro', sample_weight=sample_weight)


    return loss_val_total, f1_micro, f1_macro, model


def neural_net(train_data, train_tag, val_data, val_tag, epoch, batch_size, learning_rate, weight_decay_optimizer,
               dropout, activation_function, hidden_layer1_size, hidden_layer2_size, char_num=94, classes=9, is_nni=False):

    weighted_lst = calculate_weighted_in_train(train_tag, options=classes)
    train_loader, validation_loader = loading_data(train_data, train_tag, val_data, val_tag, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    model = Model_multiClass(char_num, hidden_layer1_size, hidden_layer2_size,
                             classes, activation_function, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_optimizer)

    best_f1mic = 0
    tr_f1mic_in_best_f1mic_test = 0
    f1mac_tr_in_best_f1mic_test = 0  # add to return
    f1mac_ts_in_best_f1mic_test = 0  # add to return
    best_model = None

    for e in range(1, epoch + 1):
        loss_train_total, f1_micro_train, f1_macro_train = train_nn(model, train_loader, train_tag, optimizer, device, weighted_lst)
        loss_val_total, f1_micro_val, f1_macro_val, temp_model = validation(model, validation_loader, val_tag, device, weighted_lst)

        print(f'Epoch {e + 0:03}: | Loss Train: {loss_train_total / len(train_loader):.7f} | '
              f'Loss Val: {loss_val_total / len(validation_loader):.7f} | '
              f'F1 Micro Train: {f1_micro_train:.5f} | F1 Micro Val: {f1_micro_val:.5f} | '
              f'F1 Macro Train: {f1_macro_train:.5f} | F1 Macro Val: {f1_macro_val:.5f}')

        if f1_micro_val > best_f1mic:  # save the best measures and the best model
            best_f1mic = f1_micro_val
            tr_f1mic_in_best_f1mic_test = f1_micro_train
            f1mac_tr_in_best_f1mic_test = f1_macro_train
            f1mac_ts_in_best_f1mic_test = f1_macro_val
            best_model = copy.deepcopy(temp_model)

        if is_nni:
            nni.report_intermediate_result(best_f1mic)  # report after each epoch

    if is_nni:
        nni.report_final_result(best_f1mic)  # report in the end of trail running
        return None, None, None
    else:
        return best_f1mic, tr_f1mic_in_best_f1mic_test, f1mac_ts_in_best_f1mic_test, f1mac_tr_in_best_f1mic_test, best_model


def running_cross_validation(train, test, train_tag, test_tag, params: list, cv_num=5,):
    """
    check stability of parameters by cross-validation
    """
    # the params order is: epoch, batch_size, learning_rate, dropout, hidden_layer1_size, hidden_layer2_size, activation_function, weight_decay_optimizer

    epoch, batch_size, learning_rate, dropout, hidden_layer1_size, hidden_layer2_size, \
    activation_function, weight_decay_optimizer = params[0], params[1], params[2], params[3], \
                                                  params[4], params[5], params[6], params[7]

    single_f1mic_ts_total, single_f1mic_tr_total, single_f1mac_ts_total, single_f1mac_tr_total = 0, 0, 0, 0
    for i in range(cv_num):
        f1mic_test, f1mic_tr, f1mac_ts, f1mac_tr, best_model = neural_net(train, train_tag, test, test_tag,
                                                                          epoch, batch_size, learning_rate, weight_decay_optimizer, dropout, activation_function,
                                                                          hidden_layer1_size, hidden_layer2_size, is_nni=False)
        single_f1mic_ts_total += f1mic_test
        single_f1mic_tr_total += f1mic_tr
        single_f1mac_ts_total += f1mac_ts
        single_f1mac_tr_total += f1mac_tr

        # torch.save(best_model.state_dict(), f'../models_best_params/multi/model_params_{ind_param}_cv_num_{i}.pth')

    print(f'avg train f1mic: {single_f1mic_tr_total / cv_num}, avg val f1mic: {single_f1mic_ts_total / cv_num}'
          f' avg train f1mac: {single_f1mac_tr_total / cv_num}, avg val f1mac: {single_f1mac_ts_total / cv_num}')



def running_nni(train, test, train_tag, test_tag):
    params = nni.get_next_parameter()
    epoch = params["epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["lr"]
    dropout = params["dropout"]
    hidden_layer1_size = params["hidden_size_01"]
    hidden_layer2_size = params["hidden_size_02"]
    if params["activation_function"] == "relu":
        activation_function = nn.ReLU()
    elif params["activation_function"] == "tanh":
        activation_function = nn.Tanh()
    elif params["activation_function"] == "elu":
        activation_function = nn.ELU()
    weight_decay_optimizer = params["WDO"]


    neural_net(train, train_tag, test, test_tag, epoch, batch_size,
               learning_rate, weight_decay_optimizer, dropout, activation_function,
               hidden_layer1_size, hidden_layer2_size, is_nni=True)


if __name__ == '__main__':
    df = data_to_df("mushrooms_data.txt")
    df = odor_to_tag(df)
    train, test, train_tag, test_tag = split_data_and_tags(df)
    train, test, train_tag, test_tag = train.to_numpy(), test.to_numpy(), train_tag.to_numpy(), test_tag.to_numpy()
    realtrain, val, realtrain_tag, val_tag = train_test_split(train, train_tag, train_size=0.8)
    running_nni(realtrain, val, realtrain_tag, val_tag)
    #running_cross_validation(train, test, train_tag, test_tag)







