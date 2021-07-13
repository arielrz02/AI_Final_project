import csv
import numpy as np
import copy

import nni
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score


from aux_functions import process_data, loading_data, calculate_weighted_in_train, \
    process_y_only, loading_data_only_test


class Model_multiClass(nn.Module):
    def __init__(self, char_num,  hid1_size, hid2_size, num_options, activation_fun, dropout):
        super(Model_multiClass, self).__init__()

        self.layer_1 = nn.Linear(char_num, hid1_size)
        self.layer_2 = nn.Linear(hid1_size, hid2_size)
        self.layer_out = nn.Linear(hid2_size, num_options)

        self.activation = activation_fun
        self.dropout = nn.Dropout(p=dropout)
        # self.batchnorm1 = nn.BatchNorm1d(hid1_size)
        # self.batchnorm2 = nn.BatchNorm1d(hid2_size)

    def forward(self, inputs):

        x = self.activation(self.layer_1(inputs))
        # x = self.batchnorm1(x)
        x = self.activation(self.layer_2(x))
        # x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def train(model, train_loader, y_train, optimizer, device, weighted_lst):
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


def main(df_X, df_Y, epoch_, batch_size_, learning_rate_, weight_decay_optimizer_, dropout_, activation_function_,
         hidden_layer1_size_, hidden_layer2_size_, is_nni=False):
    df_x, df_y = process_data(df_X, df_Y)
    x_train, x_validation, y_train, y_validation = train_test_split(df_x, df_y, test_size=0.2, stratify=df_y)
    weighted_lst = calculate_weighted_in_train(y_train)
    train_loader, validation_loader = loading_data(x_train, y_train, x_validation, y_validation, batch_size_)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    model = Model_multiClass(hidden_layer1_size_, hidden_layer2_size_, activation_function_, dropout_).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_, weight_decay=weight_decay_optimizer_)

    best_f1mic = 0
    tr_f1mic_in_best_f1mic_test = 0
    f1mac_tr_in_best_f1mic_test = 0  # add to return
    f1mac_ts_in_best_f1mic_test = 0  # add to return
    best_model = None

    for e in range(1, epoch_ + 1):
        loss_train_total, f1_micro_train, f1_macro_train = train(model, train_loader, y_train, optimizer, device, weighted_lst)
        loss_val_total, f1_micro_val, f1_macro_val, temp_model = validation(model, validation_loader, y_validation, device, weighted_lst)

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


def running_check_stability(cv_num=5):
    """
    check stability of parameters by cross-validation
    """
    # the params order is: epoch, batch_size, learning_rate, dropout, hidden_layer1_size, hidden_layer2_size, activation_function, weight_decay_optimizer
    params1 = [120, 128, 0.001, 0.1, 250, 100, nn.ReLU(), 0.000001]
    params2 = [120, 64, 0.0001, 0.3, 200, 50, nn.Tanh(), 0.0000001]

    # average measures (lists of the cross-validation): f1 micro test, f1 micro train, f1 macro in test, f1 macro in train
    # test here means to validation
    avg_f1mic_ts_lst, avg_f1mic_tr_lst, avg_f1mac_ts_lst, avg_f1mac_tr_lst = [], [], [], []

    for ind_param, params_set in enumerate([params1, params2]):
        epoch, batch_size, learning_rate, dropout, hidden_layer1_size, hidden_layer2_size, \
        activation_function, weight_decay_optimizer = params_set[0], params_set[1], params_set[2], params_set[3], \
                                                      params_set[4], params_set[5], params_set[6], params_set[7]

        single_f1mic_ts_total, single_f1mic_tr_total, single_f1mac_ts_total, single_f1mac_tr_total = 0, 0, 0, 0
        for i in range(cv_num):
            f1mic_test, f1mic_tr, f1mac_ts, f1mac_tr, best_model = main('../data/split_external_test_multi/train_x.csv',
                     '../data/split_external_test_multi/train_y.csv',
                     epoch, batch_size, learning_rate, weight_decay_optimizer, dropout, activation_function,
                     hidden_layer1_size, hidden_layer2_size, is_nni=False)
            single_f1mic_ts_total += f1mic_test
            single_f1mic_tr_total += f1mic_tr
            single_f1mac_ts_total += f1mac_ts
            single_f1mac_tr_total += f1mac_tr

            # torch.save(best_model.state_dict(), f'../models_best_params/multi/model_params_{ind_param}_cv_num_{i}.pth')

        print(f'params: {ind_param}, avg train f1mic: {single_f1mic_tr_total / cv_num}, avg val f1mic: {single_f1mic_ts_total / cv_num}'
              f' avg train f1mac: {single_f1mac_tr_total / cv_num}, avg val f1mac: {single_f1mac_ts_total / cv_num}')
        avg_f1mic_ts_lst.append(single_f1mic_ts_total / cv_num)
        avg_f1mic_tr_lst.append(single_f1mic_tr_total / cv_num)
        avg_f1mac_ts_lst.append(single_f1mac_ts_total / cv_num)
        avg_f1mac_tr_lst.append(single_f1mac_tr_total / cv_num)

    print('avg_f1mic_train_lst:')
    print(avg_f1mic_tr_lst)
    print('avg_f1mic_val_lst:')
    print(avg_f1mic_ts_lst)
    print('avg_f1mac_train_lst:')
    print(avg_f1mac_tr_lst)
    print('avg_f1mac_val_lst:')
    print(avg_f1mac_ts_lst)



def running_nni():
    params = nni.get_next_parameter()

    epoch = 120
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    hidden_layer1_size = params["hidden_layer1_size"]
    hidden_layer2_size = params["hidden_layer2_size"]
    if params["activation_function"] == "relu":
        activation_function = nn.ReLU()
    elif params["activation_function"] == "tanh":
        activation_function = nn.Tanh()
    elif params["activation_function"] == "elu":
        activation_function = nn.ELU()
    weight_decay_optimizer = params["weight_decay_optimizer"]


    _, _, _, _, _ = main('../data/Non_Normalized_fillna_remove_corr.csv', '../data/MultiClass_labels.csv',
         epoch, batch_size, learning_rate, weight_decay_optimizer, dropout, activation_function, hidden_layer1_size,
         hidden_layer2_size, is_nni=True)


def eval_best_models(test_x, test_y, y_train, batch_size_=128):
    params_models = [[250, 100, nn.ReLU()],
                     [200, 50, nn.Tanh()]]

    test_x, test_y = process_data(test_x, test_y)

    y_train = process_y_only(y_train)  # for weighted
    weighted_lst = calculate_weighted_in_train(y_train)

    test_loader = loading_data_only_test(test_x, test_y, batch_size_)

    for idx_model, params_models in enumerate(params_models):
        path = f'../models_best_params/multi/model_params_{idx_model}_cv_num_0.pth'
        hid1_size, hid2_size, activation_fun = params_models[0], params_models[1], params_models[2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(device))

        model = Model_multiClass(hid1_size, hid2_size, activation_fun, dropout=0).to(device)
        model.load_state_dict(torch.load(path))

        loss_ts_total = 0
        y_pred_ts_lst = []  # for calculate auc
        probs_for_auc = []

        model.eval()
        with torch.no_grad():
            for x_batch_ts, y_batch_ts in test_loader:
                x_batch_ts, y_batch_ts = x_batch_ts.to(device), y_batch_ts.to(device)
                y_pred_val = model(x_batch_ts)

                loss_ts = F.cross_entropy(y_pred_val, y_batch_ts.long(), weight=weighted_lst.to(device))

                loss_ts_total += loss_ts.item()

                y_pred_val_softmax = torch.log_softmax(y_pred_val, dim=1)
                _, y_pred_val_tags = torch.max(y_pred_val_softmax, dim=1)

                y_pred_ts_lst.extend(y_pred_val_tags.detach().cpu().numpy())

                y_pred_softmax_to_auc = torch.softmax(y_pred_val, dim=1)
                pred_to_auc, _ = torch.max(y_pred_softmax_to_auc, dim=1)

                probs_for_auc.extend(pred_to_auc.detach().cpu().numpy())

        sample_weight = [weighted_lst[int(i.item())] for i in test_y.ravel()]
        f1_micro = f1_score(test_y.ravel(), y_pred_ts_lst, average='micro', sample_weight=sample_weight)
        f1_macro = f1_score(test_y.ravel(), y_pred_ts_lst, average='macro', sample_weight=sample_weight)

        print(f'params: {idx_model}, f1_micro: {f1_micro}, f1_macro: {f1_macro}')


if __name__ == '__main__':
    # running_nni()
    running_check_stability()
    # eval_best_models('../data/split_external_test_multi/test_x.csv', '../data/split_external_test_multi/test_y.csv',
    #                  '../data/split_external_test_multi/train_y.csv')







