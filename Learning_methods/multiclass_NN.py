import nni
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score

from Learning_methods.NN_funcs_and_classes import loading_data, calculate_weighted_in_train, Model_multiClass

"""
Running the neural network on the training data.
"""
def train_nn(model, train_loader, optimizer, device, weighted_lst):
    loss_train_total = 0
    # for calculating f1
    y_pred_train_lst = []
    y_batch_total = []

    model.train()
    # calculating loss and using it to improve the model.
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
        y_batch_total.extend(y_batch.cpu().numpy())

    # calculating f1.
    sample_weight = [weighted_lst[int(i.item())] for i in y_batch_total]
    f1_micro = f1_score(y_batch_total, y_pred_train_lst, average='micro', sample_weight=sample_weight)
    f1_macro = f1_score(y_batch_total, y_pred_train_lst, average='macro', sample_weight=sample_weight)

    return loss_train_total, f1_micro, f1_macro


"""
Running our NN on the validation.
"""
def validation(model, validation_loader, y_validation, device, weighted_lst):
    loss_val_total = 0
    # for calculating f1.
    y_pred_val_lst = []

    model.eval()
    # calculating loss.
    with torch.no_grad():
        for x_batch_val, y_batch_val in validation_loader:
            x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
            y_pred_val = model(x_batch_val)

            loss_val = F.cross_entropy(y_pred_val, y_batch_val.long(), weight=weighted_lst.to(device))

            loss_val_total += loss_val.item()

            y_pred_val_softmax = torch.log_softmax(y_pred_val, dim=1)
            _, y_pred_val_tags = torch.max(y_pred_val_softmax, dim=1)

            y_pred_val_lst.extend(y_pred_val_tags.detach().cpu().numpy())

    # calculating f1.
    sample_weight = [weighted_lst[int(i.item())] for i in y_validation.ravel()]
    f1_micro = f1_score(y_validation.ravel(), y_pred_val_lst, average='micro', sample_weight=sample_weight)
    f1_macro = f1_score(y_validation.ravel(), y_pred_val_lst, average='macro', sample_weight=sample_weight)


    return loss_val_total, f1_micro, f1_macro, model


"""
The neural network itself.
"""
def neural_net(train_data, val_data, train_tag, val_tag, epochs=20, batch_size=64, lr=0.01, WDO=1e-7,
               dropout=0.3, activation_function=nn.ELU(), hidden_size_01=256, hidden_size_02=32,
               charsize=94, classes=9, is_nni=False):

    # getting the weights and pytorch tensors.
    weighted_lst = calculate_weighted_in_train(train_tag, options=classes)
    train_loader, validation_loader = loading_data(train_data, train_tag, val_data, val_tag, batch_size)

    # trying to use GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    # getting our model and optimizer.
    model = Model_multiClass(charsize, hidden_size_01, hidden_size_02,
                             classes, activation_function, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WDO)

    best_f1mic = 0
    tr_f1mic_in_best_f1mic_test = 0
    f1mac_tr_in_best_f1mic_test = 0

    # running the epochs on the NN.
    for e in range(1, epochs + 1):
        # running on test and validation and getting results.
        loss_train_total, f1_micro_train, f1_macro_train = train_nn(model, train_loader, optimizer, device, weighted_lst)
        loss_val_total, f1_micro_val, f1_macro_val, temp_model = validation(model, validation_loader, val_tag, device, weighted_lst)

        print(f'Epoch {e + 0:03}: | Loss Train: {loss_train_total / len(train_loader):.7f} | '
              f'Loss Val: {loss_val_total / len(validation_loader):.7f} | '
              f'F1 Micro Train: {f1_micro_train:.5f} | F1 Micro Val: {f1_micro_val:.5f} | '
              f'F1 Macro Train: {f1_macro_train:.5f} | F1 Macro Val: {f1_macro_val:.5f}')

        # save the best measures.
        if f1_micro_val > best_f1mic:
            best_f1mic = f1_micro_val
            tr_f1mic_in_best_f1mic_test = f1_micro_train
            f1mac_tr_in_best_f1mic_test = f1_macro_train
            f1mac_ts_in_best_f1mic_test = f1_macro_val

        if is_nni:
            # report after each epoch.
            nni.report_intermediate_result(best_f1mic)

    if is_nni:
        # report in the end of trail running.
        nni.report_final_result(best_f1mic)
        return None, None, None
    else:
        # returning the best f1 we got.
        return best_f1mic, tr_f1mic_in_best_f1mic_test, f1mac_ts_in_best_f1mic_test, f1mac_tr_in_best_f1mic_test

"""
check stability of parameters by cross-validation
"""
def running_cross_validation(train, test, train_tag, test_tag, params: list, cv_num=5, charsize=94):

    # the params order is: epoch, batch_size, learning_rate, dropout, hidden_layer1_size,
    # hidden_layer2_size, activation_function, weight_decay_optimizer

    epoch, batch_size, learning_rate, dropout, hidden_layer1_size, hidden_layer2_size, \
    activation_function, weight_decay_optimizer = params[0], params[1], params[2], params[3], \
                                                  params[4], params[5], params[6], params[7]

    single_f1mic_ts_total, single_f1mic_tr_total, single_f1mac_ts_total, single_f1mac_tr_total = 0, 0, 0, 0
    # running the cross validation.
    for i in range(cv_num):
        f1mic_test, f1mic_tr, f1mac_ts, f1mac_tr = neural_net(train, test, train_tag, test_tag,
                                                                          epoch, batch_size, learning_rate, weight_decay_optimizer, dropout, activation_function,
                                                                          hidden_layer1_size, hidden_layer2_size, charsize=charsize, is_nni=False)
        # summing the results.
        single_f1mic_ts_total += f1mic_test
        single_f1mic_tr_total += f1mic_tr
        single_f1mac_ts_total += f1mac_ts
        single_f1mac_tr_total += f1mac_tr

    # returning the average in every metric.
    print(f'avg train f1mic: {single_f1mic_tr_total / cv_num}, avg val f1mic: {single_f1mic_ts_total / cv_num}'
          f' avg train f1mac: {single_f1mac_tr_total / cv_num}, avg val f1mac: {single_f1mac_ts_total / cv_num}')


"""
Running an nni to find the best parameters.
"""
def running_nni(train, test, train_tag, test_tag, charsize=94):
    params = nni.get_next_parameter()
    # translating the activation function name to the function.
    if params["activation_function"] == "relu":
        params["activation_function"] = nn.ReLU()
    elif params["activation_function"] == "tanh":
        params["activation_function"] = nn.Tanh()
    elif params["activation_function"] == "elu":
        params["activation_function"] = nn.ELU()

    neural_net(train, test, train_tag, test_tag, **params, charsize=charsize, is_nni=True)
