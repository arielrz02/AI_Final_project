import pandas as pd
import numpy as np
from typing import Tuple
from itertools import product
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from Preprocess.preprocess_full import prep_missing_data, prep_whole_data
from Preprocess.split_data import split_data_and_tags


def rf_cross_val(train_data: pd.DataFrame, tags: pd.Series, rfparams: dict, fold=5):
    f1_mac_lst = []
    f1_mic_lst = []
    for i in range(fold):
        train_and_val = train_test_split(train_data, tags, test_size=1 / fold)
        f1_mac, f1_mic = rf_single_hyperparams(*train_and_val, rfparams=rfparams)
        f1_mac_lst.append(f1_mac)
        f1_mic_lst.append(f1_mic)

    f1_mac_lst = np.array(f1_mac_lst)
    mean_mac_f1 = f1_mac_lst.mean(axis=0)
    std_mac_f1 = f1_mac_lst.std(axis=0)

    f1_mic_lst = np.array(f1_mic_lst)
    mean_mic_f1 = f1_mic_lst.mean(axis=0)
    std_mic_f1 = f1_mic_lst.std(axis=0)
    return mean_mac_f1, std_mac_f1, mean_mic_f1, std_mic_f1


def rf_single_hyperparams(train_data: pd.DataFrame, test: pd.DataFrame, train_tag: pd.Series,
                          test_tag: pd.Series, rfparams: dict) -> Tuple[np.ndarray, np.ndarray]:
    rf_single_model = RandomForestClassifier(bootstrap=True, n_jobs=-1, **rfparams)
    rf_model = OneVsRestClassifier(rf_single_model).fit(train_data, train_tag)
    prediction = rf_model.predict(test)
    return evaluate(prediction, test_tag)


def evaluate(prediction, tag) -> Tuple[np.ndarray, np.ndarray]:
    f1_mac = f1_score(prediction, tag, labels=range(9), average="macro")
    f1_mic = f1_score(prediction, tag, labels=range(9), average="micro")
    return f1_mac, f1_mic

def choose_rf_params(df: pd.DataFrame, tags: pd.Series):
    n_estimators_lst = [int(x) for x in np.linspace(start=100, stop=1300, num=7)]
    max_features_lst = ['auto', 'sqrt']
    max_depth_lst = [int(x) for x in np.linspace(10, 100, num=10)]
    min_split_lst = [2, 5, 10]
    min_leaf_lst = [1, 2, 4]

    maxmacf1 = 0
    maxmicf1 = 0

    for n_est, max_feat, max_depth, min_splt, min_leaf in tqdm(product(n_estimators_lst,
        max_features_lst, max_depth_lst, min_split_lst, min_leaf_lst), total=1260):

        paramsgrid = {"n_estimators": n_est, "max_features": max_feat, "max_depth": max_depth,
                      "min_samples_split": min_splt, "min_samples_leaf": min_leaf}

        mean_mac_f1, std_mac_f1, mean_mic_f1, std_mic_f1 = rf_cross_val(df, tags, paramsgrid)

        if mean_mic_f1 > maxmicf1:
            maxmicf1 = mean_mic_f1
            micf1std = std_mic_f1
            f1_mic_params = paramsgrid

        if mean_mac_f1 > maxmacf1:
            maxmacf1 = mean_mac_f1
            macf1std = std_mac_f1
            f1_mac_params = paramsgrid

    return maxmicf1, micf1std, f1_mic_params, maxmacf1, macf1std, f1_mac_params



if __name__ == "__main__":
    train, test, train_tags, test_tags = prep_missing_data()
    res = rf_single_hyperparams(train, test, train_tags, test_tags, {'n_estimators': 500, 'max_features': 'sqrt',
                                                                   'max_depth': 20, 'min_samples_split': 2,
                                                                   'min_samples_leaf': 4})
    print(res)
    # params = choose_rf_params(train, train_tag)
    # print(f"max of f1 micro with params {params[2]}")
    # print(f"max of f1 macro with params {params[5]}")
    # print(rf_single_hyperparams(train, test, train_tag, test_tag, params[2]))
    # print(rf_single_hyperparams(train, test, train_tag, test_tag, params[5]))
