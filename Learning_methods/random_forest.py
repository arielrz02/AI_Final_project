import pandas as pd
import numpy as np
from typing import Tuple
from itertools import product
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


"""
Using cross validation on the random forest and returning
the average and standard deviation of the results.
"""
def rf_cross_val(train_data: pd.DataFrame, tags: pd.Series, rfparams: dict, fold=5):
    f1_mac_lst = []
    f1_mic_lst = []
    # running the random forest "fold" times.
    for i in range(fold):
        train_and_val = train_test_split(train_data, tags, test_size=1 / fold)
        f1_mac, f1_mic = rf_single_hyperparams(*train_and_val, rfparams=rfparams)
        f1_mac_lst.append(f1_mac)
        f1_mic_lst.append(f1_mic)

    # computing average and std.
    f1_mac_lst = np.array(f1_mac_lst)
    mean_mac_f1 = f1_mac_lst.mean(axis=0)
    std_mac_f1 = f1_mac_lst.std(axis=0)

    f1_mic_lst = np.array(f1_mic_lst)
    mean_mic_f1 = f1_mic_lst.mean(axis=0)
    std_mic_f1 = f1_mic_lst.std(axis=0)
    return mean_mac_f1, std_mac_f1, mean_mic_f1, std_mic_f1

"""
A single run of the random forest.
"""
def rf_single_hyperparams(train_data: pd.DataFrame, test: pd.DataFrame, train_tag: pd.Series,
                          test_tag: pd.Series, rfparams: dict) -> Tuple[np.ndarray, np.ndarray]:
    # first creating the singular model, then using it in a one VS rest model.
    rf_single_model = RandomForestClassifier(bootstrap=True, n_jobs=-1, **rfparams)
    rf_model = OneVsRestClassifier(rf_single_model).fit(train_data, train_tag)
    # predicting and evaluating the results.
    prediction = rf_model.predict(test)
    return evaluate(prediction, test_tag)

"""
Calculating f1 scores, both macro and micro, as evaluation.
"""
def evaluate(prediction, tag) -> Tuple[np.ndarray, np.ndarray]:
    f1_mac = f1_score(prediction, tag, labels=range(9), average="macro")
    f1_mic = f1_score(prediction, tag, labels=range(9), average="micro")
    return f1_mac, f1_mic


"""
Choosing the optimal parameters for the random forest using grid search.
"""
def choose_rf_params(df: pd.DataFrame, tags: pd.Series):
    # the five parameters we are using to maximize.
    n_estimators_lst = [int(x) for x in np.linspace(start=100, stop=1300, num=7)]
    max_features_lst = ['log2', 'sqrt']
    max_depth_lst = [int(x) for x in np.linspace(10, 100, num=10)]
    min_split_lst = [2, 5, 10]
    min_leaf_lst = [1, 2, 4]

    maxmacf1 = 0
    maxmicf1 = 0
    # running on all possible combinations.
    for n_est, max_feat, max_depth, min_splt, min_leaf in tqdm(product(n_estimators_lst,
        max_features_lst, max_depth_lst, min_split_lst, min_leaf_lst), total=1260):

        paramsgrid = {"n_estimators": n_est, "max_features": max_feat, "max_depth": max_depth,
                      "min_samples_split": min_splt, "min_samples_leaf": min_leaf}

        # running the model with cross validation, to get a more accurate score.
        mean_mac_f1, std_mac_f1, mean_mic_f1, std_mic_f1 = rf_cross_val(df, tags, paramsgrid)

        # saving the best parameters and their score.
        if mean_mic_f1 > maxmicf1:
            maxmicf1 = mean_mic_f1
            micf1std = std_mic_f1
            f1_mic_params = paramsgrid

        if mean_mac_f1 > maxmacf1:
            maxmacf1 = mean_mac_f1
            macf1std = std_mac_f1
            f1_mac_params = paramsgrid

    # returning both the best f1 micro and f1 macro scores, and their repective parameters.
    return maxmicf1, micf1std, f1_mic_params, maxmacf1, macf1std, f1_mac_params
